#!/usr/bin/env python3
"""
Analyze confidence intervals of relative errors from batch KNN results.

If the input ``batch_knn_results_*.csv`` has a ``tier`` column (currently
emitted only for type 5 anniversary events), CIs are computed per
``(step, tier, confidence_level)``. Otherwise the legacy single-CI-per-step
behaviour applies (treated as ``tier=0``).
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import sys
import glob
from pathlib import Path


def _conformal_interval(values, confidence_level):
    """Two-sided interval with a finite-sample (conformal) coverage guarantee.

    Given signed relative errors ``values`` and a target ``confidence_level``
    (e.g. 90), returns ``(lower, upper)`` such that, marginally, at least
    ``confidence_level``% of future errors are expected to fall inside.

    Why not plain np.percentile:
      - The previous implementation rounded the percentiles *inward*
        (method='higher' for lower, 'lower' for upper), shrinking the band
        and under-covering.
      - Plain percentiles also ignore that we estimate the tail from a finite
        sample, so the nominal 90th percentile of N points under-estimates the
        true 90th percentile.

    This uses the split-conformal order statistic with the ``(n+1)`` finite-
    sample correction and rounds *outward*:
        k_low  = floor((n+1) * alpha/2)
        k_high = ceil((n+1) * (1 - alpha/2))
    both clamped to [1, n]. For small n the bounds widen toward the observed
    min/max, which is the correct conservative behaviour.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return None
    alpha = 1.0 - confidence_level / 100.0
    k_low = int(np.floor((n + 1) * (alpha / 2.0)))
    k_high = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
    k_low = min(max(k_low, 1), n)
    k_high = min(max(k_high, 1), n)
    return float(v[k_low - 1]), float(v[k_high - 1])


def analyze_confidence_intervals_for_group(df_group, confidence_levels=[75, 90]):
    """Per-(step, tier, confidence_level) bounds from raw rel_errors.

    Returns ``{(step, tier, confidence_level): [lower, upper]}``. ``tier`` is
    ``0`` for non-anniversary inputs (no ``tier`` column or all zeros).

    Bounds use ``_conformal_interval`` (finite-sample, outward-rounded) so the
    nominal coverage is met more closely than plain inward-rounded percentiles.
    """
    df_group = df_group.copy()
    df_group['relative_error'] = df_group['relative_error'] / 100
    if 'tier' not in df_group.columns:
        df_group['tier'] = 0

    step_intervals = {}
    for (step, tier), g in df_group.groupby(['step', 'tier']):
        rel_err = g['relative_error'].dropna()
        if len(rel_err) == 0:
            continue
        for confidence_level in confidence_levels:
            bounds = _conformal_interval(rel_err.values, confidence_level)
            if bounds is None:
                continue
            step_intervals[(int(step), int(tier), confidence_level)] = [bounds[0], bounds[1]]

    return step_intervals


def interpolate_confidence_intervals(step_intervals, confidence_levels, start_step=55, end_step=299):
    """Interpolate bounds across ``[start_step, end_step]`` per (tier, level).

    Returns ``{(step, tier, confidence_level): [lower, upper]}``.
    """
    interpolated = {}
    tiers = sorted({tier for (_step, tier, _cl) in step_intervals.keys()})

    for tier in tiers:
        for confidence_level in confidence_levels:
            step_data = [
                (step, bounds)
                for (step, t, cl), bounds in step_intervals.items()
                if t == tier and cl == confidence_level
            ]
            if len(step_data) < 2:
                continue
            steps = [item[0] for item in step_data]
            lower_bounds = [item[1][0] for item in step_data]
            upper_bounds = [item[1][1] for item in step_data]
            lower_interp = interpolate.interp1d(
                steps, lower_bounds, kind='linear',
                bounds_error=False, fill_value='extrapolate',
            )
            upper_interp = interpolate.interp1d(
                steps, upper_bounds, kind='linear',
                bounds_error=False, fill_value='extrapolate',
            )
            for step in range(start_step, end_step + 1):
                interpolated[(step, tier, confidence_level)] = [
                    float(lower_interp(step)),
                    float(upper_interp(step)),
                ]

    return interpolated


def verify_coverage_loeo(df, confidence_levels, start_step=55, end_step=299):
    """Leave-one-event-out coverage check (honest, out-of-sample).

    For each event, build the CI from all OTHER events, then measure how often
    that held-out event's actual error falls inside the band. Aggregated across
    events, this estimates the *real* coverage the CI would achieve on a new
    event — unlike in-sample coverage, which is optimistic by construction.

    Returns ``{confidence_level: realized_coverage_fraction}``.
    """
    if 'tier' not in df.columns:
        df = df.assign(tier=0)
    events = sorted(df['event_id'].unique())
    if len(events) < 2:
        return {}

    hits = {cl: 0 for cl in confidence_levels}
    total = {cl: 0 for cl in confidence_levels}

    for held in events:
        train = df[df['event_id'] != held]
        test = df[df['event_id'] == held]
        si = analyze_confidence_intervals_for_group(train, confidence_levels)
        if not si:
            continue
        interp = interpolate_confidence_intervals(si, confidence_levels, start_step, end_step)
        for _, row in test.iterrows():
            step = int(row['step'])
            tier = int(row.get('tier', 0))
            err = row['relative_error']
            if pd.isna(err):
                continue
            err_frac = err / 100.0
            for cl in confidence_levels:
                bounds = interp.get((step, tier, cl))
                if bounds is None:
                    continue
                total[cl] += 1
                if bounds[0] <= err_frac <= bounds[1]:
                    hits[cl] += 1

    return {cl: (hits[cl] / total[cl]) if total[cl] else float('nan') for cl in confidence_levels}


# ---------------------------------------------------------------------------
# Event-aware width calibration
# ---------------------------------------------------------------------------
#
# The per-(step,tier) bands above are estimated from idol-rows (~150 per cell),
# but idols within one event are correlated (an event runs hot/cold as a whole).
# The real exchangeable unit is the EVENT, of which there are very few (~4 for
# anniversaries). Estimating a quantile from ~4 events is hopeless, but a single
# WIDTH multiplier is robustly estimable: pick the smallest factor per
# confidence level such that the leave-one-event-out coverage meets nominal in
# every step bucket. Shape comes from the rows; width is pinned by the
# event-respecting LOEO objective. A well-calibrated group gets factor 1.0.

def _inflate_interval(bounds, c):
    """Widen ``[lo, hi]`` by factor ``c`` about its midpoint (keeps center)."""
    lo, hi = bounds
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return [mid - c * half, mid + c * half]


def _step_bucket(step):
    return 'early' if step <= 115 else ('mid' if step <= 215 else 'late')


def _loeo_records(df, confidence_levels, start_step=55, end_step=299):
    """Collect leave-one-event-out (cl, bucket, lo, hi, err_frac) tuples once.

    Building the CIs is the expensive part and does not depend on the width
    factor, so we collect the held-out bounds+errors once and then evaluate any
    factor cheaply against these records.
    """
    if 'tier' not in df.columns:
        df = df.assign(tier=0)
    events = sorted(df['event_id'].unique())
    recs = []
    if len(events) < 2:
        return recs
    for held in events:
        train = df[df['event_id'] != held]
        test = df[df['event_id'] == held]
        si = analyze_confidence_intervals_for_group(train, confidence_levels)
        if not si:
            continue
        interp = interpolate_confidence_intervals(si, confidence_levels, start_step, end_step)
        for _, row in test.iterrows():
            err = row['relative_error']
            if pd.isna(err):
                continue
            step = int(row['step'])
            tier = int(row.get('tier', 0))
            ef = err / 100.0
            for cl in confidence_levels:
                b = interp.get((step, tier, cl))
                if b is not None:
                    recs.append((cl, _step_bucket(step), b[0], b[1], ef))
    return recs


def _coverage_from_records(recs, cl, c, bucket=None):
    """Realized coverage for confidence level ``cl`` at width factor ``c``."""
    hit = tot = 0
    for rcl, bk, lo, hi, e in recs:
        if rcl != cl or (bucket is not None and bk != bucket):
            continue
        ilo, ihi = _inflate_interval((lo, hi), c)
        tot += 1
        if ilo <= e <= ihi:
            hit += 1
    return (hit / tot if tot else float('nan')), tot


def calibrate_width_factors(df, confidence_levels, start_step=55, end_step=299,
                            margin=0.0, max_factor=4.0):
    """Smallest per-level width multiplier so LOEO coverage >= nominal in every
    step bucket. Returns ``({cl: factor}, records)``. Factor 1.0 means the group
    is already calibrated.
    """
    recs = _loeo_records(df, confidence_levels, start_step, end_step)
    factors = {cl: 1.0 for cl in confidence_levels}
    if not recs:
        return factors, recs
    grid = [round(1.0 + 0.05 * i, 2) for i in range(0, int((max_factor - 1.0) / 0.05) + 1)]
    buckets = ('early', 'mid', 'late')
    for cl in confidence_levels:
        target = cl / 100.0 + margin
        chosen = grid[-1]
        for c in grid:
            covs = [_coverage_from_records(recs, cl, c, bk)[0] for bk in buckets]
            covs = [x for x in covs if not np.isnan(x)]
            if covs and min(covs) >= target:
                chosen = c
                break
        factors[cl] = chosen
    return factors, recs


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence intervals of relative errors for all CSV files')
    parser.add_argument('--input-dir', default='test_results', help='Directory containing CSV files (default: test_results)')
    parser.add_argument('--output-dir', default='confidence_intervals', help='Output directory (default: confidence_intervals)')
    parser.add_argument('--confidence-levels', nargs='+', type=int, default=[75, 90], help='Confidence levels to calculate (default: 75 90)')
    parser.add_argument('--start-step', type=int, default=55, help='Starting step (default: 55)')
    parser.add_argument('--end-step', type=int, default=299, help='Ending step (default: 299)')
    parser.add_argument('--verify', action='store_true',
                        help='also report leave-one-event-out (out-of-sample) coverage per file')
    parser.add_argument('--no-width-calibration', dest='calibrate_width',
                        action='store_false', default=True,
                        help='disable event-aware width calibration (on by default)')
    parser.add_argument('--coverage-margin', type=float, default=0.0,
                        help='extra coverage headroom above nominal when calibrating width '
                             '(e.g. 0.02 targets 92%% for a 90%% band)')

    args = parser.parse_args()

    try:
        csv_files = glob.glob(f"{args.input_dir}/batch_knn_results_*.csv")
        if not csv_files:
            print(f"No CSV files found in {args.input_dir}")
            return

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        for csv_file in csv_files:
            print(f"Processing {csv_file}...")

            filename = Path(csv_file).stem
            parts = filename.split('_')
            if len(parts) >= 5:
                event_type = parts[3]
                sub_event_type = parts[4]
                border = parts[5]
            else:
                print(f"  Skipping {csv_file} - invalid filename format")
                continue

            df = pd.read_csv(csv_file)

            step_intervals = analyze_confidence_intervals_for_group(df, args.confidence_levels)
            if not step_intervals:
                continue

            interpolated = interpolate_confidence_intervals(
                step_intervals, args.confidence_levels, args.start_step, args.end_step,
            )

            # Event-aware width calibration: scale each band's width by a single
            # per-confidence-level factor so leave-one-event-out coverage meets
            # nominal in every step bucket. Honest for clustered, few-event data
            # where the band shape (from rows) is fine but the width is not.
            width_factors = {cl: 1.0 for cl in args.confidence_levels}
            if args.calibrate_width and 'event_id' in df.columns \
                    and df['event_id'].nunique() >= 2:
                width_factors, _ = calibrate_width_factors(
                    df, args.confidence_levels, args.start_step, args.end_step,
                    margin=args.coverage_margin,
                )
                applied = ", ".join(f"{cl}%×{width_factors[cl]:.2f}"
                                    for cl in args.confidence_levels)
                print(f"  width calibration (event-aware): {applied}")

            tiers = sorted({tier for (_step, tier, _cl) in interpolated.keys()})
            output_data = []
            for step in range(args.start_step, args.end_step + 1):
                for tier in tiers:
                    for confidence_level in args.confidence_levels:
                        key = (step, tier, confidence_level)
                        if key in interpolated:
                            bounds = _inflate_interval(
                                interpolated[key], width_factors.get(confidence_level, 1.0))
                            output_data.append({
                                'step': step,
                                'tier': tier,
                                'confidence_level': confidence_level,
                                'rel_error_lower_bound': bounds[0],
                                'rel_error_upper_bound': bounds[1],
                            })

            if output_data:
                output_df = pd.DataFrame(output_data)
                output_filename = f"confidence_intervals_{event_type}_{sub_event_type}_{border}.csv"
                output_path = Path(args.output_dir) / output_filename
                output_df.to_csv(output_path, index=False)
                print(f"  Saved {output_filename}")

            if args.verify and 'event_id' in df.columns:
                recs = _loeo_records(df, args.confidence_levels,
                                     args.start_step, args.end_step)
                if recs:
                    buckets = ('early', 'mid', 'late')
                    print("  leave-one-event-out coverage (raw -> calibrated):")
                    for cl in args.confidence_levels:
                        c = width_factors.get(cl, 1.0)
                        raw_all = _coverage_from_records(recs, cl, 1.0)[0]
                        cal_all = _coverage_from_records(recs, cl, c)[0]
                        per_bucket = ", ".join(
                            f"{bk} {_coverage_from_records(recs, cl, c, bk)[0]*100:.0f}%"
                            for bk in buckets
                        )
                        print(f"    {cl}% (x{c:.2f}): overall {raw_all*100:.1f}% -> "
                              f"{cal_all*100:.1f}%  [{per_bucket}]  (target {cl}%)")

        print(f"\nAll results saved to {args.output_dir}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

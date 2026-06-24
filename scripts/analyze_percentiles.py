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


def analyze_confidence_intervals_for_group(df_group, confidence_levels=[75, 90]):
    """Per-(step, tier, confidence_level) bounds from raw rel_errors.

    Returns ``{(step, tier, confidence_level): [lower, upper]}``. ``tier`` is
    ``0`` for non-anniversary inputs (no ``tier`` column or all zeros).
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
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = np.percentile(rel_err, lower_percentile, method='higher')
            upper_bound = np.percentile(rel_err, upper_percentile, method='lower')
            step_intervals[(int(step), int(tier), confidence_level)] = [lower_bound, upper_bound]

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


def main():
    parser = argparse.ArgumentParser(description='Analyze confidence intervals of relative errors for all CSV files')
    parser.add_argument('--input-dir', default='test_results', help='Directory containing CSV files (default: test_results)')
    parser.add_argument('--output-dir', default='confidence_intervals', help='Output directory (default: confidence_intervals)')
    parser.add_argument('--confidence-levels', nargs='+', type=int, default=[75, 90], help='Confidence levels to calculate (default: 75 90)')
    parser.add_argument('--start-step', type=int, default=55, help='Starting step (default: 55)')
    parser.add_argument('--end-step', type=int, default=299, help='Ending step (default: 299)')

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

            tiers = sorted({tier for (_step, tier, _cl) in interpolated.keys()})
            output_data = []
            for step in range(args.start_step, args.end_step + 1):
                for tier in tiers:
                    for confidence_level in args.confidence_levels:
                        key = (step, tier, confidence_level)
                        if key in interpolated:
                            bounds = interpolated[key]
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

        print(f"\nAll results saved to {args.output_dir}/")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

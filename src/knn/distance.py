"""
Distance computation and nearest-neighbour selection.

Given a *current* partial trajectory and a pool of *candidate* partial
trajectories, this module computes a per-candidate distance and returns
the top-k nearest candidates. Candidates shorter than ``current_step``
are dropped.

The distance metric is chosen per-stage (see ``knn_stage``): RMSE, DTW,
or a mean-absolute-difference on the lookback window (``FINAL_DIFF``). If
the group's config enables ``use_trend_weighting`` and the metric is
RMSE, a trend-difference term is blended in.

Candidate assembly (``build_candidate_set``) also lives here since it is
the input side of this pipeline.
"""

import threading
import weakref
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.knn.config import DistanceMetric, get_group_config

NeighbourId = Tuple[float, float]  # (event_id, idol_id)


# ---------------------------------------------------------------------------
# Within-event percentile rank (for the rank-gap distance penalty)
# ---------------------------------------------------------------------------

# Keyed by (id(search_df), event_id, step) -> {idol_id: percentile}.
# Same weak-reference eviction pattern as stage.py's ``_scores_cache`` so
# entries disappear once the underlying dataframe is garbage collected.
_percentile_cache: Dict[Tuple[int, float, int], Dict[float, float]] = {}
_percentile_cache_lock = threading.Lock()


def compute_event_percentiles(
    search_df: pd.DataFrame, event_id: float, step: int
) -> Dict[float, float]:
    """Percentile rank of each idol within ``event_id`` at ``step``.

    ``0.0`` = highest score (best) in the event at that step, ``1.0`` = worst.
    Idols whose trajectory is shorter than ``step`` are omitted. Used by the
    rank-gap distance penalty (see ``find_nearest_neighbors``) to detect when
    a candidate's absolute-score shape match is misleading because the
    candidate was a much more (or less) elite idol within its own event than
    the target currently is within its own event -- e.g. a transient
    scoring-rate surge can push a mid-pack idol's absolute trajectory into
    the range only elite idols reach in calmer historical events.
    """
    from src.knn.stage import scores_of

    cache_key = (id(search_df), float(event_id), int(step))
    cached = _percentile_cache.get(cache_key)
    if cached is not None:
        return cached
    with _percentile_cache_lock:
        cached = _percentile_cache.get(cache_key)
        if cached is not None:
            return cached
        sub = search_df[search_df["event_id"] == event_id]
        idols = sub["idol_id"].unique()
        scores: Dict[float, float] = {}
        for iid in idols:
            arr = scores_of(search_df, event_id, iid)
            if len(arr) >= step and step > 0:
                scores[float(iid)] = float(arr[step - 1])
        if not scores:
            result: Dict[float, float] = {}
        else:
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            n = len(ordered)
            result = {iid: rank / (n - 1) if n > 1 else 0.0
                      for rank, (iid, _) in enumerate(ordered)}
        _percentile_cache[cache_key] = result
        weakref.finalize(search_df, _percentile_cache.pop, cache_key, None)
        return result


# ---------------------------------------------------------------------------
# Contemporaneous event-scale normalization (for the distance/search space)
# ---------------------------------------------------------------------------

# Keyed by (id(search_df), event_id) -> per-step scale array. Same
# weak-reference eviction pattern as the percentile cache above.
_event_scale_cache: Dict[Tuple[int, float], np.ndarray] = {}
_event_scale_cache_lock = threading.Lock()


def compute_event_scale_series(
    search_df: pd.DataFrame, event_id: float, min_score: float = 1.0
) -> np.ndarray:
    """Per-step scale (median score across idols) for ``event_id``.

    Used to make trajectories from different events comparable in the
    neighbour-search distance space even when one event is running at a
    globally higher or lower pace than another (e.g. a scoring-rate surge
    that inflates every idol's absolute score, not just one). Dividing each
    idol's raw score by its own event's contemporaneous scale removes that
    shared, event-wide magnitude shift while preserving each idol's
    *relative* position within their event -- which is what the shape
    distance is meant to compare in the first place.

    Recomputed fresh from the population at every step (not fit to a
    parametric decay curve), so a time-varying inflation factor (e.g. a
    front-loaded surge that fades) is tracked automatically without any
    extra modelling.

    Idols whose score at a step is below ``min_score`` (not yet
    meaningfully started) are excluded from that step's median so early
    steps aren't dragged toward zero by not-yet-active idols. Any leading
    steps where nobody has started yet are back-filled with the first valid
    value.
    """
    from src.knn.stage import scores_of

    cache_key = (id(search_df), float(event_id))
    cached = _event_scale_cache.get(cache_key)
    if cached is not None:
        return cached
    with _event_scale_cache_lock:
        cached = _event_scale_cache.get(cache_key)
        if cached is not None:
            return cached
        idol_ids = search_df.loc[search_df["event_id"] == event_id, "idol_id"].unique()
        trajectories = [scores_of(search_df, event_id, iid) for iid in idol_ids]
        trajectories = [t for t in trajectories if len(t) > 0]
        if not trajectories:
            result = np.array([1.0])
        else:
            max_len = max(len(t) for t in trajectories)
            scale = np.full(max_len, np.nan)
            for t in range(max_len):
                vals = [traj[t] for traj in trajectories if len(traj) > t and traj[t] >= min_score]
                if vals:
                    scale[t] = np.median(vals)
            filled = pd.Series(scale).bfill().ffill()
            result = filled.to_numpy()
            # Guard against a pathological all-zero/NaN step (shouldn't
            # happen once back/forward-filled, but keeps division safe).
            result = np.where((result > 0) & np.isfinite(result), result, 1.0)
        _event_scale_cache[cache_key] = result
        weakref.finalize(search_df, _event_scale_cache.pop, cache_key, None)
        return result


# Cache for _idol_popularity_ranking, keyed by (id(df), event_id, step,
# rank_window). The macro-regime gate calls this heavily (every idol x every
# historical event x every persistence sample step), so without memoization
# it becomes O(n_idols^2 * n_events) per prediction. Weak-reference eviction
# like the other caches in this module.
_ranking_cache: Dict[Tuple[int, float, int, int], List[Tuple[float, float]]] = {}
_ranking_cache_lock = threading.Lock()


def _idol_popularity_ranking(
    search_df: pd.DataFrame, event_id: float, current_step: int, rank_window: int,
) -> List[Tuple[float, float]]:
    """Rank an event's idols by AVERAGE score over the trailing
    ``rank_window`` steps ending at ``current_step``. Returns
    ``[(idol_id, avg_score), ...]`` sorted descending (index 0 = most
    popular/highest standing).

    Deliberately a trailing average, not cumulative growth -- cumulative
    growth is dominated by the single latest point (since growth starts
    near zero), which would reintroduce single-point ranking instability.
    """
    from src.knn.stage import scores_of

    cache_key = (id(search_df), float(event_id), int(current_step), int(rank_window))
    cached = _ranking_cache.get(cache_key)
    if cached is not None:
        return cached
    with _ranking_cache_lock:
        cached = _ranking_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _compute_idol_popularity_ranking(search_df, event_id, current_step, rank_window)
        _ranking_cache[cache_key] = result
        weakref.finalize(search_df, _ranking_cache.pop, cache_key, None)
        return result


def _compute_idol_popularity_ranking(
    search_df: pd.DataFrame, event_id: float, current_step: int, rank_window: int,
) -> List[Tuple[float, float]]:
    """Uncached core of ``_idol_popularity_ranking`` (see its docstring)."""
    from src.knn.stage import scores_of

    idol_ids = search_df.loc[search_df["event_id"] == event_id, "idol_id"].unique()
    window_start = max(0, current_step - rank_window)
    rows: List[Tuple[float, float]] = []
    for iid in idol_ids:
        arr = scores_of(search_df, event_id, iid)
        if len(arr) == 0:
            continue
        end = min(current_step, len(arr))
        lo = min(window_start, end)
        window_vals = arr[lo:end]
        if len(window_vals) == 0:
            window_vals = arr[end - 1:end]
        if len(window_vals) == 0:
            continue
        rows.append((float(iid), float(np.mean(window_vals))))
    rows.sort(key=lambda kv: -kv[1])
    return rows


def _cumulative_growth(arr: np.ndarray, current_step: int) -> Optional[float]:
    """``score[current_step-1] - score[0]`` for a trajectory, or ``None``
    if the trajectory is too short to compute a meaningful growth value."""
    if len(arr) < 2:
        return None
    end_idx = min(current_step, len(arr)) - 1
    if end_idx < 1:
        return None
    return float(arr[end_idx] - arr[0])


def _windowed_growth(arr: np.ndarray, step: int, window: int) -> Optional[float]:
    """``score[end-1] - score[start]`` over the trailing ``window`` steps
    ending at ``step`` (NOT cumulative-since-start). ``None`` if too short."""
    end = min(step, len(arr))
    start = max(0, end - window)
    if end - start < 2:
        return None
    return float(arr[end - 1] - arr[start])


def _win_growth(
    arr: np.ndarray, step: int, window: int,
    event_id: Optional[float] = None, deseason_ir: bool = False, days: int = 13,
) -> Optional[float]:
    """``_windowed_growth`` optionally on the WEEKDAY-deseasonalized trajectory.

    When ``deseason_ir`` is set and the event's start weekday is known, the
    cumulative series is first passed through ``deseasonalize_weekday`` (each
    per-step increment divided by its calendar-weekday factor) before the
    trailing-window diff is taken. This removes the NUMERATOR's own weekday
    phase -- the residual weekend hump the cross-event denominator averaging
    can't cancel, which makes raw iR swing ~+8% on weekends / -3% midweek. Falls
    back to the plain windowed growth when off, or when the start weekday is
    unknown (no-op, byte-identical to the raw path)."""
    if deseason_ir and event_id is not None:
        a = np.asarray(arr, dtype=float)
        if len(a) >= 2:
            arr = deseasonalize_weekday(a, float(event_id), len(a), days=days)
    return _windowed_growth(arr, step, window)


def _pooled_instantaneous_rate(
    search_df: pd.DataFrame,
    event_id: float,
    pos: int,
    half_width: int,
    step: int,
    rate_window: int,
    rank_window: int,
) -> Optional[float]:
    """Mean windowed growth (see ``_windowed_growth``) across idols in the
    popularity-position neighbourhood ``[pos-half_width, pos+half_width]``
    (clipped, not extended) of ``event_id`` at ``step``. Used for BOTH the
    historical side (always pooled) and, in the reversal-detection /ewma
    path, the target's own side too -- pooling the target's own recent
    rate across its position neighbourhood (not just the single idol)
    reduces single-idol noise in the time-series used for regime
    detection specifically, which is a different concern from the
    single-idol precision the cumulative-ratio path optimizes for.
    """
    from src.knn.stage import scores_of

    ranking = _idol_popularity_ranking(search_df, event_id, step, rank_window)
    n = len(ranking)
    if n == 0 or pos >= n:
        return None
    lo, hi = max(0, pos - half_width), min(n - 1, pos + half_width)
    vals: List[float] = []
    for p in range(lo, hi + 1):
        iid, _ = ranking[p]
        arr = scores_of(search_df, event_id, iid)
        g = _windowed_growth(arr, step, rate_window)
        if g is not None:
            vals.append(g)
    return float(np.mean(vals)) if vals else None


def _detect_ratio_reversal(
    search_df: pd.DataFrame,
    target_event_id: float,
    historical_event_ids: List[float],
    pos: int,
    half_width: int,
    current_step: int,
    rank_window: int,
    rate_window: int,
    sample_spacing: int,
    short_window: int,
    long_window: int,
    min_short_magnitude: float,
) -> bool:
    """Detect a SIGN REVERSAL between a short-term and a longer-term trend
    in the pooled instantaneous ratio series -- i.e. the short-run
    direction has flipped relative to the longer-run direction (a
    "spike-then-decay" or "dip-then-rebound" shape), as opposed to a
    monotonic drift in one direction the whole time (which a cumulative
    ratio already tracks correctly and does NOT need EWMA for).

    Returns False (no reversal / stick with cumulative) whenever there
    isn't enough sampled history yet to judge -- the safe default.
    """
    sample_steps = list(range(max(0, current_step - long_window), current_step + 1, sample_spacing))
    if current_step not in sample_steps:
        sample_steps.append(current_step)
    sample_steps = sorted(set(s for s in sample_steps if s > 0))
    if len(sample_steps) < 5:
        return False

    ratios: List[Tuple[int, float]] = []
    for s in sample_steps:
        own = _pooled_instantaneous_rate(search_df, target_event_id, pos, half_width, s, rate_window, rank_window)
        hist_vals = []
        for eid in historical_event_ids:
            if eid == target_event_id:
                continue
            hv = _pooled_instantaneous_rate(search_df, eid, pos, half_width, s, rate_window, rank_window)
            if hv is not None:
                hist_vals.append(hv)
        if own is None or not hist_vals:
            continue
        hist_mean = np.mean(hist_vals)
        if hist_mean and hist_mean > 0:
            ratios.append((s, own / hist_mean))

    if len(ratios) < 5:
        return False

    def _slope(pairs: List[Tuple[int, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        x = x - x.mean()
        denom = float(np.sum(x ** 2))
        return float(np.sum(x * y) / denom) if denom > 0 else 0.0

    short_pairs = [(s, r) for s, r in ratios if current_step - s <= short_window]
    long_pairs = ratios
    if len(short_pairs) < 3:
        return False

    s_slope = _slope(short_pairs)
    l_slope = _slope(long_pairs)
    if s_slope == 0 or l_slope == 0:
        return False

    short_span = short_pairs[-1][0] - short_pairs[0][0]
    short_total_move = abs(s_slope * short_span)
    disagree = np.sign(s_slope) != np.sign(l_slope)
    return bool(disagree and short_total_move >= min_short_magnitude)


def _compute_ewma_ratio(
    search_df: pd.DataFrame,
    target_event_id: float,
    historical_event_ids: List[float],
    pos: int,
    half_width: int,
    current_step: int,
    rank_window: int,
    rate_window: int,
    sample_spacing: int,
    alpha: float,
    lookback: int,
) -> Optional[float]:
    """EWMA-smoothed instantaneous ratio: EWMA-smooth the pooled own-rate
    and pooled historical-rate series SEPARATELY (more stable than
    EWMA-ing a ratio-of-noisy-things directly), then take their ratio at
    the latest sample. Only uses samples within the trailing ``lookback``
    steps, walked forward from the oldest to build up the EWMA state --
    deliberately NOT seeded from step 0, since an early-event EWMA with
    no accumulated history is dominated by whichever noisy early sample
    happened to come first (the cold-start volatility problem measured
    directly against live event 437 data).
    """
    sample_steps = sorted(set(
        s for s in range(max(0, current_step - lookback), current_step + 1, sample_spacing) if s > 0
    ))
    if current_step not in sample_steps:
        sample_steps.append(current_step)
        sample_steps.sort()
    if len(sample_steps) < 3:
        return None

    own_prev, hist_prev = None, None
    own_smoothed, hist_smoothed = None, None
    for s in sample_steps:
        own = _pooled_instantaneous_rate(search_df, target_event_id, pos, half_width, s, rate_window, rank_window)
        hist_vals = []
        for eid in historical_event_ids:
            if eid == target_event_id:
                continue
            hv = _pooled_instantaneous_rate(search_df, eid, pos, half_width, s, rate_window, rank_window)
            if hv is not None:
                hist_vals.append(hv)
        hist = float(np.mean(hist_vals)) if hist_vals else None
        if own is not None:
            own_prev = own if own_prev is None else (alpha * own + (1 - alpha) * own_prev)
            own_smoothed = own_prev
        if hist is not None:
            hist_prev = hist if hist_prev is None else (alpha * hist + (1 - alpha) * hist_prev)
            hist_smoothed = hist_prev

    if own_smoothed is None or hist_smoothed is None or hist_smoothed <= 0:
        return None
    return own_smoothed / hist_smoothed


def compute_adaptive_scale_cap(
    search_df: pd.DataFrame,
    target_event_id: float,
    target_idol_id: float,
    current_step: int,
    historical_event_ids: List[float],
    rank_window: int,
    half_width: int,
    min_historical_events: int,
    static_cap: Tuple[float, float],
    use_reversal_gated_ewma: bool = False,
    reversal_rate_window: int = 40,
    reversal_sample_spacing: int = 10,
    reversal_short_window: int = 30,
    reversal_long_window: int = 80,
    reversal_min_short_magnitude: float = 0.2,
    ewma_alpha: float = 0.3,
    ewma_lookback: int = 80,
) -> Tuple[float, float]:
    """Loosen ONE bound of ``static_cap`` -- whichever side a live, measured
    growth ratio for ``target_idol_id`` points toward -- based on how its
    cumulative growth compares to popularity-matched idols in
    ``historical_event_ids``.

    Direction-agnostic: if the target idol is running hot relative to its
    matched historical idols (ratio > 1), the UPPER bound is loosened to
    ``max(static_upper, ratio)``. If running cold (ratio < 1), the LOWER
    bound is loosened to ``min(static_lower, ratio)``. Never tightens
    either bound. Falls back to ``static_cap`` unchanged whenever there
    isn't enough reliable data to trust the ratio (too few contributing
    historical events, degenerate/too-short trajectories, non-positive
    growth on either side).

    Popularity matching uses the SAME trailing-window ranking on both the
    current event and each historical event, then clips (never extends)
    the pooled position window ``[pos-half_width, pos+half_width]`` to
    whatever positions actually exist in that historical event's roster --
    e.g. position 1 with half_width=2 only pools positions {1,2,3}, not
    {1,2,3,4,5}, so the historical comparison set never reaches further
    into less-popular territory than the target's own popularity position.

    ``use_reversal_gated_ewma`` (default False, no behaviour change): the
    plain cumulative-growth ratio above is ~3x more stable across
    successive prediction runs than any windowed/EWMA alternative (see
    docs/relative_scale_search_normalization.md), but it LAGS when the
    true instantaneous ratio has a spike-then-decay (or dip-then-rebound)
    shape -- it stays elevated/depressed well after the real rate has
    already reversed, causing an over-correction during the decay phase
    (empirically found on event 142 at step 190). A plain monotonic drift
    (event 192's steady decline) does NOT create this problem, because
    the cumulative average naturally follows a one-directional trend.

    When enabled, a REVERSAL gate (``_detect_ratio_reversal``) checks
    whether the short-run trend has flipped sign relative to the
    longer-run trend in the pooled instantaneous-ratio series. Only when
    that reversal is detected does this function switch from the
    (default, stable) cumulative ratio to an EWMA-smoothed instantaneous
    ratio (``_compute_ewma_ratio``) for computing the loosened bound.
    Falls back to the cumulative ratio if the EWMA computation itself
    can't produce a usable value.
    """
    from src.knn.stage import scores_of

    static_lower, static_upper = static_cap

    cur_ranking = _idol_popularity_ranking(search_df, target_event_id, current_step, rank_window)
    idol_ids_ordered = [iid for iid, _ in cur_ranking]
    try:
        pos = idol_ids_ordered.index(float(target_idol_id))
    except ValueError:
        return static_cap

    target_arr = scores_of(search_df, target_event_id, target_idol_id)
    cur_growth = _cumulative_growth(target_arr, current_step)
    if cur_growth is None or not np.isfinite(cur_growth) or cur_growth <= 0:
        return static_cap

    pooled_hist_growths: List[float] = []
    contributing_events = 0
    for eid in historical_event_ids:
        if eid == target_event_id:
            continue
        hist_ranking = _idol_popularity_ranking(search_df, eid, current_step, rank_window)
        n_hist = len(hist_ranking)
        if n_hist == 0 or pos >= n_hist:
            continue
        lo = max(0, pos - half_width)
        hi = min(n_hist - 1, pos + half_width)
        event_growths: List[float] = []
        for p in range(lo, hi + 1):
            hist_iid, _ = hist_ranking[p]
            hist_arr = scores_of(search_df, eid, hist_iid)
            g = _cumulative_growth(hist_arr, current_step)
            if g is not None and np.isfinite(g) and g > 0:
                event_growths.append(g)
        if event_growths:
            pooled_hist_growths.extend(event_growths)
            contributing_events += 1

    if contributing_events < min_historical_events or not pooled_hist_growths:
        return static_cap

    hist_mean_growth = float(np.mean(pooled_hist_growths))
    if hist_mean_growth <= 0 or not np.isfinite(hist_mean_growth):
        return static_cap

    ratio = cur_growth / hist_mean_growth
    if not np.isfinite(ratio):
        return static_cap

    # Reversal-gated EWMA override (default off): only replace the stable
    # cumulative ratio with an EWMA-smoothed instantaneous ratio when a
    # short-vs-long-run trend REVERSAL is detected -- a monotonic drift
    # (no reversal) is already handled correctly by the cumulative ratio
    # above, so this only engages for the specific spike-then-decay /
    # dip-then-rebound shape that causes cumulative to lag.
    if use_reversal_gated_ewma:
        reversed_ = _detect_ratio_reversal(
            search_df=search_df,
            target_event_id=target_event_id,
            historical_event_ids=historical_event_ids,
            pos=pos,
            half_width=half_width,
            current_step=current_step,
            rank_window=rank_window,
            rate_window=reversal_rate_window,
            sample_spacing=reversal_sample_spacing,
            short_window=reversal_short_window,
            long_window=reversal_long_window,
            min_short_magnitude=reversal_min_short_magnitude,
        )
        if reversed_:
            ewma_ratio = _compute_ewma_ratio(
                search_df=search_df,
                target_event_id=target_event_id,
                historical_event_ids=historical_event_ids,
                pos=pos,
                half_width=half_width,
                current_step=current_step,
                rank_window=rank_window,
                rate_window=reversal_rate_window,
                sample_spacing=reversal_sample_spacing,
                alpha=ewma_alpha,
                lookback=ewma_lookback,
            )
            if ewma_ratio is not None and np.isfinite(ewma_ratio) and ewma_ratio > 0:
                ratio = ewma_ratio

    if ratio > 1.0:
        return (static_lower, max(static_upper, ratio))
    if ratio < 1.0:
        return (min(static_lower, ratio), static_upper)
    return static_cap


def _trimmed_mean(values: List[float], trim_pct: float) -> Optional[float]:
    """Mean of ``values`` after dropping the top and bottom ``trim_pct``
    fraction. Guarantees at least 1 element is dropped per tail whenever
    ``n >= 5`` (scipy's ``trim_mean`` floors ``n*trim_pct`` to 0 for small
    n, silently trimming nothing); falls back to a plain mean for ``n < 5``
    (too few values to trim meaningfully)."""
    if not values:
        return None
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    if n < 5:
        return float(np.mean(arr))
    n_cut = max(1, int(n * trim_pct))
    n_cut = min(n_cut, (n - 1) // 2)
    trimmed = arr[n_cut: n - n_cut]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(arr))


def _idol_position_matched_ratio(
    search_df: pd.DataFrame,
    target_event_id: float,
    idol_id: float,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
) -> Optional[float]:
    """One idol's CUMULATIVE growth ratio against history: this idol's own
    cumulative growth (score at ``current_step`` minus score at step 0)
    divided by the mean cumulative growth of the POSITION-MATCHED idol in
    each comparison event -- i.e. the idol occupying the same popularity
    rank in that historical event at the same step. ``None`` if the idol
    can't be ranked, its own growth is degenerate, or no historical match
    has usable growth.

    Position-matching (rather than same-idol-id) controls for the fact
    that which specific idol is "the popular one" changes every event.
    """
    from src.knn.stage import scores_of

    ranking = _idol_popularity_ranking(search_df, target_event_id, current_step, rank_window)
    ids_ordered = [i for i, _ in ranking]
    try:
        pos = ids_ordered.index(float(idol_id))
    except ValueError:
        return None
    arr = scores_of(search_df, target_event_id, idol_id)
    g = _cumulative_growth(arr, current_step)
    if g is None or not np.isfinite(g) or g <= 0:
        return None
    hist_growths: List[float] = []
    for eid in comparison_event_ids:
        if eid == target_event_id:
            continue
        hist_ranking = _idol_popularity_ranking(search_df, eid, current_step, rank_window)
        if pos < len(hist_ranking):
            hist_iid, _ = hist_ranking[pos]
            hist_arr = scores_of(search_df, eid, hist_iid)
            hg = _cumulative_growth(hist_arr, current_step)
            if hg is not None and np.isfinite(hg) and hg > 0:
                hist_growths.append(hg)
    if not hist_growths:
        return None
    hist_mean = float(np.mean(hist_growths))
    if hist_mean <= 0:
        return None
    ratio = g / hist_mean
    return ratio if np.isfinite(ratio) else None


def _event_ratio_and_spread(
    search_df: pd.DataFrame,
    target_event_id: float,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
    trim_pct: float,
) -> Tuple[Optional[float], Optional[float], Dict[float, float]]:
    """Compute, for ``target_event_id`` at ``current_step``:
      - ``event_ratio``  = TRIMMED MEAN over the event's idols of each
        idol's position-matched cumulative ratio (``removing top and
        tail`` so a heavily-pushed or barely-started idol doesn't dominate).
      - ``between_std``  = std ACROSS idols of those per-idol ratios
        (the cross-idol spread -- used as the "signal" variance in the
        empirical-Bayes shrinkage).
      - ``per_idol_ratios`` = ``{idol_id: ratio}`` for every idol with a
        usable ratio (returned so the caller can look up a specific idol
        without recomputing).
    ``event_ratio``/``between_std`` are ``None`` if no idol has a usable
    ratio.
    """
    idol_ids = search_df.loc[search_df["event_id"] == target_event_id, "idol_id"].unique()
    per_idol: Dict[float, float] = {}
    for iid in idol_ids:
        r = _idol_position_matched_ratio(
            search_df, target_event_id, float(iid), comparison_event_ids, current_step, rank_window,
        )
        if r is not None:
            per_idol[float(iid)] = r
    if not per_idol:
        return None, None, {}
    values = list(per_idol.values())
    event_ratio = _trimmed_mean(values, trim_pct)
    between_std = float(np.std(values)) if len(values) > 1 else 0.0
    return event_ratio, between_std, per_idol


# Cache for the event-level part of the macro-regime gate (identical for
# every idol in the event at a given step): stores
# (event_ratio, band, between_std). Weak-reference eviction like the other
# caches. The per-idol shrinkage is computed per call (cheap: one idol's
# own ratio time-series) and is NOT cached here.
_macro_regime_cache: Dict[Tuple, Tuple[Optional[float], Optional[Tuple[float, float]], Optional[float]]] = {}
_macro_regime_cache_lock = threading.Lock()

# Cache for the decay-forecast's trailing event_ratio lookup (event-level,
# identical for every idol in the event at a given step). Keyed by
# (id(search_df), event_id, step, rank_window, trim_pct). Weak-reference
# eviction like the other caches. Without this, every idol would recompute the
# same (t-window) event_ratio (a 52-idol scan) 52x per step.
_decay_ev_cache: Dict[Tuple, Optional[float]] = {}
_decay_ev_cache_lock = threading.Lock()


def compute_macro_regime_gate(
    search_df: pd.DataFrame,
    target_event_id: float,
    current_step: int,
    historical_event_ids: List[float],
    rank_window: int,
    trim_pct: float,
    min_historical_events: int,
    persistence_window: int,
    persistence_min_steps: int,
    persistence_sample_spacing: int,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[float]]:
    """Decide whether ``target_event_id`` is in a genuine, PERSISTENT
    structural regime shift relative to its historical peers.

    Returns ``(event_ratio, band, between_std)``:
      - ``event_ratio``: the event's trimmed-mean per-idol cumulative
        ratio at ``current_step`` (see ``_event_ratio_and_spread``), or
        ``None`` if uncomputable.
      - ``band = (lower, upper)``: the leave-one-out cross-event normal-
        variance band -- for each historical event, its OWN event_ratio
        against the *other* historical events, then ``mean +/- 2*std`` of
        those. ``None`` if too few historical events contribute.
      - ``between_std``: cross-idol spread of the current event's per-idol
        ratios, passed through for the empirical-Bayes shrinkage.

    **Persistence check**: samples the event_ratio at recent steps
    (spacing ``persistence_sample_spacing``, back ``persistence_window``)
    and requires at least ``persistence_min_steps`` of them to ALSO clear
    the band. If the current step clears the band but recent history
    doesn't persistently agree, ``band`` is returned as ``None`` (→ caller
    applies no correction), so a one-off spike can't trigger a breach.
    """
    hist_ids = [eid for eid in historical_event_ids if eid != target_event_id]

    event_ratio, between_std, _ = _event_ratio_and_spread(
        search_df, target_event_id, hist_ids, current_step, rank_window, trim_pct,
    )
    if event_ratio is None:
        return None, None, None

    band_ratios: List[float] = []
    for eid in hist_ids:
        others = [e for e in hist_ids if e != eid]
        r, _, _ = _event_ratio_and_spread(search_df, eid, others, current_step, rank_window, trim_pct)
        if r is not None and np.isfinite(r):
            band_ratios.append(r)

    if len(band_ratios) < min_historical_events:
        return event_ratio, None, between_std

    band_mean = float(np.mean(band_ratios))
    band_std = float(np.std(band_ratios))
    band = (band_mean - 2 * band_std, band_mean + 2 * band_std)

    # Only bother with the (expensive) persistence check if the current
    # step actually clears the band -- otherwise there's nothing to
    # confirm and the caller won't correct anyway.
    if event_ratio > band[1] or event_ratio < band[0]:
        sample_steps = [
            s for s in range(max(1, current_step - persistence_window), current_step, persistence_sample_spacing)
        ]
        if sample_steps:
            n_clearing = 0
            for s in sample_steps:
                r_s, _, _ = _event_ratio_and_spread(search_df, target_event_id, hist_ids, s, rank_window, trim_pct)
                if r_s is not None and (r_s > band[1] or r_s < band[0]):
                    n_clearing += 1
            if n_clearing < persistence_min_steps:
                # Not a persistent regime shift -- suppress by nulling the
                # band, so the caller treats it as "no correction".
                return event_ratio, None, between_std

    return event_ratio, band, between_std


# Normal-band ceiling on the EVENT-level ratio (leave-one-out cross-event
# mean+sigma*std), computed independent of the gate's persistence logic, so the
# per-idol "in-cloud" test works even on non-inflating (historical) events.
_normal_band_cache: Dict[Tuple, Optional[float]] = {}
_normal_band_cache_lock = threading.Lock()


def _normal_band_upper(
    search_df, hist_ids, current_step, rank_window, trim_pct, min_historical_events, sigma,
) -> Optional[float]:
    key = (
        id(search_df), tuple(sorted(float(e) for e in hist_ids)), int(current_step),
        int(rank_window), round(float(trim_pct), 4), int(min_historical_events),
        round(float(sigma), 4),
    )
    v = _normal_band_cache.get(key, _SENTINEL)
    if v is not _SENTINEL:
        return v
    with _normal_band_cache_lock:
        v = _normal_band_cache.get(key, _SENTINEL)
        if v is not _SENTINEL:
            return v
        band_ratios: List[float] = []
        for eid in hist_ids:
            others = [e for e in hist_ids if e != eid]
            r, _, _ = _event_ratio_and_spread(search_df, eid, others, current_step, rank_window, trim_pct)
            if r is not None and np.isfinite(r):
                band_ratios.append(r)
        res = (
            float(np.mean(band_ratios) + sigma * np.std(band_ratios))
            if len(band_ratios) >= min_historical_events else None
        )
        _normal_band_cache[key] = res
        weakref.finalize(search_df, _normal_band_cache.pop, key, None)
        return res


def idol_in_normal_cloud(
    search_df,
    target_event_id: float,
    idol_id: float,
    historical_event_ids: List[float],
    current_step: int,
    rank_window: int,
    trim_pct: float,
    min_historical_events: int,
    sigma: float = 2.0,
) -> Optional[bool]:
    """True if this idol's position-matched CUMULATIVE ratio sits within the
    historical normal band (<= mean + sigma*std of the leave-one-out cross-event
    ratios) -- i.e. the idol is still inside the main cloud of past trajectories,
    not inflated above it.

    Gates PURERAW (raw neighbour search): an in-cloud idol is well matched by
    direct raw-score neighbours, so the relative-scale rescale (added for the
    inflation regime) is unnecessary and can distort it. Returns ``None`` when
    the band or the idol's ratio can't be computed (caller keeps the default
    relative-scale behaviour)."""
    hist_ids = [e for e in historical_event_ids if e != target_event_id]
    band_upper = _normal_band_upper(
        search_df, hist_ids, current_step, rank_window, trim_pct, min_historical_events, sigma,
    )
    if band_upper is None:
        return None
    idol_cumR = _idol_position_matched_ratio(
        search_df, target_event_id, float(idol_id), hist_ids, current_step, rank_window,
    )
    if idol_cumR is None or not np.isfinite(idol_cumR):
        return None
    return bool(idol_cumR <= band_upper)


def _idol_ratio_noise(
    search_df: pd.DataFrame,
    target_event_id: float,
    idol_id: float,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
    persistence_window: int,
    persistence_sample_spacing: int,
) -> Optional[float]:
    """Within-idol NOISE of one idol's position-matched cumulative ratio,
    measured as the std of that ratio sampled at several recent steps
    (spacing ``persistence_sample_spacing``, back ``persistence_window``
    from ``current_step``). This is the "how much do I trust this idol's
    own number" signal for the empirical-Bayes shrinkage -- a ratio that
    jitters a lot step-to-step is noise-driven and should be shrunk hard
    toward the event mean; a steady one is trustworthy. ``None`` if fewer
    than 2 samples are available (can't estimate noise)."""
    sample_steps = sorted(set(
        list(range(max(1, current_step - persistence_window), current_step + 1, persistence_sample_spacing))
        + [current_step]
    ))
    samples: List[float] = []
    for s in sample_steps:
        r = _idol_position_matched_ratio(
            search_df, target_event_id, idol_id, comparison_event_ids, s, rank_window,
        )
        if r is not None and np.isfinite(r):
            samples.append(r)
    if len(samples) < 2:
        return None
    return float(np.std(samples))


def _decay_forecast(
    r_now: float, r_wago: float, w: float, p: float,
    remaining: int, window: int, floor: float,
) -> float:
    """Forecast where a destaged ratio lands by extrapolating its recent
    decline, decelerated by ``p``, blended with the current level by ``w``,
    and clamped at ``floor``:

        d_past = (r_now - r_wago) / window
        r_end  = max(floor, r_now + p * remaining * d_past)
        r_hat  = w * r_now + (1 - w) * r_end

    Only the DECLINE direction is corrected: if the ratio is flat/rising
    recently (``d_past >= 0``) ``r_now`` is returned unchanged, so this only
    ever TRIMS an over-elevated decaying anchor and never amplifies upward
    (rising accelerators are handled by the top-tier relaxation, and a
    leave-one-out backtest on the 6 historical anniversaries showed the recent
    cumulative-ratio slope is not a reliable forward signal -- ~half of
    decaying segments re-accelerate at boost/dash)."""
    if window <= 0 or remaining <= 0:
        return r_now
    d_past = (r_now - r_wago) / window
    if d_past >= 0:
        return r_now
    r_end = max(floor, r_now + p * remaining * d_past)
    return w * r_now + (1.0 - w) * r_end


def _decay_decline_persists(
    ratio_at_step,
    current_step: int,
    w_steps: int,
    persistence_window: int,
    sample_spacing: int,
    min_steps: int,
    deadband: float,
) -> bool:
    """Confirmation gate for the decay trim: True only if the ratio has been
    DECLINING (``ratio(s) < ratio(s - w_steps) - deadband``) at ``>= min_steps``
    of the steps sampled every ``sample_spacing`` back over the recent
    ``persistence_window``.

    Guards against a TENTATIVE (sub-weekly / weekday-trough) dip triggering the
    trim: the decay trim projects the recent decline over the whole remaining
    horizon, so a one/two-day lull would otherwise over-trim (measured ~66% of
    437's trim firings were such transient dips that day-6 re-heat reversed).
    Requiring the decline to hold across several samples spanning the window
    forces the decline to be sustained before the (aggressive, extrapolated)
    trim engages. ``ratio_at_step`` is a callable ``step -> Optional[float]``."""
    checked = 0
    confirmations = 0
    s = int(current_step)
    stop = int(current_step) - int(persistence_window)
    while s >= stop:
        if s - w_steps >= 1:
            r_now = ratio_at_step(s)
            r_prev = ratio_at_step(s - w_steps)
            if (r_now is not None and r_prev is not None
                    and np.isfinite(r_now) and np.isfinite(r_prev)):
                checked += 1
                if r_now < r_prev - deadband:
                    confirmations += 1
        s -= max(1, int(sample_spacing))
    return checked > 0 and confirmations >= int(min_steps)


def _cached_trailing_event_ratio(
    search_df: pd.DataFrame,
    target_event_id: float,
    hist_ids: List[float],
    step: int,
    rank_window: int,
    trim_pct: float,
) -> Optional[float]:
    """Event-wide trimmed-mean cumulative ratio at ``step`` (see
    ``_event_ratio_and_spread``), memoized per (df, event, step, rank_window,
    trim_pct) so the decay forecast's trailing lookup isn't recomputed once
    per idol."""
    key = (id(search_df), float(target_event_id), int(step), int(rank_window), float(trim_pct))
    cached = _decay_ev_cache.get(key)
    if cached is not None or key in _decay_ev_cache:
        return cached
    with _decay_ev_cache_lock:
        if key in _decay_ev_cache:
            return _decay_ev_cache[key]
        r, _, _ = _event_ratio_and_spread(
            search_df, target_event_id, hist_ids, step, rank_window, trim_pct,
        )
        _decay_ev_cache[key] = r
        weakref.finalize(search_df, _decay_ev_cache.pop, key, None)
        return r


def _idol_position_matched_interval_ratio(
    search_df: pd.DataFrame,
    target_event_id: float,
    idol_id: float,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
    window: int,
    deseason_ir: bool = False,
    days: int = 13,
    growth_end_step: Optional[int] = None,
    growth_window: Optional[int] = None,
) -> Optional[float]:
    """Like ``_idol_position_matched_ratio`` but uses WINDOWED (trailing
    ``window``-step interval) growth instead of cumulative growth: this idol's
    trailing-window score gain divided by the mean trailing-window gain of the
    position-matched idol across ``comparison_event_ids``. This is the
    deseasonalized/destaged INTERVAL ratio (iR) -- position-matching to the same
    normalized step in each historical event cancels the shared stage (boost),
    and averaging over several events + a multi-day window smooths the weekday
    effect. ``None`` if the idol can't be ranked or growth is degenerate."""
    from src.knn.stage import scores_of

    ranking = _idol_popularity_ranking(search_df, target_event_id, current_step, rank_window)
    ids_ordered = [i for i, _ in ranking]
    try:
        pos = ids_ordered.index(float(idol_id))
    except ValueError:
        return None
    arr = scores_of(search_df, target_event_id, idol_id)
    # Regime-aware growth window (B, double-count fix): by default the trailing
    # ``window`` ending at ``current_step``. When the caller passes a
    # ``growth_end_step`` / ``growth_window`` (post-2.4M-crossing routing) the
    # target's iR is measured over that window instead, and the position-matched
    # historical comparators are measured over the SAME window so the ratio stays
    # position-matched. Ranking (WHO sits at each position) always uses
    # ``current_step`` -- only the growth measurement window moves.
    ge = current_step if growth_end_step is None else int(growth_end_step)
    gw = window if growth_window is None else int(growth_window)
    g = _win_growth(arr, ge, gw, target_event_id, deseason_ir, days)
    if g is None or not np.isfinite(g) or g <= 0:
        return None
    hist_growths: List[float] = []
    for eid in comparison_event_ids:
        if eid == target_event_id:
            continue
        hist_ranking = _idol_popularity_ranking(search_df, eid, current_step, rank_window)
        if pos < len(hist_ranking):
            hist_iid, _ = hist_ranking[pos]
            hist_arr = scores_of(search_df, eid, hist_iid)
            hg = _win_growth(hist_arr, ge, gw, eid, deseason_ir, days)
            if hg is not None and np.isfinite(hg) and hg > 0:
                hist_growths.append(hg)
    if not hist_growths:
        return None
    hist_mean = float(np.mean(hist_growths))
    if hist_mean <= 0:
        return None
    ratio = g / hist_mean
    return ratio if np.isfinite(ratio) else None


# Historical normal-band ceiling for the per-idol INTERVAL ratio (Layer-2 band
# clamp). Event-level (depends only on the comparison events / step / window /
# sigma, identical for every target idol), so cached with weak-ref eviction.
_SENTINEL = object()
_interval_band_cache: Dict[Tuple, Optional[float]] = {}
_interval_band_cache_lock = threading.Lock()


def _idol_interval_band_ceiling(
    search_df: pd.DataFrame,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
    window: int,
    sigma: float,
    deseason_ir: bool = False,
    days: int = 13,
) -> Optional[float]:
    """``mean + sigma*std`` of the per-idol INTERVAL ratio across the historical
    events at ``current_step`` -- the "normal out-performance" band ceiling in
    the SAME units as ``_idol_position_matched_interval_ratio``.

    For each historical event ``e`` and rank position ``p`` present at the step,
    the leave-one-out interval ratio = event ``e``'s rank-``p`` idol trailing-
    ``window`` growth divided by the mean trailing-``window`` growth of the SAME
    rank position across the OTHER historical events. Pooled over all ``(e, p)``
    cells (~events x ranks samples), giving a well-conditioned band. Anything a
    live idol's iR_base sits above is treated by the clamp as an unsustainable
    outlier. Returns ``None`` if too few cells contribute (caller then skips the
    clamp). Position-matching cancels the shared stage; the multi-day window +
    cross-event pooling smooth the weekday effect -- consistent with the iR
    estimator itself."""
    from src.knn.stage import scores_of

    if len(comparison_event_ids) < 2:
        return None
    key = (
        id(search_df), tuple(sorted(float(e) for e in comparison_event_ids)),
        int(current_step), int(rank_window), int(window), round(float(sigma), 4),
        bool(deseason_ir),
    )
    cached = _interval_band_cache.get(key, _SENTINEL)
    if cached is not _SENTINEL:
        return cached
    with _interval_band_cache_lock:
        cached = _interval_band_cache.get(key, _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        rankings = {
            e: _idol_popularity_ranking(search_df, e, current_step, rank_window)
            for e in comparison_event_ids
        }
        ratios: List[float] = []
        for e in comparison_event_ids:
            others = [o for o in comparison_event_ids if o != e]
            for pos, (iid, _) in enumerate(rankings[e]):
                g = _win_growth(scores_of(search_df, e, iid), current_step, window, e, deseason_ir, days)
                if g is None or not np.isfinite(g) or g <= 0:
                    continue
                hgs: List[float] = []
                for o in others:
                    o_rank = rankings[o]
                    if pos < len(o_rank):
                        hg = _win_growth(
                            scores_of(search_df, o, o_rank[pos][0]), current_step, window,
                            o, deseason_ir, days,
                        )
                        if hg is not None and np.isfinite(hg) and hg > 0:
                            hgs.append(hg)
                if hgs:
                    m = float(np.mean(hgs))
                    if m > 0:
                        r = g / m
                        if np.isfinite(r):
                            ratios.append(r)
        result = (
            float(np.mean(ratios) + sigma * np.std(ratios)) if len(ratios) >= 3 else None
        )
        _interval_band_cache[key] = result
        weakref.finalize(search_df, _interval_band_cache.pop, key, None)
        return result


# Current-event cross-idol band ceiling (Layer-2 band clamp, default reference).
_event_band_cache: Dict[Tuple, Optional[float]] = {}
_event_band_cache_lock = threading.Lock()


def _event_interval_band_ceiling(
    search_df: pd.DataFrame,
    target_event_id: float,
    comparison_event_ids: List[float],
    current_step: int,
    rank_window: int,
    window: int,
    sigma: float,
    deseason_ir: bool = False,
    days: int = 13,
) -> Optional[float]:
    """``mean + sigma*std`` of the CURRENT event's per-idol INTERVAL ratio taken
    ACROSS its own idols -- the ceiling above which an idol is an outlier *even
    among this (hot) event's* idols.

    Unlike the historical band (``_idol_interval_band_ceiling``), this reference
    RISES with the event: in a broadly-hot surge the whole distribution shifts
    up, so mean+2sigma sits at the event's own upper tail and only the genuine
    extremes (thin-comparator / spike idols like idol44/36) are flagged -- not
    every above-normal idol. Each idol's ratio is the same estimator the clamp
    anchors on (``_idol_position_matched_interval_ratio``). Cached event-level;
    ``None`` if fewer than 3 idols have a usable ratio."""
    key = (
        id(search_df), float(target_event_id), int(current_step),
        int(rank_window), int(window), round(float(sigma), 4), bool(deseason_ir),
    )
    cached = _event_band_cache.get(key, _SENTINEL)
    if cached is not _SENTINEL:
        return cached
    with _event_band_cache_lock:
        cached = _event_band_cache.get(key, _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        idol_ids = search_df.loc[search_df["event_id"] == target_event_id, "idol_id"].unique()
        vals: List[float] = []
        for iid in idol_ids:
            r = _idol_position_matched_interval_ratio(
                search_df, target_event_id, float(iid), comparison_event_ids,
                current_step, rank_window, window, deseason_ir, days,
            )
            if r is not None and np.isfinite(r):
                vals.append(r)
        result = (
            float(np.mean(vals) + sigma * np.std(vals)) if len(vals) >= 3 else None
        )
        _event_band_cache[key] = result
        weakref.finalize(search_df, _event_band_cache.pop, key, None)
        return result


def _interval_cap_effective_ratio(
    search_df: pd.DataFrame,
    target_event_id: float,
    target_idol_id: float,
    current_step: int,
    hist_ids: List[float],
    rank_window: int,
    total_steps: int,
    base_window_days: float,
    reversion_frac: float,
    floor: float,
    daily_increments: Optional[np.ndarray],
    days: int = 13,
    band_clamp: bool = False,
    band_sigma: float = 2.0,
    band_clamp_frac: float = 1.0,
    band_reference: str = "current_event",
    deseason_ir: bool = False,
    ir_crossing_step: Optional[int] = None,
    skip_haircut_f: float = 0.90,
    skip_blend_enabled: bool = False,
    skip_full_weight_days: float = 2.0,
    skip_max_ratio: float = 1.0,
    skip_min_ratio: float = 0.0,
    skip_fast_weight_days: float = 0.0,
    skip_fast_ratio: float = 1.0,
    skip_surge_alpha: float = 0.0,
) -> Optional[float]:
    """Interval-anchored per-idol cap.

    Anchor on the idol's recent INTERVAL ratio ``iR_base`` (trailing
    ``base_window_days``), project it daily over the remaining horizon with a
    reversion toward ``floor``
    (``iR_k = floor + (iR_base - floor)*(1 - reversion_frac*tau_k)``; frac=0 ->
    flat, >0 -> decays toward floor, <0 -> rises), and collapse to one scalar as
    the Δḡ-weighted average ``Σ_k iR_k·Δḡ_k / Σ_k Δḡ_k`` where ``daily_increments``
    are the neighbours' per-remaining-day score gains (dash/boost days heaviest;
    absolute scale cancels in the ratio). At ``frac=0`` the cap is exactly
    ``iR_base`` (the weighting is a no-op on a flat sequence). Floored at
    ``floor``. ``None`` if the interval ratio is uncomputable (caller falls back
    to the static cap). ``days`` = normalized event length in days (13 for
    anniversaries, whose day<->step mapping is uniform).

    **Layer-2 band clamp** (``band_clamp``): if ``iR_base`` exceeds the
    historical normal-band ceiling (``mean + band_sigma*std`` of the same
    interval-ratio estimator across the historical events, see
    ``_idol_interval_band_ceiling``), pull it toward that ceiling by
    ``band_clamp_frac`` (1.0 = clamp to the ceiling, 0.5 = halfway) BEFORE the
    reversion projection. Only lowers an above-band anchor; in-band idols are
    untouched. Regression-to-the-mean for the thin-comparator / spike outliers.
    ``band_reference``: ``"current_event"`` (default) uses this event's own
    cross-idol iR distribution (flags only the event's genuine tail -- correct
    for a broadly-hot surge); ``"historical"`` uses the cross-event normal band
    (over-clamps a surge, since most idols exceed the normal band)."""
    steps_per_day = float(total_steps) / float(days)
    window = max(2, int(round(base_window_days * steps_per_day)))
    # B (double-count fix): regime-aware iR base relative to the idol's OWN
    # 2.4M crossing (``ir_crossing_step`` in normalized-step units; None = not yet
    # crossed).
    #   * not crossed        -> trailing ``window`` ending now (skip-active pace).
    #   * crossed >= 24h ago  -> OBSERVED post-crossing pace: window spans only the
    #                            post-crossing steps (the model haircut C is turned
    #                            off for these idols in predict.py, so no
    #                            double-count -- the decay is measured, not modelled).
    #   * crossed < 24h ago   -> freeze on the pre-crossing skip-active pace (window
    #                            ending AT the crossing step); C still applies the
    #                            modelled post-crossing haircut on top.
    g_end: Optional[int] = None
    g_win: Optional[int] = None
    iR_base: Optional[float] = None
    if ir_crossing_step is not None and int(ir_crossing_step) >= 1:
        day_steps = max(2, int(round(steps_per_day)))
        steps_since = int(current_step) - int(ir_crossing_step)
        if skip_blend_enabled and steps_since >= 0:
            # Observed-decay ramp (supersedes the binary 24h switch). The
            # post-crossing pace multiplier ``m`` ramps from the MODELLED haircut
            # (``skip_haircut_f``) toward the idol's OWN observed decay ratio
            # ``r_obs = iR_post / iR_pre`` as post-crossing data accrues.
            #   * iR_pre  = pre-crossing skip-active pace (window ending AT the
            #               crossing step).
            #   * iR_post = observed post-crossing pace (window = the post steps).
            # BOTH are position-matched to history at their own normalized-step
            # windows, so the shared stage/boost multiplier cancels in EACH --
            # ``r_obs`` is a pure pace-vs-history change, NOT the raw boost-day
            # jump. An idol crossing right at boost reads iR_post relative to a
            # historically-boosted comparator, so boost does not inflate r_obs.
            # ``w = clip(steps_since / (full_weight_days*day_steps), 0, 1)``;
            # ``r_obs`` capped at ``skip_max_ratio`` (1.0 -> the haircut can only
            # be removed, never turned into a boost). C is OFF for crossed idols
            # (predict.py), so ``m`` fully owns the post-crossing decay.
            iR_pre = _idol_position_matched_interval_ratio(
                search_df, target_event_id, float(target_idol_id), hist_ids,
                current_step, rank_window, window, deseason_ir, days,
                growth_end_step=int(ir_crossing_step), growth_window=window,
            )
            if iR_pre is not None and np.isfinite(iR_pre) and iR_pre > 0:
                m = float(skip_haircut_f)
                _iR_post_obs = None
                if steps_since >= 2:
                    iR_post = _idol_position_matched_interval_ratio(
                        search_df, target_event_id, float(target_idol_id), hist_ids,
                        current_step, rank_window, window, deseason_ir, days,
                        growth_end_step=int(current_step),
                        growth_window=max(2, steps_since),
                    )
                    if iR_post is not None and np.isfinite(iR_post) and iR_post > 0:
                        r_obs = min(float(skip_max_ratio),
                                    max(float(skip_min_ratio), iR_post / iR_pre))
                        # Asymmetric ramp: quick to LIFT the haircut when the idol
                        # shows little/no decay (r_obs high -> converge to observed
                        # in ``skip_fast_weight_days``), but keep the slower, more
                        # conservative horizon when genuinely cutting (r_obs low).
                        # Avoids leaving a non-decaying idol needlessly trimmed for
                        # the full window. skip_fast_weight_days<=0 disables it.
                        wdays = float(skip_full_weight_days)
                        if (float(skip_fast_weight_days) > 0.0
                                and r_obs >= float(skip_fast_ratio)):
                            wdays = float(skip_fast_weight_days)
                        full = max(1.0, wdays * day_steps)
                        w = min(1.0, float(steps_since) / full)
                        m = (1.0 - w) * float(skip_haircut_f) + w * r_obs
                        _iR_post_obs = float(iR_post)
                iR_base = m * iR_pre
                # Surge credit: the r_obs cap (<=skip_max_ratio) pins iR_base at
                # iR_pre for an idol whose OBSERVED post-crossing pace exceeds its
                # pre-crossing pace, so a genuine post-2.4M accelerator is
                # under-projected. Lift the base a fraction ``skip_surge_alpha`` of
                # the way toward the observed pace. alpha<1 = partial trust (a fresh
                # spike only leaks in by alpha). Anchor is the current (haircut)
                # base so a brand-new crosser is not un-haircut and lifted at once.
                # skip_surge_alpha=0.0 -> legacy hard-ceiling behavior. See
                # docs/skip_surge_credit_design.md.
                if (float(skip_surge_alpha) > 0.0 and _iR_post_obs is not None
                        and _iR_post_obs > iR_pre):
                    iR_base = iR_base + float(skip_surge_alpha) * (_iR_post_obs - iR_pre)
            # iR_pre uncomputable -> leave iR_base None, fall through to the
            # generic trailing-window estimator below.
        else:
            # Legacy binary regime (blend off): frozen pre-crossing pace (<24h)
            # or observed post-crossing pace (>=24h). C in predict.py flips at 24h.
            if steps_since >= day_steps:
                g_end = int(current_step)
                g_win = max(2, min(window, steps_since))
            elif steps_since >= 0:
                g_end = int(ir_crossing_step)
                g_win = window
    if iR_base is None:
        iR_base = _idol_position_matched_interval_ratio(
            search_df, target_event_id, float(target_idol_id), hist_ids,
            current_step, rank_window, window, deseason_ir, days,
            growth_end_step=g_end, growth_window=g_win,
        )
    if iR_base is None or not np.isfinite(iR_base):
        return None

    if band_clamp and band_clamp_frac > 0.0:
        if band_reference == "historical":
            ceiling = _idol_interval_band_ceiling(
                search_df, hist_ids, current_step, rank_window, window, band_sigma,
                deseason_ir, days,
            )
        else:
            ceiling = _event_interval_band_ceiling(
                search_df, target_event_id, hist_ids, current_step, rank_window, window, band_sigma,
                deseason_ir, days,
            )
        if ceiling is not None and np.isfinite(ceiling) and iR_base > ceiling:
            iR_base = iR_base - band_clamp_frac * (iR_base - ceiling)

    remaining_steps = int(total_steps) - int(current_step)
    if remaining_steps <= 0:
        return max(floor, iR_base)

    if daily_increments is not None and len(daily_increments) > 0:
        incs = np.asarray(daily_increments, dtype=float)
        K = len(incs)
    else:
        K = max(1, int(round(remaining_steps / steps_per_day)))
        incs = np.ones(K, dtype=float)

    numer = 0.0
    denom = 0.0
    for k in range(1, K + 1):
        tau = k / float(K)
        iR_k = floor + (iR_base - floor) * (1.0 - reversion_frac * tau)
        wk = float(incs[k - 1])
        numer += iR_k * wk
        denom += wk
    cap = (numer / denom) if denom > 0 else iR_base
    return max(floor, cap)


def compute_macro_regime_scale_cap(
    search_df: pd.DataFrame,
    target_event_id: float,
    target_idol_id: float,
    current_step: int,
    historical_event_ids: List[float],
    rank_window: int,
    trim_pct: float,
    min_historical_events: int,
    static_cap: Tuple[float, float],
    persistence_window: int = 40,
    persistence_min_steps: int = 3,
    persistence_sample_spacing: int = 10,
    use_eb_shrinkage: bool = True,
    use_toptier_relax: bool = False,
    toptier_relax_sigma: float = 2.0,
    toptier_relax_strength: float = 0.7,
    toptier_relax_recency_lookback: int = 46,
    toptier_relax_recency_tol: float = 0.0,
    use_decay_forecast: bool = False,
    decay_forecast_p: float = 0.8,
    decay_forecast_w: float = 0.5,
    decay_forecast_window: int = 46,
    decay_forecast_floor: float = 1.0,
    decay_persistence_enabled: bool = False,
    decay_persistence_window: int = 69,
    decay_persistence_sample_spacing: int = 17,
    decay_persistence_min_steps: int = 3,
    decay_deadband: float = 0.01,
    use_interval_cap: bool = False,
    interval_cap_base_window_days: float = 2.0,
    interval_cap_reversion_frac: float = 0.0,
    interval_cap_floor: float = 1.0,
    interval_cap_band_clamp: bool = False,
    interval_cap_band_sigma: float = 2.0,
    interval_cap_band_clamp_frac: float = 1.0,
    interval_cap_band_reference: str = "current_event",
    interval_cap_deseason_ir: bool = False,
    interval_cap_hot_only: bool = False,
    interval_cap_hot_sigma: float = 2.0,
    interval_cap_crossing_step: Optional[int] = None,
    interval_cap_skip_haircut_f: float = 0.90,
    interval_cap_skip_blend_enabled: bool = False,
    interval_cap_skip_full_weight_days: float = 2.0,
    interval_cap_skip_max_ratio: float = 1.0,
    interval_cap_skip_min_ratio: float = 0.0,
    interval_cap_skip_fast_weight_days: float = 0.0,
    interval_cap_skip_fast_ratio: float = 1.0,
    interval_cap_skip_surge_alpha: float = 0.0,
    neighbor_daily_increments: Optional[np.ndarray] = None,
    total_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """Event-level macro-regime scale cap with empirical-Bayes, stability-
    based per-idol shrinkage.

    **Step 1 -- detect (event level, cached, identical for all idols):**
    Compute the event's trimmed-mean per-idol CUMULATIVE ratio
    (``_event_ratio_and_spread``: each idol's cumulative growth vs. the
    position-matched historical idols' mean, then trimmed-mean across
    idols), and a leave-one-out cross-event normal-variance band. Require
    the event to PERSISTENTLY clear that band (see
    ``compute_macro_regime_gate``). If it doesn't, return ``static_cap``
    unchanged.

    **Step 2 -- base breach:** if it clears, the event-wide ratio becomes
    the base amount by which the static cap is allowed to be breached
    (upper bound if the event is inflating, lower bound if deflating).

    **Step 3 -- per-idol stability shrinkage (empirical Bayes):** the
    individual idol's own position-matched cumulative ratio is shrunk
    toward the event-wide ratio by a weight derived FROM DATA, not a
    hand-picked constant:

        w = between_var / (between_var + within_var_idol)
        effective_ratio = event_ratio + w * (idol_ratio - event_ratio)

    where ``within_var_idol`` is the variance of THIS idol's own ratio
    over recent steps (``_idol_ratio_noise`` -- how much its number
    jitters, i.e. how unreliable it is) and ``between_var`` is the
    cross-idol spread of ratios. A noisy idol (large within_var) gets
    ``w -> 0`` and is pulled fully to the stable event-wide ratio; a
    rock-steady idol (small within_var) gets ``w -> 1`` and keeps its own
    value. The absolute pull ``(1-w)*(idol_ratio - event_ratio)`` also
    grows with how far the idol sits from the event mean, so far-out
    idols are adjusted more in absolute terms -- both properties the
    design calls for. There is NO fixed shrink-strength constant; the
    strength is entirely a function of measured noise vs. spread.

    Direction: when ``event_ratio > 1`` (inflation), the shrink pulls a
    hot idol back down toward the event mean (conservative); when
    ``event_ratio < 1`` (deflation), it pulls a cold idol back up toward
    it. If the idol's own ratio or noise can't be computed, fall back to
    the (safe) event-wide ratio with no per-idol excess.

    Falls back to ``static_cap`` unchanged whenever there isn't enough
    reliable data (too few historical events, degenerate growth, or the
    event doesn't persistently clear the band).
    """
    static_lower, static_upper = static_cap
    hist_ids = [eid for eid in historical_event_ids if eid != target_event_id]

    cache_key = (
        id(search_df), float(target_event_id), int(current_step), int(rank_window),
        float(trim_pct), int(min_historical_events), static_cap,
        int(persistence_window), int(persistence_min_steps), int(persistence_sample_spacing),
    )
    cached = _macro_regime_cache.get(cache_key)
    if cached is None:
        with _macro_regime_cache_lock:
            cached = _macro_regime_cache.get(cache_key)
            if cached is None:
                cached = compute_macro_regime_gate(
                    search_df, target_event_id, current_step, hist_ids, rank_window,
                    trim_pct, min_historical_events, persistence_window,
                    persistence_min_steps, persistence_sample_spacing,
                )
                _macro_regime_cache[cache_key] = cached
                weakref.finalize(search_df, _macro_regime_cache.pop, cache_key, None)

    event_ratio, band, between_std = cached
    if event_ratio is None or band is None:
        return static_cap
    band_lower, band_upper = band
    if not (event_ratio > band_upper or event_ratio < band_lower):
        return static_cap

    # Interval-anchored cap (takes precedence over the cumulative decay /
    # shrinkage path below when enabled). Only meaningful in an inflation
    # regime (event_ratio > 1); 437 is inflation. The projected cap is set as
    # the upper bound DIRECTLY -- unlike the cumulative path it is allowed to be
    # BELOW static_upper, so a cooled idol in a surge event is trimmed to its
    # own recent interval pace rather than held up by the stock anchor. Falls
    # back to the static cap if the idol's interval ratio can't be computed.
    if use_interval_cap and event_ratio > 1.0 and total_steps is not None:
        # B: per-idol routing. When ``interval_cap_hot_only``, only idols ABOVE
        # the historical normal cloud (hot / crossing-bound) take the interval
        # (iR) path; in-cloud (cool) idols fall through to the cumulative DECAY
        # path below. Off -> all idols in the inflation regime take interval
        # (prior behaviour). In a broadly-hot event (437) all idols are above
        # the cloud, so all route to interval.
        take_interval = True
        if interval_cap_hot_only:
            in_cloud = idol_in_normal_cloud(
                search_df, target_event_id, float(target_idol_id), hist_ids,
                current_step, rank_window, trim_pct, min_historical_events,
                interval_cap_hot_sigma,
            )
            take_interval = (in_cloud is False)  # only above-cloud (hot) idols
        if take_interval:
            eff = _interval_cap_effective_ratio(
                search_df, target_event_id, float(target_idol_id), current_step,
                hist_ids, rank_window, int(total_steps),
                interval_cap_base_window_days, interval_cap_reversion_frac,
                interval_cap_floor, neighbor_daily_increments,
                band_clamp=interval_cap_band_clamp,
                band_sigma=interval_cap_band_sigma,
                band_clamp_frac=interval_cap_band_clamp_frac,
                band_reference=interval_cap_band_reference,
                deseason_ir=interval_cap_deseason_ir,
                ir_crossing_step=interval_cap_crossing_step,
                skip_haircut_f=interval_cap_skip_haircut_f,
                skip_blend_enabled=interval_cap_skip_blend_enabled,
                skip_full_weight_days=interval_cap_skip_full_weight_days,
                skip_max_ratio=interval_cap_skip_max_ratio,
                skip_min_ratio=interval_cap_skip_min_ratio,
                skip_fast_weight_days=interval_cap_skip_fast_weight_days,
                skip_fast_ratio=interval_cap_skip_fast_ratio,
                skip_surge_alpha=interval_cap_skip_surge_alpha,
            )
            if eff is not None and np.isfinite(eff):
                return (static_lower, max(eff, static_lower))
            return static_cap

    # This idol's own cumulative ratio, and its recent-step noise.
    idol_ratio = _idol_position_matched_ratio(
        search_df, target_event_id, float(target_idol_id), hist_ids, current_step, rank_window,
    )

    # Decay forecast: replace the cumulative (stock) anchor -- both the
    # event-wide ratio the shrinkage pulls toward, and this idol's own ratio --
    # with a forward projection of where it lands, so a decaying surge (and
    # tide-rider idols whose recent pace is back to normal) is no longer over-
    # predicted. Only trims a declining anchor (see ``_decay_forecast``); a flat/
    # rising anchor is left untouched and handled by the top-tier relaxation
    # below. Applied AFTER the (raw-cumulative) gate/persistence decision, so
    # whether we correct at all is still governed by the observed regime.
    if use_decay_forecast and total_steps is not None:
        remaining = max(0, int(total_steps) - int(current_step))
        _w_steps = int(decay_forecast_window)
        if remaining > 0 and (current_step - _w_steps) >= 1:
            ev_wago = _cached_trailing_event_ratio(
                search_df, target_event_id, hist_ids, current_step - _w_steps,
                rank_window, trim_pct,
            )
            # Persistence gate: only trim if the decline is SUSTAINED (not a
            # tentative sub-weekly dip). Applied per-anchor.
            ev_gate = True
            if decay_persistence_enabled:
                ev_gate = _decay_decline_persists(
                    lambda s: _cached_trailing_event_ratio(
                        search_df, target_event_id, hist_ids, s, rank_window, trim_pct),
                    current_step, _w_steps, decay_persistence_window,
                    decay_persistence_sample_spacing, decay_persistence_min_steps,
                    decay_deadband,
                )
            if ev_gate and ev_wago is not None and np.isfinite(ev_wago):
                event_ratio = _decay_forecast(
                    event_ratio, ev_wago, decay_forecast_w, decay_forecast_p,
                    remaining, _w_steps, decay_forecast_floor,
                )
            if idol_ratio is not None and np.isfinite(idol_ratio):
                ir_wago = _idol_position_matched_ratio(
                    search_df, target_event_id, float(target_idol_id), hist_ids,
                    current_step - _w_steps, rank_window,
                )
                idol_gate = True
                if decay_persistence_enabled:
                    idol_gate = _decay_decline_persists(
                        lambda s: _idol_position_matched_ratio(
                            search_df, target_event_id, float(target_idol_id),
                            hist_ids, s, rank_window),
                        current_step, _w_steps, decay_persistence_window,
                        decay_persistence_sample_spacing, decay_persistence_min_steps,
                        decay_deadband,
                    )
                if idol_gate and ir_wago is not None and np.isfinite(ir_wago):
                    idol_ratio = _decay_forecast(
                        idol_ratio, ir_wago, decay_forecast_w, decay_forecast_p,
                        remaining, _w_steps, decay_forecast_floor,
                    )

    def _shrunk_toward_event(ir: float) -> float:
        """Pull an OVER-the-mean idol's own ratio toward the event-wide
        ratio by an empirical-Bayes weight derived from its recent-step
        noise (noisy idol -> pulled fully to the event ratio; steady idol
        -> keeps most of its own value). If the idol's noise can't be
        estimated, pull fully to the event ratio (the conservative
        choice)."""
        if not use_eb_shrinkage:
            # Shrinkage disabled: use the idol's own (decay-forecast) ratio
            # directly, no pull toward the event mean. Makes _maybe_toptier_relax
            # a no-op (shrunk == ir), by design.
            return ir
        if between_std is None or between_std <= 0:
            return event_ratio
        within_std = _idol_ratio_noise(
            search_df, target_event_id, float(target_idol_id), hist_ids, current_step,
            rank_window, persistence_window, persistence_sample_spacing,
        )
        if within_std is None:
            return event_ratio
        between_var = between_std ** 2
        within_var = within_std ** 2
        w = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0.0
        return event_ratio + w * (ir - event_ratio)

    def _maybe_toptier_relax(shrunk: float, ir: float) -> float:
        """Recency-gated top-tier relaxation. A genuine top-tier outlier
        (own ratio more than ``toptier_relax_sigma`` cross-idol standard
        deviations above the event-wide ratio) that is ALSO still
        accelerating gets its shrinkage partially undone -- moved back
        toward its own ratio by ``toptier_relax_strength`` -- to fix
        under-prediction of the strongest idols. The acceleration test is
        WEEKDAY-ROBUST: it compares this idol's cumulative-ratio change
        over the recent window against the EVENT-WIDE cumulative-ratio
        change over the same window. Because a weekday dip (e.g. a low
        Thursday) drops the whole event's ratio too, differencing the
        idol's change against the event's change cancels that common
        weekday component -- so a bare single-window rate (which would be
        weekday/stage-confounded) is never used. The idol qualifies only
        if it did NOT lose ground relative to the field, i.e. its
        cumulative-ratio change is at least the event-wide change. Only
        ever RAISES the value; if any precondition is missing it returns
        the unrelaxed shrunk value."""
        if not use_toptier_relax or between_std is None or between_std <= 0:
            return shrunk
        # Outlier gate: far above the event mean (this is what limits the
        # relaxation to the top tail -- typically only a handful of idols).
        if ir <= event_ratio + toptier_relax_sigma * between_std:
            return shrunk
        # Recency gate (weekday-robust): idol change vs. event-wide change.
        prev_step = current_step - int(toptier_relax_recency_lookback)
        if prev_step < int(rank_window):
            return shrunk  # not enough history to judge the trend
        prev_ir = _idol_position_matched_ratio(
            search_df, target_event_id, float(target_idol_id), hist_ids, prev_step, rank_window,
        )
        if prev_ir is None:
            return shrunk
        prev_event, _, _ = _event_ratio_and_spread(
            search_df, target_event_id, hist_ids, prev_step, rank_window, trim_pct,
        )
        if prev_event is None:
            return shrunk
        # Idol must not lose ground relative to the event over the window.
        if (ir - prev_ir) < (event_ratio - prev_event) - toptier_relax_recency_tol:
            return shrunk
        relaxed = shrunk + toptier_relax_strength * (ir - shrunk)
        return max(shrunk, relaxed)

    if event_ratio > 1.0:
        # Inflation regime. Only idols running HOTTER than the event-wide
        # ratio (the "over") get pulled DOWN toward it. An idol at or below
        # the event ratio is NOT pulled up -- it keeps its own (lower)
        # ratio, so it gets a smaller breach, never a larger one than
        # warranted. Idols we can't rate at all fall back to the event ratio.
        if idol_ratio is None:
            effective_ratio = event_ratio
        elif idol_ratio > event_ratio:
            effective_ratio = _maybe_toptier_relax(_shrunk_toward_event(idol_ratio), idol_ratio)
        else:
            effective_ratio = idol_ratio
        return (static_lower, max(static_upper, effective_ratio))
    if event_ratio < 1.0:
        # Deflation regime (mirror image). Only idols running COLDER than
        # the event-wide ratio get pulled UP toward it; an idol at or above
        # the event ratio keeps its own (higher) value, never pulled down.
        if idol_ratio is None:
            effective_ratio = event_ratio
        elif idol_ratio < event_ratio:
            effective_ratio = _shrunk_toward_event(idol_ratio)
        else:
            effective_ratio = idol_ratio
        return (min(static_lower, effective_ratio), static_upper)
    return static_cap


def to_relative_trajectory(
    arr: np.ndarray, event_id: float, search_df: pd.DataFrame, min_score: float = 1.0,
    ref_step: Optional[int] = None,
) -> np.ndarray:
    """Divide ``arr`` (a raw score trajectory from ``event_id``) by a SINGLE
    scalar reference value: that event's own contemporaneous scale
    (``compute_event_scale_series``) at ``ref_step`` (default: the last index
    of ``arr``, i.e. "now" for whichever trajectory is passed in).

    Deliberately a single scalar, not an elementwise division by the whole
    per-step scale series. Dividing point-by-point by a moving scale(t)
    removes the between-event LEVEL difference but also distorts each
    trajectory's own SHAPE within the lookback window (the scale itself
    drifts over the window, which changes the effective slope) -- this
    showed up empirically as a regression on ordinary (non-surging) events,
    where there was no real cross-event mismatch to correct in the first
    place. Dividing by one fixed value removes only the level difference and
    leaves each trajectory's internal dynamics untouched, which both fixed
    that regression AND strengthened the correction on the surging event it
    was designed for (mismatched candidates separate more cleanly once
    within-window shape noise is removed).
    """
    scale = compute_event_scale_series(search_df, event_id, min_score)
    idx = (len(arr) - 1) if ref_step is None else (ref_step - 1)
    idx = max(0, min(idx, len(scale) - 1))
    ref = scale[idx]
    return arr / ref if ref > 0 else arr


# Border-100 weekday multiplier (Mon..Sun), from decay_overestimation_context.md
# §4. Weekend (Sat 1.10, Sun 1.07) scores more; Thu (0.94) least.
WD_B_BORDER100 = np.array([0.96, 0.96, 1.00, 0.94, 0.99, 1.10, 1.07])

# event_id -> start weekday (Mon=0..Sun=6). Populated by the caller
# (main.py / harness) from event_info start_at. Empty -> deseason is a no-op.
EVENT_START_WEEKDAY: Dict[float, int] = {}


def set_event_start_weekdays(mapping: Dict[float, int]) -> None:
    EVENT_START_WEEKDAY.clear()
    EVENT_START_WEEKDAY.update({float(k): int(v) for k, v in mapping.items()})


def deseasonalize_weekday(
    arr: np.ndarray,
    event_id: float,
    norm_len: int,
    wd_factors: Optional[np.ndarray] = None,
    days: int = 13,
    boost_frac: float = 6.0 / 13.0,
) -> np.ndarray:
    """Remove the weekly (weekday) seasonal hump from a normalized cumulative
    trajectory so neighbour matching is on the underlying pace, not on weekend
    bumps that land at different normalized positions for events starting on
    different weekdays.

    Each normalized step is mapped to its real event-day (piecewise-linear with
    the boost point at ``boost_frac*norm_len``), then to a calendar weekday via
    the event's start weekday; the per-step increment is divided by that
    weekday's factor and re-cumulated. Daily-resolution (the factors are daily),
    so it corrects the day-level weekend hump, not intra-day shape. No-op if the
    event's start weekday is unknown."""
    wd0 = EVENT_START_WEEKDAY.get(float(event_id))
    a = np.asarray(arr, dtype=float)
    L = len(a)
    if wd0 is None or L < 2:
        return a
    wdf = WD_B_BORDER100 if wd_factors is None else np.asarray(wd_factors, dtype=float)
    boost = boost_frac * float(norm_len)
    incs = np.empty(L, dtype=float)
    incs[0] = a[0]
    incs[1:] = np.diff(a)
    for s in range(1, L):
        if boost > 0 and s <= boost:
            real_day = (s / boost) * 6.0
        else:
            real_day = 6.0 + ((s - boost) / max(1.0, (norm_len - boost))) * 7.0
        wday = int((wd0 + int(real_day)) % 7)
        f = wdf[wday]
        if f > 0:
            incs[s] = incs[s] / f
    return np.cumsum(incs)


# ---------------------------------------------------------------------------
# Distance primitives
# ---------------------------------------------------------------------------

def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Classic dynamic-time-warping distance between two 1-D score series."""
    D = cdist(seq_a.reshape(-1, 1), seq_b.reshape(-1, 1))
    n, m = D.shape
    cost = np.zeros((n, m))
    cost[0, 0] = D[0, 0]
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + D[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + D[0, j]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = D[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[-1, -1])


# ---------------------------------------------------------------------------
# SLOPE_AWARE distance
# ---------------------------------------------------------------------------

def _slope_aware_distance(
    current_window: np.ndarray,
    candidate_window: np.ndarray,
    slope_weight: float,
) -> float:
    """Distance blending level and slope RMSEs.

        D_level = RMSE(x - y)        / (|mean(x)| + eps)
        D_slope = RMSE(diff(x) - diff(y)) / (|mean(diff(x))| + eps)
        D       = (1 - slope_weight) * D_level + slope_weight * D_slope

    Each component is relativised by the current window's own magnitude
    (or mean slope) so the two terms are commensurable before blending.
    """
    L = len(current_window)
    if L < 2:
        return float(np.sqrt(np.mean((current_window - candidate_window) ** 2)))

    a_slope = float(slope_weight)
    a_level = max(0.0, 1.0 - a_slope)

    eps = 1e-9

    # Level term
    diff_values = current_window - candidate_window
    scale_level = float(np.abs(np.mean(current_window))) + eps
    d_level = float(np.sqrt(np.mean(diff_values ** 2))) / scale_level

    # Slope term (L - 1 points)
    diff_slope = np.diff(current_window) - np.diff(candidate_window)
    scale_slope = float(np.abs(np.mean(np.diff(current_window)))) + eps
    d_slope = float(np.sqrt(np.mean(diff_slope ** 2))) / scale_slope

    return a_level * d_level + a_slope * d_slope


def trajectory_distance(
    current_partial: np.ndarray,
    candidate_partial: np.ndarray,
    lookback: int,
    metric: DistanceMetric,
    event_type: float,
    sub_types: Tuple[float],
    border: float,
    slope_weight: float,
) -> float:
    """Distance between the last ``lookback`` points of the two trajectories.

    If the group's config enables ``use_trend_weighting`` and the metric is
    RMSE, blend the raw-value RMSE with a first-difference (trend) distance.
    """
    config = get_group_config(event_type, sub_types, border)
    current_window = current_partial[-lookback:]
    candidate_window = candidate_partial[-lookback:]

    if config.use_trend_weighting and metric == DistanceMetric.RMSE:
        recent_window = min(30, lookback)
        trend_diff = np.abs(
            np.diff(current_partial[-recent_window:]) - np.diff(candidate_partial[-recent_window:])
        )
        rmse = float(np.sqrt(np.mean((current_window - candidate_window) ** 2)))
        trend_dist = float(np.mean(trend_diff))
        return (1 - config.trend_weight) * rmse + config.trend_weight * trend_dist

    if metric == DistanceMetric.DTW:
        return _dtw_distance(current_window, candidate_window)
    if metric == DistanceMetric.RMSE:
        return float(np.sqrt(np.mean((current_window - candidate_window) ** 2)))
    if metric == DistanceMetric.SLOPE_AWARE:
        return _slope_aware_distance(current_window, candidate_window, slope_weight)
    # FINAL_DIFF: mean absolute difference in the lookback window
    return float(np.mean(np.abs(current_window - candidate_window)))


# ---------------------------------------------------------------------------
# Candidate assembly & neighbour selection
# ---------------------------------------------------------------------------

def build_candidate_set(
    search_df: pd.DataFrame,
    exclude_event_id: float,
    current_step: int,
    min_event_id: float,
    min_score_at_step: float = 0.0,
    full_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[np.ndarray], List[NeighbourId]]:
    """Collect historical (event, idol) trajectories eligible as neighbours.

    A trajectory is eligible if it is not from the current event, its
    event_id is at least ``min_event_id``, its length is at least
    ``current_step``, and (when ``min_score_at_step`` > 0) its score at the
    prediction step is at least ``min_score_at_step``.

    The score gate mirrors the target-side ``MIN_CURRENT_SCORE`` check so
    "not yet started" candidates (flat-zero in the lookback window, which
    arbitrarily match low-magnitude targets by shape) don't enter the pool.

    ``full_df`` (typically the caller's ``prediction_full_df``, which is
    expected to exclude any still-in-progress event) is used as a
    completeness gate when supplied: a candidate is also required to have a
    trajectory present there, so a still-live event's partial data --
    present in ``search_df`` because the search space legitimately needs it
    to find matches against, but with no COMPLETE future to predict from --
    can never be selected as a neighbour. Without this, a neighbour with an
    empty "full" trajectory later crashes prediction/alignment (which reads
    ``full[-1]``); with a raw-score distance metric this was rare enough to
    go unnoticed, but relative/normalised distance spaces make it far more
    likely for an incomplete event's shape to look deceptively close.
    """
    historical = search_df[search_df["event_id"] != exclude_event_id]
    complete_ids = None
    if full_df is not None:
        complete_ids = set(
            map(tuple, full_df[["event_id", "idol_id"]].drop_duplicates().to_numpy())
        )
    trajectories: List[np.ndarray] = []
    ids: List[NeighbourId] = []
    for (eid, iid), group in historical.groupby(["event_id", "idol_id"]):
        if eid < min_event_id:
            continue
        if complete_ids is not None and (eid, iid) not in complete_ids:
            continue
        scores = group["score"].values
        if len(scores) < current_step:
            continue
        if min_score_at_step > 0 and scores[current_step - 1] < min_score_at_step:
            continue
        trajectories.append(scores)
        ids.append((eid, iid))
    return trajectories, ids


def _vectorized_window_distances(
    current_full: np.ndarray,
    cand_matrix: np.ndarray,
    lookback: int,
    metric: DistanceMetric,
    slope_weight: float,
    use_trend_weighting: bool,
    trend_weight: float,
) -> np.ndarray:
    """Distance from ``current_full`` to every row of ``cand_matrix`` at once.

    ``cand_matrix`` is ``(n_candidates, L)`` where every candidate has the
    same length ``L`` (they are all sliced to ``current_step`` upstream).
    Returns a length-``n_candidates`` array. Produces the same numbers as
    looping ``trajectory_distance`` over each row, for the RMSE / FINAL_DIFF /
    SLOPE_AWARE / trend-weighted-RMSE paths. DTW is handled by the caller.
    """
    cur_win = current_full[-lookback:]
    cand_win = cand_matrix[:, -lookback:]
    diff = cand_win - cur_win  # (n, w)

    if use_trend_weighting and metric == DistanceMetric.RMSE:
        recent = min(30, lookback)
        cur_recent = current_full[-recent:]
        cand_recent = cand_matrix[:, -recent:]
        cur_d = np.diff(cur_recent)
        cand_d = np.diff(cand_recent, axis=1)
        trend = np.mean(np.abs(cand_d - cur_d), axis=1)
        rmse = np.sqrt(np.mean(diff ** 2, axis=1))
        return (1 - trend_weight) * rmse + trend_weight * trend

    if metric == DistanceMetric.RMSE:
        return np.sqrt(np.mean(diff ** 2, axis=1))

    if metric == DistanceMetric.FINAL_DIFF:
        return np.mean(np.abs(diff), axis=1)

    if metric == DistanceMetric.SLOPE_AWARE:
        if lookback < 2:
            return np.sqrt(np.mean(diff ** 2, axis=1))
        eps = 1e-9
        a_slope = float(slope_weight)
        a_level = max(0.0, 1.0 - a_slope)
        scale_level = float(np.abs(np.mean(cur_win))) + eps
        d_level = np.sqrt(np.mean(diff ** 2, axis=1)) / scale_level
        cur_slope = np.diff(cur_win)
        cand_slope = np.diff(cand_win, axis=1)
        scale_slope = float(np.abs(np.mean(cur_slope))) + eps
        d_slope = np.sqrt(np.mean((cand_slope - cur_slope) ** 2, axis=1)) / scale_slope
        return a_level * d_level + a_slope * d_slope

    # Fallback (shouldn't reach here; DTW handled by caller)
    return np.sqrt(np.mean(diff ** 2, axis=1))


def find_nearest_neighbors(
    current_partial: np.ndarray,
    candidate_partials: List[np.ndarray],
    candidate_ids: List[NeighbourId],
    current_step: int,
    k: int,
    lookback: int,
    metric: DistanceMetric,
    event_type: float,
    sub_types: Tuple[float],
    border: float,
    slope_weight: float,
    target_idol_id: float = None,
    same_idol_distance_factor: float = 1.0,
    pool_k: int = None,
    rank_gap_weight: float = 0.0,
    search_df: Optional[pd.DataFrame] = None,
    target_event_id: Optional[float] = None,
    rank_gap_threshold: Optional[float] = None,
    rank_gap_max_gap: Optional[float] = None,
    rank_gap_target_inflation: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-k nearest neighbours and their distances.

    Candidates shorter than ``current_step`` are skipped. The distance is
    computed vectorised for RMSE / FINAL_DIFF / SLOPE_AWARE / trend-weighted
    RMSE (the common paths); DTW falls back to the per-candidate loop.

    If ``same_idol_distance_factor`` < 1.0 and ``target_idol_id`` is given,
    candidates sharing the target's idol_id have their distance multiplied by
    that factor, biasing selection toward the same idol across other events.

    If ``rank_gap_weight`` > 0 (requires ``search_df`` and ``target_event_id``),
    distance is multiplied by ``(1 + rank_gap_weight * |own_pctl - cand_pctl|)``
    where ``*_pctl`` is each idol's within-event percentile rank (0=best) at
    ``current_step``. Penalises candidates whose within-event standing is far
    from the target's, independent of both trajectories' absolute scale --
    see ``compute_event_percentiles``. 0.0 = off (no behaviour change).

    If ``rank_gap_threshold`` is also given, the correction is ADAPTIVE: a
    provisional (unweighted) top-k is selected first, and the mean rank-gap
    of that provisional set is compared to the threshold. The correction is
    only applied if the provisional set already looks mismatched (gap >
    threshold); otherwise the provisional, unweighted result is returned
    unchanged. This confines the correction to idol/steps that actually show
    the mismatch, rather than applying it unconditionally to every
    prediction.

    If ``rank_gap_max_gap`` is given, the correction is CATEGORICAL instead
    of magnitude-scaled: any candidate whose own rank-gap exceeds this value
    is dropped from the pool entirely (as long as enough candidates remain
    to fill ``k``), rather than having its distance multiplied by
    ``rank_gap_weight``. ``rank_gap_max_gap`` takes precedence over
    ``rank_gap_weight`` / ``rank_gap_target_inflation`` when set.

    If ``rank_gap_target_inflation`` is given (and ``rank_gap_max_gap`` is
    not), the effective weight is derived from the CURRENT mismatch severity
    instead of being a fixed constant:
        effective_weight = (rank_gap_target_inflation - 1) / mean(provisional_gaps)
    so a "typical" mismatched candidate in the provisional top-k ends up with
    its distance multiplied by approximately ``rank_gap_target_inflation``,
    whether the underlying mismatch this event/step is mild or severe. This
    avoids calibrating a fixed weight against one historical event's
    severity and having it be far too weak (or too aggressive) for a
    different event with a differently-sized mismatch. Takes precedence over
    a plain ``rank_gap_weight`` when set.

    ``pool_k`` overrides how many neighbours are returned (default ``k``). Soft
    (kernel-weighted) callers pass a larger pool so the caller can fade distant
    neighbours smoothly instead of relying on the hard rank-k cutoff.
    """
    if not candidate_partials:
        raise ValueError(f"No candidate trajectories for step {current_step}")

    kept = [(i, c) for i, c in enumerate(candidate_partials) if len(c) >= current_step]
    if not kept:
        raise ValueError(f"No comparable candidate trajectories at step {current_step}")
    kept_indices = [i for i, _ in kept]

    if metric == DistanceMetric.DTW:
        # Inherently sequential; keep the loop.
        distance_arr = np.array([
            trajectory_distance(
                current_partial, c, lookback, metric,
                event_type, sub_types, border, slope_weight,
            )
            for _, c in kept
        ])
    else:
        # All kept candidates share length current_step -> stack and vectorise.
        cand_matrix = np.vstack([c[:current_step] for _, c in kept])
        config = get_group_config(event_type, sub_types, border)
        distance_arr = _vectorized_window_distances(
            current_full=np.asarray(current_partial),
            cand_matrix=cand_matrix,
            lookback=lookback,
            metric=metric,
            slope_weight=slope_weight,
            use_trend_weighting=bool(config.use_trend_weighting),
            trend_weight=float(config.trend_weight),
        )

    id_arr = np.array(candidate_ids)[kept_indices]

    # Bias toward the same idol across other events, if requested.
    if same_idol_distance_factor != 1.0 and target_idol_id is not None and len(id_arr):
        same_idol = id_arr[:, 1] == target_idol_id
        distance_arr = distance_arr * np.where(same_idol, same_idol_distance_factor, 1.0)

    # Penalise (or exclude) candidates whose within-event standing differs a
    # lot from the target's own within-event standing, if requested.
    use_categorical = rank_gap_max_gap is not None
    use_severity_adaptive = rank_gap_target_inflation is not None
    if (rank_gap_weight > 0 or use_categorical or use_severity_adaptive) and search_df is not None and target_event_id is not None and len(id_arr):
        own_pctls = compute_event_percentiles(search_df, target_event_id, current_step)
        own_pctl = own_pctls.get(float(target_idol_id)) if target_idol_id is not None else None
        if own_pctl is not None:
            event_pctl_cache: Dict[float, Dict[float, float]] = {}

            def _gaps_for(order_ids: np.ndarray) -> np.ndarray:
                out = np.zeros(len(order_ids))
                for i, (eid, iid) in enumerate(order_ids):
                    pctls = event_pctl_cache.get(eid)
                    if pctls is None:
                        pctls = compute_event_percentiles(search_df, eid, current_step)
                        event_pctl_cache[eid] = pctls
                    cand_pctl = pctls.get(float(iid))
                    out[i] = abs(own_pctl - cand_pctl) if cand_pctl is not None else 0.0
                return out

            apply_correction = True
            mean_provisional_gap = None
            if rank_gap_threshold is not None or rank_gap_target_inflation is not None:
                # Compute the PROVISIONAL (unweighted) top-k's mean gap once;
                # used both for the adaptive gate and for severity-scaling
                # the effective weight below.
                k_provisional = min(pool_k if pool_k else k, len(distance_arr))
                provisional_order = np.argsort(distance_arr)[:k_provisional]
                provisional_gaps = _gaps_for(id_arr[provisional_order])
                mean_provisional_gap = float(np.mean(provisional_gaps))
                if rank_gap_threshold is not None:
                    apply_correction = mean_provisional_gap > rank_gap_threshold

            if apply_correction:
                gaps = _gaps_for(id_arr)
                if use_categorical:
                    # Drop candidates over the cutoff, but only if enough
                    # remain to fill k -- never shrink the pool below k.
                    keep_mask = gaps <= rank_gap_max_gap
                    if keep_mask.sum() >= min(pool_k if pool_k else k, len(distance_arr)):
                        distance_arr = distance_arr[keep_mask]
                        id_arr = id_arr[keep_mask]
                else:
                    effective_weight = rank_gap_weight
                    if rank_gap_target_inflation is not None:
                        # Derive the weight from the CURRENT event/step's own
                        # measured severity, so a fixed constant never has to
                        # be calibrated against how bad the mismatch happens
                        # to be this time.
                        if mean_provisional_gap is None:
                            k_provisional = min(pool_k if pool_k else k, len(distance_arr))
                            provisional_order = np.argsort(distance_arr)[:k_provisional]
                            mean_provisional_gap = float(np.mean(_gaps_for(id_arr[provisional_order])))
                        if mean_provisional_gap > 1e-9:
                            effective_weight = (rank_gap_target_inflation - 1.0) / mean_provisional_gap
                        else:
                            effective_weight = 0.0
                    distance_arr = distance_arr * (1.0 + effective_weight * gaps)

    k_effective = min(pool_k if pool_k else k, len(distance_arr))
    top_k_order = np.argsort(distance_arr)[:k_effective]
    return distance_arr[top_k_order], id_arr[top_k_order]

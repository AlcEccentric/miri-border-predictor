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

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

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.knn.config import DistanceMetric, get_group_config

NeighbourId = Tuple[float, float]  # (event_id, idol_id)


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
) -> Tuple[List[np.ndarray], List[NeighbourId]]:
    """Collect historical (event, idol) trajectories eligible as neighbours.

    A trajectory is eligible if it is not from the current event, its
    event_id is at least ``min_event_id``, its length is at least
    ``current_step``, and (when ``min_score_at_step`` > 0) its score at the
    prediction step is at least ``min_score_at_step``.

    The score gate mirrors the target-side ``MIN_CURRENT_SCORE`` check so
    "not yet started" candidates (flat-zero in the lookback window, which
    arbitrarily match low-magnitude targets by shape) don't enter the pool.
    """
    historical = search_df[search_df["event_id"] != exclude_event_id]
    trajectories: List[np.ndarray] = []
    ids: List[NeighbourId] = []
    for (eid, iid), group in historical.groupby(["event_id", "idol_id"]):
        if eid < min_event_id:
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-k nearest neighbours and their distances.

    Candidates shorter than ``current_step`` are skipped. The distance is
    computed vectorised for RMSE / FINAL_DIFF / SLOPE_AWARE / trend-weighted
    RMSE (the common paths); DTW falls back to the per-candidate loop.
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
    k_effective = min(k, len(distance_arr))
    top_k_order = np.argsort(distance_arr)[:k_effective]
    return distance_arr[top_k_order], id_arr[top_k_order]

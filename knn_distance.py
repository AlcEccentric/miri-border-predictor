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

from knn_config import DistanceMetric, get_group_config

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


def trajectory_distance(
    current_partial: np.ndarray,
    candidate_partial: np.ndarray,
    lookback: int,
    metric: DistanceMetric,
    event_type: float,
    sub_types: Tuple[float],
    border: float,
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
) -> Tuple[List[np.ndarray], List[NeighbourId]]:
    """Collect historical (event, idol) trajectories eligible as neighbours.

    A trajectory is eligible if it is not from the current event, its
    event_id is at least ``min_event_id``, and its length is at least
    ``current_step``.
    """
    historical = search_df[search_df["event_id"] != exclude_event_id]
    trajectories: List[np.ndarray] = []
    ids: List[NeighbourId] = []
    for (eid, iid), group in historical.groupby(["event_id", "idol_id"]):
        if eid < min_event_id:
            continue
        scores = group["score"].values
        if len(scores) >= current_step:
            trajectories.append(scores)
            ids.append((eid, iid))
    return trajectories, ids


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-k nearest neighbours and their distances.

    Candidates shorter than ``current_step`` are skipped.
    """
    if not candidate_partials:
        raise ValueError(f"No candidate trajectories for step {current_step}")

    distances: List[float] = []
    kept_indices: List[int] = []
    for idx, candidate in enumerate(candidate_partials):
        if len(candidate) < current_step:
            continue
        distances.append(trajectory_distance(
            current_partial, candidate, lookback, metric,
            event_type, sub_types, border,
        ))
        kept_indices.append(idx)

    if not distances:
        raise ValueError(f"No comparable candidate trajectories at step {current_step}")

    distance_arr = np.array(distances)
    id_arr = np.array(candidate_ids)[kept_indices]
    k_effective = min(k, len(distance_arr))
    top_k_order = np.argsort(distance_arr)[:k_effective]
    return distance_arr[top_k_order], id_arr[top_k_order]

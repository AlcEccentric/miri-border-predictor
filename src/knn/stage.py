"""
Stage parameter selection and data-source picking for ``knn.predict_curve_knn``.

A "stage" is determined by where ``current_step`` falls relative to the
``early_stage_end`` / ``mid_stage_end`` boundaries in the ``GroupConfig``.
Each stage has its own k, lookbacks, metric, and smoothing choices; this
module exposes them as a flat ``StageParams`` dataclass so callers do not
have to juggle three parallel sets of config fields.

``_DataSources`` bundles the (search/prediction, full/partial) dataframes
picked for a given stage's smoothing preferences.
"""

import threading
import weakref
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.knn.config import AlignmentMethod, DistanceMetric, GroupConfig


# ---------------------------------------------------------------------------
# Stage parameter selection
# ---------------------------------------------------------------------------

@dataclass
class StageParams:
    """Parameters picked from ``GroupConfig`` according to ``current_step``."""
    k: int
    distance_lookback: int
    align_lookback: int
    metric: DistanceMetric
    use_ensemble: bool
    method_weights: Dict[AlignmentMethod, float]
    use_smooth_for_neighbors: bool
    use_smooth_for_prediction: bool
    scale_cap: Tuple[float, float]
    slope_weight: float


def get_stage_params(config: GroupConfig, current_step: int) -> StageParams:
    """Pick early/mid/late-stage parameters by comparing ``current_step`` to stage endpoints."""
    if current_step < config.early_stage_end:
        stage = "early"
    elif current_step < config.mid_stage_end:
        stage = "mid"
    else:
        stage = "late"

    return StageParams(
        k=getattr(config, f"{stage}_stage_k"),
        distance_lookback=getattr(config, f"{stage}_stage_lookback"),
        align_lookback=getattr(config, f"{stage}_stage_lookback_for_align"),
        metric=getattr(config, f"{stage}_stage_metric"),
        use_ensemble=getattr(config, f"{stage}_stage_use_ensemble"),
        method_weights=getattr(config, f"{stage}_stage_weights"),
        use_smooth_for_neighbors=getattr(config, f"{stage}_stage_use_smooth_for_neighbors"),
        use_smooth_for_prediction=getattr(config, f"{stage}_stage_use_smooth_for_prediction"),
        scale_cap=getattr(config, f"{stage}_stage_scale_cap"),
        slope_weight=getattr(config, f"{stage}_stage_slope_weight"),
    )


# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------

@dataclass
class DataSources:
    """Dataframes picked for a given stage configuration."""
    search_partial_df: pd.DataFrame        # used to find neighbours
    prediction_full_df: pd.DataFrame       # neighbours' full trajectories used for prediction
    prediction_partial_df: pd.DataFrame    # neighbours' partial trajectories used for alignment
    current_scores: np.ndarray             # current event's scores used for prediction/alignment


def scores_of(df: pd.DataFrame, event_id: float, idol_id: float) -> np.ndarray:
    """Return the ordered score series for one ``(event_id, idol_id)`` in ``df``.

    The naive implementation ``df[(df.event_id==e) & (df.idol_id==i)]["score"]``
    is an O(N) boolean scan over the whole dataframe and gets called hundreds of
    thousands of times during a backtest. Instead we build a one-time
    ``{(event_id, idol_id): score_array}`` lookup per dataframe and cache it,
    keyed by the dataframe object via a weak reference so it is evicted
    automatically when the frame is garbage collected.

    The cache is shared across threads (the normalised frames are reused by
    every ``(event, step)`` task), so the (one-time) build is guarded by a lock.
    Group order matches ``df`` row order, so results are identical to the scan.
    """
    groups = _scores_groups_for(df)
    return groups.get((float(event_id), float(idol_id)), _EMPTY_SCORES)


_EMPTY_SCORES = np.array([], dtype=float)
# DataFrames are unhashable, so we cannot use them as dict keys directly.
# Key by id(df) and drop the entry via weakref.finalize when the frame is GC'd
# (which also guards against id reuse).
_scores_cache: Dict[int, Dict[Tuple[float, float], np.ndarray]] = {}
_scores_cache_lock = threading.Lock()


def _scores_groups_for(df: pd.DataFrame) -> Dict[Tuple[float, float], np.ndarray]:
    """Return (and build/cache) the ``(event_id, idol_id) -> scores`` map for ``df``."""
    key = id(df)
    cached = _scores_cache.get(key)
    if cached is not None:
        return cached
    with _scores_cache_lock:
        cached = _scores_cache.get(key)
        if cached is not None:
            return cached
        groups: Dict[Tuple[float, float], np.ndarray] = {
            (float(eid), float(iid)): g["score"].to_numpy()
            for (eid, iid), g in df.groupby(["event_id", "idol_id"], sort=False)
        }
        _scores_cache[key] = groups
        weakref.finalize(df, _scores_cache.pop, key, None)
        return groups


def select_data_sources(
    event_id: float,
    idol_id: float,
    norm_partial: pd.DataFrame,
    norm_full: pd.DataFrame,
    smooth_partial: Optional[pd.DataFrame],
    smooth_full: Optional[pd.DataFrame],
    use_smooth_for_neighbors: bool,
    use_smooth_for_prediction: bool,
) -> DataSources:
    """Pick the search / prediction dataframes per the stage configuration.

    Smoothed variants are used only when both the stage says so *and* the
    corresponding smoothed dataframe is available; otherwise we fall back
    to the normalised dataframes.
    """
    if use_smooth_for_neighbors and smooth_partial is not None:
        search_partial_df = smooth_partial
    else:
        search_partial_df = norm_partial

    if use_smooth_for_prediction and smooth_full is not None:
        # Original logic used smooth_partial for the "current" scores here,
        # even when only smooth_full was set. Preserve that.
        return DataSources(
            search_partial_df=search_partial_df,
            prediction_full_df=smooth_full,
            prediction_partial_df=smooth_partial,
            current_scores=scores_of(smooth_partial, event_id, idol_id),
        )
    return DataSources(
        search_partial_df=search_partial_df,
        prediction_full_df=norm_full,
        prediction_partial_df=norm_partial,
        current_scores=scores_of(norm_partial, event_id, idol_id),
    )

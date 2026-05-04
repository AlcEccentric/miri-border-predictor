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
    """Return the ordered score series for one ``(event_id, idol_id)`` in ``df``."""
    return df[(df["event_id"] == event_id) & (df["idol_id"] == idol_id)]["score"].values


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

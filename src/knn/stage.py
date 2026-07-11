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
    rank_gap_weight: float
    rank_gap_threshold: Optional[float]
    rank_gap_max_gap: Optional[float]
    rank_gap_target_inflation: Optional[float]
    use_relative_scale_for_search: bool
    use_pureraw_for_incloud: bool
    pureraw_band_sigma: float
    use_deseason_search: bool
    use_adaptive_scale_cap: bool
    adaptive_cap_rank_window: int
    adaptive_cap_half_width: int
    adaptive_cap_max_recent_events: Optional[int]
    adaptive_cap_min_historical_events: int
    use_reversal_gated_ewma: bool
    reversal_rate_window: int
    reversal_sample_spacing: int
    reversal_short_window: int
    reversal_long_window: int
    reversal_min_short_magnitude: float
    ewma_alpha: float
    ewma_lookback: int
    use_macro_regime_gate: bool
    macro_regime_rank_window: int
    macro_regime_trim_pct: float
    macro_regime_min_historical_events: int
    macro_regime_persistence_window: int
    macro_regime_persistence_min_steps: int
    macro_regime_persistence_sample_spacing: int
    use_eb_shrinkage: bool
    use_toptier_relax: bool
    toptier_relax_sigma: float
    toptier_relax_strength: float
    toptier_relax_recency_lookback: int
    toptier_relax_recency_tol: float
    use_decay_forecast: bool
    decay_forecast_p: float
    decay_forecast_w: float
    decay_forecast_window: int
    decay_forecast_floor: float
    decay_persistence_enabled: bool
    decay_persistence_window: int
    decay_persistence_sample_spacing: int
    decay_persistence_min_steps: int
    decay_deadband: float
    use_interval_cap: bool
    interval_cap_base_window_days: float
    interval_cap_reversion_frac: float
    interval_cap_floor: float
    interval_cap_band_clamp: bool
    interval_cap_band_sigma: float
    interval_cap_band_clamp_frac: float
    interval_cap_band_reference: str
    use_deseason_ir: bool
    interval_cap_hot_only: bool
    interval_cap_hot_sigma: float
    skip_haircut_f: float
    skip_observed_blend_enabled: bool
    skip_observed_full_weight_days: float
    skip_observed_max_ratio: float
    skip_observed_min_ratio: float
    skip_observed_fast_weight_days: float
    skip_observed_fast_ratio: float
    skip_surge_alpha: float


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
        rank_gap_weight=getattr(config, f"{stage}_stage_rank_gap_weight", 0.0),
        rank_gap_threshold=getattr(config, f"{stage}_stage_rank_gap_threshold", None),
        rank_gap_max_gap=getattr(config, f"{stage}_stage_rank_gap_max_gap", None),
        rank_gap_target_inflation=getattr(config, f"{stage}_stage_rank_gap_target_inflation", None),
        use_relative_scale_for_search=getattr(config, f"{stage}_stage_use_relative_scale_for_search", False),
        use_pureraw_for_incloud=getattr(config, "use_pureraw_for_incloud", False),
        pureraw_band_sigma=getattr(config, "pureraw_band_sigma", 2.0),
        use_deseason_search=getattr(config, "use_deseason_search", False),
        use_adaptive_scale_cap=getattr(config, f"{stage}_stage_use_adaptive_scale_cap", False),
        adaptive_cap_rank_window=getattr(config, f"{stage}_stage_adaptive_cap_rank_window", 24),
        adaptive_cap_half_width=getattr(config, f"{stage}_stage_adaptive_cap_half_width", 2),
        adaptive_cap_max_recent_events=getattr(config, f"{stage}_stage_adaptive_cap_max_recent_events", 4),
        adaptive_cap_min_historical_events=getattr(config, "adaptive_cap_min_historical_events", 2),
        use_reversal_gated_ewma=getattr(config, "use_reversal_gated_ewma", False),
        reversal_rate_window=getattr(config, "reversal_rate_window", 40),
        reversal_sample_spacing=getattr(config, "reversal_sample_spacing", 10),
        reversal_short_window=getattr(config, "reversal_short_window", 30),
        reversal_long_window=getattr(config, "reversal_long_window", 80),
        reversal_min_short_magnitude=getattr(config, "reversal_min_short_magnitude", 0.2),
        ewma_alpha=getattr(config, "ewma_alpha", 0.3),
        ewma_lookback=getattr(config, "ewma_lookback", 80),
        use_macro_regime_gate=getattr(config, "use_macro_regime_gate", False),
        macro_regime_rank_window=getattr(config, "macro_regime_rank_window", 24),
        macro_regime_trim_pct=getattr(config, "macro_regime_trim_pct", 0.1),
        macro_regime_min_historical_events=getattr(config, "macro_regime_min_historical_events", 2),
        macro_regime_persistence_window=getattr(config, "macro_regime_persistence_window", 40),
        macro_regime_persistence_min_steps=getattr(config, "macro_regime_persistence_min_steps", 3),
        macro_regime_persistence_sample_spacing=getattr(config, "macro_regime_persistence_sample_spacing", 10),
        use_eb_shrinkage=getattr(config, "use_eb_shrinkage", True),
        use_toptier_relax=getattr(config, "use_toptier_relax", False),
        toptier_relax_sigma=getattr(config, "toptier_relax_sigma", 2.0),
        toptier_relax_strength=getattr(config, "toptier_relax_strength", 0.7),
        toptier_relax_recency_lookback=getattr(config, "toptier_relax_recency_lookback", 46),
        toptier_relax_recency_tol=getattr(config, "toptier_relax_recency_tol", 0.0),
        use_decay_forecast=getattr(config, "use_decay_forecast", False),
        decay_forecast_p=getattr(config, "decay_forecast_p", 0.8),
        decay_forecast_w=getattr(config, "decay_forecast_w", 0.5),
        decay_forecast_window=getattr(config, "decay_forecast_window", 46),
        decay_forecast_floor=getattr(config, "decay_forecast_floor", 1.0),
        decay_persistence_enabled=getattr(config, "decay_persistence_enabled", False),
        decay_persistence_window=getattr(config, "decay_persistence_window", 69),
        decay_persistence_sample_spacing=getattr(config, "decay_persistence_sample_spacing", 17),
        decay_persistence_min_steps=getattr(config, "decay_persistence_min_steps", 3),
        decay_deadband=getattr(config, "decay_deadband", 0.01),
        use_interval_cap=getattr(config, "use_interval_cap", False),
        interval_cap_base_window_days=getattr(config, "interval_cap_base_window_days", 2.0),
        interval_cap_reversion_frac=getattr(config, "interval_cap_reversion_frac", 0.0),
        interval_cap_floor=getattr(config, "interval_cap_floor", 1.0),
        interval_cap_band_clamp=getattr(config, "interval_cap_band_clamp", False),
        interval_cap_band_sigma=getattr(config, "interval_cap_band_sigma", 2.0),
        interval_cap_band_clamp_frac=getattr(config, "interval_cap_band_clamp_frac", 1.0),
        interval_cap_band_reference=getattr(config, "interval_cap_band_reference", "current_event"),
        use_deseason_ir=getattr(config, "use_deseason_ir", False),
        interval_cap_hot_only=getattr(config, "interval_cap_hot_only", False),
        interval_cap_hot_sigma=getattr(config, "interval_cap_hot_sigma", 2.0),
        skip_haircut_f=getattr(config, "skip_haircut_f", 0.90),
        skip_observed_blend_enabled=getattr(config, "skip_observed_blend_enabled", False),
        skip_observed_full_weight_days=getattr(config, "skip_observed_full_weight_days", 2.0),
        skip_observed_max_ratio=getattr(config, "skip_observed_max_ratio", 1.0),
        skip_observed_min_ratio=getattr(config, "skip_observed_min_ratio", 0.0),
        skip_observed_fast_weight_days=getattr(config, "skip_observed_fast_weight_days", 0.0),
        skip_observed_fast_ratio=getattr(config, "skip_observed_fast_ratio", 1.0),
        skip_surge_alpha=getattr(config, "skip_surge_alpha", 0.0),
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

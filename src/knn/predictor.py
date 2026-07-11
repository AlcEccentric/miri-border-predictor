"""
KNN-based trajectory prediction - public entry point.

Flow (implemented in ``predict_curve_knn``):

    1.  Look up the (event_type, sub_types, border) config and pick
        stage-specific parameters (k, lookback, metric, etc.) based on
        ``current_step``. (``knn_stage``)
    2.  Choose which dataframes to use (normalised vs smoothed) for
        (a) the neighbour search and (b) the prediction/alignment.
        (``knn_stage``)
    3.  Build a candidate set of historical (event_id, idol_id)
        trajectories long enough to compare against the current partial
        trajectory. (``knn_distance``)
    4.  Compute a distance between the current partial trajectory and each
        candidate; keep the top-k as neighbours. (``knn_distance``)
    5.  Run either an ensemble across alignment methods or a single-method
        weighted average over the neighbours. (``knn_alignment``)

Glossary (used throughout this package):

    "current"  - the event/idol we are predicting for.
    "neighbour"- a historical event/idol chosen as one of the k-nearest.
    "partial"  - only the first ``current_step`` observations are used.
    "full"     - the complete trajectory from step 0 to the event's end.

Public symbols used by other modules: ``predict_curve_knn``, ``get_filtered_df``.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.knn.alignment import (
    ensemble_predict,
    fetch_neighbor_trajectories,
    single_method_predict,
)
from src.knn.config import AlignmentMethod, get_group_config
from src.knn.distance import (
    build_candidate_set,
    compute_adaptive_scale_cap,
    compute_macro_regime_scale_cap,
    deseasonalize_weekday,
    find_nearest_neighbors,
    idol_in_normal_cloud,
    to_relative_trajectory,
)
from src.knn.plotting import (
    is_debug_logging,
    plot_current_and_neighbors,
    plot_neighbors_full_and_prediction,
)
from src.knn.stage import get_stage_params, scores_of, select_data_sources


MIN_CURRENT_SCORE = 5000  # skip minor idols whose latest score is too low to predict


def _neighbor_daily_increments(
    neighbor_full_list: List[np.ndarray],
    weights: Optional[np.ndarray],
    current_step: int,
    days: int = 13,
) -> Optional[np.ndarray]:
    """KNN-weighted average of the neighbours' per-day score increments over
    the REMAINING days (day boundaries at ``current_step + k*steps_per_day``),
    used as the Δḡ_k weights for the interval-anchored cap.

    The cap collapses the projected daily interval ratios via
    ``Σ_k iR_k·Δḡ_k / Σ_k Δḡ_k``, weighting each day by how much a typical
    neighbour scores that day (dash/boost heaviest). Only the relative sizes
    matter (absolute scale cancels in the ratio), so the neighbours' own
    normalized increments are used directly. Returns ``None`` if uncomputable;
    the cap then falls back to equal weighting, which at reversion_frac=0 is
    identical anyway."""
    if not neighbor_full_list:
        return None
    L = min(len(f) for f in neighbor_full_list)
    if L < 2 or current_step >= L - 1:
        return None
    if weights is None or len(weights) == 0:
        w = np.ones(len(neighbor_full_list), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        w = np.ones(len(neighbor_full_list), dtype=float)
    w = w / w.sum()
    avg = np.zeros(L, dtype=float)
    for f, wi in zip(neighbor_full_list, w):
        avg += wi * np.asarray(f[:L], dtype=float)
    spd = L / float(days)
    incs: List[float] = []
    k = 1
    while True:
        start = current_step + (k - 1) * spd
        if start >= L - 1:
            break
        end = min(current_step + k * spd, L - 1)
        si = max(0, min(int(round(start)), L - 1))
        ei = max(0, min(int(round(end)), L - 1))
        incs.append(max(0.0, float(avg[ei] - avg[si])))
        if current_step + k * spd >= L:
            break
        k += 1
    return np.asarray(incs, dtype=float) if incs else None


def get_filtered_df(
    df: pd.DataFrame,
    event_type: float,
    border: float,
    sub_types: List[float],
) -> pd.DataFrame:
    """Public helper used by other modules. Filter by event_type, sub_type, border."""
    return df[
        (df["event_type"] == event_type)
        & (df["sub_event_type"].isin(sub_types))
        & (df["border"] == border)
    ]


def predict_curve_knn(
    event_id: float,
    idol_id: int,
    border: float,
    sub_types: tuple,
    current_step: int,
    norm_data: pd.DataFrame,
    norm_partial_data: pd.DataFrame,
    smooth_partial_data: pd.DataFrame,
    smooth_full_data: pd.DataFrame,
    candidate_cache: Optional[dict] = None,
    ir_crossing_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the full score trajectory for ``(event_id, idol_id, border)``.

    ``candidate_cache``: optional dict, scoped by the caller to a single
    (event, step). The neighbour candidate pool depends only on
    (search_df, exclude_event_id, current_step) — NOT on idol_id — so when
    predicting all 52 idols of an anniversary event at the same step, passing
    one shared dict lets the expensive groupby run once instead of 52x.
    Production callers leave it None (no change in behaviour).

    Returns
    -------
    prediction : np.ndarray
        The predicted full trajectory (same length as a completed event).
        Empty array if the current event's latest score is below ``MIN_CURRENT_SCORE``.
    neighbor_ids : np.ndarray
        Array of ``(event_id, idol_id)`` pairs selected as nearest neighbours.
    distances : np.ndarray
        Distance between the current trajectory and each neighbour.
    """
    logging.info(
        f"Running knn for event {event_id}, idol {idol_id}, "
        f"border {border}, step {current_step}"
    )

    # Early exit for minor idols with insufficient score
    current_partial_norm = scores_of(norm_partial_data, event_id, idol_id)
    if len(current_partial_norm) == 0 or current_partial_norm[-1] < MIN_CURRENT_SCORE:
        return np.array([]), np.array([]), np.array([])

    # Pick stage parameters
    event_type = norm_partial_data[
        (norm_partial_data["event_id"] == event_id)
        & (norm_partial_data["idol_id"] == idol_id)
    ].iloc[0]["event_type"]
    config = get_group_config(event_type, sub_types, border)

    common = dict(
        event_id=event_id, idol_id=idol_id, border=border, sub_types=sub_types,
        current_step=current_step, event_type=event_type, config=config,
        norm_data=norm_data, norm_partial_data=norm_partial_data,
        smooth_partial_data=smooth_partial_data, smooth_full_data=smooth_full_data,
        candidate_cache=candidate_cache,
        ir_crossing_step=ir_crossing_step,
    )

    # If we are within the blend half-width of a stage boundary, compute the
    # prediction for BOTH adjacent stages and linearly blend them. This removes
    # the discontinuous jump a forecast shows as current_step crosses the
    # boundary (the whole method - k, metric, ensemble, weights - changes at
    # once). Outside the transition window we run a single stage as before.
    blend = _blend_spec(current_step, config)
    if blend is None:
        stage = get_stage_params(config, current_step)
        return _predict_for_stage(stage, **common)

    boundary, w_upper = blend
    stage_lower = get_stage_params(config, boundary - 1)
    stage_upper = get_stage_params(config, boundary)
    pred_l, nids_l, dist_l = _predict_for_stage(stage_lower, **common)
    pred_u, nids_u, dist_u = _predict_for_stage(stage_upper, **common)

    if len(pred_l) == 0:
        return pred_u, nids_u, dist_u
    if len(pred_u) == 0:
        return pred_l, nids_l, dist_l

    n = min(len(pred_l), len(pred_u))
    prediction = (1.0 - w_upper) * pred_l[:n] + w_upper * pred_u[:n]
    # Report neighbour info from whichever stage dominates the blend.
    if w_upper >= 0.5:
        return prediction, nids_u, dist_u
    return prediction, nids_l, dist_l


def _blend_spec(current_step: int, config) -> Optional[Tuple[int, float]]:
    """Return ``(boundary, w_upper)`` if ``current_step`` is within the blend
    half-width of a stage boundary, else ``None``.

    ``w_upper`` ramps linearly 0->1 across ``[boundary - hw, boundary + hw)``,
    so the prediction is all-lower-stage at ``boundary - hw`` and
    all-upper-stage at ``boundary + hw``, with a 50/50 split exactly at the
    boundary. The nearest boundary wins if windows ever overlap.
    """
    hw = config.stage_blend_halfwidth
    if hw <= 0:
        return None
    best = None
    for b in (config.early_stage_end, config.mid_stage_end):
        if b - hw <= current_step < b + hw:
            dist = abs(current_step - b)
            if best is None or dist < best[0]:
                w = (current_step - (b - hw)) / (2.0 * hw)
                best = (dist, b, min(1.0, max(0.0, w)))
    if best is None:
        return None
    return best[1], best[2]


def _predict_for_stage(
    stage,
    *,
    event_id: float,
    idol_id: int,
    border: float,
    sub_types: tuple,
    current_step: int,
    event_type: float,
    config,
    norm_data: pd.DataFrame,
    norm_partial_data: pd.DataFrame,
    smooth_partial_data: pd.DataFrame,
    smooth_full_data: pd.DataFrame,
    candidate_cache: Optional[dict],
    ir_crossing_step: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce a prediction for one fully-resolved set of stage parameters."""
    # Pick dataframes for neighbour search vs prediction
    sources = select_data_sources(
        event_id=event_id, idol_id=idol_id,
        norm_partial=norm_partial_data, norm_full=norm_data,
        smooth_partial=smooth_partial_data, smooth_full=smooth_full_data,
        use_smooth_for_neighbors=stage.use_smooth_for_neighbors,
        use_smooth_for_prediction=stage.use_smooth_for_prediction,
    )
    current_scores_for_search = scores_of(sources.search_partial_df, event_id, idol_id)

    # Build candidate set (or reuse a cached pool for this event+step). The
    # pool is identical across idols of the same event/step, so a caller-
    # supplied cache avoids rebuilding it 52x for anniversary events. The
    # key includes use_smooth_for_neighbors because the search dataframe (and
    # thus the pool) differs between stages that smooth and those that don't;
    # use_smooth_for_prediction is included too since the completeness gate
    # below now filters against prediction_full_df, which also depends on it.
    cache_key = (event_id, current_step, stage.use_smooth_for_neighbors, stage.use_smooth_for_prediction)
    if candidate_cache is not None and cache_key in candidate_cache:
        candidate_partials, candidate_ids = candidate_cache[cache_key]
    else:
        candidate_partials, candidate_ids = build_candidate_set(
            search_df=sources.search_partial_df,
            exclude_event_id=event_id,
            current_step=current_step,
            min_event_id=config.least_neighbor_id,
            # Exclude only literally-unstarted candidates (score == 0 at the
            # prediction step). Don't impose the target-side MIN_CURRENT_SCORE
            # floor on neighbours — that would strip out small-magnitude
            # candidates and force the pool to be much larger than the target,
            # producing predictions inflated by ~1000x for type 5 low-popularity
            # idols at small borders.
            min_score_at_step=1.0,
            # Completeness gate: a candidate must also have a full trajectory,
            # so a still-live event's partial data (present in search_df, but
            # with no complete future) can never be selected as a neighbour.
            full_df=sources.prediction_full_df,
        )
        if candidate_cache is not None:
            candidate_cache[cache_key] = (candidate_partials, candidate_ids)
    if not candidate_partials:
        raise ValueError(f"No valid historical partial trajectories for step {current_step}")

    logging.info(f"Latest score value for current idol: {current_scores_for_search[-1]}")

    # Make trajectories from differently-inflated events comparable in the
    # SEARCH (distance) space only -- divide each trajectory by its own
    # event's contemporaneous scale (median score across idols in that
    # event) at current_step, a single scalar reference value (not an
    # elementwise per-step division, which would distort each trajectory's
    # own shape within the lookback window). Prediction/alignment continue
    # to use the raw (non-rescaled) trajectories fetched later via
    # fetch_neighbor_trajectories.
    search_current = np.array(current_scores_for_search)
    search_candidates = [c[:current_step] for c in candidate_partials]
    _use_rel = getattr(stage, "use_relative_scale_for_search", False)
    # PURERAW gate: an idol whose cumulative ratio is still inside the historical
    # normal cloud is well matched by raw-score neighbours -- skip the relative
    # rescale (added for the inflation regime) for it, keeping PURERAW. Inflated
    # (above-band) idols keep relative search.
    if _use_rel and getattr(stage, "use_pureraw_for_incloud", False):
        _all_eids = sorted({float(cid[0]) for cid in candidate_ids})
        _incloud = idol_in_normal_cloud(
            sources.search_partial_df, event_id, idol_id, _all_eids, current_step,
            getattr(stage, "macro_regime_rank_window", 24),
            getattr(stage, "macro_regime_trim_pct", 0.1),
            getattr(stage, "macro_regime_min_historical_events", 3),
            getattr(stage, "pureraw_band_sigma", 2.0),
        )
        if _incloud is True:
            _use_rel = False
    if _use_rel:
        search_current = to_relative_trajectory(search_current, event_id, sources.search_partial_df)
        search_candidates = [
            to_relative_trajectory(c, float(cid[0]), sources.search_partial_df)
            for c, cid in zip(search_candidates, candidate_ids)
        ]
    # Weekday-deseasonalize the search trajectories (remove the weekend hump so
    # events starting on different weekdays match on underlying pace, not on
    # misaligned humps). Composes with either raw (PURERAW) or relative search.
    if getattr(stage, "use_deseason_search", False):
        _nl = max((len(c) for c in candidate_partials), default=current_step)
        search_current = deseasonalize_weekday(search_current, event_id, _nl)
        search_candidates = [
            deseasonalize_weekday(c, float(cid[0]), _nl)
            for c, cid in zip(search_candidates, candidate_ids)
        ]

    # Soft (kernel-weighted) blending: pull a larger neighbour pool so distant
    # neighbours can be faded out smoothly rather than dropped at the rank-k
    # cliff. 0 => classic hard top-k.
    bw_k = getattr(config, "soft_knn_bandwidth_k", 0)
    pool_k = 3 * bw_k if bw_k > 0 else None
    distances, neighbor_ids = find_nearest_neighbors(
        current_partial=search_current,
        candidate_partials=search_candidates,
        candidate_ids=candidate_ids,
        current_step=current_step,
        k=stage.k,
        lookback=stage.distance_lookback,
        metric=stage.metric,
        event_type=event_type,
        sub_types=sub_types,
        border=border,
        slope_weight=stage.slope_weight,
        target_idol_id=idol_id,
        same_idol_distance_factor=config.same_idol_distance_factor,
        pool_k=pool_k,
        rank_gap_weight=getattr(stage, "rank_gap_weight", 0.0),
        search_df=sources.search_partial_df,
        target_event_id=event_id,
        rank_gap_threshold=getattr(stage, "rank_gap_threshold", None),
        rank_gap_max_gap=getattr(stage, "rank_gap_max_gap", None),
        rank_gap_target_inflation=getattr(stage, "rank_gap_target_inflation", None),
    )

    # Gaussian-kernel neighbour weights with bandwidth = the bw_k-th neighbour's
    # distance. A neighbour's weight then decays smoothly toward zero as it
    # recedes, so crossing the pool boundary between steps costs ~nothing. None
    # => let the alignment use its classic inverse-distance weighting.
    soft_weights = None
    if bw_k > 0 and len(distances) > 0:
        h = float(distances[min(bw_k, len(distances)) - 1])
        if h > 0 and np.isfinite(h):
            w = np.exp(-0.5 * (np.asarray(distances, dtype=float) / h) ** 2)
            total = w.sum()
            if total > 0:
                soft_weights = w / total

    if is_debug_logging():
        plot_current_and_neighbors(
            current_scores=np.array(sources.current_scores),
            neighbor_ids=neighbor_ids,
            candidate_trajectories=candidate_partials,
            candidate_ids=candidate_ids,
            current_step=current_step,
            current_event_id=event_id,
            current_idol_id=idol_id,
            border=border,
        )

    # Build the prediction from the k chosen neighbours
    neighbor_full_list, neighbor_partial_list = fetch_neighbor_trajectories(
        sources.prediction_full_df, sources.prediction_partial_df, neighbor_ids,
    )

    # Δḡ_k weights for the interval-anchored cap: KNN-weighted per-day increments
    # of the selected neighbours over the remaining days. Only computed when the
    # interval cap is enabled (else None -> the cumulative path is unaffected).
    neighbor_daily_increments = None
    if getattr(stage, "use_interval_cap", False) and neighbor_full_list:
        if soft_weights is not None:
            _inc_w = np.asarray(soft_weights, dtype=float)
        elif len(distances) > 0:
            _inc_w = 1.0 / (np.asarray(distances, dtype=float) + 1e-9)
        else:
            _inc_w = None
        neighbor_daily_increments = _neighbor_daily_increments(
            neighbor_full_list, _inc_w, current_step,
        )

    # Adaptive scale cap: loosen ONLY the bound (upper or lower) a live,
    # measured growth ratio for THIS idol points toward, instead of using a
    # fixed static cap for every idol regardless of how its current growth
    # compares to popularity-matched idols in recent historical events. See
    # compute_adaptive_scale_cap's docstring for the full method. Off by
    # default (falls straight through to stage.scale_cap unchanged).
    effective_scale_cap = stage.scale_cap
    if getattr(stage, "use_macro_regime_gate", False):
        # Macro-regime gate takes precedence over the per-idol adaptive cap
        # when both are enabled for a stage -- it's an event-wide statistic
        # (identical result for every idol in the event at this step, see
        # compute_macro_regime_scale_cap's cache), validated to avoid the
        # per-idol mechanism's failure mode of over-correcting idols whose
        # predictions were already accurate on events with only a modest
        # real deviation. See docs/relative_scale_search_normalization.md.
        all_event_ids = sorted({float(cid[0]) for cid in candidate_ids})
        max_recent = getattr(stage, "adaptive_cap_max_recent_events", None)
        historical_event_ids = all_event_ids[-max_recent:] if max_recent else all_event_ids
        effective_scale_cap = compute_macro_regime_scale_cap(
            search_df=sources.search_partial_df,
            target_event_id=event_id,
            target_idol_id=idol_id,
            current_step=current_step,
            historical_event_ids=historical_event_ids,
            rank_window=stage.macro_regime_rank_window,
            trim_pct=stage.macro_regime_trim_pct,
            min_historical_events=stage.macro_regime_min_historical_events,
            static_cap=stage.scale_cap,
            persistence_window=stage.macro_regime_persistence_window,
            persistence_min_steps=stage.macro_regime_persistence_min_steps,
            persistence_sample_spacing=stage.macro_regime_persistence_sample_spacing,
            use_eb_shrinkage=getattr(stage, "use_eb_shrinkage", True),
            use_toptier_relax=getattr(stage, "use_toptier_relax", False),
            toptier_relax_sigma=getattr(stage, "toptier_relax_sigma", 2.0),
            toptier_relax_strength=getattr(stage, "toptier_relax_strength", 0.7),
            toptier_relax_recency_lookback=getattr(stage, "toptier_relax_recency_lookback", 46),
            toptier_relax_recency_tol=getattr(stage, "toptier_relax_recency_tol", 0.0),
            use_decay_forecast=getattr(stage, "use_decay_forecast", False),
            decay_forecast_p=getattr(stage, "decay_forecast_p", 0.8),
            decay_forecast_w=getattr(stage, "decay_forecast_w", 0.5),
            decay_forecast_window=getattr(stage, "decay_forecast_window", 46),
            decay_forecast_floor=getattr(stage, "decay_forecast_floor", 1.0),
            decay_persistence_enabled=getattr(stage, "decay_persistence_enabled", False),
            decay_persistence_window=getattr(stage, "decay_persistence_window", 69),
            decay_persistence_sample_spacing=getattr(stage, "decay_persistence_sample_spacing", 17),
            decay_persistence_min_steps=getattr(stage, "decay_persistence_min_steps", 3),
            decay_deadband=getattr(stage, "decay_deadband", 0.01),
            use_interval_cap=getattr(stage, "use_interval_cap", False),
            interval_cap_base_window_days=getattr(stage, "interval_cap_base_window_days", 2.0),
            interval_cap_reversion_frac=getattr(stage, "interval_cap_reversion_frac", 0.0),
            interval_cap_floor=getattr(stage, "interval_cap_floor", 1.0),
            interval_cap_band_clamp=getattr(stage, "interval_cap_band_clamp", False),
            interval_cap_band_sigma=getattr(stage, "interval_cap_band_sigma", 2.0),
            interval_cap_band_clamp_frac=getattr(stage, "interval_cap_band_clamp_frac", 1.0),
            interval_cap_band_reference=getattr(stage, "interval_cap_band_reference", "current_event"),
            interval_cap_deseason_ir=getattr(stage, "use_deseason_ir", False),
            interval_cap_hot_only=getattr(stage, "interval_cap_hot_only", False),
            interval_cap_hot_sigma=getattr(stage, "interval_cap_hot_sigma", 2.0),
            interval_cap_crossing_step=ir_crossing_step,
            interval_cap_skip_haircut_f=getattr(stage, "skip_haircut_f", 0.90),
            interval_cap_skip_blend_enabled=getattr(stage, "skip_observed_blend_enabled", False),
            interval_cap_skip_full_weight_days=getattr(stage, "skip_observed_full_weight_days", 2.0),
            interval_cap_skip_max_ratio=getattr(stage, "skip_observed_max_ratio", 1.0),
            interval_cap_skip_min_ratio=getattr(stage, "skip_observed_min_ratio", 0.0),
            interval_cap_skip_fast_weight_days=getattr(stage, "skip_observed_fast_weight_days", 0.0),
            interval_cap_skip_fast_ratio=getattr(stage, "skip_observed_fast_ratio", 1.0),
            interval_cap_skip_surge_alpha=getattr(stage, "skip_surge_alpha", 0.0),
            neighbor_daily_increments=neighbor_daily_increments,
            # Normalized event length (full neighbour trajectories are all
            # norm_event_length long); used to size the forecast's remaining
            # horizon. None -> forecast is skipped.
            total_steps=max((len(f) for f in neighbor_full_list), default=None),
        )
    elif getattr(stage, "use_adaptive_scale_cap", False):
        all_event_ids = sorted({float(cid[0]) for cid in candidate_ids})
        # Only compare against RECENT historical events -- older anniversaries
        # can reflect a materially different game meta/player base, making
        # them a poor comparison point for "is this event's growth unusual".
        max_recent = getattr(stage, "adaptive_cap_max_recent_events", None)
        historical_event_ids = all_event_ids[-max_recent:] if max_recent else all_event_ids
        effective_scale_cap = compute_adaptive_scale_cap(
            search_df=sources.search_partial_df,
            target_event_id=event_id,
            target_idol_id=idol_id,
            current_step=current_step,
            historical_event_ids=historical_event_ids,
            rank_window=stage.adaptive_cap_rank_window,
            half_width=stage.adaptive_cap_half_width,
            min_historical_events=stage.adaptive_cap_min_historical_events,
            static_cap=stage.scale_cap,
            use_reversal_gated_ewma=getattr(stage, "use_reversal_gated_ewma", False),
            reversal_rate_window=getattr(stage, "reversal_rate_window", 40),
            reversal_sample_spacing=getattr(stage, "reversal_sample_spacing", 10),
            reversal_short_window=getattr(stage, "reversal_short_window", 30),
            reversal_long_window=getattr(stage, "reversal_long_window", 80),
            reversal_min_short_magnitude=getattr(stage, "reversal_min_short_magnitude", 0.2),
            ewma_alpha=getattr(stage, "ewma_alpha", 0.3),
            ewma_lookback=getattr(stage, "ewma_lookback", 80),
        )

    if stage.use_ensemble:
        prediction = ensemble_predict(
            current_scores=np.array(sources.current_scores),
            current_step=current_step,
            neighbor_full_list=neighbor_full_list,
            neighbor_partial_list=neighbor_partial_list,
            distances=distances,
            align_lookback=stage.align_lookback,
            method_weights=stage.method_weights,
            scale_cap=effective_scale_cap,
            disable_scale=config.disable_scale,
            neighbor_ids=neighbor_ids,
            weights=soft_weights,
        )
    else:
        prediction = single_method_predict(
            current_scores=np.array(sources.current_scores),
            current_step=current_step,
            neighbor_full_list=neighbor_full_list,
            neighbor_partial_list=neighbor_partial_list,
            distances=distances,
            method=AlignmentMethod.RATIO,
            align_lookback=stage.align_lookback,
            scale_cap=effective_scale_cap,
            disable_scale=config.disable_scale,
            neighbor_ids=neighbor_ids,
            weights=soft_weights,
        )

    if is_debug_logging():
        plot_neighbors_full_and_prediction(
            current_partial_data=np.array(current_scores_for_search),
            neighbor_ids=neighbor_ids,
            neighbor_full_list=neighbor_full_list,
            prediction=prediction,
            current_step=current_step,
            current_event_id=event_id,
            current_idol_id=idol_id,
            border=border,
            prediction_full_df=sources.prediction_full_df,
        )
    logging.info(f"Original predicted score value for current idol: {prediction[-1]}")
    return prediction, neighbor_ids, distances

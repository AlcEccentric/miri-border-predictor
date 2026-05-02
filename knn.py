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
from typing import List, Tuple

import numpy as np
import pandas as pd

from knn_alignment import (
    ensemble_predict,
    fetch_neighbor_trajectories,
    single_method_predict,
)
from knn_config import AlignmentMethod, get_group_config
from knn_distance import build_candidate_set, find_nearest_neighbors
from knn_plotting import (
    is_debug_logging,
    plot_current_and_neighbors,
    plot_neighbors_full_and_prediction,
)
from knn_stage import get_stage_params, scores_of, select_data_sources


MIN_CURRENT_SCORE = 5000  # skip minor idols whose latest score is too low to predict


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the full score trajectory for ``(event_id, idol_id, border)``.

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
    stage = get_stage_params(config, current_step)

    # Pick dataframes for neighbour search vs prediction
    sources = select_data_sources(
        event_id=event_id, idol_id=idol_id,
        norm_partial=norm_partial_data, norm_full=norm_data,
        smooth_partial=smooth_partial_data, smooth_full=smooth_full_data,
        use_smooth_for_neighbors=stage.use_smooth_for_neighbors,
        use_smooth_for_prediction=stage.use_smooth_for_prediction,
    )
    current_scores_for_search = scores_of(sources.search_partial_df, event_id, idol_id)

    # Build candidate set and find nearest neighbours
    candidate_partials, candidate_ids = build_candidate_set(
        search_df=sources.search_partial_df,
        exclude_event_id=event_id,
        current_step=current_step,
        min_event_id=config.least_neighbor_id,
    )
    if not candidate_partials:
        raise ValueError(f"No valid historical partial trajectories for step {current_step}")

    logging.info(f"Latest score value for current idol: {current_scores_for_search[-1]}")
    distances, neighbor_ids = find_nearest_neighbors(
        current_partial=np.array(current_scores_for_search),
        candidate_partials=[c[:current_step] for c in candidate_partials],
        candidate_ids=candidate_ids,
        current_step=current_step,
        k=stage.k,
        lookback=stage.distance_lookback,
        metric=stage.metric,
        event_type=event_type,
        sub_types=sub_types,
        border=border,
    )

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

    if stage.use_ensemble:
        prediction = ensemble_predict(
            current_scores=np.array(sources.current_scores),
            current_step=current_step,
            neighbor_full_list=neighbor_full_list,
            neighbor_partial_list=neighbor_partial_list,
            distances=distances,
            align_lookback=stage.align_lookback,
            method_weights=stage.method_weights,
            scale_cap=stage.scale_cap,
            disable_scale=config.disable_scale,
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
            scale_cap=stage.scale_cap,
            disable_scale=config.disable_scale,
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

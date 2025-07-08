import logging
import numpy as np
import pandas as pd
from knn_config import AlignmentMethod, DistanceMetric, get_group_config
from typing import Dict, List, Tuple, Union
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

def get_filtered_df(df: pd.DataFrame, 
                   event_type: float, 
                   border: float, 
                   sub_types: List[float]) -> pd.DataFrame:
    return df[
        (df['event_type'] == event_type) & 
        (df['sub_event_type'].isin(sub_types)) &
        (df['border'] == border)
    ]


def align_curve(neighbor: np.ndarray, 
                real_partial_data: np.ndarray, 
                start_idx: int, 
                end_idx: int, 
                method: AlignmentMethod) -> Tuple[float, float]:
    neighbor_partial = neighbor[start_idx:end_idx]
    real_partial = real_partial_data[start_idx:end_idx]
    
    if method == AlignmentMethod.LINEAR:
        scale = (real_partial[-1] - real_partial[0]) / (neighbor_partial[-1] - neighbor_partial[0])
        offset = real_partial[0] - scale * neighbor_partial[0]
    
    elif method == AlignmentMethod.AFFINE:
        X = neighbor_partial.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, real_partial)
        scale, offset = model.coef_[0], model.intercept_
    
    else:  # RATIO
        scale = (real_partial[-1]-real_partial[0]) / (neighbor_partial[-1]-neighbor_partial[0])
        offset = real_partial[-1] - neighbor_partial[-1] * scale 
        
    return (float(scale), float(offset))

def calculate_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    D = cdist(seq1.reshape(-1, 1), seq2.reshape(-1, 1))
    n, m = D.shape
    cost = np.zeros((n, m))
    cost[0, 0] = D[0, 0]
    
    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + D[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + D[0, j]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = D[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            
    return cost[-1, -1]

def calculate_distance(real_partial: np.ndarray, 
                      curve_partial: np.ndarray, 
                      lookback_window: int,
                      metric: DistanceMetric,
                      step: int,
                      event_type: float,
                      sub_types: Tuple[float],
                      border: float) -> float:

    config = get_group_config(event_type, sub_types, border)
    
    if config.use_trend_weighting:
        recent_window = min(30, lookback_window)
        trend_diff = np.abs(np.diff(real_partial[-recent_window:]) - 
                           np.diff(curve_partial[-recent_window:]))
        
        if metric == DistanceMetric.RMSE:
            regular_dist = np.sqrt(np.mean((real_partial[-lookback_window:] - 
                                          curve_partial[-lookback_window:]) ** 2))
            trend_dist = np.mean(trend_diff)
            return (1 - config.trend_weight) * regular_dist + config.trend_weight * trend_dist
    
    # Default distance calculation
    if metric == DistanceMetric.DTW:
        return calculate_dtw_distance(real_partial[-lookback_window:], 
                                    curve_partial[-lookback_window:])
    elif metric == DistanceMetric.RMSE:
        return np.sqrt(np.mean((real_partial[-lookback_window:] - 
                              curve_partial[-lookback_window:]) ** 2))
    else:  # FINAL_DIFF - average absolute difference over lookback window
        return float(np.mean(np.abs(real_partial[-lookback_window:] - curve_partial[-lookback_window:])))


def find_similar_curves(cur_trajectory: np.ndarray,
                       historical_trajectories: List[np.ndarray],
                       event_ids: List[Tuple[float, float]],
                       current_step: int,
                       k: int,
                       lookback_window: int,
                       metric: DistanceMetric,
                       event_type: float,
                       sub_types: Tuple[float],
                       border: float,) -> Tuple[np.ndarray, np.ndarray]:
    
    if not historical_trajectories:
        raise ValueError(f"No historical trajectories provided for step {current_step}")
    
    distances = []
    valid_indices = []
    
    for idx, hist_trajectory in enumerate(historical_trajectories):
        if len(hist_trajectory) >= current_step:
            distance = calculate_distance(
                cur_trajectory, 
                hist_trajectory, 
                lookback_window, 
                metric,
                current_step,
                event_type,
                sub_types,
                border
            )
            distances.append(distance)
            valid_indices.append(idx)

    if not distances:
        raise ValueError(f"No valid distances calculated for step {current_step}")

    distances = np.array(distances)
    valid_event_ids = np.array(event_ids)[valid_indices]
    k = min(k, len(distances))
    indices = np.argsort(distances)[:k]
    return distances[indices], valid_event_ids[indices]


def ensemble_prediction(real_data: np.ndarray,
                       current_step: int,
                       similar_curves: List[np.ndarray],
                       similar_partial_curves: List[np.ndarray],
                       distances: np.ndarray,
                       lookback_window: int,
                       method_weights: Dict[AlignmentMethod, float],) -> np.ndarray:
    """Enhanced ensemble prediction with stage-specific weights"""
    
    # Calculate curve weights
    if np.max(distances) > 0:
        curve_weights = 1 / (distances + 1e-6)
        weights_sum = np.sum(curve_weights)
        if weights_sum > 0:
            curve_weights = curve_weights / weights_sum
        else:
            curve_weights = np.ones(len(distances)) / len(distances)
    else:
        curve_weights = np.ones(len(distances)) / len(distances)
    
    # Get predictions using different methods
    predictions = []
    method_names = []
    # print(f"\n=== Ensemble Prediction for Event {current_event_id}, Idol {current_idol_id}, Step {current_step} ===")
    for method in AlignmentMethod:
        aligned_curves = []
        for ci, similar_curve in enumerate(similar_curves):
            similar_partial_curve = similar_partial_curves[ci]
            actual_lookback = min(lookback_window, len(real_data))
            align_start = max(0, len(real_data) - actual_lookback)
            align_end = len(real_data)

            scale, offset = align_curve(similar_partial_curve, real_data, align_start, align_end, method)
            # print(f"Method {method.value} Scale: {scale}, Offset: {offset}, Neighbor Last: {similar_partial_curve[align_end -1]}, Real Last: {real_data[align_end-1]}")
            aligned_curve = np.concatenate([
                similar_curve[:current_step],
                similar_curve[current_step:] * scale + offset
            ])
            aligned_curves.append(aligned_curve)
        
        pred = np.average(aligned_curves, axis=0, weights=curve_weights)
        predictions.append(pred)
        method_names.append(method.value)

    # Combine predictions using method weights
    final_prediction = np.zeros_like(predictions[0])
    for pred, method in zip(predictions, AlignmentMethod):
        final_prediction += pred * method_weights[method]
    
    return final_prediction

def get_prediction(real_data: np.ndarray, 
                  current_step: int, 
                  similar_curves: List[np.ndarray], 
                  distances: np.ndarray,
                  alignment_method: AlignmentMethod,
                  lookback_window: int,) -> np.ndarray:
    
    if lookback_window is None:
        lookback_window = current_step
        
    aligned_curves = []
    
    # Fix weight calculation to handle large distances
    max_distance = np.max(distances)
    if max_distance > 0:
        # Normalize distances to prevent numerical issues
        normalized_distances = distances / max_distance
        weights = 1 / (normalized_distances + 1e-6)
    else:
        # If all distances are 0, use equal weights
        weights = np.ones(len(distances))
    weights = weights / np.sum(weights)
    
    for curve in similar_curves:
        actual_lookback = min(lookback_window, len(real_data))
        align_start = max(0, len(real_data) - actual_lookback)
        align_end = len(real_data)

        scale, offset = align_curve(curve, real_data, 
                                  align_start, 
                                  align_end, 
                                  alignment_method)
        
        if alignment_method != AlignmentMethod.RATIO:
            aligned_curve = curve * scale + offset
        else:
            # For RATIO: apply scale and offset to future values, keep past values unchanged
            aligned_curve = np.concatenate([
                curve[:current_step],
                curve[current_step:] * scale + offset
            ])
        aligned_curves.append(aligned_curve)
    
    prediction = np.average(aligned_curves, axis=0, weights=weights)
    return prediction

def predict_curve_knn(event_id: float,
                     idol_id: int,
                     border: float,
                     sub_types: Tuple[float],
                     current_step: int,
                     norm_data: pd.DataFrame,
                     norm_partial_data: pd.DataFrame,
                     smooth_partial_data: pd.DataFrame,
                     smooth_full_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info(f"Running knn for event {event_id}, idol {idol_id}, border {border}, step {current_step}")
    neighbor_partial_norm = norm_partial_data
    neighbor_partial_smooth = smooth_partial_data
    prediction_full_norm = norm_data
    prediction_partial_norm = norm_partial_data
    prediction_full_smooth = smooth_full_data
    prediction_partial_smooth = smooth_partial_data
    
    current_trajectory = neighbor_partial_norm[(neighbor_partial_norm['event_id'] == event_id) & (neighbor_partial_norm['idol_id'] == idol_id)]['score'].values

    # Early return for minor idols with low scores
    if len(current_trajectory) == 0 or current_trajectory[-1] < 5000:
        return np.array([]), np.array([]), np.array([])

    event_data = neighbor_partial_norm[(neighbor_partial_norm['event_id'] == event_id) & (neighbor_partial_norm['idol_id'] == idol_id)].iloc[0]
    event_type = event_data['event_type']
    config = get_group_config(event_type, sub_types, border)
    
    # Get stage-specific parameters
    if current_step < config.early_stage_end:
        adapted_k = config.early_stage_k
        adapted_lookback = config.early_stage_lookback
        current_metric = config.early_stage_metric
        use_ensemble = config.early_stage_use_ensemble
        method_weights = config.early_stage_weights
        use_smooth_for_neighbors = config.early_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.early_stage_use_smooth_for_prediction
    elif current_step < config.mid_stage_end:
        adapted_k = config.mid_stage_k
        adapted_lookback = config.mid_stage_lookback
        current_metric = config.mid_stage_metric
        use_ensemble = config.mid_stage_use_ensemble
        method_weights = config.mid_stage_weights
        use_smooth_for_neighbors = config.mid_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.mid_stage_use_smooth_for_prediction
    else:
        adapted_k = config.late_stage_k
        adapted_lookback = config.late_stage_lookback
        current_metric = config.late_stage_metric
        use_ensemble = config.late_stage_use_ensemble
        method_weights = config.late_stage_weights
        use_smooth_for_neighbors = config.late_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.late_stage_use_smooth_for_prediction
    
    # Choose data sources based on stage configuration and availability
    if use_smooth_for_neighbors and neighbor_partial_smooth is not None:
        actual_neighbor_data = neighbor_partial_smooth
    else:
        actual_neighbor_data = neighbor_partial_norm
        
    if use_smooth_for_prediction and prediction_full_smooth is not None:
        actual_prediction_data = prediction_full_smooth
        actual_prediction_partial_data = prediction_partial_smooth
        currnet_prediction_data = neighbor_partial_smooth[(neighbor_partial_smooth['event_id'] == event_id) & (neighbor_partial_smooth['idol_id'] == idol_id)]['score'].values
    else:
        actual_prediction_data = prediction_full_norm
        actual_prediction_partial_data = prediction_partial_norm
        currnet_prediction_data = neighbor_partial_norm[(neighbor_partial_norm['event_id'] == event_id) & (neighbor_partial_norm['idol_id'] == idol_id)]['score'].values
    
    # Get current trajectory from actual_neighbor_data
    current_neighbor_data = actual_neighbor_data[(actual_neighbor_data['event_id'] == event_id) & (actual_neighbor_data['idol_id'] == idol_id)]['score'].values
    
    # Get historical trajectories from actual_neighbor_data for similarity calculation
    historical_neighbor_data = actual_neighbor_data[actual_neighbor_data['event_id'] != event_id]
    
    historical_trajectories = []
    historical_ids = []
    
    for (eid, iid), group in historical_neighbor_data.groupby(['event_id', 'idol_id']):
        if len(group['score'].values) >= current_step:
            historical_trajectories.append(group['score'].values)
            historical_ids.append((eid, iid))
    
    if not historical_trajectories:
        raise ValueError(f"No valid historical trajectories found for step {current_step}")

    should_plot = False
    logging.info(f"Latest score value for current idol: {current_neighbor_data[-1]}")
    distances, similar_ids = find_similar_curves(
        np.array(current_neighbor_data),
        [h[:current_step] for h in historical_trajectories],
        historical_ids,
        current_step,
        adapted_k,
        adapted_lookback,
        current_metric,
        event_type,
        sub_types,
        border,
    )
    
    # Get similar curves from actual_prediction_data for final prediction
    similar_curves = [
        actual_prediction_data[(actual_prediction_data['event_id'] == id[0]) & (actual_prediction_data['idol_id'] == id[1])]['score'].values 
        for id in similar_ids
    ]

    similar_partial_curves = [
        actual_prediction_partial_data[(actual_prediction_partial_data['event_id'] == id[0]) & (actual_prediction_partial_data['idol_id'] == id[1])]['score'].values 
        for id in similar_ids
    ]
    
    if use_ensemble:
        neighbor_prediction = ensemble_prediction(
            np.array(currnet_prediction_data),
            current_step,
            similar_curves,
            similar_partial_curves,
            distances,
            adapted_lookback,
            method_weights,
        )
    else:
        neighbor_prediction = get_prediction(
            np.array(currnet_prediction_data),
            current_step,
            similar_curves,
            distances,
            AlignmentMethod.RATIO,
            adapted_lookback,
        )

    logging.info(f"Calculated prediction: {neighbor_prediction[-1]}")
    return neighbor_prediction, similar_ids, distances
import logging
import math
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
                method: AlignmentMethod,
                scale_cap: Tuple[float, float],
                current_step: int) -> Tuple[float, float]:
    neighbor_partial = neighbor[start_idx:end_idx]
    real_partial = real_partial_data[start_idx:end_idx]
    scale = 1.0
    
    if method == AlignmentMethod.LINEAR:
        scale = (real_partial[-1] - real_partial[0]) / (neighbor_partial[-1] - neighbor_partial[0])
        offset = real_partial[0] - scale * neighbor_partial[0]
    
    elif method == AlignmentMethod.AFFINE:
        X = neighbor_partial.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, real_partial)
        scale, offset = model.coef_[0], model.intercept_
    
    else:  # RATIO
        real_mean = np.mean(real_partial)
        neighbor_mean = np.mean(neighbor_partial)
        scale = real_mean / neighbor_mean
        if scale > scale_cap[1]:
            logging.debug(f"Scale {scale} hit upper bound: {scale_cap[1]}")
            scale = scale_cap[1]
        if scale < scale_cap[0]:
            logging.debug(f"Scale {scale} hit lower bound: {scale_cap[0]}")
            scale = scale_cap[0]
        offset = real_partial[-1] - scale * neighbor_partial[-1]

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
                       border: float,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
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
    k3_indices = np.argsort(distances)[:3*k]
    return distances[indices], valid_event_ids[indices], valid_event_ids[k3_indices]

def get_scale_cap_from_neighbors(similar_curves: List[np.ndarray], 
                                 similar_partial_curves: List[np.ndarray], 
                                 current_step: int, 
                                 lookback_window: int) -> Tuple[float, float]:
    ratios = []
    for curve, partial in zip(similar_curves, similar_partial_curves):
        # Increasing rate in lookback window
        lookback_start = max(0, len(partial) - lookback_window)
        lookback_end = len(partial)
        neighbor_rate = partial[lookback_end-1] - partial[lookback_start]
        future_rate = curve[-1] - curve[current_step-1]
        # Avoid division by zero
        if neighbor_rate != 0:
            ratio = future_rate / neighbor_rate
            ratios.append(ratio)
    return min(ratios), max(ratios)

def get_future_increase_rate_range(curves: List[np.ndarray], current_step: int) -> Tuple[float, float]:
    rates = []
    for curve in curves:
        if len(curve) > current_step:
            future_start = curve[current_step]
            future_end = curve[-1]
            future_len = len(curve) - current_step
            if future_len > 0:
                rate = (future_end - future_start) / future_len
                rates.append(rate)
    return min(rates), max(rates)

def cap_aligned_future_increase(aligned_curve: np.ndarray, 
                               neighbor_partial: np.ndarray, 
                               current_step: int, 
                               align_start: int, 
                               align_end: int, 
                               cap_min: float, 
                               cap_max: float) -> np.ndarray:
    """
    Cap the future increase of the aligned curve so that the ratio of
    (future increase) / (neighbor lookback increase) falls within [cap_min, cap_max].
    """
    offset = aligned_curve[current_step]
    future = aligned_curve[current_step+1:] - offset
    neighbor_lookback_increase = neighbor_partial[align_end-1] - neighbor_partial[align_start]
    # Avoid division by zero
    if neighbor_lookback_increase == 0 or len(future) == 0:
        return aligned_curve

    new_curve = np.concatenate([aligned_curve[:current_step+1], future + offset])
    return new_curve

def cap_future_by_rate(aligned_curve: np.ndarray, 
                      current_step: int, 
                      min_rate: float, 
                      max_rate: float) -> np.ndarray:
    """
    Cap the future part of the aligned curve so that its per-step increasing rate
    falls within [min_rate, max_rate].
    """
    offset = aligned_curve[current_step-1]
    future_len = len(aligned_curve) - current_step
    if future_len <= 0:
        return aligned_curve

    future = aligned_curve[current_step:] - offset
    total_increase = future[-1] if len(future) > 0 else 0.0
    actual_rate = total_increase / future_len if future_len > 0 else 0.0

    # Cap the rate if needed
    if actual_rate < min_rate or actual_rate > max_rate:
        capped_rate = min_rate if actual_rate < min_rate else max_rate
        capped_increase = capped_rate * future_len
        capped_future = np.linspace(0, capped_increase, future_len)
        future = capped_future

    new_curve = np.concatenate([aligned_curve[:current_step], future + offset])
    return new_curve

def ensemble_prediction(real_data: np.ndarray,
                       current_step: int,
                       similar_curves: List[np.ndarray],
                       similar_partial_curves: List[np.ndarray],
                       distances: np.ndarray,
                       lookback_window: int,
                       method_weights: Dict[AlignmentMethod, float],
                       scale_cap: Tuple[float, float],
                       disable_scale: bool) -> np.ndarray:
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
            cap_min, cap_max = get_scale_cap_from_neighbors(similar_curves, similar_partial_curves, current_step, lookback_window)
            rate_cap_min, rate_cap_max = get_future_increase_rate_range(similar_curves, current_step)
            tolerence = 0.2
            cap_min = cap_min * (1 - tolerence)
            cap_max = cap_max * (1 + tolerence)

            scale, offset = align_curve(similar_partial_curve, real_data, align_start, align_end, method, scale_cap, current_step)
            # print(f"Method {method.value} Scale: {scale}, Offset: {offset}, Neighbor Last: {similar_partial_curve[align_end -1]}, Real Last: {real_data[align_end-1]}")
            if disable_scale:
                scale = 1 if current_step >= 270 else scale
            aligned_curve = np.concatenate([
                similar_curve[:current_step],
                similar_curve[current_step:] * scale + offset
            ])
            aligned_curve = cap_aligned_future_increase(
                aligned_curve, similar_partial_curve, current_step, align_start, align_end, cap_min, cap_max
            )
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
                  more_similar_curves: List[np.ndarray],
                  more_similar_partial_curves: List[np.ndarray],
                  similar_curves: List[np.ndarray],
                  similar_partial_curves: List[np.ndarray],
                  distances: np.ndarray,
                  alignment_method: AlignmentMethod,
                  lookback_window: int,
                  scale_cap: Tuple[float, float],
                  disable_scale: bool) -> np.ndarray:
    
    if lookback_window is None:
        lookback_window = current_step
        
    aligned_curves = []
    
    # Fix weight calculation to handle large distances
    processed_distances = np.array([abs(d) for d in distances if d > 0])
    max_distance = np.max(processed_distances)
    if max_distance > 0:
        # Normalize distances to prevent numerical issues
        normalized_distances = processed_distances / max_distance
        weights = 1 / (normalized_distances + 1e-6)
    else:
        # If all distances are 0, use equal weights
        weights = np.ones(len(processed_distances))
    weights = weights / np.sum(weights)

    cap_min, cap_max = get_scale_cap_from_neighbors(more_similar_curves, more_similar_partial_curves, current_step, lookback_window)
    rate_min, rate_max = get_future_increase_rate_range(more_similar_curves, current_step)
    for ci, curve in enumerate(similar_curves):
        similar_partial_curve = similar_partial_curves[ci]
        actual_lookback = min(lookback_window, len(real_data))
        align_start = max(0, len(real_data) - actual_lookback)
        align_end = len(real_data)

        scale, offset = align_curve(similar_partial_curve, real_data, 
                                  align_start, 
                                  align_end, 
                                  alignment_method,
                                  scale_cap,
                                  current_step)
        
        if disable_scale:
            scale = 1 if current_step >= 270 else scale
        
        if alignment_method != AlignmentMethod.RATIO:
            aligned_curve = curve * scale + offset
        else:
            # For RATIO: apply scale and offset to future values, keep past values unchanged
            aligned_curve = np.concatenate([
                curve[:current_step],
                curve[current_step:] * scale + offset
            ])

        # aligned_curve = cap_aligned_future_increase(
        #     aligned_curve, similar_partial_curve, current_step, align_start, align_end, cap_min, cap_max
        # )
        logging.debug(f"Scale: {scale}, Offset: {offset}, Neighbor Current: {curve[current_step]}, Neighbor Last: {curve[-1]}, Diff: {curve[-1] - curve[current_step]}")
        # aligned_curve = cap_future_by_rate(aligned_curve, current_step, rate_min, rate_max)
        
        aligned_curve = cap_aligned_future_increase(
            aligned_curve, similar_partial_curve, current_step, align_start, align_end, cap_min, cap_max
        )
        logging.debug(f"Scaled Neighbor Current: {aligned_curve[current_step]}, Scaled Neighbor Last: {aligned_curve[-1]}, Diff: {aligned_curve[-1] - aligned_curve[current_step]}")
        aligned_curves.append(aligned_curve)
        logging.debug(f"Weight: {weights[ci]} Distance: {processed_distances[ci]}")
    
    prediction = np.average(aligned_curves, axis=0, weights=weights)
    return prediction

def predict_curve_knn(event_id: float,
                     idol_id: int,
                     border: float,
                     sub_types: tuple,
                     current_step: int,
                     norm_data: pd.DataFrame,
                     norm_partial_data: pd.DataFrame,
                     smooth_partial_data: pd.DataFrame,
                     smooth_full_data: pd.DataFrame,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        adapted_lookback_for_align = config.early_stage_lookback_for_align
        current_metric = config.early_stage_metric
        use_ensemble = config.early_stage_use_ensemble
        method_weights = config.early_stage_weights
        use_smooth_for_neighbors = config.early_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.early_stage_use_smooth_for_prediction
        scale_cap = config.early_stage_scale_cap
    elif current_step < config.mid_stage_end:
        adapted_k = config.mid_stage_k
        adapted_lookback = config.mid_stage_lookback
        adapted_lookback_for_align = config.mid_stage_lookback_for_align
        current_metric = config.mid_stage_metric
        use_ensemble = config.mid_stage_use_ensemble
        method_weights = config.mid_stage_weights
        use_smooth_for_neighbors = config.mid_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.mid_stage_use_smooth_for_prediction
        scale_cap = config.mid_stage_scale_cap
    else:
        adapted_k = config.late_stage_k
        adapted_lookback = config.late_stage_lookback
        adapted_lookback_for_align = config.late_stage_lookback_for_align
        current_metric = config.late_stage_metric
        use_ensemble = config.late_stage_use_ensemble
        method_weights = config.late_stage_weights
        use_smooth_for_neighbors = config.late_stage_use_smooth_for_neighbors
        use_smooth_for_prediction = config.late_stage_use_smooth_for_prediction
        scale_cap = config.late_stage_scale_cap

    # Choose data sources based on stage configuration and availability
    if use_smooth_for_neighbors and neighbor_partial_smooth is not None:
        partial_data_for_neighbor_search = neighbor_partial_smooth
    else:
        partial_data_for_neighbor_search = neighbor_partial_norm
        
    if use_smooth_for_prediction and prediction_full_smooth is not None:
        full_data_for_avg_prediction = prediction_full_smooth
        partial_data_for_align = prediction_partial_smooth
        currnet_prediction_data = neighbor_partial_smooth[(neighbor_partial_smooth['event_id'] == event_id) & (neighbor_partial_smooth['idol_id'] == idol_id)]['score'].values
    else:
        full_data_for_avg_prediction = prediction_full_norm
        partial_data_for_align = prediction_partial_norm
        currnet_prediction_data = neighbor_partial_norm[(neighbor_partial_norm['event_id'] == event_id) & (neighbor_partial_norm['idol_id'] == idol_id)]['score'].values
    
    # Get current trajectory from actual_neighbor_data
    current_partial_data = partial_data_for_neighbor_search[(partial_data_for_neighbor_search['event_id'] == event_id) & (partial_data_for_neighbor_search['idol_id'] == idol_id)]['score'].values
    
    # Get historical trajectories from actual_neighbor_data for similarity calculation
    historical_partial_data = partial_data_for_neighbor_search[partial_data_for_neighbor_search['event_id'] != event_id]
    
    historical_partial_trajectories = []
    historical_ids = []
    
    for (eid, iid), group in historical_partial_data.groupby(['event_id', 'idol_id']):
        if len(group['score'].values) >= current_step:
            historical_partial_trajectories.append(group['score'].values)
            historical_ids.append((eid, iid))
    
    if not historical_partial_trajectories:
        raise ValueError(f"No valid historical partial trajectories found for step {current_step}")

    logging.info(f"Latest score value for current idol: {current_partial_data[-1]}")
    distances, similar_ids, more_similar_ids = find_similar_curves(
        np.array(current_partial_data),
        [h[:current_step] for h in historical_partial_trajectories],
        historical_ids,
        current_step,
        adapted_k,
        adapted_lookback,
        current_metric,
        event_type,
        sub_types,
        border,
    )

    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == 'DEBUG':
        plot_current_and_neighbors(
            current_neighbor_data=np.array(currnet_prediction_data),
            similar_ids=similar_ids,
            historical_trajectories=historical_partial_trajectories,
            historical_ids=historical_ids,
            current_step=current_step,
            current_event_id=event_id,
            current_idol_id=idol_id,
            border=border
        )
    
    # Get similar curves from actual_prediction_data for final prediction
    similar_curves, similar_partial_curves = get_similar_curves(full_data_for_avg_prediction,
                                                                partial_data_for_align,
                                                                similar_ids)
    k3_similar_curves, k3_similar_partial_curves = get_similar_curves(full_data_for_avg_prediction,
                                                                partial_data_for_align,
                                                                more_similar_ids)
    
    if use_ensemble:
        neighbor_prediction = ensemble_prediction(
            np.array(currnet_prediction_data),
            current_step,
            similar_curves,
            similar_partial_curves,
            distances,
            adapted_lookback_for_align,
            method_weights,
            scale_cap,
            config.disable_scale,
        )
    else:
        neighbor_prediction = get_prediction(
            np.array(currnet_prediction_data),
            current_step,
            k3_similar_curves,
            k3_similar_partial_curves,
            similar_curves,
            similar_partial_curves,
            distances,
            AlignmentMethod.RATIO,
            adapted_lookback_for_align,
            scale_cap,
            config.disable_scale,
        )
    
    if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == 'DEBUG':
        plot_neighbors_full_and_prediction(
            current_partial_data=np.array(current_partial_data),
            similar_ids=similar_ids,
            similar_curves=similar_curves,
            neighbor_prediction=neighbor_prediction,
            current_step=current_step,
            current_event_id=event_id,
            current_idol_id=idol_id,
            border=border,
            full_data_for_avg_prediction=full_data_for_avg_prediction
        )
    logging.info(f"Original predicted score value for current idol: {neighbor_prediction[-1]}")
    return neighbor_prediction, similar_ids, distances

def get_similar_curves(full_data_for_avg_prediction, partial_data_for_align, similar_ids):
    similar_curves = [
        full_data_for_avg_prediction[(full_data_for_avg_prediction['event_id'] == id[0]) & (full_data_for_avg_prediction['idol_id'] == id[1])]['score'].values 
        for id in similar_ids
    ]

    similar_partial_curves = [
        partial_data_for_align[(partial_data_for_align['event_id'] == id[0]) & (partial_data_for_align['idol_id'] == id[1])]['score'].values 
        for id in similar_ids
    ]
    
    return similar_curves,similar_partial_curves


def plot_current_and_neighbors(current_neighbor_data: np.ndarray,
                             similar_ids: np.ndarray,
                             historical_trajectories: List[np.ndarray],
                             historical_ids: List[Tuple[float, float]],
                             current_step: int,
                             current_event_id: float,
                             current_idol_id: int,
                             border: float,
                             output_dir: str = "debug",) -> None:
    """
    Plot the current event's partial trajectory and its neighbors' partial trajectories.
    
    Args:
        current_neighbor_data: Current event's partial score trajectory
        similar_ids: IDs of similar events found by KNN
        historical_trajectories: All historical trajectories (full length)
        historical_ids: All historical event IDs
        current_step: Current step (length of known data)
        current_event_id: Current event ID
        current_idol_id: Current idol ID
        border: Border value
        output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import os
    # Suppress matplotlib debug logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    # Create debug directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot current event trajectory (partial)
    steps = np.arange(len(current_neighbor_data))
    plt.plot(steps, current_neighbor_data, 
             color='red', linewidth=3, marker='o', markersize=4,
             label=f'Current Event {int(current_event_id)} (Idol {current_idol_id})', 
             zorder=10)
    
    # Plot neighbors' partial trajectories
    neighbor_colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(similar_ids)))
    
    for i, (neighbor_event_id, neighbor_idol_id) in enumerate(similar_ids):
        # Find the neighbor's full trajectory in historical data
        neighbor_trajectory = None
        for j, (hist_event_id, hist_idol_id) in enumerate(historical_ids):
            if hist_event_id == neighbor_event_id and hist_idol_id == neighbor_idol_id:
                neighbor_trajectory = historical_trajectories[j]
                break
        
        if neighbor_trajectory is not None:
            # Plot only up to current_step
            neighbor_partial = neighbor_trajectory[:current_step]
            neighbor_steps = np.arange(len(neighbor_partial))
            
            plt.plot(neighbor_steps, neighbor_partial,
                    color=neighbor_colors[i], linewidth=2, marker='s', markersize=3,
                    label=f'Neighbor Event {int(neighbor_event_id)} (Idol {neighbor_idol_id})',
                    alpha=0.7)
    
    # Add vertical line at current step
    plt.axvline(x=current_step-1, color='black', linestyle='--', alpha=0.5,
                label=f'Current Step {current_step}')
    
    # Formatting
    plt.title(f'Current Event vs Neighbors - Step {current_step}\n'
              f'Event {int(current_event_id)}, Idol {current_idol_id}, Border {int(border)}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    filename = f'current_vs_neighbors_e{int(current_event_id)}_i{current_idol_id}_s{current_step}_b{int(border)}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Current vs neighbors plot saved to {output_path}")
    
    # Also create a summary text file
    summary_path = os.path.join(output_dir, f'neighbors_summary_e{int(current_event_id)}_i{current_idol_id}_s{current_step}_b{int(border)}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Neighbor Analysis Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Current Event: {int(current_event_id)}\n")
        f.write(f"Current Idol: {current_idol_id}\n")
        f.write(f"Current Step: {current_step}\n")
        f.write(f"Border: {int(border)}\n")
        f.write(f"Current Score: {current_neighbor_data[-1]:.2f}\n\n")
        
        f.write("Neighbors Found:\n")
        f.write("-" * 20 + "\n")
        for i, (neighbor_event_id, neighbor_idol_id) in enumerate(similar_ids):
            # Find neighbor's score at current step
            for j, (hist_event_id, hist_idol_id) in enumerate(historical_ids):
                if hist_event_id == neighbor_event_id and hist_idol_id == neighbor_idol_id:
                    neighbor_score = historical_trajectories[j][current_step-1]
                    f.write(f"Event {int(neighbor_event_id)}, Idol {neighbor_idol_id}: {neighbor_score:.2f}\n")
                    break
    
    logging.info(f"Neighbor summary saved to {summary_path}")

def plot_neighbors_full_and_prediction(current_partial_data: np.ndarray,
                                     similar_ids: np.ndarray,
                                     similar_curves: List[np.ndarray],
                                     neighbor_prediction: np.ndarray,
                                     current_step: int,
                                     current_event_id: float,
                                     current_idol_id: int,
                                     border: float,
                                     full_data_for_avg_prediction: pd.DataFrame,
                                     output_dir: str = "debug") -> None:
    """
    Plot neighbors' full trajectories, prediction, and real final value.
    
    Args:
        current_neighbor_data: Current event's partial score trajectory
        similar_ids: IDs of similar events found by KNN
        similar_curves: Full trajectories of similar events
        neighbor_prediction: Predicted trajectory
        current_step: Current step (length of known data)
        current_event_id: Current event ID
        current_idol_id: Current idol ID
        border: Border value
        actual_prediction_data: Full prediction data to get real final value
        output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import os

    # Get real final value
    real_full_data_for_avg_prediction = full_data_for_avg_prediction[
        (full_data_for_avg_prediction['event_id'] == current_event_id) & 
        (full_data_for_avg_prediction['idol_id'] == current_idol_id)
    ]
    real_final_value = None
    if len(real_full_data_for_avg_prediction) > 0:
        real_final_value = real_full_data_for_avg_prediction['score'].to_numpy()[-1]

    # Compute relative error if possible
    rel_error = None
    if neighbor_prediction is not None and len(neighbor_prediction) > 0 and real_final_value is not None and real_final_value != 0:
        rel_error = abs(neighbor_prediction[-1] - real_final_value) / real_final_value * 100
    
    # Suppress matplotlib debug logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    # Create debug directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot current event trajectory (partial - known data)
    steps = np.arange(len(current_partial_data))
    plt.plot(steps, current_partial_data, 
             color='red', linewidth=3, marker='o', markersize=4,
             label=f'Current Event {int(current_event_id)} (Known)', 
             zorder=10)
    
    # Plot neighbors' full trajectories
    neighbor_colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(similar_ids)))
    
    for i, (neighbor_event_id, neighbor_idol_id) in enumerate(similar_ids):
        neighbor_trajectory = similar_curves[i]
        neighbor_steps = np.arange(len(neighbor_trajectory))
        
        plt.plot(neighbor_steps, neighbor_trajectory,
                color=neighbor_colors[i], linewidth=2, marker='s', markersize=2,
                label=f'Neighbor Event {int(neighbor_event_id)} (Idol {neighbor_idol_id})',
                alpha=0.7)
    
    # Plot prediction
    if len(neighbor_prediction) > 0:
        pred_steps = np.arange(len(neighbor_prediction))
        plt.plot(pred_steps, neighbor_prediction,
                color='blue', linewidth=3, marker='^', markersize=3,
                label=f'Prediction for Event {int(current_event_id)}',
                linestyle='--', alpha=0.9, zorder=8)
    
    # Get and plot real final value
    real_full_data_for_avg_prediction = full_data_for_avg_prediction[
        (full_data_for_avg_prediction['event_id'] == current_event_id) & 
        (full_data_for_avg_prediction['idol_id'] == current_idol_id)
    ]
    
    if len(real_full_data_for_avg_prediction) > 0:
        real_final_trajectory = real_full_data_for_avg_prediction['score'].to_numpy()
        real_final_value = real_final_trajectory[-1]
        final_step = len(real_final_trajectory) - 1
        
        plt.plot(final_step, real_final_value,
                color='green', marker='o', markersize=10,
                label=f'Real Final Value: {real_final_value:.0f}',
                zorder=12)
        
        # Also plot the full real trajectory for reference
        real_steps = np.arange(len(real_final_trajectory))
        plt.plot(real_steps, real_final_trajectory,
                color='green', linewidth=2, alpha=0.5,
                label=f'Real Full Trajectory (Event {int(current_event_id)})',
                linestyle='-', zorder=5)
    
    # Add vertical line at current step
    plt.axvline(x=current_step-1, color='black', linestyle='--', alpha=0.5,
                label=f'Current Step {current_step}')
    
    # Add shaded region for future steps
    if len(neighbor_prediction) > current_step:
        plt.axvspan(current_step-1, len(neighbor_prediction)-1, 
                   alpha=0.1, color='gray', label='Future Steps')
    
    # Formatting
    plt.title(f'Full Trajectories, Prediction & Real Final Value - Step {current_step}\n'
              f'Event {int(current_event_id)}, Idol {current_idol_id}, Border {int(border)}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    filename = f'full_prediction_e{int(current_event_id)}_i{current_idol_id}_s{current_step}_b{int(border)}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Full prediction plot saved to {output_path}")
    
    # Create detailed summary
    summary_path = os.path.join(output_dir, f'prediction_summary_e{int(current_event_id)}_i{current_idol_id}_s{current_step}_b{int(border)}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Prediction Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Current Event: {int(current_event_id)}\n")
        f.write(f"Current Idol: {current_idol_id}\n")
        f.write(f"Current Step: {current_step}\n")
        f.write(f"Border: {int(border)}\n")
        f.write(f"Current Score: {current_partial_data[-1]:.2f}\n\n")
        
        if len(neighbor_prediction) > 0:
            f.write(f"Predicted Final Score: {neighbor_prediction[-1]:.2f}\n")
        
        if len(real_full_data_for_avg_prediction) > 0:
            real_final_value = real_full_data_for_avg_prediction['score'].values[-1]
            f.write(f"Real Final Score: {real_final_value:.2f}\n")
            
            if len(neighbor_prediction) > 0:
                prediction_error = abs(neighbor_prediction[-1] - real_final_value)
                relative_error = (prediction_error / real_final_value) * 100
                f.write(f"Prediction Error: {prediction_error:.2f}\n")
                f.write(f"Relative Error: {relative_error:.2f}%\n")
        
        f.write(f"\nNeighbors Used:\n")
        f.write("-" * 20 + "\n")
        for i, (neighbor_event_id, neighbor_idol_id) in enumerate(similar_ids):
            neighbor_final = similar_curves[i][-1]
            f.write(f"Event {int(neighbor_event_id)}, Idol {neighbor_idol_id}: {neighbor_final:.2f}\n")
    
    logging.info(f"Prediction summary saved to {summary_path}")
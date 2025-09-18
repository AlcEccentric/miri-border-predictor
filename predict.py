from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from knn import get_filtered_df, predict_curve_knn
from normalization import denormalize_target_to_raw

def calculate_confidence_bounds(normalized_target: np.ndarray,
                                last_known_step_index: int,
                                confidence_intervals: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[str, np.ndarray]]:
    """Calculate confidence bounds for multiple CI levels."""
    bounds = {}
    
    for confidence_level, (lower_bound, upper_bound) in confidence_intervals.items():
        upper_series = normalized_target.copy()
        lower_series = normalized_target.copy()
        
        if last_known_step_index + 1 < len(normalized_target):
            remaining_points = len(normalized_target) - (last_known_step_index + 1)
            for i in range(last_known_step_index + 1, len(normalized_target)):
                progress = (i - last_known_step_index - 1) / (remaining_points - 1) if remaining_points > 1 else 1
                
                upper_delta = progress * upper_bound
                upper_series[i] = (1 + upper_delta) * normalized_target[i]
                
                lower_delta = progress * lower_bound
                lower_series[i] = (1 + lower_delta) * normalized_target[i]
        
        bounds[confidence_level] = {
            "upper": upper_series,
            "lower": lower_series
        }
    
    return bounds

def validate_confidence_bounds(raw_target: np.ndarray, raw_bounds: Dict[int, Dict[str, np.ndarray]], 
                              last_known_idx: int, confidence_intervals: Dict[int, Tuple[float, float]]) -> None:
    """Validate confidence bounds consistency."""
    target_at_last = raw_target[last_known_idx]
    target_final = raw_target[-1]
    
    for confidence_level, bounds in raw_bounds.items():
        upper_at_last = bounds["upper"][last_known_idx]
        lower_at_last = bounds["lower"][last_known_idx]
        
        # Check 1: Values at last_known_step_index should be same across target, upper, lower (0.5% tolerance)
        tolerance = abs(target_at_last) * 0.005
        if not (abs(target_at_last - upper_at_last) <= tolerance and abs(target_at_last - lower_at_last) <= tolerance):
            logging.error(f"CRITICAL: CI{confidence_level} values at last_known_step_index mismatch: target={target_at_last}, upper={upper_at_last}, lower={lower_at_last}")
        else:
            logging.debug(f"✓ CI{confidence_level} values at last_known_step_index match: {target_at_last}")
        
        # Check 2: Relative difference between final values should match confidence intervals
        upper_final = bounds["upper"][-1]
        lower_final = bounds["lower"][-1]
        
        upper_rel_diff = (upper_final - target_final) / target_final if target_final != 0 else 0
        lower_rel_diff = (lower_final - target_final) / target_final if target_final != 0 else 0
        
        expected_upper = confidence_intervals[confidence_level][1]
        expected_lower = confidence_intervals[confidence_level][0]
        
        upper_tolerance = abs(expected_upper) * 0.005
        lower_tolerance = abs(expected_lower) * 0.005
        
        if abs(upper_rel_diff - expected_upper) > upper_tolerance:
            logging.error(f"CRITICAL: CI{confidence_level} upper bound relative diff mismatch: got {upper_rel_diff:.6f}, expected {expected_upper:.6f}")
        else:
            logging.debug(f"✓ CI{confidence_level} upper bound relative diff matches: {upper_rel_diff:.6f}")
        
        if abs(lower_rel_diff - expected_lower) > lower_tolerance:
            logging.error(f"CRITICAL: CI{confidence_level} lower bound relative diff mismatch: got {lower_rel_diff:.6f}, expected {expected_lower:.6f}")
        else:
            logging.debug(f"✓ CI{confidence_level} lower bound relative diff matches: {lower_rel_diff:.6f}")

def load_confidence_intervals(event_type: float, sub_type: float, border: float, step: int) -> Dict[int, Tuple[float, float]]:
    """Load confidence intervals for given parameters."""
    intervals_file = Path("confidence_intervals") / f"confidence_intervals_{int(event_type)}_{int(sub_type)}_{int(border)}.csv"
    
    if not intervals_file.exists():
        raise FileNotFoundError(f"Confidence intervals file not found: {intervals_file}")
    
    df = pd.read_csv(intervals_file)
    step_data = df[df['step'] == step]
    
    intervals_map = {}
    for _, row in step_data.iterrows():
        intervals_map[int(row['confidence_level'])] = (row['rel_error_lower_bound'], row['rel_error_upper_bound'])
    
    if not intervals_map:
        raise ValueError(f"No confidence interval data found for step {step} in {intervals_file}")
    
    required_levels = {75, 90}
    missing_levels = required_levels - set(intervals_map.keys())
    if missing_levels:
        raise ValueError(f"Missing required confidence levels {sorted(missing_levels)} for step {step} in {intervals_file}")
    
    return intervals_map

def build_result_dict(
    event_id: int,
    idol_id: int,
    border: float,
    current_step: int,
    current_raw_data: np.ndarray,
    current_norm_data: np.ndarray,
    similar_ids: np.ndarray,
    distances: np.ndarray,
    filtered_norm_all: pd.DataFrame,
    norm_event_length: int,
    standard_event_length: int,
    full_event_length: int,
    actual_boost_start: int,
    smoothed_prediction: np.ndarray,
    event_name_map: Dict,
    eid_to_len_boost_ratio: Dict,
    confidence_intervals: Dict[int, Tuple[float, float]],
    standard_event_boost_ratio: float,
) -> Dict:
    # Initialize result structure
    event_name = event_name_map[event_id]
    result = {
        "metadata": {
            "raw": {
                "id": event_id,
                "last_known_step_index": len(current_raw_data) - 1,
                "name": event_name
            },
            "normalized": {
                "last_known_step_index": len(current_norm_data) - 1,
                "neighbors": {},
                "confidence_intervals": confidence_intervals
            }
        },
        "data": {
            "raw": {
                "target": [],
                "bounds": {}
            },
            "normalized": {
                "target": [],
                "neighbors": {}
            }
        }
    }

    neighbor_weights = 1 / (distances + 1e-6)
    neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
    neighbor_norm_full_curves = []
    
    for i, (neighbor_eid, neighbor_iid) in enumerate(similar_ids):
        neighbor_norm_df = filtered_norm_all[
            (filtered_norm_all['event_id'] == neighbor_eid) & 
            (filtered_norm_all['idol_id'] == neighbor_iid)
        ]
        neighbor_norm_data = np.array(neighbor_norm_df['score'].values)
        neighbor_name = event_name_map[neighbor_eid]
        
        result["metadata"]["normalized"]["neighbors"][str(i+1)] = {
            "id": int(neighbor_eid),
            "idol_id": int(neighbor_iid),
            "name": neighbor_name,
            "raw_length": eid_to_len_boost_ratio[(neighbor_eid, neighbor_iid)]['length'],
        }
        result["data"]["normalized"]["neighbors"][str(i+1)] = [round(x) for x in neighbor_norm_data.tolist()]
        neighbor_norm_full_curves.append(neighbor_norm_data)

    target_norm_final_value = smoothed_prediction[-1]

    if len(current_norm_data) > 0:
        last_known_norm = current_norm_data[-1]
        remaining_steps = norm_event_length - len(current_norm_data)
        
        if remaining_steps > 0:
            predicted_norm_part = np.zeros(remaining_steps)
            for neighbor_curve, weight in zip(neighbor_norm_full_curves, neighbor_weights):
                if len(neighbor_curve) == norm_event_length and len(neighbor_curve) > len(current_norm_data):
                    neighbor_future = neighbor_curve[len(current_norm_data):]

                    if len(neighbor_future) > 0:
                        predicted_norm_part += weight * neighbor_future
                else:
                    raise ValueError(f"Neighbor norm data has incorrect length for idol {idol_id} at border {border} for event {event_id}")

            # Ensure continuity: the prediction should start from the last known value
            # and end at the target final value
            original_start = predicted_norm_part[0]
            original_end = predicted_norm_part[-1]
            
            # Scale prediction to go from last_known_norm to target_norm_final_value
            scale = (target_norm_final_value - last_known_norm) / (original_end - original_start)
            offset = last_known_norm - scale * original_start
            predicted_norm_part = scale * predicted_norm_part + offset

            # --- BOUND INCREASING RATES ---
            # Calculate per-step increasing rates for each neighbor
            neighbor_incr_rates = []
            for neighbor_curve in neighbor_norm_full_curves:
                if len(neighbor_curve) == norm_event_length and len(neighbor_curve) > len(current_norm_data):
                    neighbor_future = neighbor_curve[len(current_norm_data):]
                    # Calculate rates: first rate is from last_known to first_future, then diff between consecutive futures
                    neighbor_last_known = neighbor_curve[len(current_norm_data)-1] if len(current_norm_data) > 0 else 0
                    rates = np.diff(neighbor_future, prepend=neighbor_last_known)
                    neighbor_incr_rates.append(rates)
            neighbor_incr_rates = np.array(neighbor_incr_rates)  # shape: (n_neighbors, remaining_steps)

            # Calculate weighted average trajectory's increasing rates
            avg_incr_rate = np.diff(predicted_norm_part, prepend=last_known_norm)

            # Bound each step's rate by min/max among neighbors
            min_rates = np.min(neighbor_incr_rates, axis=0)
            max_rates = np.max(neighbor_incr_rates, axis=0)
            bounded_incr_rate = np.clip(avg_incr_rate, min_rates, max_rates)

            # Reconstruct bounded predicted_norm_part using the bounded incremental rates
            bounded_predicted_norm_part = np.zeros(remaining_steps)
            bounded_predicted_norm_part[0] = last_known_norm + bounded_incr_rate[0]
            for i in range(1, remaining_steps):
                bounded_predicted_norm_part[i] = bounded_predicted_norm_part[i-1] + bounded_incr_rate[i]
            predicted_norm_part = bounded_predicted_norm_part
            
            # Debug: Log differences between consecutive prediction values
            last_known_to_first = predicted_norm_part[0] - last_known_norm
            first_to_second = predicted_norm_part[1] - predicted_norm_part[0] if len(predicted_norm_part) > 1 else 0
            logging.info(f"Gap (last_known → first_predicted): {round(last_known_to_first)}")
            logging.info(f"Gap (first_predicted → second_predicted): {round(first_to_second)}")
            
            normalized_target = np.concatenate([current_norm_data, predicted_norm_part])
            
            # Debug: Check gap in normalized data (should be zero now)
            if len(current_norm_data) > 0 and len(predicted_norm_part) > 0:
                last_known_norm_check = current_norm_data[-1]
                first_predicted_norm_check = predicted_norm_part[0]
                norm_gap = first_predicted_norm_check - last_known_norm_check
                logging.debug(f"Normalized gap for idol {idol_id}, border {border}: last_known={last_known_norm_check}, first_predicted={first_predicted_norm_check}, gap={norm_gap}")
                
                gap_tolerance = abs(last_known_norm_check) * 0.005
                if abs(norm_gap) > gap_tolerance:
                    logging.error(f"CRITICAL: Non-zero gap in normalized data: {norm_gap}")
                    logging.error(f"  This violates requirement #1: slice point gap should be 0")
                else:
                    logging.debug(f"✓ No gap in normalized data at slice point")
                
                # Log final values for verification
                logging.debug(f"Normalized final value: {normalized_target[-1]}")
                logging.debug(f"Target normalized final value: {target_norm_final_value}")
                
                final_tolerance = abs(target_norm_final_value) * 0.005
                if abs(normalized_target[-1] - target_norm_final_value) > final_tolerance:
                    logging.error(f"CRITICAL: Final value mismatch: got {normalized_target[-1]}, expected {target_norm_final_value}")
                else:
                    logging.debug(f"✓ Normalized final value matches target: {normalized_target[-1]}")
        else:
            raise ValueError(f"Insufficient data for idol {idol_id} at border {border} for event {event_id}")
    else:
        raise ValueError(f"No current normalized data for idol {idol_id} at border {border} for event {event_id}")

    result["data"]["normalized"]["target"] = [round(x) for x in normalized_target.tolist()]
    
    # Calculate confidence bounds for multiple CI levels
    normalized_bounds = calculate_confidence_bounds(normalized_target, len(current_norm_data) - 1, confidence_intervals)

    raw_target = denormalize_target_to_raw(
        normalized_target=normalized_target,
        current_step=current_step,
        current_raw_data=current_raw_data,
        full_norm_length=norm_event_length,
        standard_event_length=standard_event_length,
        full_event_length=full_event_length,
        actual_boost_start=actual_boost_start,
        standard_event_boost_ratio=standard_event_boost_ratio,
    )
    
    # Debug: Check gap in raw data
    if len(current_raw_data) > 0 and len(raw_target) > len(current_raw_data):
        last_known_raw = current_raw_data[-1]
        first_predicted_raw = raw_target[len(current_raw_data)]
        raw_gap = first_predicted_raw - last_known_raw
        logging.debug(f"Raw gap for idol {idol_id}, border {border}: last_known={last_known_raw}, first_predicted={first_predicted_raw}, gap={raw_gap}")
        
        gap_tolerance = abs(last_known_raw) * 0.005
        if abs(raw_gap) > gap_tolerance:
            logging.error(f"CRITICAL: Non-zero gap in raw data: {raw_gap}")
            logging.error(f"  This violates requirement #1: slice point gap should be 0")
        else:
            logging.debug(f"✓ No gap in raw data at slice point")

    result["data"]["raw"]["target"] = [round(x) for x in raw_target.tolist()]
    
    # Denormalize bounds to raw scale for each CI level
    raw_bounds = {}
    for confidence_level, bounds in normalized_bounds.items():
        raw_upper = denormalize_target_to_raw(
            normalized_target=bounds["upper"],
            current_step=current_step,
            current_raw_data=current_raw_data,
            full_norm_length=norm_event_length,
            standard_event_length=standard_event_length,
            full_event_length=full_event_length,
            actual_boost_start=actual_boost_start,
            standard_event_boost_ratio=standard_event_boost_ratio,
        )
        
        raw_lower = denormalize_target_to_raw(
            normalized_target=bounds["lower"],
            current_step=current_step,
            current_raw_data=current_raw_data,
            full_norm_length=norm_event_length,
            standard_event_length=standard_event_length,
            full_event_length=full_event_length,
            actual_boost_start=actual_boost_start,
            standard_event_boost_ratio=standard_event_boost_ratio,
        )
        
        raw_bounds[confidence_level] = {
            "upper": [round(x) for x in raw_upper.tolist()],
            "lower": [round(x) for x in raw_lower.tolist()]
        }
    
    result["data"]["raw"]["bounds"] = raw_bounds
    
    # Validate confidence bounds
    raw_bounds_arrays = {level: {"upper": np.array([x for x in bounds["upper"]]), "lower": np.array([x for x in bounds["lower"]])} for level, bounds in raw_bounds.items()}
    validate_confidence_bounds(raw_target, raw_bounds_arrays, 
                              result["metadata"]["raw"]["last_known_step_index"], 
                              confidence_intervals)
    
    # Make first value zero for all data arrays
    if result["data"]["raw"]["target"]:
        first_raw = result["data"]["raw"]["target"][0]
        result["data"]["raw"]["target"] = [x - first_raw for x in result["data"]["raw"]["target"]]
        
        # Apply zero baseline to bounds
        for confidence_level in result["data"]["raw"]["bounds"]:
            result["data"]["raw"]["bounds"][confidence_level]["upper"] = [x - first_raw for x in result["data"]["raw"]["bounds"][confidence_level]["upper"]]
            result["data"]["raw"]["bounds"][confidence_level]["lower"] = [x - first_raw for x in result["data"]["raw"]["bounds"][confidence_level]["lower"]]
    
    if result["data"]["normalized"]["target"]:
        first_norm = result["data"]["normalized"]["target"][0]
        result["data"]["normalized"]["target"] = [x - first_norm for x in result["data"]["normalized"]["target"]]
        
        for neighbor_key in result["data"]["normalized"]["neighbors"]:
            if result["data"]["normalized"]["neighbors"][neighbor_key]:
                first_neighbor = result["data"]["normalized"]["neighbors"][neighbor_key][0]
                result["data"]["normalized"]["neighbors"][neighbor_key] = [x - first_neighbor for x in result["data"]["normalized"]["neighbors"][neighbor_key]]
    
    # Final consistency check: compare normalized and denormalized final values
    norm_final = normalized_target[-1]
    raw_final = raw_target[-1]
    
    # For no score scaling case, these should be exactly equal
    scale_factor = standard_event_length / full_event_length
    logging.debug(f"Final value check - scale_factor: {scale_factor}, norm_final: {norm_final}, raw_final: {raw_final}")
    
    if abs(scale_factor - 1.0) < 1e-10:
        # No scaling case - values should be identical
        final_tolerance = abs(norm_final) * 0.005
        if abs(norm_final - raw_final) > final_tolerance:
            logging.error(f"CRITICAL: Final value mismatch (no scaling case): normalized={norm_final}, denormalized={raw_final}, diff={abs(norm_final - raw_final)}")
            logging.error(f"  This violates requirement #2: norm and denorm final values should be the same")
        else:
            logging.debug(f"✓ Final values match perfectly (no scaling): {norm_final}")
    else:
        # With scaling - raw should equal norm/scale_factor
        expected_raw_final = norm_final / scale_factor
        expected_tolerance = abs(expected_raw_final) * 0.005
        if abs(raw_final - expected_raw_final) > expected_tolerance:
            logging.error(f"CRITICAL: Final value mismatch (with scaling): normalized={norm_final}, denormalized={raw_final}, expected={expected_raw_final}")
            logging.error(f"  Scale factor: {scale_factor}, diff: {abs(raw_final - expected_raw_final)}")
        else:
            logging.debug(f"✓ Final values consistent with scaling: norm={norm_final}, raw={raw_final}, scale={scale_factor}")
    
    logging.info(f"Prediction summary for idol {idol_id} at border {border} for event {event_id}")
    logging.info(f"=>Normalized prediction {round(result['data']['normalized']['target'][-1])} with length {len(result['data']['normalized']['target'])}")
    logging.info(f"=>Denormalized prediction {round(result['data']['raw']['target'][-1])} with length {len(result['data']['raw']['target'])}")
    logging.info(f"=>Neighbors:")
    for i, neighbor in enumerate(result["metadata"]["normalized"]["neighbors"].values()):
        logging.info(f"=>{neighbor['id']} {neighbor['idol_id']} {round(result['data']['normalized']['neighbors'][str(i+1)][-1])}")

    return result

def get_predictions(
    data: Dict[str, pd.DataFrame],
    event_id: int,
    event_type: float,
    sub_type: float,
    idol_ids: List[int],
    borders: List[float],
    step: int,
    event_length: int,
    norm_event_length: int,
    standard_event_length: int,
    standard_event_boost_ratio: float,
    eid_to_len_boost_ratio: Dict,
    event_name_map: Dict,
) -> Dict:
    results = {}
    sub_types = (sub_type,)

    for idol_id in idol_ids:
        if idol_id not in results:
            results[idol_id] = {}
            
        for border in borders:
            filtered_norm_all = get_filtered_df(data['norm_all'], event_type, border, list(sub_types))
            filtered_norm_part = get_filtered_df(data['norm_part'], event_type, border, list(sub_types))
            filtered_smooth_part = get_filtered_df(data['smooth_part'], event_type, border, list(sub_types))
            filtered_smooth_all = get_filtered_df(data['smoo_all'], event_type, border, list(sub_types))
            filtered_raw = get_filtered_df(data['raw'], event_type, border, list(sub_types))

            try:
                smoothed_prediction, similar_ids, distances = predict_curve_knn(
                    event_id=event_id,
                    idol_id=idol_id,
                    border=border,
                    sub_types=sub_types,
                    current_step=step,
                    norm_data=filtered_norm_all,
                    norm_partial_data=filtered_norm_part,
                    smooth_partial_data=filtered_smooth_part,
                    smooth_full_data=filtered_smooth_all,
                )

                if len(smoothed_prediction) == 0:
                    logging.debug(f"Skipping step {step} for idol {idol_id}, border {border} due to lack of data")
                    continue

                current_raw_data = filtered_raw[
                    (filtered_raw['event_id'] == event_id) & 
                    (filtered_raw['idol_id'] == idol_id)
                ]['score'].values
                
                current_norm_data = filtered_norm_part[
                    (filtered_norm_part['event_id'] == event_id) & 
                    (filtered_norm_part['idol_id'] == idol_id)
                ]['score'].values

                current_raw_data = np.array(current_raw_data)
                current_norm_data = np.array(current_norm_data)

                confidence_intervals = load_confidence_intervals(event_type, sub_type, border, step)
                
                result = build_result_dict(
                    event_id=event_id,
                    idol_id=idol_id,
                    border=border,
                    current_step=step,
                    current_raw_data=current_raw_data,
                    current_norm_data=current_norm_data,
                    similar_ids=similar_ids,
                    distances=distances,
                    filtered_norm_all=filtered_norm_all,
                    norm_event_length=norm_event_length,
                    standard_event_length=standard_event_length,
                    full_event_length=eid_to_len_boost_ratio[(event_id, idol_id)]['length'],
                    actual_boost_start=eid_to_len_boost_ratio[(event_id, idol_id)]['boost_start'],
                    smoothed_prediction=smoothed_prediction,
                    event_name_map=event_name_map,
                    eid_to_len_boost_ratio=eid_to_len_boost_ratio,
                    confidence_intervals=confidence_intervals,
                    standard_event_boost_ratio=standard_event_boost_ratio,
                )

                results[idol_id][border] = result

            except Exception as e:
                logging.error(f"Error processing idol {idol_id}, border {border}: {e}", exc_info=True)
                continue

    return results
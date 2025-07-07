from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

from knn import get_filtered_df, predict_curve_knn
from normalization import denormalize_target_to_raw

def build_result_dict(
    event_id: int,
    idol_id: int,
    border: float,
    current_raw_data: np.ndarray,
    current_norm_data: np.ndarray,
    similar_ids: np.ndarray,
    distances: np.ndarray,
    filtered_norm_all: pd.DataFrame,
    norm_event_length: int,
    standard_event_length: int,
    full_event_length: int,
    actual_boost_start: int,
    smoothed_prediction: np.ndarray
) -> Dict:
    # Initialize result structure
    result = {
        "metadata": {
            "raw": {
                "id": event_id,
                "last_known_step_index": len(current_raw_data) - 1
            },
            "normalized": {
                "last_known_step_index": len(current_norm_data) - 1,
                "neighbors": {}
            }
        },
        "data": {
            "raw": {
                "target": []
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
        neighbor_norm_data = np.array(filtered_norm_all[
            (filtered_norm_all['event_id'] == neighbor_eid) & 
            (filtered_norm_all['idol_id'] == neighbor_iid)
        ]['score'].values)
        
        result["metadata"]["normalized"]["neighbors"][str(i)] = {
            "id": int(neighbor_eid),
            "idol_id": int(neighbor_iid)
        }
        result["data"]["normalized"]["neighbors"][str(i)] = neighbor_norm_data.tolist()
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

            original_start = predicted_norm_part[0]
            original_end = predicted_norm_part[-1]
            scale = (target_norm_final_value - last_known_norm) / (original_end - original_start)
            offset = last_known_norm - scale * original_start
            predicted_norm_part = scale * predicted_norm_part + offset
            normalized_target = np.concatenate([current_norm_data, predicted_norm_part])
        else:
            raise ValueError(f"Insufficient data for idol {idol_id} at border {border} for event {event_id}")
    else:
        raise ValueError(f"No current normalized data for idol {idol_id} at border {border} for event {event_id}")

    result["data"]["normalized"]["target"] = normalized_target.tolist()

    raw_target = denormalize_target_to_raw(
        normalized_target=normalized_target,
        current_raw_data=current_raw_data,
        full_norm_length=norm_event_length,
        standard_event_length=standard_event_length,
        full_event_length=full_event_length,
        actual_boost_start=actual_boost_start
    )

    result["data"]["raw"]["target"] = raw_target.tolist()

    return result

def get_predictions(
    data: Dict[str, pd.DataFrame],
    event_id: int,
    event_type: float,
    sub_types: Tuple[float],
    idol_ids: List[int],
    borders: List[float],
    step: int,
    event_length: int,
    norm_event_length: int,
    standard_event_length: int,
    eid_to_len_boost_ratio: Dict
) -> Dict:
    results = {}

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

                result = build_result_dict(
                    event_id=event_id,
                    idol_id=idol_id,
                    border=border,
                    current_raw_data=current_raw_data,
                    current_norm_data=current_norm_data,
                    similar_ids=similar_ids,
                    distances=distances,
                    filtered_norm_all=filtered_norm_all,
                    norm_event_length=norm_event_length,
                    standard_event_length=standard_event_length,
                    full_event_length=eid_to_len_boost_ratio[(event_id, idol_id)]['length'],
                    actual_boost_start=eid_to_len_boost_ratio[(event_id, idol_id)]['boost_start'],
                    smoothed_prediction=smoothed_prediction
                )

                results[idol_id][border] = result

            except Exception as e:
                logging.error(f"Error processing idol {idol_id}, border {border}: {e}")
                continue

    return results
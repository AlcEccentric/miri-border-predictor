import numpy as np
import pandas as pd
import logging
def normalize_consistently(df, full_norm_length, step, standard_event_length, full_event_length, actual_boost_start=None):
    """
    Normalize data while preserving the last score and maintaining target boost ratio.
    """
    target_boost_ratio = 0.55 # for normal event only
    target_boost_step = int(target_boost_ratio * full_norm_length)  # 165 for 300
    
    # Don't truncate - work with full data and preserve the last score
    original_data = df.copy()
    last_score = original_data['score'].iloc[-1]
    
    # Score scaling (if needed)
    scale_factor = standard_event_length / full_event_length
    if abs(scale_factor - 1.0) > 1e-10:
        scaled_data = original_data.copy()
        scaled_data['score'] = original_data['score'] * scale_factor
        scaled_last_score = last_score * scale_factor
    else:
        scaled_data = original_data.copy()
        scaled_last_score = last_score
    
    # Find boost start in the scaled data
    boost_start_idx = None
    if actual_boost_start is not None:
        boost_mask = scaled_data['is_boosted'] == True
        if boost_mask.any():
            boost_start_idx = boost_mask.idxmax()
    
    # Create normalized timeline with proper interpolation
    if boost_start_idx is not None:
        # Split interpolation at boost boundary
        
        # Pre-boost section: interpolate from 0 to boost_start_idx -> 0 to target_boost_step
        pre_boost_original_indices = np.arange(0, boost_start_idx)
        pre_boost_normalized_indices = np.linspace(0, target_boost_step - 1, target_boost_step, dtype=int)
        pre_boost_interp_indices = np.interp(pre_boost_normalized_indices, 
                                            np.arange(len(pre_boost_original_indices)), 
                                            pre_boost_original_indices)
        
        # Post-boost section: interpolate from boost_start_idx to end -> target_boost_step to step
        post_boost_steps = step - target_boost_step
        if post_boost_steps > 0:
            post_boost_original_indices = np.arange(boost_start_idx, len(scaled_data))
            post_boost_normalized_indices = np.linspace(0, post_boost_steps - 1, post_boost_steps, dtype=int)
            post_boost_interp_indices = np.interp(post_boost_normalized_indices,
                                                 np.arange(len(post_boost_original_indices)),
                                                 post_boost_original_indices)
            
            # Combine both sections
            all_interp_indices = np.concatenate([pre_boost_interp_indices, post_boost_interp_indices])
        else:
            all_interp_indices = pre_boost_interp_indices[:step]
    else:
        # No boost - simple linear interpolation from 0 to len-1 -> 0 to step-1
        all_interp_indices = np.linspace(0, len(scaled_data) - 1, step)
    
    # Ensure the last index points to the actual last row to preserve last score
    if step >= full_norm_length:  # Full normalization
        all_interp_indices[-1] = len(scaled_data) - 1
    
    # Interpolate all columns
    sampled_data = []
    for i, interp_idx in enumerate(all_interp_indices):
        if interp_idx == int(interp_idx):  # Exact index
            row = scaled_data.iloc[int(interp_idx)].copy()
        else:  # Interpolation needed
            lower_idx = int(np.floor(interp_idx))
            upper_idx = min(int(np.ceil(interp_idx)), len(scaled_data) - 1)
            weight = interp_idx - lower_idx
            
            # Interpolate numerical columns
            row = scaled_data.iloc[lower_idx].copy()
            for col in ['score']:  # Add other numerical columns as needed
                if lower_idx != upper_idx:
                    row[col] = (scaled_data.iloc[lower_idx][col] * (1 - weight) + 
                               scaled_data.iloc[upper_idx][col] * weight)
        
        sampled_data.append(row)
    
    # Create result dataframe
    result_df = pd.DataFrame(sampled_data).reset_index(drop=True)
    
    # Ensure the last score is exactly preserved
    if step >= full_norm_length:
        result_df.loc[result_df.index[-1], 'score'] = scaled_last_score
    
    # Set boost flags correctly
    result_df['is_boosted'] = False
    if target_boost_step < step:
        result_df.loc[target_boost_step:, 'is_boosted'] = True
    
    return result_df

def do_normalize(df, norm_event_length, step, standard_event_length, eid_to_len_boost_ratio):
    df = df.copy()
    lengths = df.groupby(['event_id', 'idol_id', 'border']).size().reset_index(name='length')

    normalized_events = []

    for _, row in lengths.iterrows():
        event_id = row['event_id']
        idol_id = row['idol_id']
        border = row['border']
        original_length = row['length']

        event_data = df[
            (df['event_id'] == event_id) &
            (df['idol_id'] == idol_id) &
            (df['border'] == border)
        ].copy()
        
        # Get event length and boost info using the new key structure
        event_key = (event_id, idol_id)
        if event_key in eid_to_len_boost_ratio:
            full_event_len = eid_to_len_boost_ratio[event_key]['length']
            actual_boost_start = eid_to_len_boost_ratio[event_key]['boost_start']
        else:
            raise ValueError(f"Event {event_id} with idol {idol_id} not found in eid_to_len_boost_ratio")
        event_type = event_data['event_type'].iloc[0]
        if event_type != 5:
            normalized_data = normalize_consistently(
                df=event_data,
                full_norm_length=norm_event_length,
                step=step,
                standard_event_length=standard_event_length,
                full_event_length=full_event_len,
                actual_boost_start=actual_boost_start,
            )
        else:
            # Event type 5 always has consistent boost ratio
            normalized_data = normalize_consistently(
                df=event_data,
                full_norm_length=norm_event_length,
                step=step,
                standard_event_length=standard_event_length,
                full_event_length=full_event_len,
            )
        normalized_events.append(normalized_data)
    
    normalized_all_data = pd.concat(normalized_events, ignore_index=True)
    normalized_all_data['time_idx'] = (normalized_all_data
                                       .sort_values('aggregated_at')
                                       .groupby(['event_id', 'idol_id', 'border'])
                                       .cumcount())
    return normalized_all_data

def denormalize_consistently(normalized_scores, full_norm_length, target_length, standard_event_length, full_event_length, actual_boost_start=None):
    """
    Revert the normalization process to get original scale scores.
    This is the INVERSE of normalize_consistently - it expands normalized data back to raw temporal resolution.
    
    Args:
        normalized_scores: normalized score array (length = full_norm_length)
        full_norm_length: target normalized length (300)
        target_length: desired output length (raw event length, e.g., 397)
        standard_event_length: standard length used in normalization
        full_event_length: original full length (397)
        actual_boost_start: actual boost start point in original data (e.g., 192)
    
    Returns:
        denormalized scores at original scale and temporal resolution
    """
    # Step 1: Check if we need score scaling
    scale_factor = standard_event_length / full_event_length
    apply_score_scaling = abs(scale_factor - 1.0) > 1e-10
    
    # Step 2: Check if we need temporal expansion
    need_temporal_expansion = target_length != len(normalized_scores)
    
    # Fast path: if no scaling or expansion is needed, return as-is
    if not apply_score_scaling and not need_temporal_expansion:
        logging.debug(f"No denormalization needed - returning normalized scores unchanged")
        logging.debug(f"Normalized final value: {normalized_scores[-1]}")
        return normalized_scores.copy()
    
    # Step 2: Expand temporal resolution using clean approach
    if need_temporal_expansion:
        # Get the normalized future part (entire curve)
        norm_future_part = normalized_scores.copy()
        
        # Get first and last values
        first_value = norm_future_part[0]
        last_value = norm_future_part[-1]
        
        # Make it relative by subtracting the first value
        relative_curve = norm_future_part - first_value
        
        # Create extended timeline with target length
        extended_curve = np.zeros(target_length)
        
        # Interpolate the relative curve to target length
        if len(relative_curve) > 1:
            norm_indices = np.linspace(0, len(relative_curve) - 1, target_length)
            extended_relative = np.interp(norm_indices, np.arange(len(relative_curve)), relative_curve)
        else:
            extended_relative = np.zeros(target_length)
        
        # Add back the first value to get the final smooth curve
        extended_curve = extended_relative + first_value
        
        # Ensure both ends are preserved exactly
        extended_curve[0] = first_value
        extended_curve[-1] = last_value
        
        expanded_scores = extended_curve
        logging.debug(f"Temporal expansion: preserved first={first_value}, last={last_value}")
    else:
        expanded_scores = normalized_scores.copy()
    
    # Step 3: Revert the score scaling (inverse of: scaled_score = original_score * scale_factor)
    if apply_score_scaling:
        denormalized_scores = expanded_scores / scale_factor
        logging.debug(f"Applied score scaling with factor: {scale_factor}")
    else:
        denormalized_scores = expanded_scores
        logging.debug(f"No score scaling applied (scale_factor = {scale_factor})")
    
    # Debug logging
    logging.debug(f"Denormalization summary:")
    logging.debug(f"  Input length: {len(normalized_scores)}, Output length: {len(denormalized_scores)}")
    logging.debug(f"  Normalized final value: {normalized_scores[-1]}")
    logging.debug(f"  Expanded final value: {expanded_scores[-1]}")
    logging.debug(f"  Denormalized final value: {denormalized_scores[-1]}")
    
    # For verification: check if the denormalized final value matches the normalized final value
    # when no score scaling is applied
    if not apply_score_scaling:
        if abs(denormalized_scores[-1] - normalized_scores[-1]) > 1e-10:
            logging.error(f"CRITICAL: Final value mismatch (no scaling case): normalized={normalized_scores[-1]}, denormalized={denormalized_scores[-1]}, diff={abs(denormalized_scores[-1] - normalized_scores[-1])}")
        else:
            logging.debug(f"✓ Final values match (no scaling case): {denormalized_scores[-1]}")
    
    return denormalized_scores

def denormalize_target_to_raw(normalized_target, current_raw_data, full_norm_length, standard_event_length, full_event_length, actual_boost_start=None):
    """
    Denormalize the full normalized target and replace the known part with actual raw data.
    Ensures continuity at the transition point.
    """
    denormalized_scores = denormalize_consistently(
        normalized_target, 
        full_norm_length, 
        full_event_length, 
        standard_event_length, 
        full_event_length, 
        actual_boost_start
    )
    
    # If we have known raw data, replace the beginning with actual values
    # while ensuring continuity at the transition point
    if len(current_raw_data) > 0:
        known_length = min(len(current_raw_data), len(denormalized_scores))
        result = denormalized_scores.copy()
        
        # Replace known part with actual raw data
        result[:known_length] = current_raw_data[:known_length]
        
        # Ensure continuity: if there's a future part, make sure it starts from the last known value
        if known_length < len(result) and known_length > 0:
            # Get the future part that needs to be smoothed
            future_part = result[known_length:].copy()
            last_known = result[known_length - 1]
            future_final = future_part[-1]
            
            # Apply the same smooth scaling approach:
            # 1. Make future part relative by subtracting first value
            future_first = future_part[0]
            relative_future = future_part - future_first
            
            # 2. Scale to match the desired gap (from last_known to future_final)
            if len(relative_future) > 1:
                original_range = relative_future[-1] - relative_future[0]  # Should be 0 since we subtracted first
                target_range = future_final - last_known
                
                # Scale the relative curve to match target range
                if abs(original_range) > 1e-10:
                    scale_factor = target_range / original_range
                    scaled_relative = relative_future * scale_factor
                else:
                    # If original range is 0, create linear interpolation
                    scaled_relative = np.linspace(0, target_range, len(relative_future))
                
                # 3. Add back the offset (last_known) to get final smooth curve
                smooth_future = scaled_relative + last_known
                
                # Ensure exact continuity
                smooth_future[0] = last_known
                smooth_future[-1] = future_final
                
                result[known_length:] = smooth_future
            else:
                # Single point case
                result[known_length] = last_known
        
        logging.debug(f"Raw denormalization summary:")
        logging.debug(f"  Known data length: {len(current_raw_data)}")
        logging.debug(f"  Full denormalized length: {len(denormalized_scores)}")
        logging.debug(f"  Result length: {len(result)}")
        if len(current_raw_data) > 0:
            logging.debug(f"  Last known raw value: {current_raw_data[-1]}")
        if len(result) > len(current_raw_data):
            logging.debug(f"  First predicted raw value: {result[len(current_raw_data)]}")
            logging.debug(f"  Final raw value: {result[-1]}")
            
            # Check for continuity
            if len(current_raw_data) > 0:
                gap = result[len(current_raw_data)] - current_raw_data[-1]
                if abs(gap) > 1e-10:
                    logging.error(f"CRITICAL: Gap in raw data at slice point: {gap}")
                else:
                    logging.debug(f"✓ No gap in raw data at slice point")
        
        return result
    else:
        return denormalized_scores

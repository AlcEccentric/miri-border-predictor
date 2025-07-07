import numpy as np
import pandas as pd
import logging

def normalize_consistently(df, full_norm_length, step, standard_event_length, full_event_length, actual_boost_start=None):
    """
    df: current data
    full_norm_length: target normalized length (300)
    step: how many steps to output
    standard_length_at_step: expected length at current step
    full_event_length: original full length (397)
    actual_boost_start: actual boost start point in original data (e.g., 192)
    """
    target_boost_ratio = 0.55
    target_boost_step = int(target_boost_ratio * full_norm_length)  # 165 for 300
    
    # Calculate corresponding original length based on boost position
    if actual_boost_start is not None:
        if step <= target_boost_step:
            # Pre-boost section
            original_step_length = int(step * actual_boost_start / target_boost_step)
        else:
            # Include both pre-boost and post-boost sections
            pre_boost_length = actual_boost_start
            post_boost_length = int((step - target_boost_step) * 
                                  (full_event_length - actual_boost_start) / 
                                  (full_norm_length - target_boost_step))
            original_step_length = pre_boost_length + post_boost_length
    else:
        # No boost point, use simple ratio
        original_step_length = int(step * full_event_length / full_norm_length)
    
    df = df.iloc[:original_step_length].copy()
    
    # Rest of the normalization logic...
    scale_factor = standard_event_length / full_event_length
    scaled_df = df.copy()
    scaled_df['score'] = df['score'] * scale_factor
    
    boost_start = scaled_df.reset_index().index[scaled_df['is_boosted'] == True][0] if True in scaled_df['is_boosted'].values else None
    
    if boost_start is None:
        indices = np.linspace(0, len(df)-1, step, dtype=int)
        sampled_df = scaled_df.iloc[indices].copy()
    else:
        if target_boost_step < step:
            pre_boost_indices = np.linspace(0, boost_start-1, target_boost_step, dtype=int)
            post_boost_indices = np.linspace(boost_start, len(df)-1, step-target_boost_step, dtype=int)
            indices = np.concatenate([pre_boost_indices, post_boost_indices])
        else:
            indices = np.linspace(0, len(df)-1, step, dtype=int)
        sampled_df = scaled_df.iloc[indices].copy()
    
    sampled_df.reset_index(drop=True, inplace=True)
    
    sampled_df['is_boosted'] = False
    if target_boost_step < step:
        sampled_df.loc[target_boost_step:, 'is_boosted'] = True
    
    return sampled_df

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
        
        normalized_data = normalize_consistently(
            df=event_data,
            full_norm_length=norm_event_length,
            step=step,
            standard_event_length=standard_event_length,
            full_event_length=full_event_len,
            actual_boost_start=actual_boost_start,
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
    target_boost_ratio = 0.55
    target_boost_step = int(target_boost_ratio * full_norm_length)  # 165 for 300
    
    # Step 1: Expand temporal resolution (inverse of the sampling done in normalization)
    if actual_boost_start is not None and target_length > actual_boost_start:
        # Handle boost timing - need to map normalized indices back to raw indices
        raw_indices = []
        
        for raw_idx in range(target_length):
            if raw_idx < actual_boost_start:
                # Pre-boost section: map raw_idx to normalized space
                norm_idx = raw_idx * target_boost_step / actual_boost_start
            else:
                # Post-boost section
                post_boost_raw_idx = raw_idx - actual_boost_start
                post_boost_length = target_length - actual_boost_start
                post_boost_norm_length = full_norm_length - target_boost_step
                norm_idx = target_boost_step + (post_boost_raw_idx * post_boost_norm_length / post_boost_length)
            
            raw_indices.append(min(norm_idx, full_norm_length - 1))
        
        # Interpolate normalized scores to raw temporal resolution
        expanded_scores = np.interp(raw_indices, np.arange(len(normalized_scores)), normalized_scores)
    else:
        # Simple linear mapping - interpolate from norm_length to target_length
        norm_indices = np.linspace(0, len(normalized_scores) - 1, target_length)
        expanded_scores = np.interp(norm_indices, np.arange(len(normalized_scores)), normalized_scores)
    
    # Step 2: Revert the score scaling (inverse of: scaled_score = original_score * scale_factor)
    scale_factor = standard_event_length / full_event_length
    denormalized_scores = expanded_scores / scale_factor
    
    # Debug: Check if scale factor is 1.0 (no score scaling case)
    if abs(scale_factor - 1.0) < 1e-10:
        # When no score scaling was applied, the final values should match exactly
        # (accounting only for potential interpolation differences)
        logging.debug(f"No score scaling applied (scale_factor = {scale_factor})")
        logging.debug(f"Normalized final value: {normalized_scores[-1]}")
        logging.debug(f"Denormalized final value: {denormalized_scores[-1]}")
    
    return denormalized_scores

def denormalize_target_to_raw(normalized_target, current_raw_data, full_norm_length, standard_event_length, full_event_length, actual_boost_start=None):
    denormalized_scores = denormalize_consistently(
        normalized_target, 
        full_norm_length, 
        full_event_length, 
        standard_event_length, 
        full_event_length, 
        actual_boost_start
    )
    
    # If we have known raw data, replace the beginning with actual values
    if len(current_raw_data) > 0:
        known_length = min(len(current_raw_data), len(denormalized_scores))
        result = denormalized_scores.copy()
        result[:known_length] = current_raw_data[:known_length]
        return result
    else:
        return denormalized_scores

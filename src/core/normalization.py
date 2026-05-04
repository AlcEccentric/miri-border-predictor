import numpy as np
import pandas as pd
import logging

def normalize_consistently(df,
                           full_norm_length, step,
                           standard_event_length,
                           full_event_length,
                           standard_event_boost_ratio,
                           actual_boost_start=None):
    """
    Normalize event data consistently while preserving boost timing and score scaling.
    
    Args:
        df: Raw event data (sorted by time)
        full_norm_length: Target normalized length (e.g., 300)
        step: Current step index (length of known data to return)
        standard_event_length: Standard event length for scaling
        full_event_length: Actual length of this event
        actual_boost_start: Actual boost start index in raw data
    
    Returns:
        Normalized dataframe with length = step
    """
    if len(df) == 0:
        raise ValueError("Input dataframe is empty!")

    # Sort and reset index
    df = df.sort_values('aggregated_at').reset_index(drop=True)

    # Step 1: Apply score scaling based on event length difference
    scale_factor = standard_event_length / full_event_length
    
    scaled_df = df.copy()
    scaled_df['score'] = df['score'] * scale_factor

    # Step 2: Determine boost timing for normalization
    target_boost_ratio = standard_event_boost_ratio  # Standard boost ratio
    target_boost_step = int(target_boost_ratio * full_norm_length)

    # Step 3: Build normalization grid using full_event_length and boost timing
    # This grid is always based on the expected full event, not the length of df
    if actual_boost_start is not None and actual_boost_start < full_event_length:
        raw_boost_idx = actual_boost_start
        pre_boost_raw_range = raw_boost_idx
        pre_boost_norm_range = target_boost_step
        post_boost_raw_range = full_event_length - raw_boost_idx
        post_boost_norm_range = full_norm_length - target_boost_step

        def get_full_raw_index(norm_idx):
            if norm_idx < target_boost_step:
                if pre_boost_norm_range <= 1:
                    return 0
                ratio = norm_idx / (pre_boost_norm_range - 1)
                return ratio * (pre_boost_raw_range - 1)
            else:
                if post_boost_norm_range <= 1:
                    return raw_boost_idx
                ratio = (norm_idx - target_boost_step) / (post_boost_norm_range - 1)
                return raw_boost_idx + ratio * (post_boost_raw_range - 1)
    else:
        def get_full_raw_index(norm_idx):
            if full_norm_length <= 1:
                return 0
            ratio = norm_idx / (full_norm_length - 1)
            return ratio * (full_event_length - 1)

    # Step 4: Interpolate available raw data onto the full normalization grid
    result_rows = []
    for norm_idx in range(step):
        full_raw_idx = get_full_raw_index(norm_idx)
        # Map full_raw_idx to available data
        # If partial data, clamp to last available index
        available_len = len(scaled_df)
        mapped_idx = min(full_raw_idx, available_len - 1)
        if mapped_idx == int(mapped_idx):
            row = scaled_df.iloc[int(mapped_idx)].copy()
        else:
            lower_idx = int(np.floor(mapped_idx))
            upper_idx = min(int(np.ceil(mapped_idx)), available_len - 1)
            if lower_idx == upper_idx:
                row = scaled_df.iloc[lower_idx].copy()
            else:
                weight = mapped_idx - lower_idx
                row = scaled_df.iloc[lower_idx].copy()
                for col in ['score']:
                    row[col] = (scaled_df.iloc[lower_idx][col] * (1 - weight) +
                               scaled_df.iloc[upper_idx][col] * weight)
        result_rows.append(row)

    # Step 5: Create result dataframe
    result_df = pd.DataFrame(result_rows).reset_index(drop=True)

    # Step 6: Set correct boost flags
    result_df['is_boosted'] = False
    if target_boost_step < step:
        result_df.loc[target_boost_step:, 'is_boosted'] = True

    # Step 7: Preserve the value at step-1 exactly
    if step > 0:
        available_len = len(scaled_df)
        last_full_raw_idx = get_full_raw_index(step - 1)
        mapped_last_idx = min(last_full_raw_idx, available_len - 1)
        if mapped_last_idx == int(mapped_last_idx):
            exact_score = scaled_df.iloc[int(mapped_last_idx)]['score']
        else:
            lower_idx = int(np.floor(mapped_last_idx))
            upper_idx = min(int(np.ceil(mapped_last_idx)), available_len - 1)
            weight = mapped_last_idx - lower_idx
            exact_score = (scaled_df.iloc[lower_idx]['score'] * (1 - weight) +
                          scaled_df.iloc[upper_idx]['score'] * weight)
        result_df.loc[step - 1, 'score'] = exact_score

    return result_df

def do_normalize(df,
                 norm_event_length,
                 step,
                 standard_event_length,
                 standard_event_boost_ratio,
                 eid_to_len_boost_ratio):
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
                standard_event_boost_ratio=standard_event_boost_ratio,
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

def denormalize_consistently(normalized_scores,
                             full_norm_length,
                             target_length,
                             standard_event_length,
                             full_event_length,
                             standard_event_boost_ratio,
                             actual_boost_start=None):
    """
    Revert the normalization process to get original scale scores.
    This matches the logic in normalize_consistently.
    """
    # Step 1: Reverse score scaling
    scale_factor = standard_event_length / full_event_length
    scores = normalized_scores / scale_factor
    logging.debug(f"Denormalized final score: {scores[-1]}, scale factor: {scale_factor}")

    # Step 2: Build mapping from normalized index to raw index (same as in normalization)
    target_boost_ratio = standard_event_boost_ratio
    target_boost_step = int(target_boost_ratio * full_norm_length)

    if actual_boost_start is not None and actual_boost_start < full_event_length:
        raw_boost_idx = actual_boost_start
        pre_boost_raw_range = raw_boost_idx
        pre_boost_norm_range = target_boost_step
        post_boost_raw_range = full_event_length - raw_boost_idx
        post_boost_norm_range = full_norm_length - target_boost_step

        def get_norm_index(raw_idx): # type: ignore
            if raw_idx < raw_boost_idx:
                if pre_boost_raw_range <= 1:
                    return 0
                ratio = raw_idx / (pre_boost_raw_range - 1)
                return ratio * (pre_boost_norm_range - 1)
            else:
                if post_boost_raw_range <= 1:
                    return target_boost_step
                ratio = (raw_idx - raw_boost_idx) / (post_boost_raw_range - 1)
                return target_boost_step + ratio * (post_boost_norm_range - 1)
    else:
        def get_norm_index(raw_idx):
            if full_event_length <= 1:
                return 0
            ratio = raw_idx / (full_event_length - 1)
            return ratio * (full_norm_length - 1)

    # Step 3: Interpolate normalized scores onto the raw timeline
    denorm_scores = []
    for raw_idx in range(target_length):
        norm_idx = get_norm_index(raw_idx)
        if norm_idx == int(norm_idx):
            score = scores[int(norm_idx)]
        else:
            lower_idx = int(np.floor(norm_idx))
            upper_idx = min(int(np.ceil(norm_idx)), len(scores) - 1)
            if lower_idx == upper_idx:
                score = scores[lower_idx]
            else:
                weight = norm_idx - lower_idx
                score = scores[lower_idx] * (1 - weight) + scores[upper_idx] * weight
        denorm_scores.append(score)
    denorm_scores = np.array(denorm_scores)

    # Ensure first and last values match exactly
    if len(denorm_scores) > 0:
        denorm_scores[0] = scores[0]
        denorm_scores[-1] = scores[-1]

    return denorm_scores

def denormalize_target_to_raw(normalized_target, current_step, current_raw_data, full_norm_length, standard_event_length, standard_event_boost_ratio, full_event_length, actual_boost_start=None):
    """
    Denormalize the full normalized target and replace the known part with actual raw data.
    Ensures continuity at the transition point.
    """
    logging.info("Last value of normalized target: {}".format(normalized_target[-1]))
    denormalized_scores = denormalize_consistently(
        normalized_scores=normalized_target, 
        full_norm_length=full_norm_length, 
        target_length=full_event_length, 
        standard_event_length=standard_event_length, 
        standard_event_boost_ratio=standard_event_boost_ratio,
        full_event_length=full_event_length, 
        actual_boost_start=actual_boost_start,
    )
    logging.info(f"Denormalized target last value: {denormalized_scores[-1]}")
    # If we have known raw data, replace the beginning with actual values
    # while ensuring continuity at the transition point
    if len(current_raw_data) > 0:
        known_length = int(min(min(len(current_raw_data), len(denormalized_scores)), len(denormalized_scores) * current_step/full_norm_length))
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
        
        return result
    else:
        return denormalized_scores

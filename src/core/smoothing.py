import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def smooth(scores, win, poly):
    return savgol_filter(scores, window_length=win, polyorder=poly)

def smooth_scores_preserve_ends(df,
                              first_half_win_ratio=0.8,
                              polyorder_1st=1,
                              second_half_win_ratio=0.8,
                              polyorder_2nd=1,
                              end_start=290,
                              end_polyorder=2,
                              overlap_size=30,
                              overlap_size1=60,):
    smoothed_df = df.copy()

    for (event_id, border), group in df.groupby(['event_id', 'border']):
        mask = (smoothed_df['event_id'] == event_id) & (smoothed_df['border'] == border)
        scores = group['score'].values
        boost_start = df.reset_index().index[df['is_boosted'] == True][0] if True in df['is_boosted'].values else 10000
        sequence_length = len(scores)
        first_half_length = min(sequence_length, boost_start)
        second_half_length = sequence_length - boost_start

        # Calculate window sizes
        win1 = int(first_half_win_ratio * first_half_length)
        if win1 % 2 == 0:
            win1 += 1
        win1 = max(win1, polyorder_1st + 2)

        overlap_end = min(first_half_length + overlap_size, sequence_length)
        first_half_s = smooth(scores[:overlap_end], win1, polyorder_1st)

        final_smoothed = np.zeros(sequence_length)
        final_smoothed[:first_half_length] = first_half_s[:first_half_length]

        if second_half_length > 0:
            win2 = int(second_half_win_ratio * (second_half_length)) + overlap_size1
            if win2 % 2 == 0:
                win2 += 1
            win2 = max(win2, polyorder_2nd + 2)

            # Smooth second section with overlap on both sides
            overlap_start = max(0, first_half_length - overlap_size1)
            second_half_end = min(sequence_length, end_start + overlap_size)
            second_half_s = smooth(scores[overlap_start:second_half_end], win2, polyorder_2nd)

            # Create weights for smooth transition
            if first_half_length > overlap_start:
                weights = np.linspace(0, 1, first_half_length - overlap_start)
                overlap_region = weights * second_half_s[:first_half_length - overlap_start] + \
                               (1 - weights) * final_smoothed[overlap_start:first_half_length]
                final_smoothed[overlap_start:first_half_length] = overlap_region

            final_smoothed[first_half_length:end_start] = second_half_s[(first_half_length - overlap_start):(end_start - overlap_start)]

        if sequence_length > end_start:
            # Smooth end section with overlap
            overlap_start = max(0, end_start - overlap_size)
            end_scores = scores[overlap_start:]
            end_win = sequence_length - end_start
            if end_win % 2 == 0:
                end_win += 1
            end_win = max(end_win, end_polyorder + 2)
            
            end_smoothed = savgol_filter(end_scores, window_length=end_win, polyorder=end_polyorder)
            
            # Create weights for smooth transition
            weights = np.linspace(0, 1, end_start - overlap_start)
            overlap_region = (1 - weights) * final_smoothed[overlap_start:end_start] + \
                           weights * end_smoothed[:(end_start - overlap_start)]
            final_smoothed[overlap_start:end_start] = overlap_region
            final_smoothed[end_start:] = end_smoothed[(end_start - overlap_start):]

        # Assign the smoothed values back to the dataframe
        smoothed_df.loc[mask, 'score'] = final_smoothed
    
    return smoothed_df

def smooth_scores_multi_range(df, ranges, end_transition_points=20):
    """
    Simplified smoothing that preserves monotonic behavior for ranking data.
    """
    smoothed_df = df.copy()

    for (event_id, border), group in df.groupby(['event_id', 'border']):
        mask = (smoothed_df['event_id'] == event_id) & (smoothed_df['border'] == border)
        scores = group['score'].values
        sequence_length = len(scores)
        original_start = scores[0]
        original_end = scores[-1]

        # Find leading zeros only
        leading_zeros_end = 0
        for i in range(sequence_length):
            if scores[i] > 0:
                break
            leading_zeros_end = i + 1
        
        final_smoothed = np.zeros(sequence_length)
        weights_sum = np.zeros(sequence_length)

        # Apply smoothing ranges
        for i, r in enumerate(ranges):
            start = r.get('start', 0)
            end = r.get('end', sequence_length)
            if start is None:
                start = 0
            if end is None:
                end = sequence_length
                
            overlap = r.get('overlap', 30)
            extended_start = max(0, start - overlap)
            extended_end = min(sequence_length, end + overlap)
            
            range_length = extended_end - extended_start
            win = int(r['win_ratio'] * range_length)
            if win % 2 == 0:
                win += 1
            win = max(win, r['poly'] + 2)
            win = min(win, range_length)
            
            if range_length > r['poly'] + 2:
                range_scores = scores[extended_start:extended_end]
                smoothed = smooth(range_scores, win, r['poly'])
                
                weights = np.ones(extended_end - extended_start)
                
                # Fade in/out weights for overlapping regions
                if extended_start < start:
                    fade_in_length = start - extended_start
                    fade_in_length = min(fade_in_length, len(weights))
                    if fade_in_length > 0:
                        weights[:fade_in_length] = np.linspace(0, 1, fade_in_length)
                
                if extended_end > end:
                    fade_out_length = extended_end - end
                    fade_out_length = min(fade_out_length, len(weights))
                    if fade_out_length > 0:
                        weights[-fade_out_length:] = np.linspace(1, 0, fade_out_length)
                
                final_smoothed[extended_start:extended_end] += smoothed * weights
                weights_sum[extended_start:extended_end] += weights

        # Normalize by weights
        mask_nonzero = weights_sum > 0
        final_smoothed[mask_nonzero] /= weights_sum[mask_nonzero]

        # Preserve leading zeros
        if leading_zeros_end > 0:
            final_smoothed[:leading_zeros_end] = 0

        # Enforce monotonic behavior after leading zeros
        if leading_zeros_end < sequence_length:
            for i in range(leading_zeros_end + 1, sequence_length):
                if final_smoothed[i] < final_smoothed[i-1]:
                    final_smoothed[i] = final_smoothed[i-1]

        # Clip any negative values to zero
        final_smoothed = np.maximum(final_smoothed, 0)

        # Handle end transitions if specified
        if end_transition_points > 0:
            actual_transition_points = min(end_transition_points, len(final_smoothed) - 1)
            
            # Preserve start point if not zero
            if original_start > 0:
                weights_start = np.linspace(0, 1, actual_transition_points)
                start_transition = (1 - weights_start) * original_start + weights_start * final_smoothed[actual_transition_points]
                final_smoothed[:actual_transition_points] = start_transition
            
            # Preserve end point if not zero
            if original_end > 0:
                weights_end = np.linspace(1, 0, actual_transition_points)
                end_transition = weights_end * final_smoothed[-actual_transition_points-1] + (1 - weights_end) * original_end
                final_smoothed[-actual_transition_points:] = end_transition

            # Ensure exact preservation of endpoints
            final_smoothed[0] = original_start
            final_smoothed[-1] = original_end

        smoothed_df.loc[mask, 'score'] = final_smoothed
    
    return smoothed_df

def get_optimized_smoothing_config(event_type: float, sub_event_type: float, border: float) -> list:
    if event_type == 3.0:
        if sub_event_type == 1.0:
            if border == 100:
                return [
                    {'start': None, 'end': 10, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 1.0, 'poly': 1, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 290, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
                ]
            else:  # border 2500
                return [
                    {'start': None, 'end': 10, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
                ]
        else:  # sub_event_type 2.0
            if border == 100:
                return [
                    {'start': None, 'end': 270, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                    {'start': 270, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
                ]
            else:  # border 2500
                return [
                   {'start': None, 'end': 270, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                    {'start': 270, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
                ]
    
    elif event_type == 4.0:
        if sub_event_type == 1.0:
            if border == 100:
                return [
                    {'start': None, 'end': 10, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 1.0, 'poly': 1, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                    {'start': 290, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
                ]
            else:  # border 2500
                return [
                    {'start': None, 'end': 10, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
                ]
        else:  # sub_event_type 2.0
            if border == 100:
                return [
                    {'start': None, 'end': 10, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 1.0, 'poly': 1, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                    {'start': 290, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
                ]
            else:  # border 2500
                return [
                    {'start': None, 'end': 10, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                    {'start': None, 'end': 170, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                    {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
                ]
    
    elif event_type == 11.0:
        if border == 100:
            return [
                {'start': None, 'end': 10, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                {'start': None, 'end': 170, 'win_ratio': 0.9, 'poly': 2, 'overlap': 10},
                {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
            ]
        else:  # border 2500
            return [
                {'start': None, 'end': 10, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                {'start': None, 'end': 170, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
            ]
    
    elif event_type == 13.0:
        if border == 100:
            return [
                {'start': None, 'end': 10, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                {'start': None, 'end': 170, 'win_ratio': 1.0, 'poly': 1, 'overlap': 10},
                {'start': 170, 'end': 290, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                {'start': 290, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
            ]
        else:  # border 2500
            return [
                {'start': None, 'end': 10, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5},
                {'start': None, 'end': 170, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                {'start': 170, 'end': 290, 'win_ratio': 0.8, 'poly': 2, 'overlap': 10},
                {'start': 290, 'end': None, 'win_ratio': 0.9, 'poly': 1, 'overlap': 5}
            ]
    elif event_type == 5.0:
        if border == 100:
            return [
                {'start': None, 'end': 170, 'win_ratio': 1.0, 'poly': 1, 'overlap': 10},
                {'start': 170, 'end': 290, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5},
                {'start': 290, 'end': None, 'win_ratio': 1.0, 'poly': 1, 'overlap': 5}
            ]
        elif border == 1000: 
            return [
                {'start': None, 'end': None, 'win_ratio': 0.6, 'poly': 1, 'overlap': 10}
            ]
    # Default return if no condition matches
    return []

def generate_smoothed_dfs(norm_full_df: pd.DataFrame, 
                          norm_part_df: pd.DataFrame, ) -> tuple:
    """Generate smoothed full and partial dataframes"""

    # Process full trajectories
    smoothed_full_df = pd.DataFrame()
    for (event_type, sub_event_type, idol_id, border), group in norm_full_df.groupby(['event_type', 'sub_event_type', 'idol_id', 'border']):
        ranges = get_optimized_smoothing_config(event_type, sub_event_type, border)
        group_smoothed = smooth_scores_multi_range(group, ranges, end_transition_points=5)
        smoothed_full_df = pd.concat([smoothed_full_df, group_smoothed])
    
    # Process partial trajectories
    smoothed_step_df = pd.DataFrame()
    for (event_type, sub_event_type, _, border), group in norm_part_df.groupby(['event_type', 'sub_event_type', 'idol_id', 'border']):
        ranges = get_optimized_smoothing_config(event_type, sub_event_type, border)
        group_smoothed = smooth_scores_multi_range(group, ranges, end_transition_points=0)
        smoothed_step_df = pd.concat([smoothed_step_df, group_smoothed])
    
    # Sort and reset index
    smoothed_full_df = smoothed_full_df.sort_values(['event_id', 'idol_id', 'border', 'time_idx']).reset_index(drop=True)
    smoothed_step_df = smoothed_step_df.sort_values(['event_id', 'idol_id', 'border', 'time_idx']).reset_index(drop=True)

    return smoothed_full_df, smoothed_step_df

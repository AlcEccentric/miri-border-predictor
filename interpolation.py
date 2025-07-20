import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def _populate_elapsed_mins(
    all_data: pd.DataFrame,
) -> pd.DataFrame:
    all_data_copy = all_data.copy()
    all_data_copy['elapsed_mins'] = (all_data_copy['aggregated_at'] - all_data_copy['start_at']).dt.total_seconds() / 60
    return all_data_copy

def _generate_expected_elapsed_mins(
    event_border_data: pd.DataFrame,
    expected_interval: int,
    interpolate_until: pd.Timestamp | None = None,
) -> pd.DataFrame:
    "Generate expected elapsed minutes for each event"
    event_border_data_row = event_border_data.iloc[0]
    start_at = event_border_data_row['start_at']
    if interpolate_until is not None:
        event_mins = (interpolate_until - start_at).total_seconds() / 60
    else:
        event_mins = (event_border_data_row['end_at'] - start_at).total_seconds() / 60

    return pd.DataFrame({
        'elapsed_mins': np.arange(0, event_mins + expected_interval, expected_interval),
    })

def _actually_interpolate_data(
    event_border_data: pd.DataFrame,
    first_row: pd.Series,
) -> pd.DataFrame:
    "Actually interpolate data"
    scores = event_border_data['score'].copy()
    is_anomaly = np.zeros_like(scores, dtype=bool)
    current_max_score = float('-inf')
    last_valid_score = None
    n_missing = 0

    # Replace invalid scores with NaNs
    for i in range(len(scores)):
        if pd.notna(scores[i]):
            if not last_valid_score:
                current_max_score = scores[i]
                last_valid_score = scores[i]
            else:
                if scores[i] < current_max_score:
                    is_anomaly[i] = True
                    scores[i] = np.nan
                else:
                    current_max_score = max(current_max_score, scores[i])
                    last_valid_score = scores[i]
        else:
            n_missing += 1

    if scores.notna().sum() < 2:
        raise ValueError(f"Too few valid points for interpolating")
    
    # Interpolate missing data
    x = np.arange(len(scores))
    valid_mask = ~is_anomaly & scores.notna()
    if np.any(valid_mask):
        x_valid = x[valid_mask]
        y_valid = scores[valid_mask]
        interpolator = PchipInterpolator(x_valid, y_valid)
        interpolated_points = interpolator(x)
        scores = np.where(valid_mask, scores, interpolated_points)

    event_border_data['original_score'] = event_border_data['score']
    event_border_data['score'] = scores

    # Copy over fields from the first row to interpolated rows
    interpolated_mask = ~valid_mask
    if np.any(interpolated_mask):
        cols_to_copy = [col for col in event_border_data.columns 
                    if col not in ['score', 'original_score', 'aggregated_at', 'elapsed_mins']]
        for col in cols_to_copy:
            event_border_data.loc[interpolated_mask, col] = first_row[col]

    # Compute aggregated_at = start_at + elapsed_mins
    if 'start_at' not in first_row:
        raise ValueError("start_at not found in data, required to recompute aggregated_at")
    event_border_data.loc[interpolated_mask, 'aggregated_at'] = (
        pd.to_datetime(first_row['start_at']) +
        pd.to_timedelta(event_border_data.loc[interpolated_mask, 'elapsed_mins'], unit='m')
    )
    return event_border_data

def interpolate(
    all_data: pd.DataFrame,
    current_event_id: int,
    expected_interval: int = 30,
) -> pd.DataFrame:
    "Interpolate missing & invalid data up to a specified time"
    all_data = _populate_elapsed_mins(all_data)

    event_border_data_list = []

    for (event_id, _, _), event_border_data in all_data.groupby(['event_id', 'idol_id', 'border']):
        first_row = event_border_data.iloc[0]
        last_row = event_border_data.iloc[-1]
        interpolate_until = None if current_event_id == None or event_id != current_event_id else last_row['aggregated_at']
        expected_elapsed_mins_df = _generate_expected_elapsed_mins(
            event_border_data, expected_interval, interpolate_until=interpolate_until
        )
        event_border_data = event_border_data.reset_index(drop=True)
        event_border_data = expected_elapsed_mins_df.merge(
            event_border_data[event_border_data.columns], 
            on='elapsed_mins', 
            how='left'
        )
        event_border_data = _actually_interpolate_data(event_border_data, first_row)
        event_border_data_list.append(event_border_data)

    return pd.concat(event_border_data_list, ignore_index=True)
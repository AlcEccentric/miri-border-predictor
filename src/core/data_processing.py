import pandas as pd

def calculate_relative_diff(group):
    group = group.sort_values('time_idx')
    score_diff = group['score'].shift(-1) - group['score']
    relative_diff = score_diff / group['score']
    relative_diff = relative_diff.fillna(0)
    return relative_diff

def process_data(all_data: pd.DataFrame):
    processed_all_data = all_data.copy()

    processed_all_data['length'] = (all_data['end_at'] - all_data['start_at']).dt.total_seconds() / (24 * 60 * 60)
    time_diff_remaining = (all_data['end_at'] - all_data['aggregated_at'])
    processed_all_data['days_remaining'] = time_diff_remaining.dt.total_seconds() / (24 * 60 * 60)

    processed_all_data['year'] = processed_all_data['aggregated_at'].dt.year - 2017
    processed_all_data['is_final_day'] = processed_all_data['days_remaining'] < 1
    processed_all_data['is_boosted'] = all_data['aggregated_at'] >= all_data['boost_at']

    processed_all_data = processed_all_data[[
        # orignal
        'event_id',
        'name',
        'event_type',
        'internal_event_type',
        'idol_id',
        'border',
        'aggregated_at',
        'score',
        'date',
        # new
        'year',
        'days_remaining',
        'length',
        'is_final_day',
        'is_boosted',
    ]]
    return processed_all_data

def combine_info(
    border_info: pd.DataFrame, 
    event_info: pd.DataFrame
) -> pd.DataFrame:
    "Construct the dataset for training"
    all_data = border_info.merge(
        event_info[['event_id', 'name', 'event_type', 'internal_event_type', 'start_at', 'end_at', 'boost_at']], 
        on='event_id',
        how='inner'
    )

    problematic_rows = all_data[all_data['aggregated_at'] >= all_data['end_at']]
    
    if not problematic_rows.empty:
        latest_problematic = problematic_rows.loc[
            problematic_rows.groupby(['event_id', 'idol_id', 'border'])['aggregated_at'].idxmax()
        ].copy()
        latest_problematic['aggregated_at'] = latest_problematic['end_at']
        all_data = all_data[all_data['aggregated_at'] < all_data['end_at']]
        all_data = pd.concat([all_data, latest_problematic], ignore_index=True)

    all_data['date'] = all_data['aggregated_at'].dt.date

    return all_data.sort_values('aggregated_at')

def purge(df, step, norm_event_length, eid_to_len_boost_ratio):
    df = df.copy()
    keep_percentage = step / norm_event_length
    processed_groups = []
    
    for (eid, iid, _), group in df.groupby(['event_id', 'idol_id', 'border']):
        
        group_sorted = group.sort_values('aggregated_at')
        full_event_length = eid_to_len_boost_ratio[(eid, iid)]['length']
        rows_to_keep = max(1, int(full_event_length * keep_percentage))
        kept_rows = group_sorted.head(rows_to_keep)
        processed_groups.append(kept_rows)
    
    return pd.concat(processed_groups, ignore_index=True)

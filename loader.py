import pandas as pd
import os
import json
import logging
from typing import Tuple, List, Optional, Dict, Any
import boto3
from io import StringIO
from data_processing import combine_info, process_data
from interpolation import interpolate
from r2_client import R2Client

BUCKET_NAME = 'mltd-border-predict'

def get_event_name_mapping_from_r2(r2_client: R2Client) -> Dict[int, str]:
    try:
        event_file_key = 'event_info/event_info_all.csv'
        event_obj = r2_client.get_object(BUCKET_NAME, event_file_key)
        event_info = pd.read_csv(StringIO(event_obj['Body'].read().decode('utf-8')))

        event_mapping = dict(zip(event_info['event_id'], event_info['name']))
        logging.info(f"Created event name mapping for {len(event_mapping)} events")
        return event_mapping
    except Exception as e:
        logging.error(f"Error creating event name mapping: {e}", exc_info=True)
        return {}

def upload_predictions_to_r2(
    r2_client: R2Client,
    results: Dict,
    event_id: int
) -> None:
    logging.info("Uploading prediction results to R2...")
    
    # Get event name mapping once
    event_mapping = get_event_name_mapping_from_r2(r2_client)
    event_name = event_mapping.get(event_id, "Unknown Event")
    
    uploaded_count = 0
    for idol_id, borders_data in results.items():
        for border, result_dict in borders_data.items():
            json_data = json.dumps(result_dict, indent=2)

            file_key = f'prediction/{idol_id}/{border}/predictions.json'
            try:
                r2_client.put_object(
                    bucket_name=BUCKET_NAME,
                    key=file_key,
                    body=json_data.encode('utf-8'),
                    ContentType='application/json'
                )
                uploaded_count += 1
                logging.info(f"Uploaded prediction for idol {idol_id}, border {border}")
            except Exception as e:
                logging.error(f"Error uploading prediction for idol {idol_id}, border {border}: {e}", exc_info=True)
    
    logging.info(f"Successfully uploaded {uploaded_count} prediction files to R2 for event: {event_name}")

def _process_border_and_event_data(
    border_info: pd.DataFrame,
    event_info: pd.DataFrame,
    event_id_range: Tuple[int, int] = (0, 1000),
    max_look_back_year: int = 100,
    exclude_event_types: Optional[List[int]] = None,
    exclude_event_ids: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if exclude_event_types is None:
        exclude_event_types = [10, 12]
    
    if exclude_event_ids is None:
        exclude_event_ids = [48, 58, 68, 72, 137, 220, 170, 296, 331]

    border_info['aggregated_at'] = pd.to_datetime(border_info['aggregated_at'])
    event_info['start_at'] = pd.to_datetime(event_info['start_at'])
    event_info['start_date'] = pd.to_datetime(event_info['start_at']).dt.date
    event_info['end_at'] = pd.to_datetime(event_info['end_at']) + pd.Timedelta(seconds=1)
    event_info['boost_at'] = pd.to_datetime(event_info['boost_at'])

    # Add border info for the start time
    earliest_rows = border_info.sort_values('aggregated_at').groupby(['event_id', 'border']).first().reset_index()
    new_rows = earliest_rows.copy()
    new_rows['score'] = 0
    new_rows['aggregated_at'] = new_rows['aggregated_at'].dt.normalize() + pd.Timedelta(hours=15)
    border_info = pd.concat([border_info, new_rows], ignore_index=True)
    border_info = border_info.sort_values(['event_id', 'border', 'aggregated_at'])
    
    # Filter event_info
    earliest_start_at = pd.Timestamp.now(tz='Asia/Tokyo') - pd.DateOffset(years=max_look_back_year)
    event_info = event_info[(event_info['event_id'] >= event_id_range[0]) & 
                            (event_info['event_id'] <= event_id_range[1]) &
                            ~event_info['event_id'].isin(exclude_event_ids) &
                            ~event_info['event_type'].isin(exclude_event_types) &
                            (event_info['start_at'] >= earliest_start_at)]

    # Adjust aggregated_at values
    merged = border_info.merge(event_info[['event_id', 'end_at']], on='event_id', how='inner')
    too_late = merged['aggregated_at'] > merged['end_at']
    if too_late.any():
        merged.loc[too_late, 'aggregated_at'] = merged.loc[too_late, 'end_at']

    # Check for duplicates
    for (_, _), border_info_per_event_border in merged.groupby(['event_id', 'border']):
        has_duplicates = border_info_per_event_border['aggregated_at'].duplicated().any()
        if has_duplicates:
            raise ValueError(f"Duplicate aggregated_at values found for event_id={_['event_id']}, border={_['border']}")
    
    return merged[border_info.columns], event_info

def _process_anniversary_data(
    border_info: pd.DataFrame,
    event_info: pd.DataFrame,
    filtered_event_ids: List[int],
    idol_ids: List[int],
    borders: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Make a copy to avoid SettingWithCopyWarning
    event_info = event_info.copy()
    
    event_info['start_at'] = pd.to_datetime(event_info['start_at'])
    event_info['start_date'] = pd.to_datetime(event_info['start_at']).dt.date
    event_info['end_at'] = pd.to_datetime(event_info['end_at']) + pd.Timedelta(seconds=1)
    event_info['boost_at'] = pd.to_datetime(event_info['boost_at'], errors='coerce')
    
    latest_valid_date = pd.Timestamp('2010-01-01', tz='Asia/Tokyo')
    valid_mask = (~event_info['boost_at'].isna()) & (event_info['boost_at'] > latest_valid_date)
    valid_event_info = event_info[valid_mask]
    
    if not border_info.empty:
        border_info['aggregated_at'] = pd.to_datetime(border_info['aggregated_at'])
        
        # Add start time rows
        start_time_rows = []
        for event_id in filtered_event_ids:
            event_start = valid_event_info[valid_event_info['event_id'] == event_id]['start_at'].iloc[0]
            for idol_id in idol_ids:
                for border in borders:
                    existing_combo = border_info[
                        (border_info['event_id'] == event_id) & 
                        (border_info['idol_id'] == idol_id) & 
                        (border_info['border'] == border)
                    ]
                    
                    if not existing_combo.empty:
                        start_time_rows.append({
                            'event_id': event_id,
                            'idol_id': idol_id,
                            'border': border,
                            'aggregated_at': event_start,
                            'score': 0
                        })
        
        if start_time_rows:
            start_time_df = pd.DataFrame(start_time_rows)
            border_info = pd.concat([border_info, start_time_df], ignore_index=True)
            border_info = border_info.sort_values(['event_id', 'idol_id', 'border', 'aggregated_at'])
        
        # Adjust aggregated_at values
        merged = border_info.merge(valid_event_info[['event_id', 'end_at']], on='event_id', how='inner')
        too_late = merged['aggregated_at'] > merged['end_at']
        if too_late.any():
            merged.loc[too_late, 'aggregated_at'] = merged.loc[too_late, 'end_at']
        
        # Check for duplicates
        for (event_id, idol_id, border), border_info_per_event_border in merged.groupby(['event_id', 'idol_id', 'border']):
            has_duplicates = border_info_per_event_border['aggregated_at'].duplicated().any()
            if has_duplicates:
                duplicated_mask = border_info_per_event_border['aggregated_at'].duplicated(keep=False)
                duplicate_rows = border_info_per_event_border[duplicated_mask].sort_values('aggregated_at')
                
                duplicate_info = []
                for _, row in duplicate_rows.iterrows():
                    duplicate_info.append(f"aggregated_at={row['aggregated_at']}, score={row['score']}")
                
                duplicate_summary = "\n    ".join(duplicate_info)
                
                raise ValueError(
                    f"Duplicate aggregated_at values found for event_id={event_id}, idol_id={idol_id}, border={border}\n"
                    f"  Duplicate entries ({len(duplicate_rows)} rows):\n"
                    f"    {duplicate_summary}"
                )
        
        border_info = merged[border_info.columns]
    
    return border_info, valid_event_info

def load_data_from_r2(
    r2_client: R2Client,
    current_event_id: int,
    borders: List[int] = [100, 2500],
    max_look_back_year: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_file_key = 'event_info/event_info_all.csv'
    event_id_range = (1, current_event_id)
    exclude_event_types = [
        10, # ツインステージ
        12, # ツインステージ   
    ]

    # Outliers which are not representative of typical behavior
    exclude_event_ids = [
        48,
        58,  # ラスト・アクトレス (too many free jewels)
        68,
        72,
        137, # アライブファクター
        220, # boost started at the beginning? wtf?? 220.0
        170, # outsider for チューン
        296,
        331, # ...
    ]
    
    logging.info(f"Loading data from R2 with range {event_id_range}")
    
    # Load event info
    event_obj = r2_client.get_object(BUCKET_NAME, event_file_key)
    event_info = pd.read_csv(StringIO(event_obj['Body'].read().decode('utf-8')))
    
    # Load border info from individual files
    border_dfs = []
    
    for event_id in range(1, current_event_id + 1):
        for border in borders:
            filename = f'border_info_{event_id}_0_{border}.csv'
            file_key = f'border_info/{filename}'
            
            try:
                obj = r2_client.get_object(BUCKET_NAME, file_key)
                df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                df['event_id'] = event_id
                df['border'] = border
                border_dfs.append(df)
            except r2_client.client.exceptions.NoSuchKey:
                continue
            except Exception as e:
                logging.warning(f"Could not load {filename} from R2: {e}", exc_info=True)
    
    if border_dfs:
        border_info = pd.concat(border_dfs, ignore_index=True)
        logging.info(f"Loaded {len(border_info)} border info records from R2")
    else:
        border_info = pd.DataFrame(columns=['event_id', 'border', 'aggregated_at', 'score'])
        logging.warning("No border info files found in R2")
    
    merged, event_info = _process_border_and_event_data(
        border_info, event_info, event_id_range, max_look_back_year,
        exclude_event_types, exclude_event_ids
    )
    
    logging.info(f"Loaded {len(event_info)} events from R2")
    return merged, event_info

def clear_anniversary_cache(local_cache_dir: str = './data_cache') -> None:
    """Clear the local cache for anniversary data"""
    border_info_cache_path = os.path.join(local_cache_dir, 'anniversary_border_info.pkl')
    event_info_cache_path = os.path.join(local_cache_dir, 'anniversary_event_info.pkl')
    
    files_removed = 0
    for cache_path in [border_info_cache_path, event_info_cache_path]:
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                files_removed += 1
                logging.info(f"Removed cache file: {cache_path}")
            except Exception as e:
                logging.error(f"Failed to remove cache file {cache_path}: {e}", exc_info=True)
    
    if files_removed > 0:
        logging.info(f"Cleared {files_removed} cache files from {local_cache_dir}")
    else:
        logging.info("No cache files to clear")

def load_anniversary_data_from_r2(
    r2_client: R2Client,
    idol_ids: List[int] = list(range(1, 53)),
    borders: List[int] = [100, 1000],
    use_local_cache: bool = False,
    local_cache_dir: str = './data_cache'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Define cache file paths
    border_info_cache_path = os.path.join(local_cache_dir, 'anniversary_border_info.pkl')
    event_info_cache_path = os.path.join(local_cache_dir, 'anniversary_event_info.pkl')
    
    # Try to load from cache if enabled
    if use_local_cache and os.path.exists(border_info_cache_path) and os.path.exists(event_info_cache_path):
        try:
            logging.info("Loading anniversary data from local cache...")
            border_info = pd.read_pickle(border_info_cache_path)
            event_info = pd.read_pickle(event_info_cache_path)
            logging.info(f"Successfully loaded {len(border_info)} border records and {len(event_info)} event records from cache")
            return border_info, event_info
        except Exception as e:
            logging.warning(f"Failed to load from cache, falling back to R2: {e}", exc_info=True)
    
    # Load from R2 (original logic)
    logging.info("Loading anniversary data from R2...")
    event_info_key = 'event_info/event_info_all.csv'
    event_obj = r2_client.get_object(BUCKET_NAME, event_info_key)
    event_info = pd.read_csv(StringIO(event_obj['Body'].read().decode('utf-8')))
    event_ids = event_info[event_info['event_type'] == 5]['event_id'].unique()

    event_info['boost_at'] = pd.to_datetime(event_info['boost_at'], errors='coerce')
    latest_valid_date = pd.Timestamp('2010-01-01', tz='Asia/Tokyo')
    valid_mask = (~event_info['boost_at'].isna()) & (event_info['boost_at'] > latest_valid_date)
    valid_event_info = event_info[valid_mask]
    
    valid_event_ids = set(valid_event_info['event_id'].unique())
    filtered_event_ids = [eid for eid in event_ids if eid in valid_event_ids and eid >= 192] # exclude early anniversary data
    
    logging.info(f"Loading data from R2 for {len(filtered_event_ids)} events (filtered from {len(event_ids)} requested)")
    
    border_dfs = []
    
    for event_id in filtered_event_ids:
        for idol_id in idol_ids:
            for border in borders:
                filename = f'border_info_{event_id}_{idol_id}_{border}.csv'
                file_key = f'border_info/{filename}'
                
                try:
                    obj = r2_client.get_object(BUCKET_NAME, file_key)
                    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    df['event_id'] = event_id
                    df['idol_id'] = idol_id
                    df['border'] = border
                    border_dfs.append(df)
                except r2_client.client.exceptions.NoSuchKey:
                    continue
                except Exception as e:
                    logging.warning(f"Could not load {filename} from R2: {e}", exc_info=True)
    
    if border_dfs:
        border_info = pd.concat(border_dfs, ignore_index=True)
        logging.info(f"Loaded {len(border_info)} border info records from R2")
    else:
        border_info = pd.DataFrame(columns=['event_id', 'idol_id', 'border', 'aggregated_at', 'score'])
        logging.warning("No border info files found in R2")
    
    border_info, valid_event_info = _process_anniversary_data(
        border_info, valid_event_info, filtered_event_ids, idol_ids, borders
    )
    
    # Save to cache if enabled
    if use_local_cache:
        try:
            os.makedirs(local_cache_dir, exist_ok=True)
            border_info.to_pickle(border_info_cache_path)
            valid_event_info.to_pickle(event_info_cache_path)
            logging.info(f"Saved anniversary data to local cache at {local_cache_dir}")
        except Exception as e:
            logging.warning(f"Failed to save to cache: {e}", exc_info=True)
    
    return border_info, valid_event_info

def load_latest_event_metadata_from_r2(r2_client: R2Client) -> Dict[str, Any]:
    metadata_key = 'metadata/latest_event_border_info.json'
    
    logging.info(f"Loading latest event metadata from R2: {BUCKET_NAME}/{metadata_key}")
    
    try:
        obj = r2_client.get_object(BUCKET_NAME, metadata_key)
        metadata = json.loads(obj['Body'].read().decode('utf-8'))
        logging.info(f"Successfully loaded metadata for event {metadata.get('EventId', 'Unknown')}: {metadata.get('EventName', 'Unknown')}")
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata from R2: {e}", exc_info=True)
        raise

def load_all_data(r2_client: R2Client, metadata: Any, use_local_cache: bool = False) -> Tuple[pd.DataFrame, Dict[int, str]]:
    logging.info("Loading data from R2...")

    target = 'anniversary' if metadata['EventType'] == 5 else 'normal'
    if target == 'normal':
        b_info, e_info = load_data_from_r2(r2_client, metadata['EventId'])
    else:
        b_info, e_info = load_anniversary_data_from_r2(r2_client, use_local_cache=use_local_cache)

    # Extract event_id to name mapping
    event_name_map = dict(zip(e_info['event_id'], e_info['name']))
    logging.info(f"Created event name mapping for {len(event_name_map)} events")

    # Combine and process data
    df = combine_info(b_info, e_info)
    df = interpolate(df, metadata['EventId'])    
    df = process_data(df)

    return df, event_name_map
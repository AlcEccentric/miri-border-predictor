import datetime
import logging
from typing import Any, Dict, List, Tuple
import pandas as pd
from dateutil import parser
import pytz
import numpy as np

# Import logger configuration first
import logger_config

from loader import load_all_data, load_latest_event_metadata_from_r2, save_predictions_to_local_debug, upload_predictions_to_r2
from data_processing import purge
from predict import get_predictions
from r2_client import R2Client
from normalization import do_normalize
from feature_engineering import add_additional_features
from smoothing import generate_smoothed_dfs

def should_skip_prediction(metadata: Dict[str, Any]) -> bool:
    # Parse event start and end times (they are in JST)
    jst = pytz.timezone('Asia/Tokyo')
    event_start = parser.isoparse(metadata['StartAt'])
    event_end = parser.isoparse(metadata['EndAt'])

    current_time = datetime.datetime.now(jst)

    if current_time >= event_end:
        logging.info(f"Event has ended at {event_end}. Current time: {current_time}. Skipping prediction.")
        return True

    last_hour_start = event_end - datetime.timedelta(hours=2.5)
    if current_time >= last_hour_start:
        logging.info(f"Current time {current_time} is within last hour of event (starting at {last_hour_start}). Skipping prediction.")
        return True

    event_duration = event_end - event_start
    first_tenth_end = event_start + (event_duration / 10)
    
    if current_time <= first_tenth_end:
        logging.info(f"Current time {current_time} is within first 1/10 of event (until {first_tenth_end}). Skipping prediction.")
        return True
    
    logging.info(f"Event running from {event_start} to {event_end}. Current time: {current_time}. Proceeding with prediction.")
    return False

def calculate_standard_event_length(df):
    """Calculate standard event length based on most frequent group length"""
    group_lengths = df.groupby(['event_id', 'idol_id', 'border']).size()
    length_counts = group_lengths.value_counts().sort_values(ascending=False)
    
    # Get the most frequent length safely
    most_frequent_item = length_counts.index[0]
    if isinstance(most_frequent_item, tuple):
        standard_event_length = int(most_frequent_item[0])
    else:
        standard_event_length = int(most_frequent_item)
    
    logging.info(f"Standard event length (most frequent): {standard_event_length}")
    logging.debug(f"Length distribution: {dict(length_counts.head())}")
    
    return standard_event_length

def prepare_data_pipeline(df: pd.DataFrame, metadata: Dict[str, Any], norm_event_length: int, current_step: int, standard_event_length: int, eid_to_len_boost_ratio: Dict) -> Dict[str, pd.DataFrame]:
    """Prepare the complete data pipeline including normalization, feature engineering, and smoothing"""
    logging.info("Starting data preparation pipeline...")

    # Normalize full event data
    logging.info("Normalizing full event data...")
    norm_df = do_normalize(df[df['event_id'] != metadata['EventId']], norm_event_length, norm_event_length, standard_event_length, eid_to_len_boost_ratio)
    
    # Normalize partial event data till current step
    logging.info("Normalizing partial event data...")
    pdf = purge(df, current_step, norm_event_length, eid_to_len_boost_ratio)
    n_pdf = do_normalize(pdf, norm_event_length, current_step, standard_event_length, eid_to_len_boost_ratio)

    # Feature engineering
    logging.info("Adding features...")
    df['time_idx'] = (df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount()) 
    feature_df = add_additional_features(df)
    feature_df['time_idx'] = (feature_df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())
    
    norm_df['time_idx'] = (norm_df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount()) 
    norm_feature_df = add_additional_features(norm_df)
    norm_feature_df['time_idx'] = (norm_feature_df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())
    
    n_pdf['time_idx'] = (n_pdf.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount()) 
    nf_pdf = add_additional_features(n_pdf)
    nf_pdf['time_idx'] = (nf_pdf.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())

    for (event_id, idol_id) in nf_pdf[['event_id', 'idol_id']].drop_duplicates().values:
        part_scores = nf_pdf[(nf_pdf['event_id'] == event_id) & (nf_pdf['idol_id'] == idol_id)]['score']
        full_scores = norm_feature_df[(norm_feature_df['event_id'] == event_id) & (norm_feature_df['idol_id'] == idol_id)]['score']
        if len(part_scores) > 0 and len(full_scores) > current_step-1:
            part_last = part_scores.iloc[-1]
            full_at_step = full_scores.iloc[current_step-1]
            if not np.isclose(part_last, full_at_step):
                print(f"DEBUG: Mismatch for event {event_id}, idol {idol_id}, step {current_step}: part_last={part_last}, full_at_step={full_at_step}")
                break

    # Smoothing
    logging.info("Generating smoothed data...")
    smoo_feature_df, snf_pdf = generate_smoothed_dfs(norm_feature_df, nf_pdf)

    # Prepare data dictionary
    new_data = {
        'raw': feature_df,
        'norm_all': norm_feature_df,
        'smoo_all': smoo_feature_df,
        'norm_part': nf_pdf,
        'smooth_part': snf_pdf
    }
    
    logging.info("Data preparation pipeline completed")
    return new_data

def run_predictions(new_data: Dict[str, pd.DataFrame], metadata: Dict[str, Any], borders: list, idol_ids: list, current_step: int, norm_event_length: int, standard_event_length: int, eid_to_len_boost_ratio: Dict, event_name_map: Dict) -> Dict:
    """Run the prediction pipeline"""
    logging.info("Starting predictions...")

    sub_event_types = get_sub_event_types(
        event_id=metadata['EventId'],
        internal_event_type=metadata['InternalEventType'],
        event_type=metadata['EventType']
    )

    logging.info(f"Sub event types: {sub_event_types}")

    results = get_predictions(
        data=new_data,
        event_id=metadata['EventId'],
        event_type=metadata['EventType'],
        sub_types=sub_event_types,
        idol_ids=idol_ids,
        borders=borders,
        step=current_step,
        event_length=norm_event_length,
        norm_event_length=norm_event_length,
        standard_event_length=standard_event_length,
        eid_to_len_boost_ratio=eid_to_len_boost_ratio,
        event_name_map=event_name_map,
    )
    
    logging.info("Predictions completed")
    return results

def check_event_length(target, df):
    group_lengths = df.groupby(['event_id', 'idol_id', 'border']).size()
    unique_lengths = group_lengths.unique()
    logging.info(f"Unique group lengths after interpolation: {sorted(unique_lengths)}")
    
    if target == 'anniversary' and len(unique_lengths) > 1:
        logging.warning(f"Inconsistent group lengths found:")
        length_counts = group_lengths.value_counts().sort_index()
        for length, count in length_counts.items():
            logging.warning(f"  Length {length}: {count} groups")
        raise ValueError(f"Anniversary data has inconsistent group lengths: {sorted(unique_lengths)}")

def get_current_step(norm_event_length: int, metadata: Dict[str, Any], df: pd.DataFrame) -> int:
    """Calculate current step based on event progress"""
    logging.debug(f"Current event id: {metadata['EventId']}")
    logging.debug(f"Ids in data: {df['event_id'].unique()}")
    current_data = df[(df['event_id'] == metadata['EventId'])]
    logging.debug(f"Norm event length: {norm_event_length}, current data shape: {current_data.shape}")
    
    event_start_time = parser.isoparse(metadata['StartAt'])
    event_end_time = parser.isoparse(metadata['EndAt'])
    current_time = current_data['aggregated_at'].max()
    logging.debug(f"Latest aggregated at: {current_data['aggregated_at'].max()}, event start time: {event_start_time}, event end time: {event_end_time}, current time: {current_time}")
    current_step = int((current_time - event_start_time).total_seconds() * norm_event_length / (event_end_time - event_start_time).total_seconds())
    logging.debug(f"Current step: {current_step}")
    return current_step

def main():
    logging.info("Starting border prediction pipeline...")
    
    r2_client = R2Client()
    metadata = load_latest_event_metadata_from_r2(r2_client=r2_client)
    testing = False

    # For testing
    if testing:
        metadata = {
            'EventId': 378,
            'EventType': metadata['EventType'],
            'InternalEventType': metadata['InternalEventType'],
            'EventName': metadata['EventName'],
            'StartAt': "2025-04-18T15:00:00+09:00",
            'EndAt': "2025-04-25T20:59:59+09:00",
        }
    
    if not testing and should_skip_prediction(metadata) :
        logging.info("Prediction skipped due to timing constraints.")
        return
    
    target = 'anniversary' if metadata['EventType'] == 5 else 'normal'
    logging.info(f"Event type: {target}")

    # Parameters
    norm_event_length = 300
    borders = [100.0, 1000.0] if target == 'anniversary' else [100.0, 2500.0]
    idol_ids = list(range(1, 53)) if target == 'anniversary' else [0]
    logging.info(f"Using borders: {borders}")
    min_event_id = get_min_event_id(metadata['InternalEventType'])
    logging.info(f"Using min event id: {min_event_id}")

    # Load data and calculate parameters
    df, eid_to_len_boost_ratio, event_name_map = load_all_data(r2_client,
                                                               metadata,
                                                               target,
                                                               idol_ids,
                                                               min_event_id,
                                                               use_local_cache=True if testing else False)
    standard_event_length = calculate_standard_event_length(df)
    
    current_step = get_current_step(norm_event_length, metadata, df)
    # For testing
    if testing:
        current_step = 270
    logging.info(f"Current step: {current_step}/{norm_event_length}")

    new_data = prepare_data_pipeline(df, metadata, norm_event_length, current_step, standard_event_length, eid_to_len_boost_ratio)

    results = run_predictions(new_data, metadata, borders, idol_ids, current_step, norm_event_length, standard_event_length, eid_to_len_boost_ratio, event_name_map)

    if results:
        if testing:
            save_predictions_to_local_debug(r2_client, results, metadata['EventId'])
        else:
            upload_predictions_to_r2(r2_client, results, metadata['EventId'])
        logging.info(f"Prediction completed and uploaded for event ID: {metadata['EventId']}")
    else:
        logging.warning("No results to upload")

    return results

def get_min_event_id(internal_event_type: int) -> int:
    if internal_event_type in [22, 23]: # TourBingo & TourSpecial
        return 250
    return 200

def get_sub_event_types(event_id: int, internal_event_type: int, event_type: float) -> Tuple[float, ...]:
    """
    Determine sub_event_type based on event_id, internal_event_type, and event_type.
    Returns the sub_event_type that would be assigned by the feature engineering functions.
    """
    if event_type == 3.0:
        # For event type 3, check if it has bonus
        boosted_internal_event_types = range(16, 21)  # 16 (Tiara), 17-20 (Trust...)
        if internal_event_type in boosted_internal_event_types:
            return (2.0,)  # has_bonus = True
        else:
            return (1.0,)  # has_bonus = False
    
    elif event_type == 4.0:
        # For event type 4, check bonus conditions
        has_bonus = (
            (internal_event_type == 22) or
            ((internal_event_type == 23) and (event_id > 300))
        )
        if has_bonus:
            return (2.0,)
        else:
            return (1.0,)
    
    elif event_type == 11.0:
        # For event type 11, all events have sub_event_type = 1.0
        return (1.0,)
    
    elif event_type == 13.0:
        # For event type 13, all events have sub_event_type = 1.0
        return (1.0,)
    
    elif event_type == 5.0:
        # For event type 5, all events have sub_event_type = 1.0
        return (1.0,)
    
    else:
        # Default case (not explicitly handled in feature engineering)
        return (1.0,)

if __name__ == "__main__":
    main()
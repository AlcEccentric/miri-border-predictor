import datetime
import logging
from typing import Any, Dict, Tuple
import pandas as pd
from dateutil import parser
import pytz

# Import logger configuration first
import logger_config

from loader import load_all_data, load_latest_event_metadata_from_r2, upload_predictions_to_r2
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

    last_hour_start = event_end - datetime.timedelta(hours=1)
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
    
    logging.debug(f"Standard event length (most frequent): {standard_event_length}")
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
    feature_df = add_additional_features(df)
    feature_df['time_idx'] = (feature_df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())
    norm_feature_df = add_additional_features(norm_df)
    norm_feature_df['time_idx'] = (norm_feature_df.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())
    nf_pdf = add_additional_features(n_pdf)
    nf_pdf['time_idx'] = (nf_pdf.sort_values('aggregated_at').groupby(['border', 'event_id', 'idol_id']).cumcount())

    # debug
    logging.debug(f"Idol 44 max score in df: {df[(df['event_id'] == metadata['EventId']) & (df['idol_id'] == 44)]['score'].max()}")
    logging.debug(f"Idol 44 max score in pdf: {pdf[(pdf['event_id'] == metadata['EventId']) & (pdf['idol_id'] == 44)]['score'].max()}")
    logging.debug(f"Idol 44 max score in n_pdf: {n_pdf[(n_pdf['event_id'] == metadata['EventId']) & (n_pdf['idol_id'] == 44)]['score'].max()}")

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
    
    internal_event_type_to_sub_types = {
        5.0: (1.0,)
    }

    results = get_predictions(
        data=new_data,
        event_id=metadata['EventId'],
        event_type=metadata['EventType'],
        sub_types=internal_event_type_to_sub_types[metadata['InternalEventType']],
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

def get_len_and_boost_ratio(df: pd.DataFrame, current_event_id: int) -> Dict[Tuple[int, int], Dict[str, Any]]:
    eid_to_len_boost_ratio = {}
    
    # Get current event type info
    current_event_data = df[df['event_id'] == current_event_id]
    current_event_type = current_event_data['event_type'].iloc[0] if len(current_event_data) > 0 else None
    current_internal_event_type = current_event_data['internal_event_type'].iloc[0] if len(current_event_data) > 0 else None
    
    # Find template event (most recent similar event with same event type and internal event type)
    template_params = None
    if current_event_type is not None:
        similar_events = df[(df['event_id'] != current_event_id) & (df['event_type'] == current_event_type) & (df['internal_event_type'] == current_internal_event_type)]
        if len(similar_events) > 0:
            template_event_id = similar_events['event_id'].max()
            template_idol_id = similar_events[similar_events['event_id'] == template_event_id]['idol_id'].max()
            template_data = similar_events[(similar_events['event_id'] == template_event_id) & (similar_events['idol_id'] == template_idol_id) & (similar_events['border'] == 100.0)]
            if len(template_data) > 0:
                boost_start = template_data.reset_index().index[template_data['is_boosted'] == True][0] if True in template_data['is_boosted'].values else None
                template_params = {
                    'boost_ratio': boost_start/len(template_data) if boost_start else None,
                    'length': len(template_data),
                    'boost_start': boost_start,
                }
    
    # Process all events
    for event_id in df['event_id'].unique():
        for idol_id in df['idol_id'].unique():
            if event_id == current_event_id:
                # Use template params for current event
                eid_to_len_boost_ratio[(event_id, idol_id)] = template_params or {'boost_ratio': None, 'length': None, 'boost_start': None}
            else:
                # Calculate params for historical events
                edf = df[(df['event_id'] == event_id) & (df['idol_id'] == idol_id) & (df['border'] == 100.0)]
                if len(edf) > 0:
                    boost_start = edf.reset_index().index[edf['is_boosted'] == True][0] if True in edf['is_boosted'].values else None
                    eid_to_len_boost_ratio[(event_id, idol_id)] = {
                        'boost_ratio': boost_start/len(edf) if boost_start else None,
                        'length': len(edf),
                        'boost_start': boost_start,
                    }
    return eid_to_len_boost_ratio

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
    current_data = df[(df['event_id'] == metadata['EventId'])]
    
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
    
    if should_skip_prediction(metadata):
        logging.info("Prediction skipped due to timing constraints.")
        return
    
    target = 'anniversary' if metadata['EventType'] == 5 else 'normal'
    logging.info(f"Event type: {target}")

    # Parameters
    norm_event_length = 300
    borders = [100.0, 1000.0] if target == 'anniversary' else [100.0, 2500.0]
    idol_ids = list(range(1, 53)) if target == 'anniversary' else [0]
    logging.info(f"Using borders: {borders}")

    # Load data and calculate parameters
    df, event_name_map = load_all_data(r2_client, metadata, use_local_cache=False)
    
    eid_to_len_boost_ratio = get_len_and_boost_ratio(df, metadata['EventId'])
    standard_event_length = calculate_standard_event_length(df)
    
    current_step = get_current_step(norm_event_length, metadata, df)
    logging.info(f"Current step: {current_step}/{norm_event_length}")

    new_data = prepare_data_pipeline(df, metadata, norm_event_length, current_step, standard_event_length, eid_to_len_boost_ratio)

    results = run_predictions(new_data, metadata, borders, idol_ids, current_step, norm_event_length, standard_event_length, eid_to_len_boost_ratio, event_name_map)

    if results:
        upload_predictions_to_r2(r2_client, results, metadata['EventId'])
        logging.info(f"Prediction completed and uploaded for event ID: {metadata['EventId']}")
    else:
        logging.warning("No results to upload")

    return results


if __name__ == "__main__":
    main()
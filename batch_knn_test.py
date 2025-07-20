#!/usr/bin/env python3
"""
Batch KNN Testing Script
Allows testing KNN predictions on multiple events and steps with flexible event selection.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import sys
from datetime import datetime

# Import your existing modules
from knn import predict_curve_knn
from data_processing import combine_info, process_data, purge
from normalization import do_normalize
from interpolation import interpolate
from smoothing import generate_smoothed_dfs
from feature_engineering import add_additional_features
from logger_config import setup_logging

class BatchKNNTester:
    def __init__(self, event_type: float, sub_event_types: List[float], border: float, min_event_id: int):
        self.event_type = event_type
        self.sub_event_types = sub_event_types
        self.border = border
        self.min_event_id = min_event_id
        self.results = []
        self.norm_event_length = 300

    def filter_event_info(self, e_info: pd.DataFrame) -> pd.DataFrame:
        """Filter event info based on event type, sub event types, and border."""
        return e_info[(e_info['event_type'] == self.event_type) & 
                      (e_info['event_id'] >= self.min_event_id)]

    def get_normalized_full_and_part_data(
        self,
        raw_data: pd.DataFrame,
        steps: List[int],
        standard_event_length: int,
        eid_to_len_boost_ratio: Dict,) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        # Create full normalized data (excluding current event)

        norm_df = do_normalize(
            raw_data, 
            self.norm_event_length, 
            self.norm_event_length, 
            standard_event_length, 
            eid_to_len_boost_ratio
        )
        norm_feature_df = add_additional_features(norm_df)
        norm_feature_df['time_idx'] = (norm_feature_df.sort_values('aggregated_at')
                                        .groupby(['border', 'event_id', 'idol_id']).cumcount())
        norm_feature_df = norm_feature_df[norm_feature_df['sub_event_type'].isin(self.sub_event_types)]
        logging.debug(f"Max score for {386.0} is {norm_feature_df[norm_feature_df['event_id'] == 386.0]['score'].max()}")
        
        step_to_norm_part_df = {}
        for step in steps:
            pdf = purge(raw_data, step, self.norm_event_length, eid_to_len_boost_ratio)
            n_pdf = do_normalize(pdf, self.norm_event_length, step, standard_event_length, eid_to_len_boost_ratio)
            nf_pdf = add_additional_features(n_pdf)
            nf_pdf['time_idx'] = (nf_pdf.sort_values('aggregated_at')
                                .groupby(['border', 'event_id', 'idol_id']).cumcount())
            nf_pdf = nf_pdf[nf_pdf['sub_event_type'].isin(self.sub_event_types)]
            step_to_norm_part_df[step] = nf_pdf
        
        for step, norm_part_df in step_to_norm_part_df.items():
            # For each event/idol, check the last value in norm_part_df matches the value at 'step-1' in norm_feature_df
            for (event_id, idol_id) in norm_part_df[['event_id', 'idol_id']].drop_duplicates().values:
                if event_id != 386.0:
                    continue
                part_scores = norm_part_df[(norm_part_df['event_id'] == event_id) & (norm_part_df['idol_id'] == idol_id)]['score']
                full_scores = norm_feature_df[(norm_feature_df['event_id'] == event_id) & (norm_feature_df['idol_id'] == idol_id)]['score']
                if len(part_scores) > 0 and len(full_scores) > step-1:
                    part_last = part_scores.iloc[-1]
                    full_at_step = full_scores.iloc[step-1]
                    if not np.isclose(part_last, full_at_step):
                        print(f"DEBUG: Mismatch for event {event_id}, idol {idol_id}, step {step}: part_last={part_last}, full_at_step={full_at_step}")
        return norm_feature_df, step_to_norm_part_df
        
    def load_and_process_data(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """Load and process data from local test_data directory."""
        logging.info("Loading data from local test_data directory...")
        
        test_data_dir = Path("test_data")
        if not test_data_dir.exists():
            raise FileNotFoundError("test_data directory not found. Please ensure it exists with the same structure as R2 bucket.")
        
        # Load event info
        event_info_path = test_data_dir / "event_info" / "event_info_all.csv"
        if not event_info_path.exists():
            raise FileNotFoundError(f"Event info file not found at {event_info_path}")
        
        e_info = pd.read_csv(event_info_path)
        e_info = self.filter_event_info(e_info)
        
        logging.info(f"Loaded event info with {len(e_info)} events")
        
        # Load border info files
        border_info_dir = test_data_dir / "border_info"
        if not border_info_dir.exists():
            raise FileNotFoundError(f"Border info directory not found at {border_info_dir}")
        
        # Find all border info CSV files
        border_files = []
        iids = [0] if self.event_type != 5 else range(1, 53)
        for eid in e_info['event_id'].unique():
            for iid in iids:
                file_path = border_info_dir / f"border_info_{eid}_{iid}_{int(self.border)}.csv"
                if file_path.exists():
                    border_files.append(file_path)
                else:
                    logging.warning(f"Border info file not found for event {eid}")
        if not border_files:
            raise FileNotFoundError(f"No border info files found in {border_info_dir}")
        
        logging.info(f"Found {len(border_files)} border info files")
        
        # Load and combine all border info files
        border_dataframes = []
        for file_path in border_files:
            try:
                df = pd.read_csv(file_path)
                border_dataframes.append(df)
                logging.debug(f"Loaded {file_path} with {len(df)} rows")
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not border_dataframes:
            raise ValueError("No border info files could be loaded")
        
        # Combine all border info
        b_info = pd.concat(border_dataframes, ignore_index=True)
        logging.info(f"Combined border info with {len(b_info)} total rows")
        
        # Ensure datetime conversion for aggregated_at column
        if 'aggregated_at' in b_info.columns:
            b_info['aggregated_at'] = pd.to_datetime(b_info['aggregated_at'])
        
        # Ensure datetime conversion for event info columns
        datetime_columns = ['start_at', 'end_at', 'boost_at']
        for col in datetime_columns:
            if col in e_info.columns:
                e_info[col] = pd.to_datetime(e_info[col])
        e_info['end_at'] = e_info['end_at'] + pd.Timedelta(seconds=1) # round data to o'clock time
        
        # Extract event_id to name mapping
        event_name_map = dict(zip(e_info['event_id'], e_info['name']))
        logging.info(f"Created event name mapping for {len(event_name_map)} events")
        
        # Combine and process data
        df = combine_info(b_info, e_info)
        df = interpolate(df, None) # type: ignore
        df = process_data(df)
        
        return df, event_name_map
        
    def find_matching_events(self, raw_data: pd.DataFrame, 
                           test_event_ids: Optional[List[float]] = None,
                           recent_count: Optional[int] = None) -> List[float]:
        matching_events = raw_data['event_id'].unique()
        if test_event_ids is not None:
            available_events = [eid for eid in test_event_ids if eid in matching_events]
            if not available_events:
                raise ValueError(f"None of the specified event IDs {test_event_ids} match the criteria")
            return available_events
        elif recent_count is not None:
            sorted_events = sorted(matching_events, reverse=True)
            return sorted_events[:recent_count]
        else:
            return list(raw_data['event_id'].unique())
    
    def calculate_standard_event_length(self, df: pd.DataFrame) -> int:
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
        return standard_event_length
    
    def get_len_and_boost_ratio(self, df: pd.DataFrame) -> Dict:
        """Calculate event length and boost ratio mappings"""
        eid_to_len_boost_ratio = {}
        
        for event_id in df['event_id'].unique():
            for idol_id in df['idol_id'].unique():
                # Calculate params for historical events
                edf = df[(df['event_id'] == event_id) & (df['idol_id'] == idol_id) & (df['border'] == self.border)] # it's fine to hard code border since different borders have same length
                if len(edf) > 0:
                    boost_start = edf.reset_index().index[edf['is_boosted'] == True][0] if True in edf['is_boosted'].values else None
                    eid_to_len_boost_ratio[(event_id, idol_id)] = {
                        'boost_ratio': boost_start/len(edf) if boost_start else None,
                        'length': len(edf),
                        'boost_start': boost_start,
                    }
                    logging.debug(f"Calculated len and boost ratio for event {event_id}, idol {idol_id}: {eid_to_len_boost_ratio[(event_id, idol_id)]}")
        
        return eid_to_len_boost_ratio
    
    def get_event_final_scores(self, norm_data: pd.DataFrame, event_ids: List[float]) -> dict:
        """Get final scores for each event/idol combination."""
        final_scores = {}
        
        for event_id in event_ids:
            event_data = norm_data[norm_data['event_id'] == event_id]
            event_final_scores = {}
            
            for idol_id in event_data['idol_id'].unique():
                idol_data = event_data[event_data['idol_id'] == idol_id]
                if len(idol_data) > 0:
                    final_score = idol_data['score'].iloc[-1]  # Last score
                    event_final_scores[idol_id] = final_score
            
            final_scores[event_id] = event_final_scores
        
        return final_scores
    
    def run_predictions(self, 
                       test_event_ids: List[float],
                       test_steps: List[int],
                       raw_data: pd.DataFrame,
                       norm_df: pd.DataFrame,
                       nf_pdf_by_step: Dict[int, pd.DataFrame],
                       standard_event_length: int,
                       eid_to_len_boost_ratio: Dict) -> List[dict]:
        results = []
        total_combinations = len(test_event_ids) * len(test_steps)
        current_combination = 0
        
        for event_id in test_event_ids:
            logging.info(f"Processing event {event_id}")
            
            # Get all idols for this event
            event_idols = norm_df[norm_df['event_id'] == event_id]['idol_id'].unique()
            
            for step in test_steps:
                current_combination += 1
                logging.info(f"Progress: {current_combination}/{total_combinations} - Event {event_id}, Step {step}")
                nf_pdf = nf_pdf_by_step[step]
                try:
                    # Generate smoothed data
                    smoo_df, snf_pdf = generate_smoothed_dfs(norm_df, nf_pdf)
                    
                    # Get final scores from the original data
                    final_scores = {}
                    event_data = norm_df[norm_df['event_id'] == event_id]
                    for idol_id in event_idols:
                        idol_data = event_data[
                            (event_data['idol_id'] == idol_id) & 
                            (event_data['border'] == self.border)
                        ]
                        if len(idol_data) > 0:
                            final_scores[idol_id] = idol_data['score'].iloc[-1]
                    
                    # Run predictions for each idol
                    logging.debug(f"final_scores: {final_scores}")
                    for idol_id in event_idols:
                        if idol_id not in final_scores:
                            continue
                            
                        try:
                            # Check if we have enough data
                            current_trajectory = nf_pdf[
                                (nf_pdf['event_id'] == event_id) & 
                                (nf_pdf['idol_id'] == idol_id) &
                                (nf_pdf['border'] == self.border)
                            ]['score'].values
                            
                            if len(current_trajectory) < step:
                                continue
                            
                            # Run KNN prediction
                            # Convert sub_event_types to tuple of correct format (single element tuple)
                            sub_types_tuple = tuple(self.sub_event_types)
                            
                            prediction, similar_ids, distances = predict_curve_knn(
                                event_id=event_id,
                                idol_id=idol_id,
                                border=self.border,
                                sub_types=sub_types_tuple,
                                current_step=step,
                                norm_data=norm_df,
                                norm_partial_data=nf_pdf,
                                smooth_partial_data=snf_pdf,
                                smooth_full_data=smoo_df,
                            )
                            
                            # Extract prediction final score
                            actual_final = final_scores[idol_id]
                            
                            if len(prediction) > 0:
                                predicted_final = prediction[-1]
                            else:
                                predicted_final = np.nan
                            print(predicted_final)
                            
                            # Store result
                            result = {
                                'event_id': event_id,
                                'idol_id': idol_id,
                                'border': self.border,
                                'step': step,
                                'prediction': predicted_final,
                                'actual': actual_final,
                                'relative_error': ((predicted_final - actual_final) / actual_final * 100) if actual_final != 0 else np.nan,
                                'absolute_error': abs(predicted_final - actual_final),
                                'num_neighbors': len(similar_ids) if similar_ids is not None else 0,
                                'avg_distance': np.mean(distances) if distances is not None and len(distances) > 0 else np.nan
                            }
                            
                            results.append(result)
                            
                        except Exception as e:
                            logging.error(f"Error predicting event {event_id}, idol {idol_id}, step {step}: {str(e)}")
                            raise e
                
                except Exception as e:
                    logging.error(f"Error processing event {event_id}, step {step}: {str(e)}")
                    raise e
        
        return results
    
    def save_results(self, results: List[dict], output_file: str):
        """Save results to CSV file."""
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        # Print summary statistics
        if len(df) > 0:
            print(f"\n=== BATCH KNN TEST RESULTS ===")
            print(f"Total predictions: {len(df)}")
            print(f"Unique events: {df['event_id'].nunique()}")
            print(f"Unique idols: {df['idol_id'].nunique()}")
            print(f"Steps tested: {sorted(df['step'].unique())}")
            print(f"Border: {self.border}")
            print(f"Event type: {self.event_type}")
            print(f"Sub event types: {self.sub_event_types}")
            print(f"\nError Statistics:")
            print(f"Mean absolute error: {df['absolute_error'].mean():.2f}")
            print(f"Mean relative error: {df['relative_error'].mean():.2f}%")
            print(f"Median relative error: {df['relative_error'].median():.2f}%")
            print(f"RMSE: {np.sqrt((df['absolute_error'] ** 2).mean()):.2f}")
            print(f"Results saved to: {output_file}")
        else:
            print("No results generated!")
    
    def save_md_summary(self, results: List[dict], output_md: str):
        """Save step-wise error summary to a Markdown file, including event IDs with error > 10% and their errors."""
        df = pd.DataFrame(results)
        if len(df) == 0:
            with open(output_md, "w") as f:
                f.write("# KNN Batch Test Summary\n\nNo results generated!\n")
            print(f"Markdown summary saved to: {output_md}")
            return

        md_lines = [
            "# KNN Batch Test Summary\n",
            "| Step | Avg Abs Rel Error (%) | Median Abs Rel Error (%) | Rel Error Range (%) | Coverage 5% | Coverage 10% | Event IDs >10% Error |",
            "|------|----------------------|-------------------------|--------------------|-------------|--------------|----------------------|"
        ]
        for step in sorted(df['step'].unique()):
            step_df = df[df['step'] == step]
            abs_rel_err = step_df['relative_error'].abs()
            avg_abs_rel = abs_rel_err.mean()
            median_abs_rel = abs_rel_err.median()
            rel_err_min = abs_rel_err.min()
            rel_err_max = abs_rel_err.max()
            coverage_5 = (abs_rel_err <= 5).mean() * 100
            coverage_10 = (abs_rel_err <= 10).mean() * 100
            # Get event IDs with error > 10% and their errors
            event_ids_out = step_df.loc[abs_rel_err > 10, ['event_id', 'relative_error']]
            if not event_ids_out.empty:
                event_ids_str = ", ".join(
                    f"{int(row['event_id'])}({row['relative_error']:.2f}%)"
                    for _, row in event_ids_out.iterrows()
                )
            else:
                event_ids_str = "-"
            md_lines.append(
                f"| {step} | {avg_abs_rel:.2f} | {median_abs_rel:.2f} | [{rel_err_min:.2f}, {rel_err_max:.2f}] | {coverage_5:.1f}% | {coverage_10:.1f}% | {event_ids_str} |"
            )
        with open(output_md, "w") as f:
            f.write("\n".join(md_lines))
        print(f"Markdown summary saved to: {output_md}")

    def plot_all_trajectories(self, norm_df: pd.DataFrame, output_dir: str = "debug"):
        """Plot idol 0's score trajectories for all events in norm_df and save to debug folder."""
        import matplotlib.pyplot as plt
        import os
        plt.set_loglevel("warning")
        norm_df['time_idx'] = (norm_df.sort_values('aggregated_at')
                                .groupby(['border', 'event_id', 'idol_id']).cumcount()) 
        # Create debug directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter data for the current border and idol 0 only
        border_data = norm_df[(norm_df['border'] == self.border) & (norm_df['idol_id'] == 0)].copy()
        
        if len(border_data) == 0:
            logging.warning(f"No data found for border {self.border} and idol 0")
            return
        
        # Create a simple single plot
        plt.figure(figsize=(12, 12))
        
        # Generate colors for different events
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(border_data['event_id'].unique())))
        
        # Plot trajectory for each event
        for i, event_id in enumerate(sorted(border_data['event_id'].unique())):
            event_data = border_data[border_data['event_id'] == event_id].sort_values('time_idx')
            
            if len(event_data) > 0:
                plt.plot(event_data['time_idx'], event_data['score'], 
                        color=colors[i], linewidth=2, marker='o', markersize=3,
                        label=f'Event {int(event_id)}')
        
        plt.title(f'Idol 0 Score Trajectories - Border {int(self.border)}', fontsize=14, fontweight='bold')
        plt.xlabel('Time Index (Normalized Steps)', fontsize=12)
        plt.ylabel('Normalized Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'idol0_trajectories_border_{int(self.border)}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Idol 0 trajectories plot saved to {output_path}")


def main():
    # Hardcoded configuration - modify these values as needed
    CONFIG = {
        'event_type': 3.0,
        'sub_event_types': [2.0],
        'border': 100.0,
        'steps': [270],
        'test_event_ids': [329, 341],  # Set to [388, 390, 392] for specific events, or None
        'recent_count': None,  # Set to None if using test_event_ids or testing all eventsu
        'output': 'batch_knn_results.csv',
        'log_level': 'DEBUG'
    }
    
    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, CONFIG['log_level']))  # Force DEBUG level for troubleshooting
    
    # Validate configuration
    if CONFIG['test_event_ids'] is not None and CONFIG['recent_count'] is not None:
        raise ValueError("Cannot specify both test_event_ids and recent_count")
    
    # Create tester instance
    tester = BatchKNNTester(
        event_type=CONFIG['event_type'],
        sub_event_types=CONFIG['sub_event_types'],
        border=CONFIG['border'],
        min_event_id=100,
    )
    
    try:
        # Load and process data
        logging.info("Loading and processing data...")
        raw_data, event_name_map = tester.load_and_process_data()
        
        # Calculate standard event length and boost ratios
        standard_event_length = tester.calculate_standard_event_length(raw_data)
        eid_to_len_boost_ratio = tester.get_len_and_boost_ratio(raw_data)
        
        # Find matching events
        test_event_ids = tester.find_matching_events(
            raw_data=raw_data,
            test_event_ids=CONFIG['test_event_ids'],
            recent_count=CONFIG['recent_count']
        )
        
        logging.info(f"Found {len(test_event_ids)} events to test: {test_event_ids}")

        norm_df, nf_pdf_by_step = tester.get_normalized_full_and_part_data(
            raw_data=raw_data,
            steps=CONFIG['steps'],
            standard_event_length=standard_event_length,
            eid_to_len_boost_ratio=eid_to_len_boost_ratio
        )
        
        # Run predictions
        all_results = tester.run_predictions(
            test_event_ids=test_event_ids,
            test_steps=CONFIG['steps'],
            raw_data=raw_data,
            norm_df=norm_df,
            nf_pdf_by_step=nf_pdf_by_step,
            standard_event_length=standard_event_length,
            eid_to_len_boost_ratio=eid_to_len_boost_ratio
        )
        tester.plot_all_trajectories(raw_data)
        tester.plot_all_trajectories(norm_df)
        
        # Save results
        tester.save_results(all_results, CONFIG['output'])
        tester.save_md_summary(all_results, "batch_knn_summary.md")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

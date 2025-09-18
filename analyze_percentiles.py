#!/usr/bin/env python3
"""
Analyze confidence intervals of relative errors from batch KNN results.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import argparse
import sys
import glob
from pathlib import Path

def analyze_confidence_intervals_for_group(df_group, confidence_levels=[50, 75, 90]):
    """
    Analyze confidence intervals of relative errors for each step across all events.
    
    Args:
        df_group: DataFrame containing all data for analysis
        confidence_levels: List of confidence levels to calculate
    
    Returns:
        Dictionary mapping (step, confidence_level) to [lower_bound, upper_bound]
    """
    # Convert relative_error from percentage to decimal
    df_group = df_group.copy()
    df_group['relative_error'] = df_group['relative_error'] / 100
    
    # Group by step and calculate confidence intervals
    step_intervals = {}
    
    for step in sorted(df_group['step'].unique()):
        step_data = df_group[df_group['step'] == step]['relative_error']
        
        if len(step_data) == 0:
            continue
            
        for confidence_level in confidence_levels:
            # Calculate the bounds for the given confidence level
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile
            
            lower_bound = np.percentile(step_data, lower_percentile, method='higher')
            upper_bound = np.percentile(step_data, upper_percentile, method='lower')
            
            step_intervals[(step, confidence_level)] = [lower_bound, upper_bound]
    
    return step_intervals

def interpolate_confidence_intervals(step_intervals, confidence_levels, start_step=55, end_step=299):
    """
    Interpolate confidence intervals for all steps from start_step to end_step.
    
    Args:
        step_intervals: Dictionary from analyze_confidence_intervals_for_group
        confidence_levels: List of confidence levels
        start_step: Starting step for interpolation
        end_step: Ending step for interpolation
    
    Returns:
        Dictionary with interpolated confidence intervals for all steps
    """
    interpolated = {}
    
    for confidence_level in confidence_levels:
        # Extract existing steps and bounds for this confidence level
        step_data = [(step, bounds) for (step, cl), bounds in step_intervals.items() if cl == confidence_level]
        
        if len(step_data) < 2:
            continue
            
        steps = [item[0] for item in step_data]
        lower_bounds = [item[1][0] for item in step_data]
        upper_bounds = [item[1][1] for item in step_data]
        
        # Create interpolation functions
        lower_interp = interpolate.interp1d(steps, lower_bounds, kind='linear', 
                                           bounds_error=False, fill_value='extrapolate')
        upper_interp = interpolate.interp1d(steps, upper_bounds, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')
        
        # Generate interpolated values
        for step in range(start_step, end_step + 1):
            lower = float(lower_interp(step))
            upper = float(upper_interp(step))
            interpolated[(step, confidence_level)] = [lower, upper]
    
    return interpolated

def main():
    parser = argparse.ArgumentParser(description='Analyze confidence intervals of relative errors for all CSV files')
    parser.add_argument('--input-dir', default='test_results', help='Directory containing CSV files (default: test_results)')
    parser.add_argument('--output-dir', default='confidence_intervals', help='Output directory (default: confidence_intervals)')
    parser.add_argument('--confidence-levels', nargs='+', type=int, default=[50, 75, 90], help='Confidence levels to calculate (default: 50 75 90)')
    parser.add_argument('--start-step', type=int, default=55, help='Starting step (default: 55)')
    parser.add_argument('--end-step', type=int, default=299, help='Ending step (default: 299)')
    
    args = parser.parse_args()
    
    try:
        # Find all CSV files matching the pattern
        csv_files = glob.glob(f"{args.input_dir}/batch_knn_results_*.csv")
        
        if not csv_files:
            print(f"No CSV files found in {args.input_dir}")
            return
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        for csv_file in csv_files:
            print(f"Processing {csv_file}...")
            
            # Extract event_type, sub_event_type, border from filename
            filename = Path(csv_file).stem
            parts = filename.split('_')
            if len(parts) >= 5:
                event_type = parts[3]
                sub_event_type = parts[4]
                border = parts[5]
            else:
                print(f"  Skipping {csv_file} - invalid filename format")
                continue
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Analyze confidence intervals for the entire file (all events combined)
            step_intervals = analyze_confidence_intervals_for_group(df, args.confidence_levels)
            
            if not step_intervals:
                continue
            
            # Interpolate for all steps
            interpolated = interpolate_confidence_intervals(step_intervals, args.confidence_levels, args.start_step, args.end_step)
            
            # Prepare output data
            output_data = []
            for step in range(args.start_step, args.end_step + 1):
                for confidence_level in args.confidence_levels:
                    if (step, confidence_level) in interpolated:
                        bounds = interpolated[(step, confidence_level)]
                        output_data.append({
                            'step': step,
                            'confidence_level': confidence_level,
                            'rel_error_lower_bound': bounds[0],
                            'rel_error_upper_bound': bounds[1]
                        })
            
            # Save to CSV
            if output_data:
                output_df = pd.DataFrame(output_data)
                output_filename = f"confidence_intervals_{event_type}_{sub_event_type}_{border}.csv"
                output_path = Path(args.output_dir) / output_filename
                output_df.to_csv(output_path, index=False)
                print(f"  Saved {output_filename}")
        
        print(f"\nAll results saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
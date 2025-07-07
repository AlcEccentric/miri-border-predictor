import pandas as pd
import numpy as np
from typing import Callable, Dict

def add_relative_score_diff(df):
    df = df.copy()
    
    def calculate_relative_diff(group):
        # Sort by time_idx to ensure correct order
        group = group.sort_values('time_idx')
        
        # Calculate next step's score - current score
        score_diff = group['score'].shift(-1) - group['score'] 
        
        # Divide by current score to get relative difference
        relative_diff = score_diff / group['score'].replace(0, 1)  # avoid division by zero
        
        # Set first step's relative diff to 1
        relative_diff.iloc[0] = 1
        
        # Set last step's relative diff to 0
        relative_diff.iloc[-1] = 0
        
        # Fill any remaining NaN (shouldn't be any, but just in case)
        relative_diff = relative_diff.fillna(0)
        
        return relative_diff
    
    df['relative_score_diff'] = df.groupby(['border', 'event_id']).apply(
        lambda x: calculate_relative_diff(x)
    ).reset_index(level=[0,1], drop=True)
    
    return df

def inject_bonus_smoothly(df, condition, bonus_range, bonus_total=15000):
    df = df.copy()
    target_df = df[condition].copy()

    for (event_id, border), group_df in target_df.groupby(['event_id', 'border']):
        group_df = group_df.sort_values('time_idx')
        score = group_df['score'].values
        time_idx = group_df['time_idx'].values
        cur_bonus_total = bonus_total if len(time_idx) >= bonus_range[1] else bonus_total * len(time_idx) / bonus_range[1]

        if bonus_range is None:
            # Legacy behavior: inject into is_boosted == 0.0
            first_half_mask = group_df['is_boosted'] == False
            second_half_mask = group_df['is_boosted'] == True
        else:
            start_idx, end_idx = bonus_range
            first_half_mask = (time_idx >= start_idx) & (time_idx <= end_idx)
            second_half_mask = time_idx > end_idx

        original_first_half = score[first_half_mask]

        # Create evenly distributed bonus points
        if len(original_first_half) > 0:
            cumulative_bonus = np.linspace(0, cur_bonus_total, len(original_first_half))
        else:
            cumulative_bonus = np.array([])

        adjusted_first_half = original_first_half + cumulative_bonus

        # Shift second half to keep smooth
        if len(original_first_half) > 0:
            shift_amount = adjusted_first_half[-1] - original_first_half[-1]
            adjusted_second_half = score[second_half_mask] + shift_amount
        else:
            adjusted_second_half = score[second_half_mask]

        # Update scores in group_df
        group_df.loc[first_half_mask, 'score'] = adjusted_first_half
        group_df.loc[second_half_mask, 'score'] = adjusted_second_half

        df.loc[group_df.index, 'score'] = group_df['score']

    return df

def scale(df, condition, scale_factor_by_border):
    scaled_dfs = []
    target_df = df[condition].copy()
    origin_df = df[~condition].copy()
    
    for (event_id, idol_id, border), group_df in target_df.groupby(['event_id', 'idol_id', 'border']):
        group_df = group_df.sort_values('time_idx').copy()
        scale_factor = scale_factor_by_border.get(border, 1.0)  # default to 1.0 if not found
        group_df['score'] = group_df['score'] * scale_factor
        scaled_dfs.append(group_df)

    for (event_id, idol_id, border), group_df in origin_df.groupby(['event_id', 'idol_id', 'border']):
        group_df = group_df.sort_values('time_idx').copy()
        scaled_dfs.append(group_df)
    
    return pd.concat(scaled_dfs).sort_index()

def common_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Features to add for all samples regardless of event_type & border
    df = df.copy()
    df["early_boost"] = 0
    df.loc[df['length'] != 7.25, 'early_boost'] = 1
    df["is_mid_boosted"] = 0

    for (event_id, border), group_df in df.groupby(["event_id", "border"]):
        sorted_group = group_df.sort_values("time_idx").copy()

        # --- is_mid_boosted ---
        boost_series = sorted_group["is_boosted"].values
        boost_change_idx = None
        for i in range(1, len(boost_series)):
            if boost_series[i - 1] == 0 and boost_series[i] == 1:
                boost_change_idx = i
                break
        if boost_change_idx is not None:
            boosted_rows = sorted_group.iloc[boost_change_idx:boost_change_idx + 15]
            df.loc[boosted_rows.index, "is_mid_boosted"] = 1

    # --- is_final_stage ---
    df["is_final_stage"] = (df["days_remaining"] < 0.25).astype(int)
    df["is_final_hour"] = (df["days_remaining"] < 0.125).astype(int)

    return df

def check_column_value(df, column_name, expected_value):
    unique_values = df[column_name].unique()
    if len(unique_values) != 1 or unique_values[0] != expected_value:
        raise ValueError(f"Column {column_name} contains unexpected values. Expected {expected_value}, got {unique_values}")
    return True

def feature_engineering_for_event_type_3(df: pd.DataFrame) -> pd.DataFrame:
    check_column_value(df, 'event_type', 3.0)
    boosted_internal_event_types= range(16, 21) # 16 (Tiara), 17(Trsut..), 18(Trsut..), 19(Trsut..), 20(Trsut)

    df = common_feature_engineering(df)

    # --- has_bonus ---
    df['has_bonus'] = 0
    has_bonus_condition = (df['internal_event_type'].isin(boosted_internal_event_types))
    df.loc[has_bonus_condition, 'has_bonus'] = 1

    # --- is_bonus_boosted ---
    df['is_bonus_boosted'] = 0
    is_bonus_boosted_condition = has_bonus_condition & (df['time_idx'] < 15) & (df['border'] == 2500.0)
    df.loc[is_bonus_boosted_condition, 'is_bonus_boosted'] = 1

    # --- sub_type ---
    df['sub_event_type'] = 1.0
    df.loc[has_bonus_condition, 'sub_event_type'] = 2.0

    # Starting from event 367, i.e., プラチナスターシアター～夜に輝く星座のように～
    # bonus live is also available in events with type プラチナスターシアター (i.e., event_type == 3.0 & sub_event_type == 1.0)
    # The code below evenly distribute 15000 bonus pts to theater events prior to 367.
    inject_bonus_condition = (df['event_id'] < 367)& (df['internal_event_type'] != 15.0) & (df['sub_event_type'] == 1.0) & ((df['event_type'] == 3.0))
    df = inject_bonus_smoothly(df, inject_bonus_condition, bonus_total=15000, bonus_range=(0, 165))
    return df

def feature_engineering_for_event_type_4(df: pd.DataFrame) -> pd.DataFrame:
    check_column_value(df, 'event_type', 4.0)

    df = common_feature_engineering(df)
    
    # --- has_bonus ---
    df['has_bonus'] = 0
    has_bonus_condition = (
        (df['internal_event_type'] == 22) |
        ((df['internal_event_type'] == 23) & (df['event_id'] > 300))
    )
    df.loc[has_bonus_condition, 'has_bonus'] = 1

    # --- is_bonus_boosted ---
    df['is_bonus_boosted'] = 0
    boosted_range_condition = (
        (df['time_idx'] >= 24) & 
        (df['time_idx'] < 45) & 
        (df['border'] == 2500.0)
    )
    is_bonus_boosted_condition = has_bonus_condition & boosted_range_condition
    df.loc[is_bonus_boosted_condition, 'is_bonus_boosted'] = 1

    # --- sub_type ---
    df['sub_event_type'] = 1.0
    df.loc[has_bonus_condition, 'sub_event_type'] = 2.0
    
    return df

def feature_engineering_for_event_type_11(df: pd.DataFrame) -> pd.DataFrame:
    check_column_value(df, 'event_type', 11.0)
    
    df = common_feature_engineering(df)

    # --- has_bonus ---
    df['has_bonus'] = 0
    has_bonus_condition = (df['event_id'] > 294)
    df.loc[has_bonus_condition, 'has_bonus'] = 1

    # --- is_bonus_boosted ---
    df['is_bonus_boosted'] = 0
    is_bonus_boosted_condition = has_bonus_condition & (df['time_idx'] < 15) & (df['border'] == 2500.0)
    df.loc[is_bonus_boosted_condition, 'is_bonus_boosted'] = 1

    # --- sub_type ---
    df['sub_event_type'] = 1.0

    inject_bonus_condition = (df['event_id'] <= 294) & ((df['event_type'] == 11.0))
    df = inject_bonus_smoothly(df, inject_bonus_condition, (0, 6), 10000)
    df = inject_bonus_smoothly(df, inject_bonus_condition, (15, 18), 3000)
    df = inject_bonus_smoothly(df, inject_bonus_condition, (42, 47), 2000)
    df = scale(df, inject_bonus_condition, {2500.0: 1.3, 100.0: 1.3})
    return df

def feature_engineering_for_event_type_13(df: pd.DataFrame) -> pd.DataFrame:
    check_column_value(df, 'event_type', 13.0)
    
    df = common_feature_engineering(df)

    # --- has_bonus ---
    df['has_bonus'] = 1

    # --- is_bonus_boosted ---
    df['is_bonus_boosted'] = 0
    is_bonus_boosted_condition = (df['time_idx'] < 15) & (df['border'] == 2500.0)
    df.loc[is_bonus_boosted_condition, 'is_bonus_boosted'] = 1

    # --- sub_type ---
    df['sub_event_type'] = 1.0
    return df

def feature_engineering_for_event_type_5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sub_event_type'] = 1.0
    return df

def default_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = common_feature_engineering(df)
    df['has_bonus'] = 0
    df["is_bonus_boosted"] = 0
    return df

# Registry of functions
FEATURE_ENGINEERING_FUNCS: Dict[float, Callable[[pd.DataFrame], pd.DataFrame]] = {
    3.0: feature_engineering_for_event_type_3,
    4.0: feature_engineering_for_event_type_4,
    5.0: feature_engineering_for_event_type_5,
    11.0: feature_engineering_for_event_type_11,
    13.0: feature_engineering_for_event_type_13,
}

def add_additional_features(df):
    processed_dfs = []
    for event_type, feature_func in FEATURE_ENGINEERING_FUNCS.items():
        event_type_df = df[df['event_type'] == event_type].copy()
        if not event_type_df.empty:
            processed_df = feature_func(event_type_df)
            processed_dfs.append(processed_df)
    return pd.concat(processed_dfs, ignore_index=True)

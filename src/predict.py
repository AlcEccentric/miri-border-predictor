from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.knn import get_filtered_df, predict_curve_knn
from src.knn.config import get_group_config
from src.core.normalization import denormalize_target_to_raw


def _apply_skip_haircut(arr: np.ndarray, first_pred_idx: int, crossing: float, f: float) -> np.ndarray:
    """C: discount an absolute cumulative trajectory's PREDICTED growth beyond
    ``crossing`` by factor ``f`` (piecewise). Once the running cumulative passes
    ``crossing`` (skip-pass exhaustion point), each further predicted increment
    is scaled by ``f``; the increment that straddles the crossing is split so
    only its above-crossing part is discounted. The observed portion
    (``arr[:first_pred_idx]``) is untouched. Lands exactly at
    ``crossing + f*(final - crossing)``. No-op when the trajectory never crosses
    or ``f>=1``."""
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if first_pred_idx >= n or f >= 1.0 or first_pred_idx < 1:
        return a
    out = a.copy()
    prev = float(out[first_pred_idx - 1])
    for i in range(first_pred_idx, n):
        inc = float(a[i] - a[i - 1])
        start, end = prev, prev + inc
        if start >= crossing:
            inc_adj = inc * f
        elif end <= crossing:
            inc_adj = inc
        else:  # straddles the crossing
            inc_adj = (crossing - start) + (end - crossing) * f
        prev = start + inc_adj
        out[i] = prev
    return out


def calculate_confidence_bounds(normalized_target: np.ndarray,
                                last_known_step_index: int,
                                confidence_intervals: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[str, np.ndarray]]:
    """Calculate confidence bounds for multiple CI levels."""
    bounds = {}
    
    for confidence_level, (lower_bound, upper_bound) in confidence_intervals.items():
        upper_series = normalized_target.copy()
        lower_series = normalized_target.copy()
        
        if last_known_step_index + 1 < len(normalized_target):
            current_value = normalized_target[last_known_step_index]
            remaining_points = len(normalized_target) - (last_known_step_index + 1)
            
            for i in range(last_known_step_index + 1, len(normalized_target)):
                progress = (i - last_known_step_index - 1) / (remaining_points - 1) if remaining_points > 1 else 1
                
                # Apply upper bound normally
                upper_delta = progress * upper_bound
                upper_series[i] = (1 + upper_delta) * normalized_target[i]
                
                # Apply lower bound with proper constraints
                lower_delta = progress * lower_bound
                unconstrained_lower = (1 + lower_delta) * normalized_target[i]
                predicted_value = normalized_target[i]
                
                # Calculate progress from current step to this step
                steps_from_current = i - last_known_step_index
                total_future_steps = len(normalized_target) - last_known_step_index - 1
                step_progress = steps_from_current / total_future_steps if total_future_steps > 0 else 1
                
                # Ensure lower bound is sensible:
                # 1. Never below current value (with small minimum growth)
                # 2. Never above the prediction itself
                min_reasonable_bound = current_value * (1 + step_progress * 0.01)  # At least 1% total growth
                max_reasonable_bound = predicted_value  # Never above prediction
                
                # Apply constraints in order
                constrained_lower = max(unconstrained_lower, min_reasonable_bound)
                constrained_lower = min(constrained_lower, max_reasonable_bound)
                
                lower_series[i] = constrained_lower
        
        bounds[confidence_level] = {
            "upper": upper_series,
            "lower": lower_series
        }
    
    return bounds

def validate_confidence_bounds(raw_target: np.ndarray, raw_bounds: Dict[int, Dict[str, np.ndarray]], 
                              last_known_idx: int, confidence_intervals: Dict[int, Tuple[float, float]]) -> None:
    """Validate confidence bounds consistency."""
    target_at_last = raw_target[last_known_idx]
    target_final = raw_target[-1]
    
    for confidence_level, bounds in raw_bounds.items():
        upper_at_last = bounds["upper"][last_known_idx]
        lower_at_last = bounds["lower"][last_known_idx]
        
        # Check 1: Values at last_known_step_index should be same across target, upper, lower (0.5% tolerance)
        tolerance = abs(target_at_last) * 0.005
        if not (abs(target_at_last - upper_at_last) <= tolerance and abs(target_at_last - lower_at_last) <= tolerance):
            logging.error(f"CRITICAL: CI{confidence_level} values at last_known_step_index mismatch: target={target_at_last}, upper={upper_at_last}, lower={lower_at_last}")
        else:
            logging.debug(f"✓ CI{confidence_level} values at last_known_step_index match: {target_at_last}")
        
        # Check 2: Relative difference between final values should match confidence intervals
        upper_final = bounds["upper"][-1]
        lower_final = bounds["lower"][-1]
        
        upper_rel_diff = (upper_final - target_final) / target_final if target_final != 0 else 0
        lower_rel_diff = (lower_final - target_final) / target_final if target_final != 0 else 0
        
        expected_upper = confidence_intervals[confidence_level][1]
        expected_lower = confidence_intervals[confidence_level][0]
        
        upper_tolerance = abs(expected_upper) * 0.005
        lower_tolerance = abs(expected_lower) * 0.005
        
        if abs(upper_rel_diff - expected_upper) > upper_tolerance:
            logging.error(f"CRITICAL: CI{confidence_level} upper bound relative diff mismatch: got {upper_rel_diff:.6f}, expected {expected_upper:.6f}")
        else:
            logging.debug(f"✓ CI{confidence_level} upper bound relative diff matches: {upper_rel_diff:.6f}")
        
        if abs(lower_rel_diff - expected_lower) > lower_tolerance:
            # For lower bound, check if it was constrained by minimum growth or max prediction
            lower_growth_from_current = (lower_final - target_at_last) / target_at_last if target_at_last != 0 else 0
            if lower_growth_from_current >= 0.01 and lower_final <= target_final:  # If constraints were applied
                logging.info(f"CI{confidence_level} lower bound was constrained: got {lower_rel_diff:.6f}, expected {expected_lower:.6f}, growth from current: {lower_growth_from_current:.6f}")
            else:
                logging.error(f"CRITICAL: CI{confidence_level} lower bound relative diff mismatch: got {lower_rel_diff:.6f}, expected {expected_lower:.6f}")
        else:
            logging.debug(f"✓ CI{confidence_level} lower bound relative diff matches: {lower_rel_diff:.6f}")
        
        # Check 3: Lower bound shows sensible growth from current value and doesn't exceed prediction
        lower_growth_from_current = (lower_final - target_at_last) / target_at_last if target_at_last != 0 else 0
        if lower_growth_from_current < 0:
            logging.warning(f"CI{confidence_level} lower bound shows negative growth from current: {lower_growth_from_current:.6f}")
        elif lower_growth_from_current < 0.005:
            logging.warning(f"CI{confidence_level} lower bound shows very low growth from current: {lower_growth_from_current:.6f}")
        else:
            logging.debug(f"✓ CI{confidence_level} lower bound shows reasonable growth from current: {lower_growth_from_current:.6f}")
        
        # Check 4: Lower bound should never exceed prediction
        if lower_final > target_final:
            logging.error(f"CRITICAL: CI{confidence_level} lower bound ({lower_final}) exceeds prediction ({target_final})")
        else:
            logging.debug(f"✓ CI{confidence_level} lower bound is below prediction: lower={lower_final}, pred={target_final}")

def cap_confidence_intervals(confidence_intervals: Dict[int, Tuple[float, float]], 
                            current_value: float,
                            final_value: float,
                            current_step: int,
                            total_steps: int) -> Dict[int, Tuple[float, float]]:
    """Cap confidence intervals to ensure lower bound is at least ratio * current value."""
    if current_step < 170:
        return confidence_intervals
    capped_intervals = {}
    ratio = total_steps / current_step
    min_final_value = min(ratio * current_value, final_value)
    
    for level, (lower_bound, upper_bound) in confidence_intervals.items():
        min_lower_bound = (min_final_value / final_value) - 1 if final_value > 0 else lower_bound
        
        if lower_bound < min_lower_bound:
            capped_intervals[level] = (min_lower_bound, upper_bound)
            logging.debug(f"Capped CI{level} lower bound from {lower_bound:.3f} to {min_lower_bound:.3f}")
        else:
            capped_intervals[level] = (lower_bound, upper_bound)
    
    return capped_intervals

_CI_CACHE: Dict[Tuple[float, float, float], pd.DataFrame] = {}


def _resolve_tier(df: pd.DataFrame, requested_tier: int) -> int:
    """Pick an effective tier that actually exists in the CI dataframe.

    - No ``tier`` column (legacy / non-anniversary CSV): return 0.
    - Requested tier present: use it.
    - Requested tier absent: fall back to the nearest available tier by
      numeric distance, breaking ties toward the lower (wider, more
      conservative) tier.
    """
    if 'tier' not in df.columns:
        return 0
    available = sorted(int(t) for t in df['tier'].unique())
    if requested_tier in available:
        return requested_tier
    nearest = min(available, key=lambda t: (abs(t - requested_tier), t))
    logging.warning(
        f"Requested CI tier {requested_tier} not present (available: {available}); "
        f"falling back to nearest tier {nearest}."
    )
    return nearest


def load_confidence_intervals(
    event_type: float,
    sub_type: float,
    border: float,
    step: int,
    tier: int = 0,
    r2_client=None,
) -> Dict[int, Tuple[float, float]]:
    """Load confidence intervals for given parameters from R2.

    The per-(event_type, sub_type, border) CSV is fetched once per process
    and cached in ``_CI_CACHE``. If R2 access is unavailable and the caller
    did not supply a client, raises (no local-disk fallback by design).

    ``tier`` defaults to ``0`` for non-anniversary CIs (single CI per step).
    For anniversary (type 5) CIs, callers should pass the target idol's
    quartile (1..4) computed from the current event's smoothed score
    ranking at the prediction step. If the requested tier is missing from
    the CSV, the nearest available tier is used (see ``_resolve_tier``).
    """
    from src.storage.loader import load_ci_csv_from_r2
    from src.storage.r2_client import R2Client

    cache_key = (float(event_type), float(sub_type), float(border))
    if cache_key not in _CI_CACHE:
        client = r2_client or R2Client()
        _CI_CACHE[cache_key] = load_ci_csv_from_r2(client, event_type, sub_type, border)
    df = _CI_CACHE[cache_key]

    has_tier = 'tier' in df.columns
    effective_tier = _resolve_tier(df, tier)

    def _rows_for(step_value: int) -> pd.DataFrame:
        if has_tier:
            return df[(df['step'] == step_value) & (df['tier'] == effective_tier)]
        return df[df['step'] == step_value]

    step_data = _rows_for(step)

    # If exact step not found, find closest step within 25 steps (within the
    # resolved tier so the fallback stays self-consistent).
    if step_data.empty:
        if has_tier:
            available_steps = df[df['tier'] == effective_tier]['step'].unique()
        else:
            available_steps = df['step'].unique()
        if len(available_steps) == 0:
            raise ValueError(
                f"No confidence interval rows for tier {effective_tier} "
                f"(event_type={event_type}, sub_type={sub_type}, border={border})"
            )
        closest_step = min(available_steps, key=lambda x: abs(x - step))

        if abs(closest_step - step) > 25:
            raise ValueError(
                f"No confidence interval data found within 25 steps of step {step} "
                f"for (event_type={event_type}, sub_type={sub_type}, border={border}, "
                f"tier={effective_tier}). Closest available step: {closest_step}"
            )

        step_data = _rows_for(closest_step)
        logging.info(f"Using closest step {closest_step} for requested step {step}")

    intervals_map = {}
    for _, row in step_data.iterrows():
        intervals_map[int(row['confidence_level'])] = (row['rel_error_lower_bound'], row['rel_error_upper_bound'])

    required_levels = {75, 90}
    missing_levels = required_levels - set(intervals_map.keys())
    if missing_levels:
        raise ValueError(
            f"Missing required confidence levels {sorted(missing_levels)} for step {step} "
            f"(event_type={event_type}, sub_type={sub_type}, border={border}, tier={effective_tier})"
        )

    return intervals_map

def build_result_dict(
    event_id: int,
    event_type: float,
    idol_id: int,
    border: float,
    current_step: int,
    current_raw_data: np.ndarray,
    current_norm_data: np.ndarray,
    similar_ids: np.ndarray,
    distances: np.ndarray,
    filtered_norm_all: pd.DataFrame,
    norm_event_length: int,
    standard_event_length: int,
    full_event_length: int,
    actual_boost_start: int,
    smoothed_prediction: np.ndarray,
    event_name_map: Dict,
    eid_to_len_boost_ratio: Dict,
    confidence_intervals: Dict[int, Tuple[float, float]],
    standard_event_boost_ratio: float,
    use_skip_ceiling: bool = False,
    skip_crossing_score: float = 2_400_000.0,
    skip_haircut_f: float = 0.90,
    skip_observed_blend_enabled: bool = False,
) -> Dict:
    # Initialize result structure
    event_name = event_name_map[event_id]
    result = {
        "metadata": {
            "raw": {
                "id": event_id,
                "last_known_step_index": len(current_raw_data) - 1,
                "name": event_name
            },
            "normalized": {
                "last_known_step_index": len(current_norm_data) - 1,
                "neighbors": {}
            }
        },
        "data": {
            "raw": {
                "target": [],
                "bounds": {}
            },
            "normalized": {
                "target": [],
                "neighbors": {}
            }
        }
    }

    neighbor_weights = 1 / (distances + 1e-6)
    neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
    neighbor_norm_full_curves = []
    
    for i, (neighbor_eid, neighbor_iid) in enumerate(similar_ids):
        neighbor_norm_df = filtered_norm_all[
            (filtered_norm_all['event_id'] == neighbor_eid) & 
            (filtered_norm_all['idol_id'] == neighbor_iid)
        ]
        neighbor_norm_data = np.array(neighbor_norm_df['score'].values)
        neighbor_name = event_name_map[neighbor_eid]
        
        result["metadata"]["normalized"]["neighbors"][str(i+1)] = {
            "id": int(neighbor_eid),
            "idol_id": int(neighbor_iid),
            "name": neighbor_name,
            "raw_length": eid_to_len_boost_ratio[(neighbor_eid, neighbor_iid)]['length'],
        }
        result["data"]["normalized"]["neighbors"][str(i+1)] = [round(x) for x in neighbor_norm_data.tolist()]
        neighbor_norm_full_curves.append(neighbor_norm_data)

    target_norm_final_value = smoothed_prediction[-1]

    if len(current_norm_data) > 0:
        last_known_norm = current_norm_data[-1]
        remaining_steps = norm_event_length - len(current_norm_data)
        
        if remaining_steps > 0:
            predicted_norm_part = np.zeros(remaining_steps)
            for neighbor_curve, weight in zip(neighbor_norm_full_curves, neighbor_weights):
                if len(neighbor_curve) == norm_event_length and len(neighbor_curve) > len(current_norm_data):
                    neighbor_future = neighbor_curve[len(current_norm_data):]

                    if len(neighbor_future) > 0:
                        predicted_norm_part += weight * neighbor_future
                else:
                    raise ValueError(f"Neighbor norm data has incorrect length for idol {idol_id} at border {border} for event {event_id}")

            # Ensure continuity: the prediction should start from the last known value
            # and end at the target final value
            original_start = predicted_norm_part[0]
            original_end = predicted_norm_part[-1]
            
            # Scale prediction to go from last_known_norm to target_norm_final_value
            scale = (target_norm_final_value - last_known_norm) / (original_end - original_start)
            offset = last_known_norm - scale * original_start
            predicted_norm_part = scale * predicted_norm_part + offset

            # --- BOUND INCREASING RATES ---
            # Calculate per-step increasing rates for each neighbor
            neighbor_incr_rates = []
            for neighbor_curve in neighbor_norm_full_curves:
                if len(neighbor_curve) == norm_event_length and len(neighbor_curve) > len(current_norm_data):
                    neighbor_future = neighbor_curve[len(current_norm_data):]
                    # Calculate rates: first rate is from last_known to first_future, then diff between consecutive futures
                    neighbor_last_known = neighbor_curve[len(current_norm_data)-1] if len(current_norm_data) > 0 else 0
                    rates = np.diff(neighbor_future, prepend=neighbor_last_known)
                    neighbor_incr_rates.append(rates)
            neighbor_incr_rates = np.array(neighbor_incr_rates)  # shape: (n_neighbors, remaining_steps)

            # Calculate weighted average trajectory's increasing rates
            avg_incr_rate = np.diff(predicted_norm_part, prepend=last_known_norm)

            # Bound each step's rate by min/max among neighbors
            min_rates = np.min(neighbor_incr_rates, axis=0)
            max_rates = np.max(neighbor_incr_rates, axis=0)
            bounded_incr_rate = np.clip(avg_incr_rate, min_rates, max_rates)

            # Reconstruct bounded predicted_norm_part using the bounded incremental rates
            bounded_predicted_norm_part = np.zeros(remaining_steps)
            bounded_predicted_norm_part[0] = last_known_norm + bounded_incr_rate[0]
            for i in range(1, remaining_steps):
                bounded_predicted_norm_part[i] = bounded_predicted_norm_part[i-1] + bounded_incr_rate[i]
            predicted_norm_part = bounded_predicted_norm_part
            
            # Debug: Log differences between consecutive prediction values
            last_known_to_first = predicted_norm_part[0] - last_known_norm
            first_to_second = predicted_norm_part[1] - predicted_norm_part[0] if len(predicted_norm_part) > 1 else 0
            logging.info(f"Gap (last_known → first_predicted): {round(last_known_to_first)}")
            logging.info(f"Gap (first_predicted → second_predicted): {round(first_to_second)}")
            
            normalized_target = np.concatenate([current_norm_data, predicted_norm_part])
            
            # Debug: Check gap in normalized data (should be zero now)
            if len(current_norm_data) > 0 and len(predicted_norm_part) > 0:
                last_known_norm_check = current_norm_data[-1]
                first_predicted_norm_check = predicted_norm_part[0]
                norm_gap = first_predicted_norm_check - last_known_norm_check
                logging.debug(f"Normalized gap for idol {idol_id}, border {border}: last_known={last_known_norm_check}, first_predicted={first_predicted_norm_check}, gap={norm_gap}")
                
                gap_tolerance = abs(last_known_norm_check) * 0.005
                if abs(norm_gap) > gap_tolerance:
                    logging.error(f"CRITICAL: Non-zero gap in normalized data: {norm_gap}")
                    logging.error(f"  This violates requirement #1: slice point gap should be 0")
                else:
                    logging.debug(f"✓ No gap in normalized data at slice point")
                
                # Log final values for verification
                logging.debug(f"Normalized final value: {normalized_target[-1]}")
                logging.debug(f"Target normalized final value: {target_norm_final_value}")
                
                final_tolerance = abs(target_norm_final_value) * 0.005
                if abs(normalized_target[-1] - target_norm_final_value) > final_tolerance:
                    logging.error(f"CRITICAL: Final value mismatch: got {normalized_target[-1]}, expected {target_norm_final_value}")
                else:
                    logging.debug(f"✓ Normalized final value matches target: {normalized_target[-1]}")
        else:
            raise ValueError(f"Insufficient data for idol {idol_id} at border {border} for event {event_id}")
    else:
        raise ValueError(f"No current normalized data for idol {idol_id} at border {border} for event {event_id}")

    result["data"]["normalized"]["target"] = [round(x) for x in normalized_target.tolist()]
    
    # Cap confidence intervals to ensure lower bound is at least ratio * current value
    final_norm_value = normalized_target[-1]
    current_norm_value = normalized_target[len(current_norm_data) - 1]
    capped_confidence_intervals = cap_confidence_intervals(confidence_intervals,
                                                           current_norm_value,
                                                           final_norm_value,
                                                           len(current_norm_data),
                                                           norm_event_length)
    
    result["metadata"]["normalized"]["confidence_intervals"] = capped_confidence_intervals
    
    # Calculate confidence bounds for multiple CI levels
    normalized_bounds = calculate_confidence_bounds(normalized_target, len(current_norm_data) - 1, capped_confidence_intervals)

    raw_target = denormalize_target_to_raw(
        normalized_target=normalized_target,
        current_step=current_step,
        current_raw_data=current_raw_data,
        full_norm_length=norm_event_length,
        standard_event_length=standard_event_length,
        full_event_length=full_event_length,
        actual_boost_start=actual_boost_start,
        standard_event_boost_ratio=standard_event_boost_ratio,
    )
    
    # Debug: Check gap in raw data
    if len(current_raw_data) > 0 and len(raw_target) > len(current_raw_data):
        last_known_raw = current_raw_data[-1]
        first_predicted_raw = raw_target[len(current_raw_data)]
        raw_gap = first_predicted_raw - last_known_raw
        logging.debug(f"Raw gap for idol {idol_id}, border {border}: last_known={last_known_raw}, first_predicted={first_predicted_raw}, gap={raw_gap}")
        
        gap_tolerance = abs(last_known_raw) * 0.005
        if abs(raw_gap) > gap_tolerance:
            logging.error(f"CRITICAL: Non-zero gap in raw data: {raw_gap}")
            logging.error(f"  This violates requirement #1: slice point gap should be 0")
        else:
            logging.debug(f"✓ No gap in raw data at slice point")

    # C: skip-pass haircut -- discount predicted growth beyond the crossing
    # (2.4M, raw/absolute space). Only fires when the central prediction crosses;
    # cool idols (pred below crossing) are untouched. Bounds haircut consistently
    # below. DOUBLE-COUNT GUARD. When the observed-blend is ON, any idol that has
    # ALREADY crossed has its post-crossing decay fully owned by the regime-aware
    # base (the ramp multiplier m) in distance.py -> C is turned OFF for it. Only
    # not-yet-crossed idols (base still skip-active) get C, which haircuts the
    # PROJECTED future crossing. When the blend is OFF, fall back to the legacy
    # binary: C off only once >= 24h of observed post-crossing data exists.
    _c_off = False
    if use_skip_ceiling and len(current_raw_data) >= 1:
        _crossed = np.where(np.asarray(current_raw_data, dtype=float) >= skip_crossing_score)[0]
        if len(_crossed) > 0:
            if skip_observed_blend_enabled:
                _c_off = True  # crossed -> decay owned by the regime-aware base
            elif len(current_raw_data) >= 2:
                _spd_raw = max(2.0, float(full_event_length) / 13.0)
                _c_off = ((len(current_raw_data) - 1) - int(_crossed[0])) >= _spd_raw
    _skip_applied = False
    if (use_skip_ceiling and not _c_off and len(current_raw_data) > 0
            and len(raw_target) > len(current_raw_data)
            and float(raw_target[-1]) > skip_crossing_score):
        raw_target = _apply_skip_haircut(
            raw_target, len(current_raw_data), skip_crossing_score, skip_haircut_f)
        _skip_applied = True

    result["data"]["raw"]["target"] = [round(x) for x in raw_target.tolist()]
    
    # Denormalize bounds to raw scale for each CI level
    raw_bounds = {}
    for confidence_level, bounds in normalized_bounds.items():
        raw_upper = denormalize_target_to_raw(
            normalized_target=bounds["upper"],
            current_step=current_step,
            current_raw_data=current_raw_data,
            full_norm_length=norm_event_length,
            standard_event_length=standard_event_length,
            full_event_length=full_event_length,
            actual_boost_start=actual_boost_start,
            standard_event_boost_ratio=standard_event_boost_ratio,
        )
        
        raw_lower = denormalize_target_to_raw(
            normalized_target=bounds["lower"],
            current_step=current_step,
            current_raw_data=current_raw_data,
            full_norm_length=norm_event_length,
            standard_event_length=standard_event_length,
            full_event_length=full_event_length,
            actual_boost_start=actual_boost_start,
            standard_event_boost_ratio=standard_event_boost_ratio,
        )
        
        raw_bounds[confidence_level] = {
            "upper": [round(x) for x in (_apply_skip_haircut(raw_upper, len(current_raw_data), skip_crossing_score, skip_haircut_f) if _skip_applied else raw_upper).tolist()],
            "lower": [round(x) for x in (_apply_skip_haircut(raw_lower, len(current_raw_data), skip_crossing_score, skip_haircut_f) if _skip_applied else raw_lower).tolist()]
        }
    
    result["data"]["raw"]["bounds"] = raw_bounds
    
    # Validate confidence bounds
    raw_bounds_arrays = {level: {"upper": np.array([x for x in bounds["upper"]]), "lower": np.array([x for x in bounds["lower"]])} for level, bounds in raw_bounds.items()}
    validate_confidence_bounds(raw_target, raw_bounds_arrays, 
                              result["metadata"]["raw"]["last_known_step_index"], 
                              capped_confidence_intervals)
    
    # Make first value zero for all data arrays
    if result["data"]["raw"]["target"]:
        first_raw = result["data"]["raw"]["target"][0]
        result["data"]["raw"]["target"] = [x - first_raw for x in result["data"]["raw"]["target"]]
        
        # Apply zero baseline to bounds
        for confidence_level in result["data"]["raw"]["bounds"]:
            result["data"]["raw"]["bounds"][confidence_level]["upper"] = [x - first_raw for x in result["data"]["raw"]["bounds"][confidence_level]["upper"]]
            result["data"]["raw"]["bounds"][confidence_level]["lower"] = [x - first_raw for x in result["data"]["raw"]["bounds"][confidence_level]["lower"]]
    
    if result["data"]["normalized"]["target"]:
        first_norm = result["data"]["normalized"]["target"][0]
        result["data"]["normalized"]["target"] = [x - first_norm for x in result["data"]["normalized"]["target"]]
        
        for neighbor_key in result["data"]["normalized"]["neighbors"]:
            if result["data"]["normalized"]["neighbors"][neighbor_key]:
                first_neighbor = result["data"]["normalized"]["neighbors"][neighbor_key][0]
                result["data"]["normalized"]["neighbors"][neighbor_key] = [x - first_neighbor for x in result["data"]["normalized"]["neighbors"][neighbor_key]]

    # For type 5 anniversary events, the per-step CI bound trajectories balloon
    # the JSON payload (52 idols x 2 borders x 2 levels x 300 steps). Frontends
    # only need the final value of each bound, so collapse to last-value-only
    # after all zero-baseline shifts have been applied.
    if event_type == 5:
        for confidence_level in result["data"]["raw"]["bounds"]:
            bounds = result["data"]["raw"]["bounds"][confidence_level]
            result["data"]["raw"]["bounds"][confidence_level] = {
                "upper_final": bounds["upper"][-1],
                "lower_final": bounds["lower"][-1],
            }

    # Final consistency check: compare normalized and denormalized final values
    norm_final = normalized_target[-1]
    raw_final = raw_target[-1]
    
    # For no score scaling case, these should be exactly equal
    scale_factor = standard_event_length / full_event_length
    logging.debug(f"Final value check - scale_factor: {scale_factor}, norm_final: {norm_final}, raw_final: {raw_final}")
    
    if abs(scale_factor - 1.0) < 1e-10:
        # No scaling case - values should be identical
        final_tolerance = abs(norm_final) * 0.005
        if _skip_applied:
            logging.debug(f"Skip haircut applied: raw_final={raw_final} intentionally below norm_final={norm_final}")
        elif abs(norm_final - raw_final) > final_tolerance:
            logging.error(f"CRITICAL: Final value mismatch (no scaling case): normalized={norm_final}, denormalized={raw_final}, diff={abs(norm_final - raw_final)}")
            logging.error(f"  This violates requirement #2: norm and denorm final values should be the same")
        else:
            logging.debug(f"✓ Final values match perfectly (no scaling): {norm_final}")
    else:
        # With scaling - raw should equal norm/scale_factor
        expected_raw_final = norm_final / scale_factor
        expected_tolerance = abs(expected_raw_final) * 0.005
        if not _skip_applied and abs(raw_final - expected_raw_final) > expected_tolerance:
            logging.error(f"CRITICAL: Final value mismatch (with scaling): normalized={norm_final}, denormalized={raw_final}, expected={expected_raw_final}")
            logging.error(f"  Scale factor: {scale_factor}, diff: {abs(raw_final - expected_raw_final)}")
        else:
            logging.debug(f"✓ Final values consistent with scaling: norm={norm_final}, raw={raw_final}, scale={scale_factor}")
    
    logging.info(f"Prediction summary for idol {idol_id} at border {border} for event {event_id}")
    logging.info(f"=>Normalized prediction {round(result['data']['normalized']['target'][-1])} with length {len(result['data']['normalized']['target'])}")
    logging.info(f"=>Denormalized prediction {round(result['data']['raw']['target'][-1])} with length {len(result['data']['raw']['target'])}")
    logging.info(f"=>Neighbors:")
    for i, neighbor in enumerate(result["metadata"]["normalized"]["neighbors"].values()):
        logging.info(f"=>{neighbor['id']} {neighbor['idol_id']} {round(result['data']['normalized']['neighbors'][str(i+1)][-1])}")

    return result

def _compute_tier_by_idol_border(
    data: Dict[str, pd.DataFrame],
    event_id: int,
    event_type: float,
    sub_type: float,
    borders: List[float],
    n_tiers: int = 4,
) -> Dict[Tuple[int, float], int]:
    """Per-(idol, border) quartile tier for type 5 anniversary events.

    Returns ``{}`` for non-anniversary types. For type 5: rank the event's
    52 idols by their smoothed score at the latest known step (last value
    of the smoothed partial trajectory). Tier 1 = bottom quartile, tier 4
    = top quartile. Same idol can land in different tiers across events.
    """
    if event_type != 5:
        return {}
    sub_types = (sub_type,)
    out: Dict[Tuple[int, float], int] = {}
    for border in borders:
        filtered = get_filtered_df(data['smooth_part'], event_type, border, list(sub_types))
        cur = filtered[filtered['event_id'] == event_id]
        if cur.empty:
            continue
        per_idol_score = (
            cur.sort_values('aggregated_at')
               .groupby('idol_id')['score']
               .last()
        )
        ranks = per_idol_score.rank(method='first')
        n = len(ranks)
        if n == 0:
            continue
        for iid, r in ranks.items():
            tier = max(1, min(n_tiers, int(np.ceil(r / n * n_tiers))))
            out[(int(iid), float(border))] = tier
    return out


def get_predictions(
    data: Dict[str, pd.DataFrame],
    event_id: int,
    event_type: float,
    sub_type: float,
    idol_ids: List[int],
    borders: List[float],
    step: int,
    event_length: int,
    norm_event_length: int,
    standard_event_length: int,
    standard_event_boost_ratio: float,
    eid_to_len_boost_ratio: Dict,
    event_name_map: Dict,
    r2_client=None,
) -> Dict:
    results = {}
    sub_types = (sub_type,)

    # Pre-compute per-(idol, border) tier for type 5 anniversary events.
    # Empty dict for other event types -> tier=0 used everywhere downstream.
    tier_by_idol_border = _compute_tier_by_idol_border(
        data=data, event_id=event_id, event_type=event_type,
        sub_type=sub_type, borders=borders,
    )

    for idol_id in idol_ids:
        if idol_id not in results:
            results[idol_id] = {}
            
        for border in borders:
            filtered_norm_all = get_filtered_df(data['norm_all'], event_type, border, list(sub_types))
            filtered_norm_part = get_filtered_df(data['norm_part'], event_type, border, list(sub_types))
            filtered_smooth_part = get_filtered_df(data['smooth_part'], event_type, border, list(sub_types))
            filtered_smooth_all = get_filtered_df(data['smoo_all'], event_type, border, list(sub_types))
            filtered_raw = get_filtered_df(data['raw'], event_type, border, list(sub_types))

            try:
                # Detect whether THIS idol's observed border score has already
                # crossed the skip-exhaustion point (2.4M). If so, map the crossing
                # to a normalized-step index so B bases the iR on the correct
                # regime: pre-crossing skip-active pace while it hasn't crossed /
                # just crossed, or the OBSERVED post-crossing pace once >= 24h past
                # it (avoids the "look-back straddles the crossing" double-count).
                _grp_cfg = get_group_config(float(event_type), (float(sub_type),), float(border))
                _pre_raw = np.asarray(filtered_raw[
                    (filtered_raw['event_id'] == event_id) &
                    (filtered_raw['idol_id'] == idol_id)
                ]['score'].values, dtype=float)
                ir_crossing_step = None
                if getattr(_grp_cfg, "use_skip_ceiling", False) and len(_pre_raw) >= 2:
                    _above = np.where(_pre_raw >= getattr(_grp_cfg, "skip_crossing_score", 2_400_000.0))[0]
                    if len(_above) > 0 and (len(_pre_raw) - 1) > 0:
                        ir_crossing_step = int(round(int(_above[0]) / (len(_pre_raw) - 1) * step))

                smoothed_prediction, similar_ids, distances = predict_curve_knn(
                    event_id=event_id,
                    idol_id=idol_id,
                    border=border,
                    sub_types=sub_types,
                    current_step=step,
                    norm_data=filtered_norm_all,
                    norm_partial_data=filtered_norm_part,
                    smooth_partial_data=filtered_smooth_part,
                    smooth_full_data=filtered_smooth_all,
                    ir_crossing_step=ir_crossing_step,
                )

                if len(smoothed_prediction) == 0:
                    logging.debug(f"Skipping step {step} for idol {idol_id}, border {border} due to lack of data")
                    continue

                current_raw_data = filtered_raw[
                    (filtered_raw['event_id'] == event_id) & 
                    (filtered_raw['idol_id'] == idol_id)
                ]['score'].values
                
                current_norm_data = filtered_norm_part[
                    (filtered_norm_part['event_id'] == event_id) & 
                    (filtered_norm_part['idol_id'] == idol_id)
                ]['score'].values

                current_raw_data = np.array(current_raw_data)
                current_norm_data = np.array(current_norm_data)

                confidence_intervals = load_confidence_intervals(
                    event_type, sub_type, border, step,
                    tier=tier_by_idol_border.get((int(idol_id), float(border)), 0),
                    r2_client=r2_client,
                )
                
                _grp_cfg = get_group_config(float(event_type), (float(sub_type),), float(border))
                result = build_result_dict(
                    event_id=event_id,
                    event_type=event_type,
                    idol_id=idol_id,
                    border=border,
                    current_step=step,
                    current_raw_data=current_raw_data,
                    current_norm_data=current_norm_data,
                    similar_ids=similar_ids,
                    distances=distances,
                    filtered_norm_all=filtered_norm_all,
                    norm_event_length=norm_event_length,
                    standard_event_length=standard_event_length,
                    full_event_length=eid_to_len_boost_ratio[(event_id, idol_id)]['length'],
                    actual_boost_start=eid_to_len_boost_ratio[(event_id, idol_id)]['boost_start'],
                    smoothed_prediction=smoothed_prediction,
                    event_name_map=event_name_map,
                    eid_to_len_boost_ratio=eid_to_len_boost_ratio,
                    confidence_intervals=confidence_intervals,
                    standard_event_boost_ratio=standard_event_boost_ratio,
                    use_skip_ceiling=getattr(_grp_cfg, "use_skip_ceiling", False),
                    skip_crossing_score=getattr(_grp_cfg, "skip_crossing_score", 2_200_000.0),
                    skip_haircut_f=getattr(_grp_cfg, "skip_haircut_f", 0.90),
                    skip_observed_blend_enabled=getattr(_grp_cfg, "skip_observed_blend_enabled", False),
                )

                results[idol_id][border] = result

            except Exception as e:
                logging.error(f"Error processing idol {idol_id}, border {border}: {e}", exc_info=True)
                continue

    return results

"""
Alignment and per-neighbour prediction.

Given the top-k neighbours' full and partial trajectories, align each
neighbour's curve to the current event and combine them into a single
predicted trajectory.

Two strategies are exposed:

    ``single_method_predict`` - distance-weighted average of neighbours,
        aligned using one alignment method (matches the original
        ``get_prediction``).

    ``ensemble_predict`` - compute a distance-weighted average per
        alignment method, then blend the per-method results using
        ``method_weights`` (matches the original ``ensemble_prediction``).

The two strategies share the per-neighbour alignment helper
``_align_neighbor_curve`` but with a subtly different branch: the
ensemble path always keeps the neighbour's past untouched (splicing
the scaled future onto ``neighbor_full[:current_step]``), while the
single-method path rescales the entire curve for LINEAR/AFFINE.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.knn.config import AlignmentMethod
from src.knn.stage import scores_of


# ---------------------------------------------------------------------------
# Alignment primitives
# ---------------------------------------------------------------------------

def _fit_alignment(
    neighbor_scores: np.ndarray,
    current_scores: np.ndarray,
    align_start: int,
    align_end: int,
    method: AlignmentMethod,
    scale_cap: Tuple[float, float],
) -> Tuple[float, float]:
    """Fit (scale, offset) so that ``neighbor * scale + offset`` matches ``current`` in the align window.

    ``neighbor_scores`` is the neighbour's partial trajectory (matching the
    length of ``current_scores``). Only the window ``[align_start:align_end]``
    is used for the fit.

    The ``scale_cap`` applies to all three methods. When a fitted scale is
    clamped, the offset is recomputed to keep a natural anchor point:
        - LINEAR: anchors the first point of the align window.
        - AFFINE / RATIO: anchors the last point of the align window.
    """
    neighbor_window = neighbor_scores[align_start:align_end]
    current_window = current_scores[align_start:align_end]
    scale_low, scale_high = scale_cap

    def _clamped(raw_scale: float) -> float:
        if raw_scale > scale_high:
            logging.debug(f"[{method.value}] scale {raw_scale:.4f} hit upper bound {scale_high}")
            return scale_high
        if raw_scale < scale_low:
            logging.debug(f"[{method.value}] scale {raw_scale:.4f} hit lower bound {scale_low}")
            return scale_low
        return raw_scale

    if method == AlignmentMethod.LINEAR:
        raw_scale = (current_window[-1] - current_window[0]) / (neighbor_window[-1] - neighbor_window[0])
        scale = _clamped(float(raw_scale))
        # Anchor the first point.
        offset = current_window[0] - scale * neighbor_window[0]
        return float(scale), float(offset)

    if method == AlignmentMethod.AFFINE:
        model = LinearRegression().fit(neighbor_window.reshape(-1, 1), current_window)
        raw_scale = float(model.coef_[0])
        raw_offset = float(model.intercept_)
        scale = _clamped(raw_scale)
        # If the scale was clamped, re-anchor the last point so the aligned
        # curve still lines up with the current trajectory's present value.
        if scale == raw_scale:
            offset = raw_offset
        else:
            offset = float(current_window[-1] - scale * neighbor_window[-1])
        return float(scale), offset

    # RATIO: scale = mean(current) / mean(neighbor), clamped, then offset
    # chosen so the last scaled point matches the last current point.
    raw_scale = float(np.mean(current_window) / np.mean(neighbor_window))
    scale = _clamped(raw_scale)
    offset = float(current_window[-1] - scale * neighbor_window[-1])
    return scale, offset


def _cap_future_growth_ratio(
    aligned_curve: np.ndarray,
    neighbor_partial: np.ndarray,
    current_step: int,
    align_start: int,
    align_end: int,
    ratio_cap_lower: float,
    ratio_cap_upper: float,
) -> np.ndarray:
    """Cap the future growth so its ratio to the neighbour's lookback growth is in range.

    NOTE: In the current implementation this function reconstructs the same
    curve it is given (``future + offset == aligned_curve[current_step+1:]``)
    and the ``ratio_cap_*`` arguments are unused. It is effectively a no-op
    today, preserved as a placeholder so the call sites still exist for a
    future reinstatement of the cap. Behaviour is unchanged from before the
    refactor; this docstring just flags the dead-code status so future
    readers do not spend time puzzling over it.
    """
    offset = aligned_curve[current_step]
    future = aligned_curve[current_step + 1:] - offset
    neighbor_lookback_increase = neighbor_partial[align_end - 1] - neighbor_partial[align_start]
    if neighbor_lookback_increase == 0 or len(future) == 0:
        return aligned_curve
    return np.concatenate([aligned_curve[:current_step + 1], future + offset])


def _neighbor_scale_cap(
    neighbor_full_list: List[np.ndarray],
    neighbor_partial_list: List[np.ndarray],
    current_step: int,
    lookback: int,
) -> Tuple[float, float]:
    """Range of (future growth) / (lookback growth) ratios across the neighbours."""
    ratios: List[float] = []
    for full, partial in zip(neighbor_full_list, neighbor_partial_list):
        lookback_start = max(0, len(partial) - lookback)
        lookback_growth = partial[-1] - partial[lookback_start]
        future_growth = full[-1] - full[current_step - 1]
        if lookback_growth != 0:
            ratios.append(future_growth / lookback_growth)
    return min(ratios), max(ratios)


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

def _single_method_weights(distances: np.ndarray) -> np.ndarray:
    """Weight scheme used by ``single_method_predict``.

    Matches the original ``get_prediction`` formula exactly:
      - keep only strictly-positive distances (take abs),
      - normalise by max to avoid blow-ups,
      - invert with a 1e-6 epsilon,
      - re-normalise to sum to 1.

    If all distances are zero/empty, fall back to uniform weights over the
    retained entries.
    """
    positive = np.array([abs(d) for d in distances if d > 0])
    if len(positive) == 0 or np.max(positive) == 0:
        n = max(len(positive), 1)
        return np.ones(n) / n
    normalised = positive / np.max(positive)
    weights = 1.0 / (normalised + 1e-6)
    return weights / np.sum(weights)


def _ensemble_weights(distances: np.ndarray) -> np.ndarray:
    """Weight scheme used by ``ensemble_predict`` (no max-normalisation).

    Matches the original ``ensemble_prediction`` formula exactly.
    """
    if np.max(distances) > 0:
        w = 1.0 / (distances + 1e-6)
        total = np.sum(w)
        if total > 0:
            return w / total
    return np.ones(len(distances)) / len(distances)


# ---------------------------------------------------------------------------
# Per-neighbour alignment
# ---------------------------------------------------------------------------

def _align_neighbor_curve(
    neighbor_full: np.ndarray,
    neighbor_partial: np.ndarray,
    current_scores: np.ndarray,
    current_step: int,
    align_lookback: int,
    method: AlignmentMethod,
    scale_cap: Tuple[float, float],
    disable_scale: bool,
    always_keep_past: bool,
) -> Tuple[np.ndarray, int, int]:
    """Apply (scale, offset) to ``neighbor_full`` and return the aligned trajectory.

    Alignment itself is fit against ``neighbor_partial`` (the neighbour's
    trajectory restricted to the first ``current_step`` points) paired with
    ``current_scores`` in the same window. ``neighbor_full`` is then scaled
    so the future portion (and sometimes the past portion too) overlays the
    current trajectory.

    ``always_keep_past``:
        True  -> always splice ``neighbor_full[:current_step]`` with
                 the scaled future. Used by the ensemble path.
        False -> only splice for RATIO; for LINEAR/AFFINE rescale the
                 entire curve (past + future). Used by the single-method
                 path to match the original ``get_prediction``.

    Returns ``(aligned_curve, align_start, align_end)`` so callers can reuse
    the exact same window when post-processing (e.g. capping growth).
    """
    effective_lookback = min(align_lookback, len(current_scores))
    align_start = max(0, len(current_scores) - effective_lookback)
    align_end = len(current_scores)

    scale, offset = _fit_alignment(
        neighbor_partial, current_scores, align_start, align_end, method, scale_cap,
    )
    if disable_scale and current_step >= 270:
        scale = 1.0

    if always_keep_past or method == AlignmentMethod.RATIO:
        aligned = np.concatenate([
            neighbor_full[:current_step],
            neighbor_full[current_step:] * scale + offset,
        ])
    else:
        aligned = neighbor_full * scale + offset

    logging.debug(
        f"method={method.value} scale={scale:.4f} offset={offset:.1f} "
        f"neighbor[current_step]={neighbor_full[current_step]:.0f} "
        f"neighbor[-1]={neighbor_full[-1]:.0f} "
        f"diff={neighbor_full[-1] - neighbor_full[current_step]:.0f}"
    )
    return aligned, align_start, align_end


# ---------------------------------------------------------------------------
# Neighbour trajectory lookup
# ---------------------------------------------------------------------------

def fetch_neighbor_trajectories(
    prediction_full_df: pd.DataFrame,
    prediction_partial_df: pd.DataFrame,
    neighbor_ids: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Look up each neighbour's full and partial score trajectories."""
    full = [scores_of(prediction_full_df, eid, iid) for eid, iid in neighbor_ids]
    partial = [scores_of(prediction_partial_df, eid, iid) for eid, iid in neighbor_ids]
    return full, partial


# ---------------------------------------------------------------------------
# The two prediction strategies
# ---------------------------------------------------------------------------

def single_method_predict(
    current_scores: np.ndarray,
    current_step: int,
    neighbor_full_list: List[np.ndarray],
    neighbor_partial_list: List[np.ndarray],
    distances: np.ndarray,
    method: AlignmentMethod,
    align_lookback: Optional[int],
    scale_cap: Tuple[float, float],
    disable_scale: bool,
    neighbor_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted average of neighbour curves aligned with a single method."""
    if align_lookback is None:
        align_lookback = current_step

    weights = _single_method_weights(distances)
    ratio_lower, ratio_upper = _neighbor_scale_cap(
        neighbor_full_list, neighbor_partial_list, current_step, align_lookback,
    )

    aligned_curves: List[np.ndarray] = []
    for neighbor_full, neighbor_partial in zip(neighbor_full_list, neighbor_partial_list):
        aligned, align_start, align_end = _align_neighbor_curve(
            neighbor_full, neighbor_partial, current_scores, current_step,
            align_lookback, method, scale_cap, disable_scale,
            always_keep_past=False,
        )
        aligned = _cap_future_growth_ratio(
            aligned, neighbor_partial, current_step,
            align_start, align_end, ratio_lower, ratio_upper,
        )
        logging.debug(
            f"Scaled Neighbor Current: {aligned[current_step]:.0f} "
            f"Scaled Neighbor Last: {aligned[-1]:.0f} "
            f"Diff: {aligned[-1] - aligned[current_step]:.0f}"
        )
        aligned_curves.append(aligned)

    prediction = np.average(aligned_curves, axis=0, weights=weights)
    _log_neighbor_summary(
        neighbor_ids=neighbor_ids,
        distances=distances,
        weights=weights,
        neighbor_full_list=neighbor_full_list,
        aligned_curves_by_method={method: aligned_curves},
        method_weights={method: 1.0},
        final_prediction=prediction,
        strategy="single_method",
    )
    return prediction


def ensemble_predict(
    current_scores: np.ndarray,
    current_step: int,
    neighbor_full_list: List[np.ndarray],
    neighbor_partial_list: List[np.ndarray],
    distances: np.ndarray,
    align_lookback: int,
    method_weights: Dict[AlignmentMethod, float],
    scale_cap: Tuple[float, float],
    disable_scale: bool,
    neighbor_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend per-method predictions using ``method_weights``.

    For each alignment method we compute a distance-weighted average across
    neighbours, then blend the per-method results by ``method_weights``.
    """
    TOLERANCE = 0.2
    neighbor_weights = _ensemble_weights(distances)

    ratio_lower, ratio_upper = _neighbor_scale_cap(
        neighbor_full_list, neighbor_partial_list, current_step, align_lookback,
    )
    ratio_lower *= (1 - TOLERANCE)
    ratio_upper *= (1 + TOLERANCE)

    per_method_predictions: List[np.ndarray] = []
    aligned_by_method: Dict[AlignmentMethod, List[np.ndarray]] = {}
    for method in AlignmentMethod:
        aligned_curves: List[np.ndarray] = []
        for neighbor_full, neighbor_partial in zip(neighbor_full_list, neighbor_partial_list):
            aligned, align_start, align_end = _align_neighbor_curve(
                neighbor_full, neighbor_partial, current_scores, current_step,
                align_lookback, method, scale_cap, disable_scale,
                always_keep_past=True,
            )
            aligned = _cap_future_growth_ratio(
                aligned, neighbor_partial, current_step,
                align_start, align_end, ratio_lower, ratio_upper,
            )
            aligned_curves.append(aligned)
        aligned_by_method[method] = aligned_curves
        per_method_predictions.append(
            np.average(aligned_curves, axis=0, weights=neighbor_weights)
        )

    blended = np.zeros_like(per_method_predictions[0])
    for pred, method in zip(per_method_predictions, AlignmentMethod):
        blended += pred * method_weights[method]

    _log_neighbor_summary(
        neighbor_ids=neighbor_ids,
        distances=distances,
        weights=neighbor_weights,
        neighbor_full_list=neighbor_full_list,
        aligned_curves_by_method=aligned_by_method,
        method_weights=method_weights,
        final_prediction=blended,
        strategy="ensemble",
    )
    return blended


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _log_neighbor_summary(
    neighbor_ids: Optional[np.ndarray],
    distances: np.ndarray,
    weights: np.ndarray,
    neighbor_full_list: List[np.ndarray],
    aligned_curves_by_method: Dict[AlignmentMethod, List[np.ndarray]],
    method_weights: Dict[AlignmentMethod, float],
    final_prediction: np.ndarray,
    strategy: str,
) -> None:
    """Emit a compact per-neighbour summary at INFO level.

    Shows distance, normalized weight, each neighbour's own final score
    (pre-alignment), and the aligned final score per alignment method.
    Helpful when one neighbour dominates the prediction unexpectedly.
    """
    if not logging.getLogger().isEnabledFor(logging.INFO):
        return

    n = len(distances)
    method_list = list(aligned_curves_by_method.keys())
    method_col_width = 10

    header = f"KNN neighbours ({strategy}, k={n}):"
    # Column layout: idx | (eid,iid) | distance | weight | own_final | <per-method aligned_final>
    fixed = f"{'idx':>3} {'id':>14} {'dist':>10} {'weight':>7} {'own_final':>11}"
    method_header = " ".join(f"{m.value:>{method_col_width}}" for m in method_list)
    logging.info(header + "\n  " + fixed + " " + method_header)

    for i in range(n):
        eid, iid = (neighbor_ids[i] if neighbor_ids is not None else (-1, -1))
        own_final = float(neighbor_full_list[i][-1])
        aligned_finals = " ".join(
            f"{float(aligned_curves_by_method[m][i][-1]):>{method_col_width}.0f}"
            for m in method_list
        )
        logging.info(
            f"  {i:>3d} ({int(eid):>4d},{int(iid):>2d}) "
            f"{float(distances[i]):>10.2f} {float(weights[i]):>7.3f} "
            f"{own_final:>11.0f} {aligned_finals}"
        )

    if strategy == "ensemble":
        # Per-method weighted-average final, pre-blend
        per_method_final = {
            m: float(np.average([c[-1] for c in aligned_curves_by_method[m]], weights=weights))
            for m in method_list
        }
        parts = " ".join(f"{m.value}={per_method_final[m]:.0f}(w={method_weights[m]:.2f})" for m in method_list)
        logging.info(f"  per-method finals: {parts}")

    logging.info(f"  blended final prediction: {float(final_prediction[-1]):.0f}")


from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, Optional, Tuple


class DistanceMetric(Enum):
    RMSE = 'rmse'
    DTW = 'dtw'
    FINAL_DIFF = 'final_diff'
    SLOPE_AWARE = 'slope_aware'

class AlignmentMethod(Enum):
    LINEAR = 'linear'
    AFFINE = 'affine'
    RATIO = 'ratio'

@dataclass
class GroupConfig:
    """Configuration for specific event type, sub_event_type, and border combination"""
    early_stage_end: int = 150
    mid_stage_end: int = 230

    # Half-width (in steps) of the linear blend across each stage boundary.
    # 0 disables blending (hard switch, original behaviour). When > 0, within
    # ``[boundary - hw, boundary + hw)`` the prediction is a linear blend of the
    # two adjacent stages' predictions, removing the discontinuous "jump" a
    # forecast otherwise shows as ``current_step`` crosses a stage boundary.
    stage_blend_halfwidth: int = 0

    # Multiplier applied to the KNN distance of candidates that share the
    # target's idol_id. 1.0 = no preference (idol identity ignored, original
    # behaviour). Values < 1.0 pull same-idol candidates closer so they are
    # preferred as neighbours; ->0 makes same-idol candidates always rank
    # first, with cross-idol shape matches filling any remaining k slots.
    # Motivated by anniversary (type 5) events, where an idol is far more
    # consistent with its own past events than with its event-mates.
    same_idol_distance_factor: float = 1.0

    # Soft (kernel-weighted) neighbour blending. 0 = off (hard top-k with the
    # near-uniform inverse-distance weights). When > 0, a larger neighbour pool
    # is Gaussian-weighted with bandwidth = the ``soft_knn_bandwidth_k``-th
    # neighbour's distance, so a neighbour's influence fades smoothly to zero
    # as it recedes instead of being dropped at the rank-k cliff. Removes the
    # discrete-swap jumps in the forecast-over-time (top-k churn drives ~35% of
    # step-to-step swing and ~89% of the largest jumps) at no accuracy cost.
    soft_knn_bandwidth_k: int = 0
    
    early_stage_k: int = 5
    mid_stage_k: int = 4
    late_stage_k: int = 3
    
    early_stage_lookback: int = 40
    mid_stage_lookback: int = 30
    late_stage_lookback: int = 23

    early_stage_lookback_for_align: int = 40
    mid_stage_lookback_for_align: int = 30
    late_stage_lookback_for_align: int = 23    

    early_stage_metric: DistanceMetric = DistanceMetric.RMSE
    mid_stage_metric: DistanceMetric = DistanceMetric.RMSE
    late_stage_metric: DistanceMetric = DistanceMetric.FINAL_DIFF    

    early_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    mid_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    late_stage_weights: Dict[AlignmentMethod, float] = field(default_factory=lambda: {
        AlignmentMethod.AFFINE: 0.4,
        AlignmentMethod.LINEAR: 0.3,
        AlignmentMethod.RATIO: 0.3
    })
    
    use_trend_weighting: bool = False
    disable_scale: bool = False
    trend_weight: float = 0.3
    smoothing_window: Optional[int] = None
    outlier_threshold: float = 2.5

    # --- SLOPE_AWARE distance metric -------------------------------------
    # Active only where a stage's metric is DistanceMetric.SLOPE_AWARE.
    # D = (1 - slope_weight) * D_level + slope_weight * D_slope
    # Each component is an RMSE divided by a scale derived from the current
    # window's own magnitude / mean |slope|.
    # Stage-specific so you can blend differently in early vs. mid vs. late.
    early_stage_slope_weight: float = 0.5
    mid_stage_slope_weight: float = 0.5
    late_stage_slope_weight: float = 0.5


    early_stage_scale_cap: Tuple[float, float] = (0.8, 1.2)
    mid_stage_scale_cap: Tuple[float, float] = (0.8, 1.2)
    late_stage_scale_cap: Tuple[float, float] = (0.8, 1.2)

    # Penalize candidates whose within-event standing (percentile rank among
    # idols in their own event, 0=best/highest score, 1=worst) differs a lot
    # from the target's own within-event standing at the same step. Distance
    # is multiplied by (1 + weight * |own_percentile - candidate_percentile|).
    # Motivated by anniversary events: a transient scoring-rate surge can
    # make a mid-pack idol's absolute trajectory shape resemble an ELITE
    # idol's trajectory from a calmer historical event, producing
    # systematically too-elite neighbour matches. 0.0 = off (no behaviour
    # change vs. before this field existed).
    early_stage_rank_gap_weight: float = 0.0
    mid_stage_rank_gap_weight: float = 0.0
    late_stage_rank_gap_weight: float = 0.0

    # Adaptive gating for the rank-gap penalty above. ``None`` (default) =
    # unconditional: whenever ``*_rank_gap_weight`` > 0, the penalty always
    # applies (the original, non-adaptive behaviour). When set to a float,
    # the penalty is applied to an idol/step ONLY if a first (unweighted)
    # neighbour search already shows a rank-gap larger than this threshold
    # for the PROVISIONAL top-k -- i.e. only when that idol currently looks
    # mismatched. This avoids penalising idol/steps that aren't actually
    # mismatched (backtesting showed the unconditional penalty could hurt
    # accuracy right around the early/mid stage boundary).
    early_stage_rank_gap_threshold: Optional[float] = None
    mid_stage_rank_gap_threshold: Optional[float] = None
    late_stage_rank_gap_threshold: Optional[float] = None

    # Categorical alternative to ``*_rank_gap_weight``. When set (and the
    # adaptive gate above fires, i.e. the provisional pool already looks
    # mismatched), candidates whose OWN rank-gap exceeds this value are
    # excluded from the pool entirely rather than having their distance
    # scaled by a weight. This avoids the severity-calibration problem a
    # multiplicative weight has: a fixed weight tuned against a mild
    # mismatch (e.g. a normal event) can be far too weak to matter for a
    # severe one (e.g. an event-wide surge), or too aggressive for the mild
    # case if tuned against the severe one. A hard cutoff is calibration-free
    # in that sense -- it always removes a badly-mismatched candidate,
    # regardless of how large the mismatch is. ``None`` = off (no behaviour
    # change; falls back to the weight-based penalty if that is set).
    early_stage_rank_gap_max_gap: Optional[float] = None
    mid_stage_rank_gap_max_gap: Optional[float] = None
    late_stage_rank_gap_max_gap: Optional[float] = None

    # Severity-adaptive alternative to a fixed ``*_rank_gap_weight``. Instead
    # of a constant multiplier tuned against one historical event's mismatch
    # severity (which may be far too weak or far too strong for a different
    # event -- e.g. a normal year's mild mismatch vs. an event-wide surge's
    # severe one), the effective weight is derived AT PREDICTION TIME from
    # how bad the mismatch actually looks for this idol/step:
    #
    #     effective_weight = (target_inflation - 1) / mean_provisional_gap
    #
    # where ``mean_provisional_gap`` is the same quantity already computed
    # for the adaptive gate (mean rank-gap of the unweighted top-k). This
    # means a "typical" mismatched candidate in the pool always ends up with
    # its distance multiplied by approximately ``target_inflation``,
    # regardless of whether the underlying mismatch is mild or severe --
    # the correction strength is calibrated by the current event's own
    # measured severity, not by history. ``None`` = off (falls back to
    # ``*_rank_gap_weight`` if set, else no correction).
    early_stage_rank_gap_target_inflation: Optional[float] = None
    mid_stage_rank_gap_target_inflation: Optional[float] = None
    late_stage_rank_gap_target_inflation: Optional[float] = None

    # When True, neighbour SEARCH (distance computation only -- not
    # alignment/prediction, which stay on raw scores) divides every idol's
    # trajectory by its own event's contemporaneous scale (median score
    # across idols in that event, at each step) before computing distances.
    # This makes trajectories from differently-inflated events comparable,
    # fixing the root cause of a scoring-rate surge making a mid-pack idol's
    # absolute curve resemble an elite idol's curve from a calmer year --
    # rather than the rank-gap fields above, which penalise/exclude
    # already-mismatched candidates after the (biased) distance is computed.
    # False (default) = off, no behaviour change.
    early_stage_use_relative_scale_for_search: bool = False
    mid_stage_use_relative_scale_for_search: bool = False
    late_stage_use_relative_scale_for_search: bool = False

    # Adaptive scale cap: instead of a fixed per-stage scale_cap bound,
    # loosen ONLY the bound in the direction a live, measured ratio points
    # for THIS idol -- direction-agnostic (works for both inflation and
    # deflation, no per-border branching needed):
    #   - rank this idol's popularity position among other idols in the
    #     current live event, using its average score over a trailing
    #     window (`adaptive_cap_rank_window` steps) -- NOT cumulative
    #     growth, which is dominated by the single latest point and would
    #     reintroduce single-point ranking instability.
    #   - find the same popularity position (+/- `adaptive_cap_half_width`,
    #     clipped at the edges, never extended) in recent complete
    #     historical events.
    #   - ratio = this idol's cumulative growth (score_now - score_at_step_0)
    #     / mean across historical events of their matched idols' pooled
    #     cumulative growth over the same window. Cumulative (not trailing)
    #     is used here specifically because it was empirically ~3x more
    #     stable across successive prediction runs than a trailing-window
    #     growth delta.
    #   - if ratio > 1: new upper bound = max(static_upper, ratio).
    #     if ratio < 1: new lower bound = min(static_lower, ratio).
    #     Only ever loosens a bound, never tightens.
    # Falls back to the static scale_cap unchanged if there isn't enough
    # reliable data (too few historical events/idols, target idol's own
    # data too short/degenerate). False (default) = off, no behaviour change.
    early_stage_use_adaptive_scale_cap: bool = False
    mid_stage_use_adaptive_scale_cap: bool = False
    late_stage_use_adaptive_scale_cap: bool = False

    # Trailing window (in normalized steps) used ONLY to rank idols by
    # current popularity/standing -- see note above on why this must be a
    # trailing average, not cumulative growth.
    early_stage_adaptive_cap_rank_window: int = 24
    mid_stage_adaptive_cap_rank_window: int = 24
    late_stage_adaptive_cap_rank_window: int = 24

    # Half-width of the popularity-position neighbourhood pooled on the
    # historical side of the ratio (reduces single-idol noise). Clipped,
    # not extended, at either edge of the popularity ranking.
    early_stage_adaptive_cap_half_width: int = 2
    mid_stage_adaptive_cap_half_width: int = 2
    late_stage_adaptive_cap_half_width: int = 2

    # Only compare against the N most recent historical events in the KNN
    # candidate pool (by event_id), not the full lookback window (which for
    # type 5 can span ~225 events back). Older anniversaries can reflect a
    # materially different game meta/player base, making them a poor
    # comparison point for "is THIS event's growth unusual". None = no cap
    # (use every historical event the candidate pool already contains).
    early_stage_adaptive_cap_max_recent_events: Optional[int] = 4
    mid_stage_adaptive_cap_max_recent_events: Optional[int] = 4
    late_stage_adaptive_cap_max_recent_events: Optional[int] = 4

    # Safety-gate constant (not stage-specific): minimum number of distinct
    # historical events that must contribute a usable pooled growth value
    # before the adaptive ratio is trusted. Below this, falls back to the
    # static cap unchanged.
    adaptive_cap_min_historical_events: int = 2

    # Reversal-gated EWMA override for the adaptive scale cap (default off,
    # no behaviour change). The plain cumulative ratio (above) is stable
    # but LAGS when the true instantaneous ratio spikes then decays (or
    # dips then rebounds) -- found empirically on event 142 around step
    # 190, where the cumulative ratio stayed elevated well after the real
    # rate had already reversed, causing an over-correction. A monotonic
    # drift (e.g. event 192's steady decline) does NOT trigger this
    # problem, since cumulative growth naturally follows a one-directional
    # trend. When enabled, a reversal gate checks whether the short-run
    # trend has flipped sign relative to the longer-run trend in the
    # pooled instantaneous-ratio series; ONLY when a reversal is detected
    # does the cumulative ratio get replaced by an EWMA-smoothed
    # instantaneous ratio for that idol/step.
    use_reversal_gated_ewma: bool = False
    reversal_rate_window: int = 40
    reversal_sample_spacing: int = 10
    reversal_short_window: int = 30
    reversal_long_window: int = 80
    reversal_min_short_magnitude: float = 0.2
    ewma_alpha: float = 0.3
    ewma_lookback: int = 80

    # Macro-regime gate (default off, no behaviour change). Unlike the
    # per-idol adaptive cap above (which loosens EACH idol's own bound
    # based on that idol's individual ratio), this computes ONE
    # event-wide, TRIMMED-MEAN cumulative growth ratio across ALL idols
    # (no popularity top-N cutoff), checked for PERSISTENCE over recent
    # steps (not a single snapshot), and only applies a correction if the
    # event as a whole clears a leave-one-out cross-event variance band.
    # If it clears, the cap's bound is set DIRECTLY to that event-wide
    # ratio (no separate proportional-strength ramp or unit-mismatched
    # correction target -- one ratio, one comparison, one cap value).
    # An individual idol whose OWN position-matched ratio runs FURTHER
    # than the event-wide number in the regime's direction (hotter when
    # inflating, colder when deflating) is pulled back TOWARD the
    # event-wide ratio via empirical-Bayes shrinkage: the shrink weight is
    # derived from that idol's own recent-step ratio noise vs. the
    # cross-idol spread (a noisy idol is pulled almost fully to the
    # event-wide ratio; a steady one keeps most of its own value) -- there
    # is NO fixed shrink-strength constant. An idol on the LESS-extreme
    # side of the event ratio is left at its own value (never pulled the
    # other way). See docs/relative_scale_search_normalization.md,
    # "Macro-regime detector" for the full validation (leave-one-out
    # backtest + broad holdout check against live event data). When both
    # this and ``use_adaptive_scale_cap`` are enabled for the same stage,
    # this macro gate takes precedence (see predictor.py wiring).
    use_macro_regime_gate: bool = False
    # Trailing-window length (in steps) used to rank idols by popularity
    # for position-matching the current event's idols to historical
    # events' idols.
    macro_regime_rank_window: int = 24
    # Fraction of the event's per-idol cumulative-ratio distribution
    # trimmed from both tails before averaging into the event-wide ratio
    # (e.g. 0.1 = drop top/bottom 10%). Removes a few unusually high/low
    # idols (e.g. a heavily-pushed center, or a barely-started idol) from
    # dominating the event-level number.
    macro_regime_trim_pct: float = 0.1
    # Minimum number of historical events that must contribute a usable
    # leave-one-out ratio before the band (and thus any correction) is
    # trusted. Below this, falls back to the static cap unchanged.
    macro_regime_min_historical_events: int = 2
    # Persistence check: the event's ratio must ALSO clear the same
    # leave-one-out band at enough recent sample steps (not just the
    # current step) before being trusted -- protects against a one-off
    # spike. The same recent-steps window is reused to measure each
    # idol's own ratio NOISE for the empirical-Bayes shrinkage (a noisy
    # idol is shrunk harder toward the event-wide ratio). There is no
    # fixed shrink-strength constant -- the strength is derived from
    # measured within-idol noise vs. cross-idol spread.
    macro_regime_persistence_window: int = 40
    macro_regime_persistence_min_steps: int = 3
    macro_regime_persistence_sample_spacing: int = 10

    # Empirical-Bayes per-idol shrinkage (see compute_macro_regime_scale_cap).
    # When False, an idol's own (possibly decay-forecast) ratio is used
    # directly as its cap anchor with NO pull toward the event-wide ratio.
    # Shrinkage was originally a coarse guard against letting an uncertain
    # "hot" outlier scale as much as it wanted (overshoot risk while we could
    # not tell if the hotness was persistent). For border-100 it is superseded
    # by the decay forecast, which estimates the actual future ratio directly
    # (and self-corrects run-to-run), so the coarse noise-shrink is no longer
    # wanted. NOTE: disabling this ALSO makes ``use_toptier_relax`` inert --
    # top-tier relaxation only ever *undoes* shrinkage, so with no shrinkage
    # there is nothing to undo (kept enabled but a no-op).
    use_eb_shrinkage: bool = True

    # Recency-gated top-tier relaxation. Inside an already-fired inflation
    # gate, a genuine top-tier outlier (own cumulative ratio above the
    # event-wide ratio by more than ``toptier_relax_sigma`` cross-idol
    # standard deviations) is normally shrunk hard toward the event mean,
    # which UNDER-predicts a real top accelerator. When enabled, if the
    # idol is ALSO still accelerating -- judged by a WEEKDAY-ROBUST signal,
    # the TREND of its cumulative ratio (cumulative ratio now vs.
    # ``toptier_relax_recency_lookback`` steps earlier, both being
    # whole-window ratios that average over weekdays, so NOT a
    # weekday/stage-confounded single-window rate) being non-decreasing --
    # the shrinkage is relaxed by ``toptier_relax_strength`` back toward
    # the idol's own ratio. Only ever RAISES the cap for confirmed top
    # accelerators; never applied on events where the gate has not fired.
    use_toptier_relax: bool = False
    toptier_relax_sigma: float = 2.0
    toptier_relax_strength: float = 0.7
    # Lookback (in steps) for the weekday-robust acceleration test: this
    # idol's cumulative ratio now vs. this many steps earlier must be
    # non-decreasing. ~24 steps (~25h) so it can fire once there are a
    # couple of days of data; still a cumulative-ratio trend, not a
    # single-window rate, so it is not weekday/stage-confounded.
    toptier_relax_recency_lookback: int = 24
    toptier_relax_recency_tol: float = 0.0

    # Decay forecast for the macro-regime cap anchor (default off, no
    # behaviour change; only meaningful when ``use_macro_regime_gate`` has
    # fired). The macro gate's anchor is a CUMULATIVE (stock) ratio, so it
    # keeps a front-loaded early surge fully weighted forever even after the
    # live interval pace has cooled -- over-predicting a decaying surge (and
    # propping up "tide-rider" idols whose own recent pace is already back to
    # normal). When enabled, the anchor ratio (both the event-wide ratio and
    # each idol's own) is replaced by a forward FORECAST of where it lands:
    #
    #   d_past = (R_now - R_{now-window}) / window          # recent slope
    #   R_end  = clamp_lo(floor, R_now + p * remaining * d_past)
    #   R_hat  = w * R_now + (1-w) * R_end
    #
    # Only the DECLINE direction is corrected: if the anchor is flat/rising
    # recently (d_past >= 0) it is left unchanged (rising accelerators are
    # handled by ``use_toptier_relax``, and a leave-one-out backtest on the 6
    # historical anniversaries showed the recent cumulative-ratio slope is NOT
    # a reliable forward signal -- ~half of decaying segments re-accelerate at
    # boost/dash -- so we never extrapolate a decline upward, only trim an
    # over-elevated decaying anchor). ``p`` (<1) decelerates the projected
    # decline (event 437 has no historical surge-decay analog to fit ``p`` on,
    # so it is a conservative policy value, refined on 437's own rolling
    # self-holdout as more data arrives). ``floor`` prevents projecting the
    # finish below historical normal. ``window`` is in normalized steps
    # (~23/day); 46 (~2 days) is more robust than 1-day (weekday-noisy).
    use_decay_forecast: bool = False
    decay_forecast_p: float = 0.8
    decay_forecast_w: float = 0.5
    decay_forecast_window: int = 46
    decay_forecast_floor: float = 1.0

    early_stage_use_ensemble: bool = True
    mid_stage_use_ensemble: bool = True
    late_stage_use_ensemble: bool = False

    early_stage_use_smooth_for_neighbors: bool = True
    mid_stage_use_smooth_for_neighbors: bool = True
    late_stage_use_smooth_for_neighbors: bool = False

    early_stage_use_smooth_for_prediction: bool = True
    mid_stage_use_smooth_for_prediction: bool = True
    late_stage_use_smooth_for_prediction: bool = True

    use_error_bounds: bool = False
    error_bound_std_multiplier: float = 2.0
    least_neighbor_id: float = 100.0
    # Lower bound on event_id for historical data loaded for this group.
    # Used by main.py's data loader to decide how far back to fetch. Must be
    # set per group; main.resolve_min_event_id raises if left as None.
    min_event_id: Optional[float] = None

def get_default_group_configs() -> Dict[Tuple[float, Tuple[float], float], GroupConfig]:
    """Get default configurations for all groups"""
    configs = {}

    # For シアター & シアタースペシャル
    # TODO: Try slope aware
    configs[(3.0, (1.0,), 100.0)] = GroupConfig(
        early_stage_end=170,
        mid_stage_end=270,
        early_stage_k=5,
        mid_stage_k=3,
        late_stage_k=3,
        disable_scale=False,
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.8, 1.2),
        late_stage_scale_cap=(0.9, 1.1),
        early_stage_lookback=5,
        mid_stage_lookback=5,
        late_stage_lookback=5,
        early_stage_lookback_for_align=100,
        mid_stage_lookback_for_align=100,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=False,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=False,
        mid_stage_use_smooth_for_prediction=False,
        late_stage_use_smooth_for_prediction=False,
        least_neighbor_id=275,
    )

    configs[(3.0, (1.0,), 2500.0)] = GroupConfig(
        early_stage_end=170,
        mid_stage_end=270,
        early_stage_k=4,
        mid_stage_k=4,
        late_stage_k=4,
        disable_scale=False,
        early_stage_scale_cap=(0.99, 1.01),
        mid_stage_scale_cap=(0.8, 1.2),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_lookback=50,
        mid_stage_lookback=24,
        late_stage_lookback=24,
        early_stage_lookback_for_align=45,
        mid_stage_lookback_for_align=24,
        late_stage_lookback_for_align=24, 
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=True,
        late_stage_use_smooth_for_neighbors=True,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=False,
        late_stage_use_smooth_for_prediction=False,
        least_neighbor_id=200,
    )
    
    # For Tiara & Trust
    configs[(3.0, (2.0,), 100.0)] = GroupConfig(
        early_stage_end=160,
        mid_stage_end=220,
        early_stage_k=5,
        mid_stage_k=5,
        late_stage_k=5,
        disable_scale=False,
        early_stage_lookback=50,
        mid_stage_lookback=35,
        late_stage_lookback=24,
        early_stage_lookback_for_align=60,
        mid_stage_lookback_for_align=35,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.SLOPE_AWARE,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_slope_weight=0.6,
        mid_stage_slope_weight=0.6,
        late_stage_slope_weight=0.6,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        mid_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_scale_cap=(0.5, 2),
        mid_stage_scale_cap=(0.8, 1.2),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    configs[(3.0, (2.0,), 2500.0)] = GroupConfig(
        early_stage_end=170,
        mid_stage_end=270,
        early_stage_k=5,
        mid_stage_k=5,
        late_stage_k=5,
        disable_scale=False,
        early_stage_lookback=50,
        mid_stage_lookback=35,
        late_stage_lookback=24,
        early_stage_lookback_for_align=45,
        mid_stage_lookback_for_align=35,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.SLOPE_AWARE,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        mid_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.9, 1.1),
        late_stage_scale_cap=(0.9, 1.1),
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    # For ツアービンゴスペシャル and ツアービンゴ
    # TODO: Try slope aware
    configs[(4.0, (2.0,), 100.0)] = GroupConfig(
        early_stage_end=255,
        mid_stage_end=255,
        early_stage_k=4,
        mid_stage_k=4,
        late_stage_k=4,
        disable_scale=False,
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.95, 1.15),
        late_stage_scale_cap=(0.9, 1.1),
        early_stage_lookback=50,
        mid_stage_lookback=30,
        late_stage_lookback=30,
        early_stage_lookback_for_align=80,
        mid_stage_lookback_for_align=30,
        late_stage_lookback_for_align=30, 
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_slope_weight=0.55,
        late_stage_slope_weight=0.55,
        early_stage_use_ensemble=False,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=False,
        mid_stage_use_smooth_for_prediction=False,
        late_stage_use_smooth_for_prediction=False
    )

    configs[(4.0, (2.0,), 2500.0)] = GroupConfig(
        early_stage_end=120,
        mid_stage_end=270, # not included
        early_stage_k=5,
        mid_stage_k=3,
        late_stage_k=3,
        disable_scale=False,
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.9, 1.1),
        late_stage_scale_cap=(0.9, 1.1),
        early_stage_lookback=50,
        mid_stage_lookback=24,
        late_stage_lookback=24,
        early_stage_lookback_for_align=96,
        mid_stage_lookback_for_align=48,
        late_stage_lookback_for_align=24, 
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_slope_weight=0.2,
        early_stage_use_ensemble=False,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=False,
        mid_stage_use_smooth_for_prediction=False,
        late_stage_use_smooth_for_prediction=False
    )

    configs[(5.0, (1.0,), 100.0)] = GroupConfig(
        early_stage_end=120,
        mid_stage_end=220,
        stage_blend_halfwidth=15,
        same_idol_distance_factor=0.25,
        soft_knn_bandwidth_k=5,
        early_stage_k=5,
        mid_stage_k=5,
        late_stage_k=5,
        early_stage_lookback=60,
        mid_stage_lookback=45,
        late_stage_lookback=25,
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.9, 1.1),
        late_stage_scale_cap=(0.95, 1.05),
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.RMSE,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_use_ensemble=False,
        mid_stage_use_ensemble=True,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False, 
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True,
        mid_stage_weights={
            AlignmentMethod.RATIO: 1.0,
            AlignmentMethod.AFFINE: 0.0,
            AlignmentMethod.LINEAR: 0.0
        },
        # Enabled after backtesting on 6 anniversary events (142/192/241/
        # 290/339/388): fixes surge-driven over-prediction (e.g. idol45/437,
        # ~34% inflated) by making neighbour search scale-invariant across
        # events. Early stage consistently improves; mid stage has a small
        # universal cost (worst single step +0.52pp MAE, worst pooled +0.28pp
        # per event); late stage is flat on MAE with a minor cov5% cost. Net
        # accepted as a tradeoff -- see TODO.md / commit history for detail.
        early_stage_use_relative_scale_for_search=True,
        mid_stage_use_relative_scale_for_search=True,
        late_stage_use_relative_scale_for_search=True,
        # Macro-regime gate + proportional correction: enabled after (1) a
        # leave-one-out backtest on 142/192/241/290/339/388 showing
        # ramp_scale=0.67 leaves all 5 "normal" historical events
        # byte-identical to off (their own excess vs. the leave-one-out
        # band is 0) and reproduces near-zero regression on 142 (the one
        # historical event with real, modest excess), and (2) a real
        # holdout validation on live event 437 itself: using 437's own
        # 20h/40h/60h data to predict "now" and comparing against the
        # actual observed score, pooled across all 52 idols and P90-
        # outlier-filtered (n=82), the model's raw (uncorrected)
        # prediction undershoots by ~1.17x on average -- consistent with
        # (slightly more conservative than) this mechanism's own computed
        # correction (~1.09x) for the same event/step. See
        # docs/relative_scale_search_normalization.md, "Macro-regime
        # detector + proportional correction" for the full validation
        # chain. Fixes the static (0.9, 1.1) cap's upper bound being hit
        # on effectively every RATIO-alignment fit for 437 (raw_scale
        # measured at 1.1-2.9 vs. the 1.1 ceiling) -- confirmed the cap,
        # not neighbour selection, was the active bottleneck.
        use_macro_regime_gate=True,
        # Recency-gated relaxation of the per-idol shrinkage for genuine
        # top-tier accelerators, to fix under-prediction of the strongest
        # idols (e.g. the pushed centre). Only raises the cap, only for
        # >2sigma outliers that are still accelerating (cumulative-ratio
        # trend non-decreasing -- a weekday-robust signal), and only inside
        # an already-fired inflation gate.
        use_toptier_relax=True,
        # Decay forecast: trim the cumulative (stock) anchor toward a
        # recency-weighted forward projection so a decaying surge (and
        # "tide-rider" idols whose own recent pace is back to normal) is no
        # longer over-predicted. Conservative policy p=0.8 (somewhat-
        # decelerated decline), 2-day window, floor 1.0; only ever trims a
        # declining anchor, never lifts (rising idols -> use_toptier_relax).
        use_decay_forecast=True,
        # EB shrinkage removed for border-100: the decay forecast now estimates
        # the actual future ratio per idol (and self-corrects each run), so the
        # coarse "shrink uncertain outliers toward the event mean" guard is no
        # longer needed. This also makes use_toptier_relax inert (nothing to
        # undo) -- left enabled but a no-op.
        use_eb_shrinkage=False,
        # Rank-gap categorical exclusion: drop candidate neighbours whose
        # within-event percentile rank differs from the target's by more than
        # 0.2. Fixes rank-jumper idols (e.g. idol21, mid-tier historically but
        # rank-1 in 437) whose same-idol past selves are standing-mismatched
        # (relative level ~0.9 vs target ~2.0) and otherwise inflate raw_scale
        # to ~2.9. Validated: historical 6-anniversary MAE 5.301->5.291%,
        # idol21 96h->120h error -7.22%->-4.80%.
        early_stage_rank_gap_max_gap=0.2,
        mid_stage_rank_gap_max_gap=0.2,
        late_stage_rank_gap_max_gap=0.2,
    )


    configs[(5.0, (1.0,), 1000.0)] = GroupConfig(
        early_stage_end=120,
        mid_stage_end=235,
        stage_blend_halfwidth=15,
        same_idol_distance_factor=0.25,
        soft_knn_bandwidth_k=5,
        early_stage_k=5,  
        mid_stage_k=4,    
        late_stage_k=3,
        early_stage_lookback=60,  
        mid_stage_lookback=40,    
        late_stage_lookback=25,
        early_stage_scale_cap=(0.95, 1.05),
        mid_stage_scale_cap=(0.95, 1.05),
        late_stage_scale_cap=(0.92, 1.08),
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.SLOPE_AWARE,
        late_stage_metric=DistanceMetric.FINAL_DIFF,
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=True,  
        late_stage_use_ensemble=True,
        early_stage_use_smooth_for_neighbors=False,   
        mid_stage_use_smooth_for_neighbors=False,     
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=False,
        mid_stage_use_smooth_for_prediction=False,
        late_stage_use_smooth_for_prediction=False
    )

    # For Tune
    configs[(11.0, (1.0,), 100.0)] = GroupConfig(
        early_stage_end=140,
        mid_stage_end=190,
        early_stage_k=6,
        mid_stage_k=3,
        late_stage_k=3,
        disable_scale=False,
        early_stage_lookback=50,
        mid_stage_lookback=35,
        late_stage_lookback=24,
        early_stage_lookback_for_align=60,
        mid_stage_lookback_for_align=35,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_scale_cap=(0.65, 1.35),
        mid_stage_scale_cap=(0.8, 1.2),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    configs[(11.0, (1.0,), 2500.0)] = GroupConfig(
        early_stage_end=190,
        mid_stage_end=250,
        early_stage_k=3,
        mid_stage_k=5,
        late_stage_k=5,
        disable_scale=False,
        early_stage_lookback=50,
        mid_stage_lookback=15,
        late_stage_lookback=25,
        early_stage_lookback_for_align=75,
        mid_stage_lookback_for_align=25,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.RMSE,
        mid_stage_metric=DistanceMetric.SLOPE_AWARE,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.7,
            AlignmentMethod.LINEAR: 0.2,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_scale_cap=(0.9, 1.1),
        mid_stage_scale_cap=(0.85, 1.15),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        
        early_stage_use_smooth_for_neighbors=True,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    # For Tale
    configs[(13.0, (1.0,), 100.0)] = GroupConfig(
        early_stage_end=100,
        mid_stage_end=165,
        early_stage_k=4,
        mid_stage_k=4,
        late_stage_k=4,
        disable_scale=False,
        early_stage_lookback=50,
        mid_stage_lookback=25,
        late_stage_lookback=24,
        early_stage_lookback_for_align=60,
        mid_stage_lookback_for_align=55,
        late_stage_lookback_for_align=25, 
        early_stage_scale_cap=(0.7, 1.3),
        mid_stage_scale_cap=(0.75, 1.25),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_metric=DistanceMetric.FINAL_DIFF,
        mid_stage_metric=DistanceMetric.FINAL_DIFF,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_slope_weight=0.6,
        mid_stage_slope_weight=0.6,
        late_stage_slope_weight=0.6,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.8,
            AlignmentMethod.LINEAR: 0.1,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    configs[(13.0, (1.0,), 2500.0)] = GroupConfig(
        early_stage_end=115,
        mid_stage_end=200,
        early_stage_k=5,
        mid_stage_k=3,
        late_stage_k=3,
        disable_scale=False,
        early_stage_scale_cap=(0.8, 1.2),
        mid_stage_scale_cap=(0.8, 1.2),
        late_stage_scale_cap=(0.8, 1.2),
        early_stage_lookback=50,
        mid_stage_lookback=50,
        late_stage_lookback=25,
        early_stage_lookback_for_align=50,
        mid_stage_lookback_for_align=25,
        late_stage_lookback_for_align=25, 
        early_stage_metric=DistanceMetric.SLOPE_AWARE,
        mid_stage_metric=DistanceMetric.SLOPE_AWARE,
        late_stage_metric=DistanceMetric.SLOPE_AWARE,
        early_stage_slope_weight=0.7,
        mid_stage_slope_weight=0.4,
        late_stage_slope_weight=0.4,
        early_stage_weights={
            AlignmentMethod.AFFINE: 0.1,
            AlignmentMethod.LINEAR: 0.8,
            AlignmentMethod.RATIO: 0.1
        },
        early_stage_use_ensemble=True,
        mid_stage_use_ensemble=False,
        late_stage_use_ensemble=False,
        early_stage_use_smooth_for_neighbors=False,
        mid_stage_use_smooth_for_neighbors=False,
        late_stage_use_smooth_for_neighbors=False,
        early_stage_use_smooth_for_prediction=True,
        mid_stage_use_smooth_for_prediction=True,
        late_stage_use_smooth_for_prediction=True
    )

    return configs

GROUP_CONFIGS: Dict[Tuple[float, Tuple[float], float], GroupConfig] = get_default_group_configs()

# Optional overlay populated at runtime from R2 by main.py. A mapping from
# (event_type, sub_event_types, border) to a dict of {field_name: value} that
# should shadow the source config. Today only ``min_event_id`` is overlaid.
_DYNAMIC_OVERLAY: Dict[Tuple[float, Tuple[float, ...], float], Dict[str, object]] = {}


def set_dynamic_overlay(overlay: Dict[Tuple[float, Tuple[float, ...], float], Dict[str, object]]) -> None:
    """Install a dynamic overlay. Replaces any previous overlay."""
    global _DYNAMIC_OVERLAY
    _DYNAMIC_OVERLAY = dict(overlay)


def clear_dynamic_overlay() -> None:
    global _DYNAMIC_OVERLAY
    _DYNAMIC_OVERLAY = {}


def get_group_config(event_type: float, sub_types: Tuple[float], border: float) -> GroupConfig:
    """Get configuration for specific group, with dynamic overlay applied."""
    key = (event_type, sub_types, border)
    if key not in GROUP_CONFIGS:
        logging.warning(f'Group config not found for {key}')
        GROUP_CONFIGS[key] = GroupConfig()
    base = GROUP_CONFIGS[key]

    overlay = _DYNAMIC_OVERLAY.get(key)
    if not overlay:
        return base

    # Shallow copy with overlay fields replaced
    from dataclasses import replace
    return replace(base, **{k: v for k, v in overlay.items() if hasattr(base, k)})

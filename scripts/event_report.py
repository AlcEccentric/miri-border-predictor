#!/usr/bin/env python3
"""
Post-event honesty report -> one-pager PNG for sharing (e.g. Twitter).

Consumes artifacts already produced by the backtest pipeline:
  - test_results/batch_knn_results_{et}_{sub}_{border}.csv   (per-step pred vs actual, + tier)
  - confidence_intervals/confidence_intervals_{et}_{sub}_{border}.csv (tiered CI bounds)

It filters those to a single finished event and renders coverage / accuracy
stats as a 16:9 image.

Typical use (after running batch_knn_test for the finished event):
    python3 -m scripts.event_report --event-id 388 --event-type 5 --sub 1
    python3 -m scripts.event_report --event-id 435 --event-type 4 --sub 2

Notes:
  - Coverage is checked against the tiered CI table: a prediction is "covered"
    at level L if the realized relative error falls within that step+tier's
    [lower, upper] L% bounds.
  - relative_error in the results CSV is in PERCENT; CI bounds are FRACTIONS.
"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_jp_font() -> Optional[str]:
    """Try to find a CJK-capable font so Japanese labels render.

    Returns the font family name if found, else None (labels fall back to
    whatever matplotlib's default is; pass --lang en to avoid tofu).
    """
    candidates = [
        "Hiragino Sans", "Hiragino Maru Gothic Pro", "YuGothic",
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoPGothic",
        "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None

# ---------------------------------------------------------------------------
# Localised labels
# ---------------------------------------------------------------------------

L = {
    "jp": {
        "title": "予測精度レポート",
        "event": "イベント",
        "border": "ボーダー",
        "cov90": "90%予測区間の的中率",
        "cov75": "75%予測区間の的中率",
        "final_err": "終盤予測の平均誤差",
        "hit5": "±5%以内に到達",
        "hit10": "±10%以内に到達",
        "step": "ステップ",
        "mean_abs_err": "平均絶対誤差 (%)",
        "convergence": "予測誤差の推移",
        "coverage_by_step": "ステップ別 的中率",
        "worst": "最も外れたケース",
        "idol": "アイドル",
        "na": "未到達",
        "disclaimer": "過去データに基づく推定値です。実際の結果とは異なる場合があります。",
        "from_step": "ステップ70以降",
        "predicted_vs_actual": "予測スコアと実際の最終スコア",
        "predicted": "予測（最終）",
        "actual": "実際の最終スコア",
        "band90": "90%予測区間",
        "band75": "75%予測区間",
        "score": "スコア",
        "remaining_hours": "終了までの残り時間",
        "hours_unit": "h",
        "err5": "実際比 ±5%",
        "err10": "実際比 ±10%",
    },
    "en": {
        "title": "Prediction Accuracy Report",
        "event": "Event",
        "border": "Border",
        "cov90": "90% CI hit rate",
        "cov75": "75% CI hit rate",
        "final_err": "Final-stretch mean error",
        "hit5": "Reached +/-5%",
        "hit10": "Reached +/-10%",
        "step": "Step",
        "mean_abs_err": "Mean abs error (%)",
        "convergence": "Error over time",
        "coverage_by_step": "Coverage by step",
        "worst": "Worst miss",
        "idol": "Idol",
        "na": "n/a",
        "disclaimer": "Estimates based on past data; actual results may differ.",
        "from_step": "from step 70",
        "predicted_vs_actual": "Predicted vs actual final score",
        "predicted": "Predicted (final)",
        "actual": "Actual final",
        "band90": "90% interval",
        "band75": "75% interval",
        "score": "Score",
        "remaining_hours": "Hours remaining",
        "hours_unit": "h",
        "err5": "actual +/-5%",
        "err10": "actual +/-10%",
    },
}

# ---------------------------------------------------------------------------
# Data loading + coverage
# ---------------------------------------------------------------------------

NORM_EVENT_LENGTH = 300


def _event_hours(data_dir: str, event_id: int) -> Optional[float]:
    """Total event duration in hours from event_info_all.csv, or None if unavailable."""
    path = Path(data_dir) / "event_info" / "event_info_all.csv"
    if not path.exists():
        return None
    e = pd.read_csv(path)
    row = e[e["event_id"] == event_id]
    if row.empty:
        return None
    start = pd.to_datetime(row["start_at"].iloc[0])
    end = pd.to_datetime(row["end_at"].iloc[0])
    return (end - start).total_seconds() / 3600.0


def _step_to_remaining_hours(step, total_hours: float, norm_len: int = NORM_EVENT_LENGTH):
    """Remaining hours until event end at a normalized step."""
    return (1.0 - np.asarray(step) / norm_len) * total_hours


DEFAULT_STEPS = [70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290]


def _detect_event_type_sub(data_dir: str, event_id: int, sub_override: Optional[int]):
    """Read event_type and derive sub_event_type for the event.

    sub is derived from internal_event_type via the same rules main.py uses
    (get_sub_event_types), so e.g. type-4 bonus events correctly resolve to
    sub 2. An explicit --sub overrides the derivation.
    """
    path = Path(data_dir) / "event_info" / "event_info_all.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; pass --event-type/--sub explicitly or fix --data-dir."
        )
    e = pd.read_csv(path)
    row = e[e["event_id"] == event_id]
    if row.empty:
        raise ValueError(f"event_id {event_id} not in {path}")
    et = float(row["event_type"].iloc[0])
    if sub_override is not None:
        return et, float(sub_override)
    internal = int(row["internal_event_type"].iloc[0])
    from src.main import get_sub_event_types
    sub = get_sub_event_types(event_id=int(event_id), internal_event_type=internal, event_type=et)[0]
    return et, float(sub)


def _run_backtest(event_type: float, sub: float, border: int, event_id: int,
                  data_dir: str) -> pd.DataFrame:
    """Run the KNN backtest in-memory for one (event, border). No CSVs touched.

    Predicts only the target event using all other loaded events as the
    neighbour pool, mirroring batch_knn_test but without writing the shared
    results files (so generating a report never clobbers them).
    """
    from scripts.batch_knn_test import BatchKNNTester
    from scripts.refresh_config import lookback_for_event_type

    tester = BatchKNNTester(
        event_type=event_type,
        sub_event_types=[sub],
        border=float(border),
        look_back_event_cnt=lookback_for_event_type(event_type),
        data_dir=data_dir,
    )
    raw, _ = tester.load_and_process_data()
    eid_to = tester.get_len_and_boost_ratio(raw)
    test_ids = tester.find_matching_events(
        raw_data=raw, test_event_ids=[float(event_id)], recent_count=None,
    )
    length_to = tester.get_normalized_full_and_part_data(raw, DEFAULT_STEPS, eid_to)
    results = tester.run_predictions(
        test_event_ids=test_ids, test_steps=DEFAULT_STEPS,
        temp_idol_id=1 if event_type == 5 else 0,
        raw_data=raw, length_to_df_data=length_to, eid_to_len_boost_ratio=eid_to,
    )
    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError(f"Backtest produced no predictions for event {event_id} border {border}")
    if "tier" not in df.columns:
        df["tier"] = 0
    df["border"] = float(border)
    df["abs_rel_error"] = df["relative_error"].abs()
    return df


def _event_end_utc(data_dir: str, event_id: int) -> Optional[pd.Timestamp]:
    """Event end time as a UTC timestamp (assumes JST if the CSV is tz-naive)."""
    path = Path(data_dir) / "event_info" / "event_info_all.csv"
    if not path.exists():
        return None
    e = pd.read_csv(path)
    row = e[e["event_id"] == event_id]
    if row.empty:
        return None
    end = pd.to_datetime(row["end_at"].iloc[0])
    if end.tzinfo is None:
        end = end.tz_localize("Asia/Tokyo")
    return end.tz_convert("UTC")


def _resolve_ci_history_ts(event_end_utc: Optional[pd.Timestamp]) -> Optional[str]:
    """Pick the ci/history/<ts>/ snapshot that holds the CI live during the event.

    The snapshot taken at the FIRST promotion after the event ended captured
    the then-current ci/latest, which had been live through the event's run
    (assuming no promotion happened mid-event). Returns that <ts>, or None to
    mean "no later promotion -> ci/latest is still what the event used".
    """
    if event_end_utc is None:
        return None
    from src.storage.r2_client import R2Client
    from src.storage.loader import BUCKET_NAME

    r2 = R2Client()
    resp = r2.list_objects(BUCKET_NAME, prefix="ci/history/")
    tss = set()
    for obj in resp.get("Contents", []):
        parts = obj["Key"].split("/")  # ci/history/<ts>/file.csv
        if len(parts) >= 3 and parts[2]:
            tss.add(parts[2])

    later = []
    for ts in tss:
        try:
            t = pd.to_datetime(ts, format="%Y%m%dT%H%M%SZ", utc=True)
        except ValueError:
            continue
        if t > event_end_utc:
            later.append((t, ts))
    if not later:
        return None
    later.sort()
    return later[0][1]



def _load_results(results_dir: str, et: int, sub: int, border: int,
                  event_id: int, min_step: int) -> pd.DataFrame:
    path = Path(results_dir) / f"batch_knn_results_{et}_{sub}_{border}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run batch_knn_test for this group first."
        )
    df = pd.read_csv(path)
    df = df[(df["event_id"] == event_id) & (df["step"] >= min_step)].copy()
    if df.empty:
        raise ValueError(f"No rows for event {event_id} (step>={min_step}) in {path}")
    if "tier" not in df.columns:
        df["tier"] = 0
    df["border"] = border
    df["abs_rel_error"] = df["relative_error"].abs()
    return df


def _load_ci(ci_dir: str, et: int, sub: int, border: int,
             allow_r2: bool = True, r2_history_ts: Optional[str] = None) -> pd.DataFrame:
    """Load the CI table for a group.

    Prefers the local file under ``ci_dir``; if missing and ``allow_r2`` is
    set, fetches from R2. The report should be scored against the SAME CI
    that production used to predict the event, not a CI regenerated later.

    ``r2_history_ts``: when set, read from ``ci/history/<ts>/`` instead of
    ``ci/latest/``. Use this when the CI table was updated after the event,
    so the report reflects the bounds the event was actually predicted
    against.
    """
    fname = f"confidence_intervals_{et}_{sub}_{border}.csv"
    path = Path(ci_dir) / fname
    if path.exists() and r2_history_ts is None:
        ci = pd.read_csv(path)
    elif allow_r2 or r2_history_ts is not None:
        from src.storage.loader import BUCKET_NAME
        from src.storage.r2_client import R2Client
        r2 = R2Client()
        if r2_history_ts is not None:
            key = f"ci/history/{r2_history_ts}/{fname}"
            print(f"Reading CI from R2 history snapshot: {key}")
            obj = r2.get_object(BUCKET_NAME, key)
            ci = pd.read_csv(obj["Body"])
        else:
            from src.storage.loader import load_ci_csv_from_r2
            print(f"Local CI {path} not found; fetching ci/latest from R2 for "
                  f"(et={et}, sub={sub}, border={border})")
            ci = load_ci_csv_from_r2(r2, float(et), float(sub), float(border))
    else:
        raise FileNotFoundError(
            f"Missing CI file {path} and R2 fallback disabled. "
            f"Run analyze_percentiles or enable --ci-source r2."
        )
    if "tier" not in ci.columns:
        ci["tier"] = 0
    return ci


def _attach_coverage(results: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    """Add covered_75 / covered_90 boolean columns by joining on (step, tier).

    A row is covered at level L if its realized relative error (as a fraction)
    falls within that step+tier's [lower, upper] L% bounds.
    """
    out = results.copy()
    out["rel_frac"] = out["relative_error"] / 100.0
    for level in (75, 90):
        sub_ci = ci[ci["confidence_level"] == level][
            ["step", "tier", "rel_error_lower_bound", "rel_error_upper_bound"]
        ]
        merged = out.merge(sub_ci, on=["step", "tier"], how="left")
        covered = (
            (merged["rel_frac"] >= merged["rel_error_lower_bound"])
            & (merged["rel_frac"] <= merged["rel_error_upper_bound"])
        )
        out[f"covered_{level}"] = covered.values
        # Keep the bounds so the normal-mode chart can draw the band.
        out[f"lower_{level}"] = merged["rel_error_lower_bound"].values
        out[f"upper_{level}"] = merged["rel_error_upper_bound"].values
    return out

def _milestone_step(by_step: pd.DataFrame, threshold: float) -> Optional[int]:
    """Earliest step from which mean abs error stays <= threshold (%) for the rest."""
    steps = sorted(by_step["step"].unique())
    means = by_step.set_index("step")["abs_rel_error"]
    for i, s in enumerate(steps):
        if all(means[ss] <= threshold for ss in steps[i:]):
            return int(s)
    return None


def compute_stats(df: pd.DataFrame) -> dict:
    """Aggregate coverage / accuracy stats across all (idol, border, step) rows."""
    by_step = df.groupby("step").agg(
        abs_rel_error=("abs_rel_error", "mean"),
        cov75=("covered_75", "mean"),
        cov90=("covered_90", "mean"),
        n=("abs_rel_error", "count"),
    ).reset_index()

    last_step = int(by_step["step"].max())
    final_band = df[df["step"] >= max(last_step - 40, by_step["step"].min())]

    worst_idx = df["abs_rel_error"].idxmax()
    worst = df.loc[worst_idx]

    return {
        "by_step": by_step,
        "cov75": df["covered_75"].mean() * 100,
        "cov90": df["covered_90"].mean() * 100,
        "final_err": final_band["abs_rel_error"].mean(),
        "hit5": _milestone_step(by_step, 5.0),
        "hit10": _milestone_step(by_step, 10.0),
        "worst": {
            "idol_id": int(worst["idol_id"]),
            "border": int(worst["border"]),
            "step": int(worst["step"]),
            "rel_error": float(worst["relative_error"]),
        },
        "per_border_err": df.groupby("border")["abs_rel_error"].mean().to_dict(),
        "n_rows": len(df),
    }

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PRIMARY = "#6C5CE7"
GOOD = "#00B894"
WARN = "#E17055"
INK = "#2D3436"
MUTED = "#636E72"


def render(df: pd.DataFrame, stats: dict, *, event_id: int, event_type: int,
           borders: List[int], lang: str, out_path: str, mode: str,
           total_hours: Optional[float] = None,
           event_name: Optional[str] = None) -> None:
    """Dispatch to the normal or anniversary layout."""
    if mode == "anniversary":
        render_anniversary(df, stats, event_id=event_id, event_type=event_type,
                           borders=borders, lang=lang, out_path=out_path,
                           event_name=event_name)
    else:
        render_normal(df, stats, event_id=event_id, event_type=event_type,
                      borders=borders, lang=lang, out_path=out_path,
                      total_hours=total_hours, event_name=event_name)


def render_normal(df: pd.DataFrame, stats: dict, *, event_id: int, event_type: int,
                  borders: List[int], lang: str, out_path: str,
                  total_hours: Optional[float] = None,
                  event_name: Optional[str] = None) -> None:
    """Layout for single-idol events (1 idol x N borders).

    Centerpiece: predicted-final vs actual-final over time, with the 90% and
    75% intervals shaded, one panel per border. The x-axis is hours remaining
    until event end when ``total_hours`` is known, else falls back to step.
    Coverage is a headline number rather than per-step bars (too few samples
    per step to be meaningful).
    """
    t = L[lang]

    def fmt_step(step: Optional[int]) -> str:
        if step is None:
            return t["na"]
        if total_hours is not None:
            rem = (1.0 - step / NORM_EVENT_LENGTH) * total_hours
            return f'{t["remaining_hours"]} {rem:.0f}{t["hours_unit"]}'
        return f'{t["step"]} {step}'

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    n_b = len(borders)
    gs = fig.add_gridspec(
        2, 1 + n_b, height_ratios=[0.85, 2.6],
        left=0.06, right=0.96, top=0.90, bottom=0.12, hspace=0.45, wspace=0.32,
    )

    # ---- Header ----
    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.7, t["title"], fontsize=26, fontweight="bold", color=INK, va="center")
    name_str = event_name or f'{t["event"]} {event_id}'
    border_str = " / ".join(str(b) for b in borders)
    head.text(0, 0.04, f'{name_str}    {t["border"]}: {border_str}    ({t["from_step"]})',
              fontsize=13, color=MUTED, va="center")

    # ---- Callouts (left column of row 2) ----
    stat_ax = fig.add_subplot(gs[1, 0]); stat_ax.axis("off")
    cards = [
        (t["cov90"], f'{stats["cov90"]:.0f}%', GOOD),
        (t["final_err"], f'{stats["final_err"]:.1f}%', INK),
        (t["hit5"], fmt_step(stats["hit5"]), WARN),
        (t["hit10"], fmt_step(stats["hit10"]), PRIMARY),
    ]
    for i, (label, value, color) in enumerate(cards):
        y = 0.94 - i * 0.245
        stat_ax.text(0.02, y, value, fontsize=20, fontweight="bold", color=color, va="top")
        stat_ax.text(0.02, y - 0.10, label, fontsize=10.5, color=MUTED, va="top")

    use_hours = total_hours is not None

    # ---- One predicted-vs-actual panel per border ----
    for bi, b in enumerate(borders):
        ax = fig.add_subplot(gs[1, 1 + bi])
        bdf = df[df["border"] == b].sort_values("step")
        if bdf.empty:
            continue
        steps = bdf["step"].values
        x = _step_to_remaining_hours(steps, total_hours) if use_hours else steps
        pred = bdf["prediction"].values
        actual = float(bdf["actual"].iloc[-1])  # realized final (same across steps)
        # Band consistent with coverage def: actual-in-band <=> rel_error in [lower, upper]
        low90 = pred / (1 + bdf["upper_90"].values)
        high90 = pred / (1 + bdf["lower_90"].values)
        low75 = pred / (1 + bdf["upper_75"].values)
        high75 = pred / (1 + bdf["lower_75"].values)

        ax.fill_between(x, low90, high90, color=GOOD, alpha=0.15, label=t["band90"])
        ax.fill_between(x, low75, high75, color=PRIMARY, alpha=0.20, label=t["band75"])
        # Fixed tolerance bands around the realized final (reference, not prediction-driven)
        ax.axhspan(actual * 0.90, actual * 1.10, color=WARN, alpha=0.06, label=t["err10"])
        ax.axhspan(actual * 0.95, actual * 1.05, color=WARN, alpha=0.10, label=t["err5"])
        ax.plot(x, pred, color=PRIMARY, linewidth=2, marker="o", markersize=3, label=t["predicted"])
        ax.axhline(actual, color=WARN, linestyle="--", linewidth=1.6, label=t["actual"])
        ax.set_title(f'{t["border"]} {b}', fontsize=13, color=INK)
        ax.set_xlabel(t["remaining_hours"] if use_hours else t["step"], fontsize=10, color=MUTED)
        if use_hours:
            ax.invert_xaxis()  # time flows left->right toward event end (0h remaining)
        if bi == 0:
            ax.set_ylabel(t["score"], fontsize=10, color=MUTED)
        ax.legend(fontsize=8.5, frameon=False, loc="best")
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="y", style="plain")

    # ---- Worst-case + disclaimer footer ----
    w = stats["worst"]
    fig.text(0.06, 0.045,
             f'{t["worst"]}: {w["rel_error"]:+.1f}%  ({t["border"]} {w["border"]}, {fmt_step(w["step"])})',
             fontsize=10, color=WARN)
    fig.text(0.06, 0.015, t["disclaimer"], fontsize=9, color=MUTED)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved report to {out_path}")


def render_anniversary(df: pd.DataFrame, stats: dict, *, event_id: int, event_type: int,
           borders: List[int], lang: str, out_path: str,
           event_name: Optional[str] = None) -> None:
    t = L[lang]
    is_anniversary = event_type == 5

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)  # 1280x720, 16:9
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.9, 2.0, 1.3], width_ratios=[1, 1, 1],
        left=0.06, right=0.96, top=0.92, bottom=0.10, hspace=0.55, wspace=0.30,
    )

    # ---- Header ----
    head = fig.add_subplot(gs[0, :]); head.axis("off")
    head.text(0, 0.7, t["title"], fontsize=26, fontweight="bold", color=INK, va="center")
    name_str = event_name or f'{t["event"]} {event_id}'
    border_str = " / ".join(f"{b}" for b in borders)
    head.text(0, 0.05, f'{name_str}    {t["border"]}: {border_str}    ({t["from_step"]})',
              fontsize=13, color=MUTED, va="center")

    # ---- Stat callouts (row 2, left column split into 2x2 via text) ----
    stat_ax = fig.add_subplot(gs[1, 0]); stat_ax.axis("off")
    cards = [
        (t["cov90"], f'{stats["cov90"]:.0f}%', GOOD),
        (t["cov75"], f'{stats["cov75"]:.0f}%', PRIMARY),
        (t["final_err"], f'{stats["final_err"]:.1f}%', INK),
        (t["hit5"], (f'{t["step"]} {stats["hit5"]}' if stats["hit5"] is not None else t["na"]), WARN),
    ]
    for i, (label, value, color) in enumerate(cards):
        y = 0.92 - i * 0.245
        stat_ax.text(0.02, y, value, fontsize=24, fontweight="bold", color=color, va="top")
        stat_ax.text(0.02, y - 0.10, label, fontsize=11, color=MUTED, va="top")

    # ---- Convergence chart (row 2, center+right) ----
    conv = fig.add_subplot(gs[1, 1:])
    by_step = stats["by_step"]
    for b in borders:
        bdf = df[df["border"] == b].groupby("step")["abs_rel_error"].mean()
        conv.plot(bdf.index, bdf.values, marker="o", markersize=4,
                  linewidth=2, label=f"{t['border']} {b}")
    conv.axhline(5, color=GOOD, linestyle="--", linewidth=1, alpha=0.7)
    conv.axhline(10, color=WARN, linestyle="--", linewidth=1, alpha=0.7)
    conv.set_title(t["convergence"], fontsize=14, color=INK)
    conv.set_xlabel(t["step"], fontsize=11, color=MUTED)
    conv.set_ylabel(t["mean_abs_err"], fontsize=11, color=MUTED)
    conv.legend(fontsize=10, frameon=False)
    conv.grid(True, alpha=0.25)
    conv.set_ylim(bottom=0)

    # ---- Bottom-left: coverage-by-step bars ----
    cov_ax = fig.add_subplot(gs[2, :2])
    width = (by_step["step"].diff().median() or 20) * 0.35
    cov_ax.bar(by_step["step"] - width / 2, by_step["cov90"] * 100, width=width,
               color=GOOD, alpha=0.85, label="90%")
    cov_ax.bar(by_step["step"] + width / 2, by_step["cov75"] * 100, width=width,
               color=PRIMARY, alpha=0.85, label="75%")
    cov_ax.axhline(90, color=GOOD, linestyle=":", linewidth=1)
    cov_ax.axhline(75, color=PRIMARY, linestyle=":", linewidth=1)
    cov_ax.set_title(t["coverage_by_step"], fontsize=13, color=INK)
    cov_ax.set_xlabel(t["step"], fontsize=10, color=MUTED)
    cov_ax.set_ylim(0, 105)
    cov_ax.legend(fontsize=9, frameon=False, ncol=2)
    cov_ax.grid(True, axis="y", alpha=0.25)

    # ---- Bottom-right: worst-case + idol-count context ----
    info = fig.add_subplot(gs[2, 2]); info.axis("off")
    w = stats["worst"]
    worst_who = (f'{t["idol"]} {w["idol_id"]}  ' if is_anniversary else "")
    info.text(0.0, 0.95, t["worst"], fontsize=12, color=MUTED, va="top")
    info.text(0.0, 0.72, f'{w["rel_error"]:+.1f}%', fontsize=22, fontweight="bold",
              color=WARN, va="top")
    info.text(0.0, 0.52, f'{worst_who}{t["border"]} {w["border"]}  @ {t["step"]} {w["step"]}',
              fontsize=10, color=MUTED, va="top")
    info.text(0.0, 0.30,
              (f'{t["hit10"]}: {t["step"]} {stats["hit10"]}'
               if stats["hit10"] is not None else f'{t["hit10"]}: {t["na"]}'),
              fontsize=10, color=INK, va="top")

    # ---- Footer disclaimer ----
    fig.text(0.06, 0.025, t["disclaimer"], fontsize=9, color=MUTED)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved report to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BORDERS = {5: [100, 1000]}  # anniversary; others default to [100, 2500]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event-id", type=int, required=True)
    p.add_argument("--event-type", type=int, default=None,
                   help="event type; auto-detected from event_info if omitted")
    p.add_argument("--sub", type=int, default=None, help="sub_event_type (default 1; needed for type 3/4 bonus)")
    p.add_argument("--borders", type=int, nargs="+", default=None,
                   help="override borders; default 100/1000 for type 5, else 100/2500")
    p.add_argument("--min-step", type=int, default=70, help="report coverage from this step (default 70)")
    p.add_argument("--source", choices=["backtest", "csv"], default="backtest",
                   help="backtest: run predictions in-memory (default, one-click); "
                        "csv: read precomputed batch_knn_results_*.csv")
    p.add_argument("--results-dir", default="test_results")
    p.add_argument("--data-dir", default="test_data",
                   help="dir with event_info + border_info (for backtest, duration, type detection)")
    p.add_argument("--event-hours", type=float, default=None,
                   help="override total event duration in hours (else read from event_info)")
    p.add_argument("--ci-dir", default="confidence_intervals")
    p.add_argument("--ci-source", choices=["auto", "local", "r2"], default="auto",
                   help="where to read CI: auto=auto-resolve history snapshot from event end, "
                        "local=local files, r2=ci/latest")
    p.add_argument("--ci-history-ts", default=None,
                   help="force reading CI from ci/history/<ts>/ (overrides --ci-source)")
    p.add_argument("--event-name", default=None, help="display name for the header")
    p.add_argument("--lang", choices=["jp", "en"], default="jp")
    p.add_argument("--mode", choices=["auto", "normal", "anniversary"], default="auto",
                   help="report layout; auto picks anniversary for event-type 5, else normal")
    p.add_argument("--out", default=None, help="output PNG path")
    p.add_argument("--debug-table", action="store_true",
                   help="print per-(border, step) prediction/actual/band/coverage numbers")
    args = p.parse_args()

    # Resolve event_type / sub. sub is derived from event_info's
    # internal_event_type unless explicitly overridden with --sub.
    detected_et, detected_sub = _detect_event_type_sub(args.data_dir, args.event_id, args.sub)
    event_type = args.event_type if args.event_type is not None else int(detected_et)
    sub = detected_sub
    sub_int = int(sub)
    print(f"Using event_type={event_type}, sub_event_type={sub_int} for event {args.event_id}")

    borders = args.borders or DEFAULT_BORDERS.get(event_type, [100, 2500])
    mode = args.mode
    if mode == "auto":
        mode = "anniversary" if event_type == 5 else "normal"

    if args.lang == "jp":
        font = _setup_jp_font()
        if font is None:
            print("WARNING: no CJK font found; Japanese labels may render as boxes. "
                  "Use --lang en or install a Japanese font.")

    # Resolve which CI table to score against.
    history_ts = args.ci_history_ts
    if history_ts is None and args.ci_source == "auto":
        end_utc = _event_end_utc(args.data_dir, args.event_id)
        history_ts = _resolve_ci_history_ts(end_utc)
        if history_ts is not None:
            print(f"Auto-selected CI history snapshot ci/history/{history_ts}/ "
                  f"(first promotion after event end).")
        else:
            print("No CI promotion after event end; using ci/latest.")

    frames = []
    for b in borders:
        if args.source == "csv":
            res = _load_results(args.results_dir, event_type, sub_int, b,
                                args.event_id, args.min_step)
        else:
            res = _run_backtest(float(event_type), sub, b, args.event_id, args.data_dir)
            res = res[res["step"] >= args.min_step].copy()
            if res.empty:
                raise ValueError(f"No predictions at step>={args.min_step} for border {b}")

        if history_ts is not None:
            ci = _load_ci(args.ci_dir, event_type, sub_int, b,
                          allow_r2=True, r2_history_ts=history_ts)
        elif args.ci_source == "local":
            ci = _load_ci(args.ci_dir, event_type, sub_int, b, allow_r2=False)
        else:  # r2, or auto with no later promotion -> ci/latest
            ci = _load_ci("___force_r2___", event_type, sub_int, b, allow_r2=True)
        frames.append(_attach_coverage(res, ci))
    df = pd.concat(frames, ignore_index=True)

    stats = compute_stats(df)
    total_hours = args.event_hours
    if total_hours is None:
        total_hours = _event_hours(args.data_dir, args.event_id)
        if total_hours is None:
            print(f"Could not determine event duration (no event_info in {args.data_dir} "
                  f"and no --event-hours); x-axis will use step instead of remaining hours.")

    out = args.out or f"test_results/event_report_{args.event_id}_{event_type}_{sub_int}.png"

    if args.debug_table:
        dbg = df.copy()
        dbg["band_low_90"] = dbg["prediction"] / (1 + dbg["upper_90"])
        dbg["band_high_90"] = dbg["prediction"] / (1 + dbg["lower_90"])
        dbg["band_low_75"] = dbg["prediction"] / (1 + dbg["upper_75"])
        dbg["band_high_75"] = dbg["prediction"] / (1 + dbg["lower_75"])
        if total_hours is not None:
            dbg["rem_h"] = (1 - dbg["step"] / NORM_EVENT_LENGTH) * total_hours
        cols = ["border", "step"] + (["rem_h"] if total_hours is not None else []) + [
            "idol_id", "prediction", "actual", "relative_error",
            "lower_75", "upper_75", "lower_90", "upper_90",
            "band_low_90", "band_high_90", "covered_75", "covered_90",
        ]
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
        dbg_sorted = dbg.sort_values(["border", "step"])[cols]
        for b in borders:
            print(f"\n=== border {b} ===")
            print(dbg_sorted[dbg_sorted["border"] == b].to_string(index=False))
        dbg_path = (Path(out).parent /
                    f"event_report_debug_{args.event_id}_{event_type}_{sub_int}.csv")
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg_sorted.to_csv(dbg_path, index=False)
        print(f"\nDebug metrics CSV: {dbg_path}")

    render(df, stats, event_id=args.event_id, event_type=event_type,
           borders=borders, lang=args.lang, out_path=out, mode=mode,
           total_hours=total_hours, event_name=args.event_name)

    # Also echo the headline numbers to stdout for quick copy-paste.
    print(f"  90% coverage: {stats['cov90']:.1f}%   75% coverage: {stats['cov75']:.1f}%")
    print(f"  final-stretch mean abs error: {stats['final_err']:.2f}%")
    print(f"  reached +/-5% by step: {stats['hit5']}   +/-10% by step: {stats['hit10']}")
    print(f"  worst miss: {stats['worst']}")


if __name__ == "__main__":
    main()

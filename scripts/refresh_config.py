#!/usr/bin/env python3
"""
Refresh dynamic config + CI files.

Replaces the manual workflow:
    1) run r2_downloader.py
    2) manually run batch_knn_test.py per (event_type, sub_types, border)
    3) run analyze_percentiles.py
    4) commit CI files to git

Modes
-----

``local`` - intended for debugging. Event/border info is downloaded to
``local_cache/`` and cached; batch results, CI files, diff, and manifest
are written to ``local_cache/output/<generation_id>/`` only. Nothing is
uploaded. A second run reuses cached event/border data.

``standard`` - intended for production refresh. Always pulls fresh
event/border data from R2 into ``local_cache/`` and points
``BatchKNNTester`` at that directory. Outputs (CI csvs, new dynamic
config, diff, manifest) go only to R2 staging paths. On exit, the local
cache is removed so no stale data lingers. ``test_data/`` is never
touched by this script.

Every run gets a generation_id like ``20260503T120000Z-ab12cd`` which is
embedded in every R2 path for the run.

Usage:
    python3 refresh_config.py --mode local
    python3 refresh_config.py --mode standard
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scripts.analyze_percentiles import (
    analyze_confidence_intervals_for_group,
    interpolate_confidence_intervals,
)
from scripts.batch_knn_test import BatchKNNTester
from src.storage.dynamic_config import (
    DynamicConfig,
    DynamicConfigEntry,
    R2_CONFIG_LATEST_KEY,
    load_dynamic_config_from_r2,
    r2_ci_staging_key,
    r2_staging_config_key,
    r2_staging_diff_key,
    r2_staging_manifest_key,
    upload_dynamic_config_to_r2,
)
from src.knn.config import GROUP_CONFIGS
from src.storage.loader import BUCKET_NAME, load_ci_csv_from_r2, upload_ci_csv_to_r2
from logger_config import setup_logging
from src.storage.notifier import send_notification
from src.storage.r2_client import R2Client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-event-type lookback for min_event_id.
# Mirrors the old ``get_min_event_id(internal_event_type)`` constants but
# keyed on ``event_type`` (the top-level column), not the internal variant.
LOOK_BACK_EVENT_CNT_BY_TYPE: Dict[float, int] = {
    3.0: 225,   # Theater / Tiara / Trust and their specials
    4.0: 175,   # Tour / Tour Bingo (incl. specials)
    5.0: 225,   # Anniversary
    11.0: 250,  # Tune
    13.0: 275,  # Tale / Team / Time
}


def lookback_for_event_type(event_type: float) -> int:
    """Raise if the event_type has no configured lookback — no silent defaults."""
    if event_type not in LOOK_BACK_EVENT_CNT_BY_TYPE:
        raise KeyError(
            f"No LOOK_BACK_EVENT_CNT configured for event_type={event_type}. "
            f"Add it to LOOK_BACK_EVENT_CNT_BY_TYPE in refresh_config.py."
        )
    return LOOK_BACK_EVENT_CNT_BY_TYPE[event_type]
NORM_EVENT_LENGTH = 300
DEFAULT_STEPS = [70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290]
DEFAULT_CONFIDENCE_LEVELS = [75, 90]
DEFAULT_START_STEP = 55
DEFAULT_END_STEP = 299

LOCAL_CACHE_DIR = Path("local_cache")
LOCAL_EVENT_INFO_DIR = LOCAL_CACHE_DIR / "event_info"
LOCAL_BORDER_INFO_DIR = LOCAL_CACHE_DIR / "border_info"
LOCAL_OUTPUT_DIR = LOCAL_CACHE_DIR / "output"


# ---------------------------------------------------------------------------
# Generation id
# ---------------------------------------------------------------------------

def make_generation_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


# ---------------------------------------------------------------------------
# Data fetching / caching
# ---------------------------------------------------------------------------

def ensure_local_cache(
    r2_client: R2Client,
    groups: List[Tuple[float, Tuple[float, ...], float]],
    force_refresh: bool = False,
) -> None:
    """Download event_info + all border_info needed by the groups into the local cache.

    ``force_refresh=False`` (local mode): keep files that already exist; only
    fetch missing ones. Subsequent runs are cheap.

    ``force_refresh=True`` (standard mode): delete cached files first so we
    always pull the latest data from R2.
    """
    if force_refresh and LOCAL_CACHE_DIR.exists():
        logging.info(f"force_refresh=True, removing existing {LOCAL_CACHE_DIR}")
        import shutil
        shutil.rmtree(LOCAL_CACHE_DIR)

    LOCAL_EVENT_INFO_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_BORDER_INFO_DIR.mkdir(parents=True, exist_ok=True)

    # Event info
    event_info_local = LOCAL_EVENT_INFO_DIR / "event_info_all.csv"
    if not event_info_local.exists():
        logging.info("Downloading event_info to local cache")
        obj = r2_client.get_object(BUCKET_NAME, "event_info/event_info_all.csv")
        event_info_local.write_bytes(obj["Body"].read())

    event_info = pd.read_csv(event_info_local)
    event_ids_needed = set(event_info["event_id"].astype(int).tolist())
    borders_needed = {int(b) for (_et, _sub, b) in groups}
    idol_ids_needed = {0}  # extend to range(1,53) if anniversary groups get added

    # Border files: one call per missing file. That's the same pattern r2_downloader.py uses.
    for eid in event_ids_needed:
        for iid in idol_ids_needed:
            for border in borders_needed:
                name = f"border_info_{eid}_{iid}_{border}.csv"
                local_path = LOCAL_BORDER_INFO_DIR / name
                if local_path.exists():
                    continue
                try:
                    obj = r2_client.get_object(BUCKET_NAME, f"border_info/{name}")
                    local_path.write_bytes(obj["Body"].read())
                    logging.debug(f"cached {name}")
                except Exception:
                    # Many (eid, iid, border) combos don't exist; that's fine.
                    pass


# ---------------------------------------------------------------------------
# Per-group pipeline: batch_knn_test -> analyze_percentiles
# ---------------------------------------------------------------------------

def run_batch_knn_for_group(
    event_type: float,
    sub_event_types: Tuple[float, ...],
    border: float,
    data_dir: str,
) -> pd.DataFrame:
    """Run batch KNN test over recent events and return the results dataframe."""
    look_back_event_cnt = lookback_for_event_type(event_type)
    tester = BatchKNNTester(
        event_type=event_type,
        sub_event_types=list(sub_event_types),
        border=border,
        look_back_event_cnt=look_back_event_cnt,
        data_dir=data_dir,
    )
    raw_data, _event_name_map = tester.load_and_process_data()
    eid_to_len_boost_ratio = tester.get_len_and_boost_ratio(raw_data)
    test_event_ids = tester.find_matching_events(raw_data=raw_data, recent_count=None, test_event_ids=None)
    logging.info(f"  testing {len(test_event_ids)} events for group ({event_type},{sub_event_types},{border})")

    length_to_df_data = tester.get_normalized_full_and_part_data(
        raw_data=raw_data,
        steps=DEFAULT_STEPS,
        eid_to_len_boost_ratio=eid_to_len_boost_ratio,
    )
    temp_idol_id = 1 if event_type == 5 else 0
    results = tester.run_predictions(
        test_event_ids=test_event_ids,
        test_steps=DEFAULT_STEPS,
        temp_idol_id=temp_idol_id,
        raw_data=raw_data,
        length_to_df_data=length_to_df_data,
        eid_to_len_boost_ratio=eid_to_len_boost_ratio,
    )
    return pd.DataFrame(results)


def build_ci_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Run the percentile + step-interpolation step over batch results."""
    step_intervals = analyze_confidence_intervals_for_group(results_df, DEFAULT_CONFIDENCE_LEVELS)
    if not step_intervals:
        return pd.DataFrame()
    interpolated = interpolate_confidence_intervals(
        step_intervals, DEFAULT_CONFIDENCE_LEVELS, DEFAULT_START_STEP, DEFAULT_END_STEP,
    )
    out_rows = []
    for step in range(DEFAULT_START_STEP, DEFAULT_END_STEP + 1):
        for cl in DEFAULT_CONFIDENCE_LEVELS:
            if (step, cl) in interpolated:
                lower, upper = interpolated[(step, cl)]
                out_rows.append({
                    "step": step,
                    "confidence_level": cl,
                    "rel_error_lower_bound": lower,
                    "rel_error_upper_bound": upper,
                })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Dynamic config generation
# ---------------------------------------------------------------------------

def compute_dynamic_config(
    groups: List[Tuple[float, Tuple[float, ...], float]],
    event_info: pd.DataFrame,
    generation_id: str,
) -> DynamicConfig:
    """Compute a fresh DynamicConfig with per-event-type ``min_event_id``.

    ``min_event_id = latest_past_event_id - lookback_for_event_type(event_type)``.
    Ongoing events (``end_at >= now``) are excluded so a currently-running
    event doesn't shift the window forward before it's actually done.
    """
    ei = event_info.copy()
    ei["end_at"] = pd.to_datetime(ei["end_at"])
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    past = ei[ei["end_at"] < now]
    dropped = len(ei) - len(past)
    if dropped:
        logging.info(f"compute_dynamic_config: ignoring {dropped} ongoing event(s)")

    latest_event_id = int(past["event_id"].max())
    entries: List[DynamicConfigEntry] = []
    for (et, sub, border) in groups:
        lookback = lookback_for_event_type(et)
        min_event_id = latest_event_id - lookback
        logging.info(
            f"  ({et}, {sub}, {border}) latest={latest_event_id} lookback={lookback} "
            f"-> min_event_id={min_event_id}"
        )
        entries.append(DynamicConfigEntry(
            event_type=et, sub_event_types=sub, border=border, min_event_id=min_event_id,
        ))
    return DynamicConfig(generation_id=generation_id, entries=entries)


# ---------------------------------------------------------------------------
# Diff generation
# ---------------------------------------------------------------------------

def diff_dynamic_configs(old: DynamicConfig, new: DynamicConfig) -> str:
    """Produce a human-readable markdown diff."""
    old_map = old.by_key()
    new_map = new.by_key()
    all_keys = sorted(set(old_map) | set(new_map))

    lines = [
        f"# Dynamic config diff",
        f"- old generation_id: `{old.generation_id}`",
        f"- new generation_id: `{new.generation_id}`",
        "",
        "| event_type | sub_event_types | border | old min_event_id | new min_event_id | delta |",
        "|------------|-----------------|--------|------------------|------------------|-------|",
    ]
    for k in all_keys:
        o = old_map.get(k)
        n = new_map.get(k)
        et, sub, border = k
        ov = o.min_event_id if o else None
        nv = n.min_event_id if n else None
        delta = (nv - ov) if (ov is not None and nv is not None) else "-"
        lines.append(f"| {et} | {sub} | {border} | {ov} | {nv} | {delta} |")
    return "\n".join(lines) + "\n"


def diff_ci_csvs(old_df: pd.DataFrame, new_df: pd.DataFrame, label: str) -> str:
    """Per-step summary of changes in CI bounds."""
    if old_df.empty and new_df.empty:
        return f"## {label}: both empty\n\n"
    if old_df.empty:
        return f"## {label}: new file (was empty)\n\n"
    if new_df.empty:
        return f"## {label}: new file is empty\n\n"

    merged = old_df.merge(
        new_df, on=["step", "confidence_level"], suffixes=("_old", "_new"), how="outer",
    )
    merged["lower_delta"] = merged["rel_error_lower_bound_new"] - merged["rel_error_lower_bound_old"]
    merged["upper_delta"] = merged["rel_error_upper_bound_new"] - merged["rel_error_upper_bound_old"]
    grouped = (
        merged.groupby("confidence_level")[["lower_delta", "upper_delta"]]
        .agg(["mean", "min", "max", "count"])
    )
    lines = [f"## {label}", "", grouped.to_markdown(), ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Output writers (local vs. R2)
# ---------------------------------------------------------------------------

def write_local_outputs(
    gen_dir: Path,
    new_cfg: DynamicConfig,
    ci_by_group: Dict[Tuple[float, Tuple[float, ...], float], pd.DataFrame],
    config_diff: str,
    ci_diff: str,
    manifest: dict,
) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "dynamic_config.json").write_text(new_cfg.to_json())
    (gen_dir / "manifest.json").write_text(
        pd.Series(manifest).to_json(indent=2)
    )
    (gen_dir / "diff.md").write_text(config_diff + "\n" + ci_diff)
    ci_dir = gen_dir / "ci"
    ci_dir.mkdir(exist_ok=True)
    for (et, sub, border), df in ci_by_group.items():
        fname = f"confidence_intervals_{int(et)}_{int(sub[0])}_{int(border)}.csv"
        df.to_csv(ci_dir / fname, index=False)
    logging.info(f"Wrote local outputs to {gen_dir}")


def upload_standard_outputs(
    r2_client: R2Client,
    generation_id: str,
    new_cfg: DynamicConfig,
    ci_by_group: Dict[Tuple[float, Tuple[float, ...], float], pd.DataFrame],
    config_diff: str,
    ci_diff: str,
    manifest: dict,
) -> None:
    upload_dynamic_config_to_r2(r2_client, BUCKET_NAME, r2_staging_config_key(generation_id), new_cfg)
    r2_client.put_object(
        bucket_name=BUCKET_NAME,
        key=r2_staging_manifest_key(generation_id),
        body=pd.Series(manifest).to_json(indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    r2_client.put_object(
        bucket_name=BUCKET_NAME,
        key=r2_staging_diff_key(generation_id),
        body=(config_diff + "\n" + ci_diff).encode("utf-8"),
        ContentType="text/markdown",
    )
    for (et, sub, border), df in ci_by_group.items():
        upload_ci_csv_to_r2(
            r2_client,
            r2_ci_staging_key(generation_id, et, sub[0], border),
            df,
        )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(mode: str, notify: bool = True) -> None:
    generation_id = make_generation_id()
    logging.info(f"=== generation_id={generation_id}, mode={mode} ===")

    r2_client = R2Client()

    # Enumerate groups from the source GroupConfigs
    groups = sorted(GROUP_CONFIGS.keys())
    logging.info(f"Refreshing {len(groups)} groups")

    # Populate the cache. Local mode reuses; standard mode forces refresh.
    force = mode == "standard"
    if force:
        logging.info(f"Pulling fresh data to {LOCAL_CACHE_DIR} for standard-mode batch_knn_test")
    ensure_local_cache(r2_client, groups, force_refresh=force)

    # Read event_info from the cache (single source of truth in both modes).
    event_info = pd.read_csv(LOCAL_EVENT_INFO_DIR / "event_info_all.csv")

    # Build the new dynamic config
    new_cfg = compute_dynamic_config(groups, event_info, generation_id)

    # Load the current latest for diffing
    try:
        old_cfg = load_dynamic_config_from_r2(r2_client, BUCKET_NAME, R2_CONFIG_LATEST_KEY)
    except Exception as ex:
        logging.warning(f"No existing latest dynamic_config; empty baseline ({ex})")
        old_cfg = DynamicConfig(generation_id="none", entries=[])

    config_diff = diff_dynamic_configs(old_cfg, new_cfg)

    try:
        # Run batch KNN + CI per group. BatchKNNTester is pointed at
        # LOCAL_CACHE_DIR explicitly; refresh_config never touches test_data/.
        ci_by_group: Dict[Tuple[float, Tuple[float, ...], float], pd.DataFrame] = {}
        ci_diff_parts: List[str] = []
        for (et, sub, border) in groups:
            logging.info(f"Running batch_knn_test for ({et}, {sub}, {border})")
            try:
                results_df = run_batch_knn_for_group(et, sub, border, data_dir=str(LOCAL_CACHE_DIR))
            except Exception as ex:
                logging.error(f"batch_knn_test failed for ({et}, {sub}, {border}): {ex}", exc_info=True)
                continue
            ci_df = build_ci_from_results(results_df)
            ci_by_group[(et, sub, border)] = ci_df

            # Compare with latest CI in R2 (if any)
            try:
                old_ci = load_ci_csv_from_r2(r2_client, et, sub[0], border)
            except Exception:
                old_ci = pd.DataFrame()
            label = f"CI diff: ({et}, {sub}, {border})"
            ci_diff_parts.append(diff_ci_csvs(old_ci, ci_df, label))

        ci_diff = "\n".join(ci_diff_parts)

        manifest = {
            "generation_id": generation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "look_back_event_cnt_by_type": LOOK_BACK_EVENT_CNT_BY_TYPE,
            "num_groups": len(groups),
            "ci_groups_generated": len(ci_by_group),
        }

        if mode == "local":
            write_local_outputs(
                LOCAL_OUTPUT_DIR / generation_id,
                new_cfg, ci_by_group, config_diff, ci_diff, manifest,
            )
        else:
            upload_standard_outputs(
                r2_client, generation_id, new_cfg, ci_by_group, config_diff, ci_diff, manifest,
            )
            if notify:
                _notify_standard_run_completed(generation_id, len(ci_by_group), len(groups))
    finally:
        # Standard mode: wipe the ephemeral cache. Local mode keeps it.
        if mode == "standard" and LOCAL_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(LOCAL_CACHE_DIR)
            logging.info(f"Removed {LOCAL_CACHE_DIR}")

    logging.info(f"=== done: generation_id={generation_id} ===")


def _notify_standard_run_completed(generation_id: str, groups_done: int, groups_total: int) -> None:
    """Email the operator so they know a generation is ready to promote."""
    subject = f"[refresh_config] staged generation {generation_id}"
    body = (
        f"refresh_config --mode standard completed.\n\n"
        f"generation_id: {generation_id}\n"
        f"groups: {groups_done}/{groups_total} produced CI\n\n"
        f"Staged files in R2:\n"
        f"  config/staging/{generation_id}/dynamic_config.json\n"
        f"  config/staging/{generation_id}/manifest.json\n"
        f"  config/staging/{generation_id}/diff.md\n"
        f"  ci/staging/{generation_id}/confidence_intervals_*.csv\n\n"
        f"Review diff.md, then promote with:\n"
        f"  python3 -m scripts.promote_config --generation-id {generation_id}\n"
    )
    send_notification(subject, body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "standard"], required=True)
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="In standard mode, skip emailing NOTIFICATION_TARGET after a successful run",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        run(args.mode, notify=not args.no_notify)
    except Exception as ex:
        logging.error(f"Refresh failed: {ex}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

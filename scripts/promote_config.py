#!/usr/bin/env python3
"""
Promote a staged generation to ``latest/``.

Before overwriting the in-use config and CI files, back up what's there
into ``config/history/<ts>/`` and ``ci/history/<ts>/`` so you always have
a rollback point.

Usage:
    python3 promote_config.py --generation-id 20260503T120000Z-ab12cd
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import List

from src.storage.dynamic_config import (
    R2_CONFIG_LATEST_KEY,
    R2_CONFIG_LATEST_MANIFEST_KEY,
    r2_staging_config_key,
    r2_staging_manifest_key,
)
from src.storage.loader import BUCKET_NAME
from logger_config import setup_logging
from src.storage.r2_client import R2Client


def _copy_object(r2: R2Client, src_key: str, dst_key: str) -> None:
    obj = r2.get_object(BUCKET_NAME, src_key)
    body = obj["Body"].read()
    content_type = obj.get("ContentType", "application/octet-stream")
    r2.put_object(bucket_name=BUCKET_NAME, key=dst_key, body=body, ContentType=content_type)
    logging.info(f"copied {src_key} -> {dst_key} ({len(body)} bytes)")


def _list_keys(r2: R2Client, prefix: str) -> List[str]:
    resp = r2.list_objects(BUCKET_NAME, prefix)
    return [o["Key"] for o in resp.get("Contents", [])]


def promote(generation_id: str) -> None:
    r2 = R2Client()

    # 1. Verify staging exists
    staging_cfg_key = r2_staging_config_key(generation_id)
    try:
        r2.get_object(BUCKET_NAME, staging_cfg_key)
    except Exception as ex:
        raise FileNotFoundError(f"No staged config at {staging_cfg_key}: {ex}")

    staging_ci_prefix = f"ci/staging/{generation_id}/"
    staged_ci_keys = _list_keys(r2, staging_ci_prefix)
    if not staged_ci_keys:
        raise FileNotFoundError(f"No staged CI files under {staging_ci_prefix}")

    # 2. Snapshot current latest into history/<ts>/
    promote_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_config_prefix = f"config/history/{promote_ts}/"
    history_ci_prefix = f"ci/history/{promote_ts}/"

    logging.info(f"Snapshotting current latest into history/{promote_ts}/")
    latest_cfg_keys = _list_keys(r2, "config/latest/")
    for k in latest_cfg_keys:
        _copy_object(r2, k, history_config_prefix + k[len("config/latest/"):])
    latest_ci_keys = _list_keys(r2, "ci/latest/")
    for k in latest_ci_keys:
        _copy_object(r2, k, history_ci_prefix + k[len("ci/latest/"):])

    # 3. Copy staging/<gen>/ into latest/
    logging.info(f"Promoting generation {generation_id} into latest/")
    _copy_object(r2, staging_cfg_key, R2_CONFIG_LATEST_KEY)
    _copy_object(r2, r2_staging_manifest_key(generation_id), R2_CONFIG_LATEST_MANIFEST_KEY)

    for src in staged_ci_keys:
        dst = "ci/latest/" + src[len(staging_ci_prefix):]
        _copy_object(r2, src, dst)

    logging.info(f"Promote complete. history backup at history/{promote_ts}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", required=True, help="staging generation id to promote")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        promote(args.generation_id)
    except Exception as ex:
        logging.error(f"Promote failed: {ex}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Dynamic KNN configuration overlay.

Fields that are periodically refreshed (today: only ``min_event_id``) are
kept out of the static ``GroupConfig`` in ``knn_config.py`` and stored in
R2 instead. Production reads from R2 at startup; a refresh script writes
new values to a staging path, and a promote script moves staged values to
the "latest" path that production reads.

JSON schema (``config/latest/dynamic_config.json``):

    {
      "generation_id": "YYYYMMDDTHHMMSSZ-<short>",
      "entries": [
        {"event_type": 11.0, "sub_event_types": [1.0], "border": 2500.0, "min_event_id": 200},
        ...
      ]
    }

Only ``min_event_id`` is overlaid today. New dynamic fields can be added
to ``DynamicConfigEntry`` without breaking readers (unknown fields on
read are ignored; missing fields on a stored entry are treated as
"not overridden").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

GroupKey = Tuple[float, Tuple[float, ...], float]  # (event_type, sub_event_types, border)


@dataclass
class DynamicConfigEntry:
    event_type: float
    sub_event_types: Tuple[float, ...]
    border: float
    min_event_id: Optional[int] = None

    def key(self) -> GroupKey:
        return (self.event_type, self.sub_event_types, self.border)

    def to_json(self) -> dict:
        return {
            "event_type": self.event_type,
            "sub_event_types": list(self.sub_event_types),
            "border": self.border,
            "min_event_id": self.min_event_id,
        }

    @staticmethod
    def from_json(d: dict) -> "DynamicConfigEntry":
        return DynamicConfigEntry(
            event_type=float(d["event_type"]),
            sub_event_types=tuple(float(x) for x in d["sub_event_types"]),
            border=float(d["border"]),
            min_event_id=(int(d["min_event_id"]) if d.get("min_event_id") is not None else None),
        )


@dataclass
class DynamicConfig:
    generation_id: str
    entries: List[DynamicConfigEntry]

    def by_key(self) -> Dict[GroupKey, DynamicConfigEntry]:
        return {e.key(): e for e in self.entries}

    def to_json(self) -> str:
        return json.dumps(
            {
                "generation_id": self.generation_id,
                "entries": [e.to_json() for e in self.entries],
            },
            indent=2,
            sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> "DynamicConfig":
        payload = json.loads(text)
        return DynamicConfig(
            generation_id=str(payload.get("generation_id", "unknown")),
            entries=[DynamicConfigEntry.from_json(x) for x in payload.get("entries", [])],
        )


# ---------------------------------------------------------------------------
# R2 paths (callers pass these explicitly)
# ---------------------------------------------------------------------------

R2_CONFIG_LATEST_KEY = "config/latest/dynamic_config.json"
R2_CONFIG_LATEST_MANIFEST_KEY = "config/latest/manifest.json"


def r2_staging_config_key(generation_id: str) -> str:
    return f"config/staging/{generation_id}/dynamic_config.json"


def r2_staging_manifest_key(generation_id: str) -> str:
    return f"config/staging/{generation_id}/manifest.json"


def r2_staging_diff_key(generation_id: str) -> str:
    return f"config/staging/{generation_id}/diff.md"


def r2_ci_latest_key(event_type: float, sub: float, border: float) -> str:
    return f"ci/latest/confidence_intervals_{int(event_type)}_{int(sub)}_{int(border)}.csv"


def r2_ci_staging_key(generation_id: str, event_type: float, sub: float, border: float) -> str:
    return (
        f"ci/staging/{generation_id}/"
        f"confidence_intervals_{int(event_type)}_{int(sub)}_{int(border)}.csv"
    )


# ---------------------------------------------------------------------------
# R2 I/O helpers
# ---------------------------------------------------------------------------

def load_dynamic_config_from_r2(r2_client, bucket_name: str, key: str = R2_CONFIG_LATEST_KEY) -> DynamicConfig:
    logging.info(f"Loading dynamic config from r2://{bucket_name}/{key}")
    obj = r2_client.get_object(bucket_name, key)
    body = obj["Body"].read().decode("utf-8")
    return DynamicConfig.from_json(body)


def upload_dynamic_config_to_r2(r2_client, bucket_name: str, key: str, cfg: DynamicConfig) -> None:
    body = cfg.to_json().encode("utf-8")
    r2_client.put_object(
        bucket_name=bucket_name,
        key=key,
        body=body,
        ContentType="application/json",
    )
    logging.info(f"Wrote dynamic config to r2://{bucket_name}/{key} ({len(body)} bytes)")


# ---------------------------------------------------------------------------
# Overlay resolution
# ---------------------------------------------------------------------------

def resolve_min_event_id_for_group(
    dynamic_cfg: Optional[DynamicConfig],
    event_type: float,
    sub_event_types: Tuple[float, ...],
    border: float,
) -> Optional[int]:
    """Return the dynamically-configured ``min_event_id`` for a group or None."""
    if dynamic_cfg is None:
        return None
    entry = dynamic_cfg.by_key().get((float(event_type), tuple(float(x) for x in sub_event_types), float(border)))
    if entry is None or entry.min_event_id is None:
        return None
    return int(entry.min_event_id)

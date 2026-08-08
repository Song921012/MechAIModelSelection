"""Versioned, resumable records for the JUQ numerical study."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2.0"


def protocol_hash(protocol: Any) -> str:
    payload = asdict(protocol) if is_dataclass(protocol) else protocol
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def result_path(root: Path, study: str, key: str) -> Path:
    path = root / "results" / "records" / "submission" / study / "raw" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_compatible(
    path: Path,
    expected_hash: str,
    overwrite: bool,
    *,
    replace_incompatible: bool = False,
) -> dict | None:
    if not path.exists() or overwrite:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION:
        if replace_incompatible:
            return None
        raise RuntimeError(f"schema mismatch in {path}")
    if record.get("protocol_hash") != expected_hash:
        if replace_incompatible:
            return None
        raise RuntimeError(f"protocol mismatch in {path}")
    return record


def write_record(path: Path, protocol_digest: str, record: dict) -> None:
    payload = _json_safe({
        "schema_version": SCHEMA_VERSION,
        "protocol_hash": protocol_digest,
        **record,
    })
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def finite_or_none(value: float) -> float | None:
    return float(value) if value == value and abs(value) != float("inf") else None


def objective_gap(diagnostics: list[dict]) -> float | None:
    values = sorted(
        float(item["objective"]) for item in diagnostics
        if item.get("finite") and item.get("objective") is not None
        and item.get("stage", "initialization") == "initialization"
    )
    return values[1] - values[0] if len(values) > 1 else None


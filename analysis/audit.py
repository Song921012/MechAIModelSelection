"""Integrity checks for versioned numerical records and aggregate tables."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from mechai_experiments.records import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION = {
    "core": 2000, "phase_diagram": 2160, "crossdomain": 600,
    "predictive": 320, "waic": 500,
}
SUBMISSION_SEED_LIMITS = {
    "core": 80, "phase_diagram": 20, "crossdomain": 30,
    "predictive": 40, "waic": 20,
}
LOGICAL_KEYS = {
    "core": ("scenario", "candidate", "seed"),
    "phase_diagram": ("n_times", "noise", "trajectories", "candidate", "seed"),
    "crossdomain": ("study", "candidate", "seed"),
    "predictive": ("scenario", "candidate", "seed"),
    "waic": ("scenario", "candidate", "seed"),
}
FINITE_FIELDS = {
    "core": ("objective", "deviance", "gic_eff", "bic", "d_obs"),
    "phase_diagram": ("objective", "deviance", "gic_eff", "bic", "d_obs"),
    "crossdomain": ("objective", "deviance", "gic_eff", "bic", "d_obs"),
}


def audit(profile: str) -> dict:
    result_root = ROOT / "results" / "records" / "submission"
    report = {"profile": profile, "studies": {}, "errors": []}
    for study in ("core", "phase_diagram", "crossdomain", "predictive", "waic"):
        files = sorted((result_root / study / "raw").glob("*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        eligible_records = records
        if profile == "submission":
            eligible_records = [
                record for record in records
                if int(record.get("seed", -1)) < SUBMISSION_SEED_LIMITS[study]
            ]
        keys = [tuple(record.get(field) for field in LOGICAL_KEYS[study]) for record in records]
        protocols = Counter(record.get("protocol_hash") for record in records)
        failed = [record for record in records if record.get("status") == "failed"]
        nonfinite_scores = [record for record in records if record.get("status") == "nonfinite_score"]
        invalid_schema = [record for record in records if record.get("schema_version") != SCHEMA_VERSION]
        missing_protocol = [record for record in records if not record.get("protocol_hash")]
        invalid_numeric = []
        for record, logical_key in zip(records, keys):
            if record.get("status") != "ok":
                continue
            for field in FINITE_FIELDS.get(study, ()):
                value = record.get(field)
                if value is None or not math.isfinite(float(value)):
                    invalid_numeric.append((logical_key, field))
                    break
        duplicate_count = len(keys) - len(set(keys))
        report["studies"][study] = {
            "records": len(records),
            "eligible_records": len(eligible_records),
            "ok": sum(record.get("status") == "ok" for record in records),
            "failed": len(failed), "nonfinite_scores": len(nonfinite_scores),
            "duplicate_keys": duplicate_count, "invalid_schema": len(invalid_schema),
            "missing_protocol": len(missing_protocol), "invalid_numeric": len(invalid_numeric),
            "protocol_hashes": dict(protocols),
        }
        if duplicate_count:
            report["errors"].append(f"{study}: duplicate keys")
        if failed:
            report["errors"].append(f"{study}: {len(failed)} failed records")
        if invalid_schema:
            report["errors"].append(f"{study}: {len(invalid_schema)} schema mismatches")
        if missing_protocol:
            report["errors"].append(f"{study}: {len(missing_protocol)} missing protocol hashes")
        if invalid_numeric:
            report["errors"].append(f"{study}: {len(invalid_numeric)} invalid required numeric records")
        if profile == "submission" and len(eligible_records) != EXPECTED_SUBMISSION[study]:
            report["errors"].append(
                f"{study}: expected {EXPECTED_SUBMISSION[study]} eligible records, "
                f"found {len(eligible_records)}"
            )
    confidence_files = sorted((ROOT / "results" / "records" / "confidence" / "raw").glob("*.json"))
    confidence_records = [json.loads(path.read_text(encoding="utf-8")) for path in confidence_files]
    confidence_keys = [(row.get("scenario"), row.get("sample"), row.get("seed")) for row in confidence_records]
    confidence_invalid = [row for row in confidence_records if row.get("schema_version") != SCHEMA_VERSION or not row.get("protocol_hash") or not math.isfinite(float(row.get("statistic", math.nan)))]
    confidence_duplicates = len(confidence_keys) - len(set(confidence_keys))
    report["studies"]["confidence"] = {
        "records": len(confidence_files), "duplicate_keys": confidence_duplicates,
        "invalid_records": len(confidence_invalid),
    }
    if confidence_duplicates:
        report["errors"].append("confidence: duplicate logical keys")
    if confidence_invalid:
        report["errors"].append(f"confidence: {len(confidence_invalid)} invalid records")
    if profile == "submission" and len(confidence_files) != 800:
        report["errors"].append(
            f"confidence: expected 800 records, found {len(confidence_files)}"
        )
    if profile == "submission":
        table_root = ROOT / "results" / "summary" / "numerical"
        aggregate_specs = {
            "core_selections.csv": (("scenario", "seed", "criterion"), 3200),
            "phase_diagram_selections.csv": (
                ("n_times", "noise", "trajectories", "seed", "criterion"), 2700
            ),
            "crossdomain_selections.csv": (("study", "seed", "criterion"), 900),
            "predictive_model_averaging.csv": (("scenario", "seed", "method"), 480),
            "reference_sensitivity_summary.csv":
                (("scenario", "rho_nn", "resolution", "gamma"), 900),
        }
        aggregate_report = {}
        for filename, (fields, expected) in aggregate_specs.items():
            path = table_root / filename
            if not path.exists():
                report["errors"].append(f"aggregate: missing {filename}")
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            keys = [tuple(row.get(field) for field in fields) for row in rows]
            duplicate_count = len(keys) - len(set(keys))
            aggregate_report[filename] = {
                "records": len(rows), "expected": expected,
                "duplicate_keys": duplicate_count,
            }
            if len(rows) != expected:
                report["errors"].append(
                    f"aggregate: {filename} expected {expected} rows, found {len(rows)}"
                )
            if filename == "reference_sensitivity_summary.csv":
                invalid_counts = sum(int(row.get("count", -1)) != 80 for row in rows)
                aggregate_report[filename]["invalid_cell_counts"] = invalid_counts
                if invalid_counts:
                    report["errors"].append(
                        f"aggregate: {filename} has {invalid_counts} cells not based on 80 seeds"
                    )
            if duplicate_count:
                report["errors"].append(
                    f"aggregate: {filename} has {duplicate_count} duplicate selection keys"
                )
        report["aggregates"] = aggregate_report
    report_path = ROOT / "reports" / "submission_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "submission"), default="smoke")
    args = parser.parse_args()
    report = audit(args.profile)
    counts = ", ".join(
        f"{name}={details.get('eligible_records', details.get('records', 0))}"
        for name, details in report["studies"].items()
    )
    print(f"submission audit: {counts}; errors={len(report['errors'])}")
    print(f"full report: {ROOT / 'reports' / 'submission_audit.json'}")
    if report["errors"] and args.profile == "submission":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


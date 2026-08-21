"""Reaggregate saved fits under the derived first-principles criteria.

This module never solves or refits a dynamical system. It translates the
versioned JSON records into the notation used by the paper and checks that
new names reproduce scores stored during the original computation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "summary" / "first_principles"
STUDIES = {
    "core": {
        "directory": ROOT / "results" / "records" / "submission" / "core" / "raw",
        "group": ("scenario", "seed"),
        "seed_limit": 80,
    },
    "phase_diagram": {
        "directory": ROOT / "results" / "records" / "submission" / "phase_diagram" / "raw",
        "group": ("n_times", "noise", "trajectories", "seed"),
        "seed_limit": 20,
    },
    "biological_systems": {
        "directory": ROOT / "results" / "records" / "submission" / "crossdomain" / "raw",
        "group": ("study", "seed"),
        "seed_limit": 30,
    },
}
CRITERIA = ("aic", "aicc", "bic", "gic_pred", "gic_evid", "gic_eff_logn")


def _number(record: dict, canonical: str, legacy: str) -> float:
    value = record.get(canonical, record.get(legacy, math.nan))
    return float(value)


def canonical_row(record: dict, study: str) -> dict:
    """Return scalar fields with canonical and compatibility score names."""
    excluded = {
        "information", "theta", "start_diagnostics", "profile", "traceback",
        "information_eigenvalues", "generalized_eigenvalues",
    }
    row = {
        key: value for key, value in record.items()
        if key not in excluded and not isinstance(value, (dict, list))
    }
    row["study_group"] = study
    row["effective_dimension"] = _number(record, "effective_dimension", "d_obs")
    row["relative_log_volume"] = _number(record, "relative_log_volume", "c_obs")
    row["gic_pred"] = _number(record, "gic_pred", "ogic_p")
    row["gic_evid"] = _number(record, "gic_evid", "ogic_e")
    row["gic_eff_logn"] = _number(record, "gic_eff_logn", "gic_eff")
    row["d_obs"] = row["effective_dimension"]
    row["c_obs"] = row["relative_log_volume"]
    row["ogic_p"] = row["gic_pred"]
    row["ogic_e"] = row["gic_evid"]
    return row


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _selection_tables(
    frame: pd.DataFrame, group_fields: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[dict] = []
    support: list[dict] = []
    grouper = group_fields[0] if len(group_fields) == 1 else list(group_fields)
    for keys, group in frame.groupby(grouper, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identifiers = dict(zip(group_fields, keys))
        truth = str(group.iloc[0]["truth"])
        for criterion in CRITERIA:
            values = pd.to_numeric(group[criterion], errors="coerce")
            finite = group.loc[np.isfinite(values)].copy()
            if finite.empty:
                continue
            criterion_values = finite[criterion].astype(float).to_numpy()
            order = np.argsort(criterion_values)
            winner = finite.iloc[int(order[0])]
            gap = (
                float(criterion_values[order[1]] - criterion_values[order[0]])
                if len(order) > 1 else math.nan
            )
            shifted = criterion_values - np.min(criterion_values)
            weights = np.exp(-0.5 * np.clip(shifted, 0.0, 1400.0))
            weights /= weights.sum()
            entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-300, None))))
            selections.append({
                **identifiers,
                "criterion": criterion,
                "truth": truth,
                "selected": winner["candidate"],
                "correct": int(str(winner["candidate"]) == truth),
                "selection_gap": gap,
                "support_entropy": entropy,
            })
            if criterion in ("gic_pred", "gic_evid"):
                interpretation = (
                    "predictive_support" if criterion == "gic_pred"
                    else "equal_prior_posterior_approximation"
                )
                for (_, candidate_row), weight in zip(finite.iterrows(), weights):
                    support.append({
                        **identifiers,
                        "criterion": criterion,
                        "interpretation": interpretation,
                        "candidate": candidate_row["candidate"],
                        "weight": float(weight),
                    })
    selected = pd.DataFrame(selections)
    supports = pd.DataFrame(support)
    summary_rows: list[dict] = []
    summary_groups = [field for field in group_fields if field != "seed"] + ["criterion"]
    for keys, group in selected.groupby(summary_groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identifiers = dict(zip(summary_groups, keys))
        successes = int(group["correct"].sum())
        total = int(len(group))
        low, high = _wilson(successes, total)
        counts = group["selected"].value_counts()
        summary_rows.append({
            **identifiers,
            "recovery_rate": successes / total,
            "n": total,
            "recovery_ci_low": low,
            "recovery_ci_high": high,
            "modal_selection": counts.index[0],
            "modal_frequency": float(counts.iloc[0] / total),
            "mean_selection_gap": float(group["selection_gap"].mean()),
            "mean_support_entropy": float(group["support_entropy"].mean()),
        })
    return selected, pd.DataFrame(summary_rows), supports


def aggregate_study(name: str, specification: dict) -> dict:
    paths = sorted(specification["directory"].glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    eligible = [
        record for record in records
        if record.get("status") == "ok"
        and int(record.get("seed", -1)) < specification["seed_limit"]
    ]
    frame = pd.DataFrame(canonical_row(record, name) for record in eligible)
    required = set(CRITERIA) | {"candidate", "truth", *specification["group"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing fields {missing}")
    duplicate_count = int(
        frame.duplicated(list(specification["group"]) + ["candidate"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"{name}: {duplicate_count} duplicate candidate fits")
    nonfinite = {
        criterion: int(
            (~np.isfinite(pd.to_numeric(frame[criterion], errors="coerce"))).sum()
        )
        for criterion in CRITERIA
    }
    selected, summary, support = _selection_tables(frame, specification["group"])
    frame.to_csv(OUTPUT / f"{name}_scores.csv", index=False)
    selected.to_csv(OUTPUT / f"{name}_selections.csv", index=False)
    summary.to_csv(OUTPUT / f"{name}_selection_summary.csv", index=False)
    support.to_csv(OUTPUT / f"{name}_model_support.csv", index=False)
    return {
        "raw_records": len(records),
        "eligible_records": len(frame),
        "failed_or_excluded": len(records) - len(frame),
        "duplicate_keys": duplicate_count,
        "nonfinite": nonfinite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES) + ("all",), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = STUDIES if args.study == "all" else {args.study: STUDIES[args.study]}
    report = {
        name: aggregate_study(name, specification)
        for name, specification in names.items()
    }
    (OUTPUT / "aggregation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for name, details in report.items():
        print(
            f"{name}: {details['eligible_records']} eligible fits; "
            f"nonfinite={details['nonfinite']}"
        )


if __name__ == "__main__":
    main()

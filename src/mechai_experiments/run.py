"""Unified experiment launcher with an explicit long-run profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "core": "core.py",
    "phase": "phase.py",
    "biological-systems": "cross_domain.py",
    "cross-domain": "cross_domain.py",
    "predictive": "predictive.py",
    "confidence": "confidence.py",
    "waic": "waic.py",
    "reference": "reference.py",
    "scalability": "scalability.py",
    "intervention": "intervention.py",
    "capacity": "capacity.py",
    "metric-boundary": "metric_boundary.py",
    "glucose": "glucose.py",
}
ORDER = [key for key in SCRIPTS if key != "cross-domain"]


def load_profile(name: str) -> dict:
    with (ROOT / "configs" / f"{name}.toml").open("rb") as handle:
        return tomllib.load(handle)


def command(
    study: str, profile: str, workers: int, resume: bool, cfg: dict
) -> list[str]:
    cmd = [sys.executable, str(ROOT / "experiments" / SCRIPTS[study])]
    underlying = profile
    if study in {"core", "phase", "biological-systems", "cross-domain", "predictive"}:
        seed_key = "cross-domain" if study == "biological-systems" else study
        cmd += [
            "--profile",
            underlying,
            "--workers",
            str(workers),
            "--seeds",
            str(cfg["seeds"][seed_key]),
        ]
        cmd += ["--resume" if resume else "--overwrite"]
    elif study == "waic":
        cmd += [
            "--draws",
            str(cfg["waic"]["draws"]),
            "--seeds",
            str(cfg["seeds"]["waic"]),
            "--workers",
            str(workers),
            "--resume" if resume else "--overwrite",
        ]
    elif study == "confidence":
        cmd += [
            "--replicates",
            str(cfg["confidence"]["replicates"]),
            "--workers",
            str(workers),
        ]
    elif study == "scalability":
        cmd += ["--profile", underlying]
    elif study == "reference":
        cmd += ["--source", "submission"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["all", *ORDER], default="core")
    parser.add_argument("--profile", choices=["smoke", "submission"], default="smoke")
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1))
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cfg = load_profile(args.profile)
    studies = ORDER if args.study == "all" else [args.study]
    if args.profile == "submission":
        print(
            "Submission profile selected explicitly; existing compatible records can be resumed.",
            flush=True,
        )
    for study in studies:
        cmd = command(study, args.profile, args.workers, args.resume, cfg)
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

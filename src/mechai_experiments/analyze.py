"""Rebuild first-principles summaries, figures, and consistency reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def call(name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / name), *args],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--tables", action="store_true",
                        help="rebuild canonical CSV summaries from fit records")
    parser.add_argument("--audit", action="store_true",
                        help="run record and protocol consistency checks")
    args = parser.parse_args()
    all_steps = not (args.figures or args.tables or args.audit)

    if args.tables or all_steps:
        call("first_principles.py")
    if args.figures or all_steps:
        call("figures.py")
        call("supplement_figures.py")
    if args.audit or all_steps:
        call("audit.py", "--profile", "submission")


if __name__ == "__main__":
    main()

# MechAIModelSelection

[![Reproduction checks](https://github.com/Song921012/MechAIModelSelection/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Song921012/MechAIModelSelection/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproduction code, audited records, aggregate data, and submission figures for
*Observable Geometry for Model Selection in Mechanism-AI Dynamical Systems*.
The repository covers epidemic, ecological, biochemical, and
electrophysiological candidate families, controlled information studies,
confidence calibration, predictive model averaging, and computational scaling.

## Install

```bash
python -m pip install -e .
```

## Safe default

The default profile is deliberately small:

```bash
python -m mechai_experiments.run --study core --profile smoke --workers 1
```

A full study starts only when `submission` is explicit:

```bash
python -m mechai_experiments.run --study all --profile submission --workers 9 --resume
```

The complete archive is already included. Rebuild aggregates, figures, and the
integrity report without refitting:

```bash
python -m mechai_experiments.analyze --figures --tables --audit
```

## Repository map

- `src/mechai_experiments/`: shared model, fitting, scoring, record, and CLI code.
- `experiments/`: study-specific entry points.
- `analysis/`: aggregation, submission figures, tables, and audits.
- `configs/`: smoke and submission protocols.
- `results/records/`: independent JSON records, including explicit failures.
- `results/summary/`: selected source-data CSV files.
- `figures/`: PDF, SVG, and PNG submission figures. TIFF files are omitted from Git.
- `docs/`: experiment definitions, schema, runtime, and repository guide.

The fitted-record schema and protocol hashes remain compatible with the
manuscript archive. Public file names omit historical revision suffixes; the
stored `schema_version` is a data contract, not a manuscript version.

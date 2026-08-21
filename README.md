# MechAIModelSelection

[![Reproduction checks](https://github.com/Song921012/MechAIModelSelection/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Song921012/MechAIModelSelection/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproduction code and fit-level records for *Geometric Model Selection for
Mechanism-AI Dynamical Systems*. The repository contains epidemic,
biochemical, ecological, and electrophysiological candidate families,
controlled information studies, confidence calibration, model averaging, and
post-fit computational scaling.

## Install

    python -m pip install -e .

## Fast verification

The smoke profile fits one seed per study and is the default for checking an
installation:

    python -m mechai_experiments.run --study core --profile smoke --workers 1

The complete study starts only when the submission profile is explicit:

    python -m mechai_experiments.run --study all --profile submission --workers 9 --resume

The fit archive is included. Rebuild canonical summaries, all five main
figures, all eight supplementary figures, and the consistency report without
refitting:

    python -m mechai_experiments.analyze --tables --figures --audit

The three first-principles calibration calculations are lightweight and have
separate entry points:

    python analysis/optimism.py
    python analysis/evidence.py --workers 9
    python experiments/metric_loss.py --workers 9

## Repository map

- src/mechai_experiments/: shared model, fitting, scoring, record, and CLI code.
- experiments/: study-specific fitting and lightweight calibration entry points.
- analysis/first_principles.py: canonical aggregation and compatibility mapping.
- analysis/figures.py: the five main manuscript figures.
- analysis/supplement_figures.py: the eight supplementary figures.
- results/records/: independent JSON records, including explicit failures.
- results/summary/first_principles/: canonical GIC-pred, GIC-evid, and geometric
  BIC summaries.
- results/summary/numerical/: confidence, prediction, WAIC, sensitivity, and
  scaling summaries.
- figures/: PDF, SVG, and PNG outputs. Journal TIFF files stay with the
  manuscript archive.
- docs/: study definitions, record schema, runtime notes, and repository guide.

## Criterion fields

New analyses use gic_pred, gic_evid, gic_eff_logn, effective_dimension, and
relative_log_volume. Legacy JSON fields ogic_p, ogic_e, gic_eff, gic_vol_050,
d_obs, and c_obs remain readable so the released fit archive does not need to
be rewritten. Scores with an arbitrary volume coefficient are reported only as
sensitivity analyses.

The historical CLI key cross-domain remains accepted for compatibility.
biological-systems is the preferred public name for the biochemical,
ecological, and electrophysiological study group.

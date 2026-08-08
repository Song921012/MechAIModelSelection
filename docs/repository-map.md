# Repository map

`src/mechai_experiments` owns shared numerical code. Files in `experiments` are
thin study entry points. `analysis` never fits a model; it consumes JSON or CSV
records. Raw records and derived summaries are separated so aggregate changes
can be audited without altering fitted results.

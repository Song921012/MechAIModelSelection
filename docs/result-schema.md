# Result schema

Each JSON record includes `schema_version`, `protocol_hash`, `status`, the
logical study key, seed, candidate, fit diagnostics, score components, and
observable-geometry quantities. Valid statuses are `ok`, `failed`, and
`nonfinite_score`. Aggregation checks logical-key uniqueness and required finite
fields. A protocol hash change indicates a different experiment and must not be
mixed with existing records.

# Runtime and profiles

`smoke` runs one seed and reduced iteration budgets. `submission` reproduces the
released design and can take many hours; it is never selected implicitly. Use
`--resume` to reuse compatible records. Runtime depends strongly on CPU count,
BLAS threading, and the ODE candidate. The released fit records avoid any need
to repeat long optimization when rebuilding figures.

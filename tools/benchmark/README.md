# Benchmarks

This directory contains the benchmark runners, native microbenchmark sources,
budgets, and saved measurement artifacts. The usual entry point is:

```bash
python -m flowedge_dev bench policy --help
```

Build native benchmarks with `-DFLOWEDGE_BENCH=ON`. Saved reports under
`artifacts/` are evidence snapshots; check each report's hardware, revision,
and workload before comparing results.

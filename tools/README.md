# Development tools

The installable API is `flowedge` plus optional packages under `integrations/`.
Checkout-only commands have one entry point:

```bash
python -m flowedge_dev --help
```

Implementation is grouped here: `benchmark/`, `conversion/`, `scripts/`, and
`verification/`. Use the API and guides for applications; these tools are for
development, profiling, and release checks.

# Contribute

Start with the repository [CONTRIBUTING.md](https://github.com/reforcemind/FlowEdge/blob/main/CONTRIBUTING.md)
and the [Code of Conduct](https://github.com/reforcemind/FlowEdge/blob/main/CODE_OF_CONDUCT.md).

Look for `good first issue` and `help wanted`. Large features need an issue before a PR.
Open starters: [Windows first-run](https://github.com/reforcemind/FlowEdge/issues/185),
[gym PushT GIF](https://github.com/reforcemind/FlowEdge/issues/186).
CPU wheels live on the [v0.1.2](https://github.com/reforcemind/FlowEdge/releases/tag/v0.1.2) Release.
Runtime changes keep the zero-allocation hot path. Parity (ULP, convert, diffusion) is local:
`tools/scripts/verify_all.sh` or `tools/scripts/verify_local.ps1`. GitHub Actions is compile + `ctest`.

| Goal | Guide |
|---|---|
| New action head | [Add a head](guides/add-a-head) |
| New backbone | [Add a backbone](guides/add-a-backbone) |
| Local verify | [Verification](guides/verification) |
| Edge JSON | [Edge benchmarks](guides/edge-benchmarks) |
| Policy vs PyTorch | [Policy evaluation](guides/policy-evaluation) |

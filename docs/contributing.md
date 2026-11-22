# Contributing Guide

Contributions to FlowEdge are welcome across kernel optimization, model support, hardware backends, benchmarking, and documentation.

Before contributing, please review the repository [CONTRIBUTING.md](https://github.com/reforcemind/FlowEdge/blob/main/CONTRIBUTING.md) and [Code of Conduct](https://github.com/reforcemind/FlowEdge/blob/main/CODE_OF_CONDUCT.md).

---

## Starter Issues

If you are new to the codebase, look for issues labeled:
- `good first issue` — scoped tasks suitable for first-time contributors.
- `help wanted` — areas where community expertise and testing on varied hardware are especially valuable.

For questions, proposals, or community discussions, visit [GitHub Discussions](https://github.com/reforcemind/FlowEdge/discussions).

---

## Core Guidelines

- **Hot-Path Allocations**: Changes to Core inference paths must preserve the zero-allocation runtime guarantee.
- **Verification Gates**: Check numerical parity against PyTorch references before submitting pull requests. See the [Verification Guide](guides/verification).
- **Benchmark Claims**: Performance claims must include reproducible before/after measurements on identical hardware configurations.

---

## Reference Guides

| Task | Guide |
|---|---|
| **Adding a Policy Head** | [Add a Head](guides/add-a-head) |
| **Adding a Backbone** | [Add a Backbone](guides/add-a-backbone) |
| **Local Verification** | [Verification Contract](guides/verification) |
| **Benchmarking Policies** | [Policy Evaluation](guides/policy-evaluation) |
| **Standardized Reports** | [Edge Benchmarks](guides/edge-benchmarks) |

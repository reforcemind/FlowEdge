---
name: verify-local
description: Run FlowEdge's local verification gate (ctest, ULP, convert, diffusion) on this machine. Use for Grok Bot, before commits or PRs, and when the user asks to verify, checkcorrect, or replace GitHub Actions parity.
---

# Local verification (Grok Bot)

Parity does not live in GitHub Actions. This host runs the gate. Use local
execution, not the cloud computer: checkpoints under `models/` and the
Release tree are here.

## Commands

Linux or Git Bash:

```bash
FLOWEDGE_BUILD_DIR=build-verify ./tools/scripts/verify_all.sh models/mamba_flow.safetensors
```

Windows (this repo's Clang gnu-target tree):

```powershell
powershell -File tools/scripts/verify_local.ps1 -BuildDir build-win-clang
```

Kernel-only slice after a CPU op change:

```powershell
.\build-win-clang\flowedge_tests.exe --gtest_filter=Matmul*:Conv*:DenseConv1d*:DiffusionConvolution*:Mish*:GroupNorm*:Gelu*:LayerNorm*:Softmax*:CachedCausalAttention*:Transformer.*
```

## Rules

1. Read `.cursor/skills/verify-local/SKILL.md` and `docs/guides/verification.md`.
2. Do not treat a green GitHub `Test` job as ULP or convert proof.
3. Fail closed on a red local gate. Do not skip `verify_ulp.py` because CI no longer runs it.
4. Kernel or op work also follows the measure-first loop in
   `docs/guides/verification.md` (real shapes, isolate, layout before inner loop).
   CUDA DP kernels use the mix + native DDIM gate in that same guide.
5. Report the exact command and pass/fail. For performance, report the median
   and keep the raw command.

## What CI still does

Compile, `ctest`, import, lint, sanitizers, Docker, ARM64. Manual workflow
`.github/workflows/parity.yml` can mirror ULP on a runner if someone asks.
The merge bar for kernels and heads is this local run.

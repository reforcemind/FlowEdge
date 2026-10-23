# Agent notes

Verification is local. Grok Bot runs it on this machine (`/verify-local`).
GitHub Actions compiles and runs `ctest`; it does not own PyTorch ULP, convert
round-trip, or diffusion parity.

- Full gate: `tools/scripts/verify_all.sh` (Linux/Git Bash) or `tools/scripts/verify_local.ps1` (Windows)
- Maintainer CLI: `python -m flowedge_dev` (bench, verify, convert, rollout)
- Skill: `.cursor/skills/verify-local/SKILL.md`
- Contract: `docs/guides/verification.md`

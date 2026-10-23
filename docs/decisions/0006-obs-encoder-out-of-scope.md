# ADR 0006: Observation encoders are out of scope

Status: Accepted. Scope: `src/core/heads/`, `tools/conversion/`.

## Context

Real policies condition on an observation encoder. Diffusion Policy and ACT use a ResNet. VLAs use a vision-language model. These are large and varied.

## Decision

- The engine covers the backbone and the action head.
- A head takes a precomputed conditioning vector. The observation encoder is out of scope for v1.
- The converter maps the backbone and head weights, not the encoder.

## Consequences

- The engine stays small and focused.
- Running a full vision policy end to end needs the encoder as a separate follow-up.

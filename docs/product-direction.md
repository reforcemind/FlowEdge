# Product Direction

FlowEdge helps teams qualify and deploy a *trained* policy behind a robot control
loop: verify the converted output, measure the deployment budget, and make action
delivery observable. Core runs supported native heads; Relay provides scheduling
and delivery contracts. LeRobot (or your stack) trains, encodes observations, and
owns robot limits and e-stop. Allocation guarantees apply to the tested native
paths, not to the whole camera/Python/driver pipeline.

## Aim: qualified policy deployment

The problem is not a universal inference time. A policy can have correct outputs
offline and still deliver late or stale actions, exhaust memory, or silently use
an unintended backend on the deployment machine.

Our target user is a LeRobot team with a trained checkpoint and a working robot
task. The deliverable is a reproducible deployment report and an execution path
whose limits and failure behavior are explicit:

- **Correct output:** bind checkpoint, encoder, normalization and action schema;
  verify converted actions against the source implementation before actuation.
- **Declared resource budget:** report the actual backend, residency, setup
  memory and measured latency distribution on the chosen host. Reject unsupported
  configurations instead of turning them into successful benchmark results.
- **Traceable delivery:** distinguish fresh-chunk generation from consuming a
  buffered action. Measure observation age, action expiry, underruns and deadline
  misses. Validate the existing replacement, cancellation and fallback contracts.
- **Task evidence:** compare success and timing separately on matched tasks.
  Faster inference that breaks the task is not an improvement.

These are qualification goals, not a claim that every existing adapter supplies
every measurement. LeRobot already provides
[asynchronous inference](https://huggingface.co/docs/lerobot/async); another async
queue alone is not FlowEdge's differentiator. The value to demonstrate is native
execution plus explicit, tested deployment contracts and independently useful
evidence. This is not a safety-certified or hard-real-time controller.

## Next milestones

1. Ship the measured CUDA improvements, regression tests and matched benchmark
   tooling. Keep same-checkpoint performance comparisons separate from model changes.
2. Complete the [SO-101 pilot](https://github.com/reforcemind/FlowEdge/issues/193):
   ACT and DP in LeRobot, plus the same DP in FlowEdge, on one host and task.
   Start with offline parity; keep limits, supervised commissioning, suitable
   first-action fallback and stop behavior in the robot adapter.
3. Turn that pilot into a deployment qualification workflow: one run configuration,
   preflight checks and a machine-readable result connecting model identity,
   memory, timing, action age/misses and task outcomes. Reuse existing Core/Relay
   mechanisms; do not build a second scheduling stack.
4. Validate on a second host or external user's setup before generalizing the
   results. Optimize the measured deployment bottleneck, not an isolated headline.

The first milestone of product usefulness is an external contributor reproducing
the matched pilot and understanding its limits without editing Core. Select the
control period, acceptable action age and task-quality criteria before each run.
Report both native sampling time and observation-to-action time; camera, transport
and actuation costs must be named when they are excluded.

## What we are dropping

The universal **10 ms fresh-policy-inference goal** is retired. The GTX 1650
experiments did not achieve it for the unchanged ten-step PushT policy; that is
not proof that 10 ms is impossible for every policy or machine. Historical
artifacts retain their original targets and failed gates, but new native A/B
reports impose no fixed latency target.

Training and student-policy experiments are outside this runtime roadmap.
There is no change to the production checkpoint, README performance headline,
or robot-support claims. A fast control tick consuming buffered actions must
never be advertised as equally fast fresh inference.

## Existing scope

```{image} _static/figures/purpose.svg
:alt: Hardware, FlowEdge, LeRobot
:class: fe-fig
```

| Keep | Leave |
|---|---|
| Hardware: CPU now; Jetson/ARM via plugin; CUDA Mamba, flow, and DP heads; matched GPU replay ([#163](https://github.com/reforcemind/FlowEdge/issues/163)) and CUDA Core period loop ([#164](https://github.com/reforcemind/FlowEdge/issues/164)) on GTX 1650 | Robot e-stop and joint limits |
| FlowEdge: flow matching, DP U-Net, Transformer decoder fixture, optional Relay | Training, datasets, graph compilers |
| LeRobot: plugin, encoders, drivers | Native kernels and the arena |

The sequence is convert → matched CPU replay vs PyTorch → Transformer fixture (#10) → CUDA device-resident flow ([#161](https://github.com/reforcemind/FlowEdge/issues/161)) and DP ([#162](https://github.com/reforcemind/FlowEdge/issues/162)) → matched GPU replay ([#163](https://github.com/reforcemind/FlowEdge/issues/163)) → CUDA DP conv occupancy ([#174](https://github.com/reforcemind/FlowEdge/issues/174), [#176](https://github.com/reforcemind/FlowEdge/issues/176), [#179](https://github.com/reforcemind/FlowEdge/issues/179)) → a period loop on Jetson or SO-100 with miss counts. Tenstorrent stays behind that GPU evidence. Success is an external team doing that without FlowEdge training code.

```{image} _static/figures/sequence.svg
:alt: convert, CPU replay, Jetson or SO-100, same contracts
:class: fe-fig
```

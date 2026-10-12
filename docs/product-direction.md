# Product Direction

FlowEdge is a fast, memory-efficient inference runtime for trained robot
policies on edge hardware. It takes the latency-critical action path out of a
heavy Python framework while LeRobot (or another robotics stack) continues to
own training, datasets, robot drivers, limits, and emergency stop.

## Aim: a policy accelerator for edge robots

FlowEdge should be the shortest supported path from a trained LeRobot policy to
a fast, reproducible edge deployment. It should not try to become another
general graph runtime. TensorRT, ONNX Runtime, and ExecuTorch already solve that
broader problem across many models and devices.

The useful specialization is the complete robot-policy path: conversion,
fixed-shape action-head execution, iterative sampling, action chunks, and a
same-device comparison against the source policy. Native kernels are one
backend, not the product by themselves. Vendor runtimes may be used when they
are faster on the target device.

The primary measurement is fresh action-chunk latency on the target device:
p50, p95, and p99, with the backend, precision, model, power mode, and thermal
state recorded. Also report encoder-to-action latency, resident memory, startup
time, and hot-path allocations. Buffered delivery is useful, but it is not fresh
inference and must be reported separately.

Output parity and task success are performance guardrails. They prevent a
faster kernel, reduced-precision path, or conversion from silently changing the
policy. They are not the product objective.

## Build next

1. **Make one edge path easy to install.** Support the current stable LeRobot
   release, publish the FlowEdge and policy-plugin packages, and provide a
   pinned Jetson container or wheel. A new user should be able to convert and
   benchmark a compatible checkpoint without editing the repository.
2. **Remove the Python/CPU seam on CUDA.** Add an explicit device-tensor API
   (for example DLPack or a checked CUDA buffer view) so a GPU encoder can pass
   its condition directly to the native head. Keep condition, noise, sampler
   state, and intermediate actions resident until the final action copy.
3. **Reduce the dominant weight traffic.** Add FP16/BF16 storage and execution
   for the diffusion path, using cuDNN, cuBLASLt, CUTLASS, or TensorRT where it
   beats the native implementation. Select implementations from measurements on
   the target device instead of expanding a parallel general-purpose operator
   library.
4. **Reduce repeated launch cost.** Capture the fixed-shape denoising schedule
   in a CUDA Graph after setup, then replay it without per-operator host launch
   overhead. Measure it independently from precision and operator changes.
5. **Complete the [SO-101 pilot](https://github.com/reforcemind/FlowEdge/issues/193).**
   Compare the best working LeRobot/PyTorch path with FlowEdge on the same edge
   device, checkpoint, observations, power mode, and physical task.
6. **Add policies from real demand.** Generalize conversion from the SO-101
   checkpoint first. Add ACT only after that pilot identifies it as the useful
   comparison; defer broad VLA and new-accelerator work until the deployment
   path is repeatable.

The first product milestone is a real SO-101 deployment where FlowEdge:

- materially reduces fresh action-chunk latency against the best working
  PyTorch/LeRobot baseline on the same device;
- fits the declared memory and power envelope;
- keeps p95 latency within the task's chosen replanning period; and
- preserves task success within a predeclared bound.

A useful initial target is at least a 2x policy-latency improvement without a
task-success regression. This is a target for the selected checkpoint and edge
device, not a universal claim.

## Refactor for adoption

- Separate **Core** (model package, execution, backends, tensor API) from the
  optional **Relay** service in the public navigation and release artifacts.
  Relay should not be required to understand or install the accelerator.
- Replace compile-time backend ambiguity with an explicit requested/resolved
  backend in the API and reports. A requested accelerator must fail closed
  instead of silently producing a CPU performance result.
- Define one versioned policy-package manifest containing architecture, shapes,
  normalization, precision, source identity, and supported backends. Keep raw
  tensor names behind the converter/runtime boundary.
- Turn the current benchmark scripts into one `flowedge bench` command that can
  run native, PyTorch eager/compiled, and an available vendor backend on the
  same inputs, then emit one small machine-readable report.
- Keep the public compatibility matrix small and current: policy, LeRobot
  version, device, precision, package, and measured status. Remove stale
  examples and archived experiment logs from the main user path.

## What we are dropping

The universal **10 ms fresh-policy-inference goal** is retired. The unchanged
ten-step FP32 PushT policy did not reach it on the GTX 1650, and its roughly 1 GB
of weights makes that target bandwidth-limited in the current implementation.
That result does not make performance secondary and does not say 10 ms is
impossible for a smaller policy, reduced precision, or different accelerator.

Training and student-policy work remain outside Core. FlowEdge should first win
through runtime and edge-deployment engineering on an existing checkpoint;
compression or distillation can later be a separate, explicit integration.

FlowEdge is not a trainer, robot driver, safety controller, or universal model
compiler.

## Scope

| FlowEdge owns | The robotics stack owns |
|---|---|
| Policy conversion and package contract | Training and datasets |
| Fast action-head execution and device residency | Robot drivers, limits, and emergency stop |
| Edge backend selection and performance reports | Task definition and safe commissioning |
| Optional action scheduling through Relay | General model serving and distributed training |

The development sequence is: converted checkpoint -> same-device framework and
FlowEdge baseline -> device-resident profiling -> precision and operator work ->
SO-101 task run -> second edge device. Every speedup names the exact checkpoint,
device, precision, backend, and measured boundary.

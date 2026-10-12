# Roadmap

FlowEdge is building a fast policy-inference path for edge robots, beginning
with LeRobot Diffusion Policy on NVIDIA hardware. The roadmap is deliberately
narrow: make one useful path easy, fast, and reproducible before adding model
families or accelerators.

See [product direction](product-direction) for the product boundary and success
criteria.

## Available now

- Native C++ Core with fixed-memory CPU and CUDA execution.
- Flow matching, Mamba, and LeRobot ConditionalUnet1D Diffusion Policy heads.
- C, C++, and Python APIs plus an out-of-tree LeRobot policy plugin.
- Converted `.safetensors` packages and matched PyTorch/FlowEdge benchmarks.
- Optional Relay scheduling and action delivery; it is not required by Core.

The current CUDA measurements are desktop GTX 1650 results. They are not Jetson
or physical-robot results.

## P0: usable edge package

Before adding another head:

1. Update the plugin to the current stable LeRobot public API.
2. Publish installable packages and a pinned Jetson build/container.
3. Convert a non-PushT SO-101 Diffusion Policy checkpoint without source edits.
4. Make requested and resolved backend/precision explicit; fail closed when the
   requested accelerator is unavailable.
5. Provide one command that converts, benchmarks, and writes a compact report.

Done means an external user can install FlowEdge, convert their compatible
checkpoint, and reproduce a same-device comparison without editing Core.

## P1: remove edge-path overhead

1. Add a device-tensor boundary between the LeRobot encoder and FlowEdge.
2. Keep condition, noise, sampler state, and intermediate actions on the GPU.
3. Capture the fixed denoising schedule in a reusable CUDA Graph.
4. Report transfer, synchronization, operator, and full action-chunk time
   separately so optimization follows the real bottleneck.

## P2: reduce memory traffic

1. Add FP16/BF16 diffusion weights and activations on supported GPUs.
2. Benchmark native kernels against cuDNN/cuBLASLt, CUTLASS, and TensorRT for
   the actual policy shapes; keep the fastest maintainable implementation.
3. Measure resident device memory, power mode, temperature/throttling, and
   p50/p95/p99 on a Jetson-class target.

The performance gate is same checkpoint, same device, same inputs, and preserved
task quality. A useful initial target is at least 2x lower policy latency than
the best working PyTorch/LeRobot baseline on that device.

## P3: physical SO-101 result

Complete [issue #193](https://github.com/reforcemind/FlowEdge/issues/193) with
matched LeRobot and FlowEdge runs. Choose the replanning period from the task,
then report fresh chunk latency, missed periods, memory, and task success.

After the Diffusion Policy pilot:

- add ACT if the same SO-101 users need it ([#194](https://github.com/reforcemind/FlowEdge/issues/194));
- repeat on a second edge device before generalizing performance claims;
- consider aarch64/Qualcomm only with a concrete device and backend owner
  ([#195](https://github.com/reforcemind/FlowEdge/issues/195)).

## Deferred

Full VLA execution, pi0, Tenstorrent, generic graph compilation, ROS 2 transport,
and additional Relay services do not outrank the installable Jetson/SO-101 path.
They should enter the active roadmap only with a target checkpoint, device,
maintainer, and measured baseline.

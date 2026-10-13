# GEMV confirmation gate

The first three alternating trials produced a 0.256% median native 10-step
gain but only one of three paired wins. Do not treat this as a stable gain.
The expanded diagnostic mix did not identify a smaller projection regression
that would justify narrowing the dispatch.

Before running confirmation, declare: five alternating trials, 10 warmups,
100 measured samples per trial, ten DDIM steps, seeds 107 through 111.
Require at least four of five candidate p50 wins AND more than 1% improvement
in median p50. All actions must satisfy the existing parity gate. A retained
kernel also requires matched upstream PyTorch replay and local verification.
If this confirmation fails, remove the experimental kernel and dispatch;
keep independent correctness tests and missing real-shape mix coverage.

No checkpoint, sampler schedule, normalization, precision, or action horizon
changes are permitted. CUDA residency and Release builds are required.

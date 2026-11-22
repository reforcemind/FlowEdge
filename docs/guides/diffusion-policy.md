# Diffusion Policy Guide

A Diffusion Policy generates actions by iteratively denoising an action trajectory conditioned on environment observations.

LeRobot trains a 1D convolutional U-Net that predicts noise given a noisy trajectory chunk, timestep, and observation condition. FlowEdge executes this U-Net natively inside a pre-allocated C++ memory arena using deterministic DDIM sampling (or seeded DDPM) and applies dataset MIN_MAX un-normalization. Upstream observation encoding remains with LeRobot.

```{image} ../_static/figures/policies.svg
:alt: Comparison of Flow matching ODE versus Diffusion Policy DDIM
:class: fe-fig
```

---

## Model Contract & Architecture

- **U-Net Architecture**: 1D convolutions (`Conv1D`, `ConvTranspose1D`), GroupNorm, Mish activations, sinusoidal timestep embeddings, and FiLM conditioning projections.
- **Noise Schedulers**: Deterministic DDIM (default, zero RNG on hot path) and seeded DDPM.
- **Un-normalization**: Fixed `MIN_MAX` un-normalization statistics imported directly during conversion.
- **Reference Configuration** (`diffusion_pusht`): Horizon 16, action dimension 2, action execution steps 8, observation steps 2, U-Net channel widths 512 / 1024 / 2048, timestep embedding dimension 128.

---

## Observation Boundary

FlowEdge operates on the flattened observation condition vector produced by upstream vision or state encoders:

For the reference `diffusion_pusht` policy, two observation steps each contribute a 2-value normalized robot state and a 64-value image feature (totaling 132 condition elements). The converter validates condition widths from checkpoint FiLM projections.

---

## Conversion and Inference

### 1. Convert Checkpoint

```bash
# Download reference checkpoint
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht

# Convert weights and metadata into FlowEdge format
python -m flowedge_dev pipeline convert models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
```

### 2. Run Standalone Sampling

```bash
# Run standalone C++ inference sample (10 DDIM steps)
./build/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
```

With `FLOWEDGE_BACKEND=cuda`, U-Net weights and intermediate scratch remain device-resident on the GPU, avoiding host roundtrips.

---

## Verification & Benchmarks

Validate numerical parity against the PyTorch reference implementation:

```bash
# Check denoiser, DDIM, DDPM, and un-normalization tolerances
python -m flowedge_dev verify diffusion build

# Verify against public reference checkpoint
python -m flowedge_dev verify diffusion-public \
  models/diffusion_pusht/model.safetensors \
  models/diffusion_pusht.flowedge.safetensors --build build
```

For full policy latency benchmarks and comparison with PyTorch, see [Performance Documentation](../performance).

---

## Specifications & Constraints

- **Batch Size**: 1 (optimized for single-robot real-time control loops).
- **Shapes**: Fixed at conversion time; scratch buffers are sized once at model initialization.
- **Data Types**: FP32 for activations and normalization; BF16 supported for linear projections; calibrated FP16 deep convolution layers available on CUDA.

# Contributing to FlowEdge

Thank you for your interest in contributing to FlowEdge.

FlowEdge welcomes contributions across kernel optimization, model support, hardware backends, benchmarking, APIs, documentation, and robotics integrations.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

If you have questions or want to discuss an idea before opening an issue or pull request, please start a discussion in [GitHub Discussions](https://github.com/reforcemind/FlowEdge/discussions).

---

## Where to Start

- **Issues**: Check issues tagged with `good first issue` or `help wanted` for well-scoped starter tasks.
- **Architectural Changes**: For large features or interface modifications, please open an issue first to discuss the design and scope.

---

## Development Standards & Principles

1. **Deterministic Hot Paths**: Inference hot paths in Core must remain free of dynamic memory allocations (`malloc`/`free`) after initialization.
2. **Minimal Overhead**: Favor standard C++23 facilities, explicit buffers, and zero-cost abstractions.
3. **Reproducible Parity**: Changes to mathematical operators or policy heads must include parity checks against PyTorch/reference implementations.
4. **Clean Layer Separation**: `FlowEdge::Core` does not depend on `FlowEdge::Relay`. Keep dependencies unidirectional.

---

## Development Environment

### Requirements

- CMake 3.21+
- C++23 compliant compiler:
  - Clang 16+
  - GCC 13+
  - MSVC 19.38+
- Python 3.10–3.12 (for building Python bindings)

### Building and Testing

```bash
# Configure and build with tests enabled
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_TESTS=ON
cmake --build build -j

# Run C++ unit tests
ctest --test-dir build --output-on-failure
```

---

## Submitting Pull Requests

1. **Branching**: Fork the repository and create a descriptive branch from `main`.
2. **Focus**: Keep pull requests small and focused on a single logical change.
3. **Testing**: Run unit tests and verification scripts before submitting:
   - Linux / macOS: `./tools/scripts/verify_all.sh models/mamba_flow.safetensors`
   - Windows: `powershell -File tools/scripts/verify_local.ps1 -BuildDir build`
4. **Code Style**: Follow existing C++ and Python formatting conventions (`./tools/scripts/lint.sh`).
5. **Performance Changes**: Any optimization or kernel change must provide before-and-after benchmark results measured on the same machine, including compiler, device, and configuration details.

---

## Bug Reports & Feature Requests

- **Bug Reports**: Please include the FlowEdge version or commit SHA, operating system, compiler, GPU/driver details (if applicable), and a minimal reproduction command or script.
- **Feature Requests**: Describe the use case, why it belongs in FlowEdge, and how the feature can be verified with automated tests.

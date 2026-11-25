# Installation

## Python

Install from PyPI (Python 3.10+, including free-threaded Python 3.14t):

```sh
pip install ketrs
```

To use the Qiskit converter (`ketrs.qiskit`), install the `qiskit` extra:

```sh
pip install "ketrs[qiskit]"
```

### Prebuilt Wheels & GPU Support

| Platform | Architecture | CPU Engines | Vulkan GPU (`GpuStatevector`, `mps.gpu`) |
|---|---|---|---|
| **Linux** (glibc 2.28+) | x86-64, arm64 | Included | Included (needs Vulkan driver at runtime) |
| **Windows** | x86-64 | Included | Included (needs Vulkan driver at runtime) |
| **macOS** | Apple Silicon | Included | Included (bundles MoltenVK; `GpuStatevector` in `complex64`) |

On platforms without a matching wheel, `pip` builds from source (requires Rust 1.88+).

!!! tip "Checking GPU availability"
    ```python
    import ketrs
    print(ketrs.gpu_device())  # Prints your GPU name, or None if unavailable
    ```
    No CUDA or ROCm toolkit is needed; Linux/Windows need only a Vulkan 1.2 driver, and macOS wheels bundle MoltenVK.

---

## Rust

Add `ketrs` to your `Cargo.toml` (MSRV: Rust 1.88):

```sh
cargo add ketrs
```

To enable `GpuStatevector` and GPU-accelerated MPS sweeps (`--features gpu`):

```sh
cargo add ketrs --features gpu
```

!!! note "Build-time shader compiler for `--features gpu`"
    Building from source with `--features gpu` compiles GLSL compute shaders using `glslc`.
    Install `glslc` / `shaderc` from your package manager (`brew install shaderc molten-vk` on macOS) or the [Vulkan SDK](https://vulkan.lunarg.com/),
    or point `GLSLC=/path/to/glslc` to it. Once built, running needs only a Vulkan driver (or MoltenVK on macOS).

---

## Building from Source (Development)

```sh
git clone https://github.com/Aethar01/ketrs.git
cd ketrs

uv venv && uv pip install maturin pytest qiskit qiskit-aer qiskit-qasm3-import numpy scipy
maturin develop --release     # add --features gpu for Vulkan support
cargo test --workspace        # add --all-features for GPU tests
pytest
```

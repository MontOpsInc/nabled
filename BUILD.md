# Building pynabled

## Default build

```bash
maturin develop
```

Builds and installs pynabled in editable mode with default features.

## Building with features

Cargo features control optional capabilities. Use `--features` with maturin:

```bash
# OpenBLAS-backed LAPACK via system libraries
maturin develop --features openblas-system

# Statically linked OpenBLAS provider
maturin develop --features openblas-static

# Netlib LAPACK via system libraries
maturin develop --features netlib-system

# Statically linked Netlib LAPACK provider
maturin develop --features netlib-static

# NVIDIA MAGMA provider (requires CUDA/MAGMA toolchain)
maturin develop --features magma-system

# Parallel CPU kernels
maturin develop --features accelerator-rayon

# WGPU-backed accelerator kernels
maturin develop --features accelerator-wgpu

# Combined
maturin develop --features "openblas-system accelerator-rayon accelerator-wgpu"

# PyArrow/Arrow interop (install pyarrow separately)
maturin develop --features arrow
```

## pip install from source

For source installs via `pip install .`, pass features via `MATURIN_PEP517_ARGS`:

```bash
MATURIN_PEP517_ARGS="--features openblas-system" pip install .
```

Feature flags can be combined in the same string, for example:

```bash
MATURIN_PEP517_ARGS='--features "openblas-system accelerator-rayon arrow"' pip install .
```

Published PyPI wheels use the default `pynabled` Cargo feature set unless the release workflow is
changed. Provider/backend/Arrow feature builds are therefore source-build workflows.

There are no Python extras that enable Rust Cargo features. You must pass `--features` explicitly
when building from source.

## Optional dependencies

- `dev`: pytest, pytest-cov, and pyarrow for local testing/development

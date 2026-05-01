# Building pynabled

## Default build

```bash
maturin develop
```

Builds and installs pynabled in editable mode with default features. Default builds include the
Rust `arrow` feature; Python Arrow workflows still require `pyarrow` at runtime.

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

# Default build without Arrow support, if needed
maturin develop --no-default-features
```

## pip install from source

For source installs, `pynabled` now exposes a thin PEP 517 shim over `maturin`.

Preferred `pip` forms:

```bash
# One provider feature
PYNABLED_PROVIDER=openblas-system python -m pip install --no-binary pynabled pynabled

# Combined LAPACK + MAGMA provider features
PYNABLED_PROVIDER="openblas-system magma-system" python -m pip install --no-binary pynabled pynabled

# Accelerator features
PYNABLED_ACCELERATORS=rayon python -m pip install --no-binary pynabled pynabled

# Escape hatch for additional feature names
PYNABLED_FEATURES=arrow python -m pip install --no-binary pynabled pynabled
```

You can pass the same settings via frontend config settings instead of environment variables:

```bash
python -m pip install --no-binary pynabled \
  -C pynabled-provider="openblas-system,magma-system" \
  -C pynabled-accelerators=rayon \
  pynabled
```

Published PyPI wheels use the default `pynabled` Cargo feature set unless the release workflow is
changed. Arrow Rust support is compiled into the default wheel. Provider/backend feature builds
remain source-build workflows.

`PYNABLED_PROVIDER` / `pynabled-provider` accepts:

- `openblas-system`
- `openblas-static`
- `netlib-system`
- `netlib-static`
- `magma-system`

Values may be space- or comma-separated. At most one LAPACK provider may be selected at a time;
`magma-system` can be combined with one LAPACK provider.

`PYNABLED_ACCELERATORS` / `pynabled-accelerators` accepts:

- `rayon` or `accelerator-rayon`
- `wgpu` or `accelerator-wgpu`

`PYNABLED_FEATURES` / `pynabled-features` is the escape hatch for explicit non-default feature
names.

If you need raw `maturin` control, `build-args` / `MATURIN_PEP517_ARGS` still work. Do not mix
them with the friendly `pynabled-*` settings in the same build.

## uv project configuration

`uv` can hide most of the source-build command surface in project or user config:

```toml
[tool.uv]
no-binary-package = ["pynabled"]
config-settings-package = { pynabled = { pynabled-provider = "openblas-system,magma-system", pynabled-accelerators = "rayon" } }
```

With that in place, `uv sync` or `uv add pynabled` can drive the source build without repeating
the flags on every command.

## Optional dependencies

- `dev`: pytest, pytest-cov, and pyarrow for local testing/development

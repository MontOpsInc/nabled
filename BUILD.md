# Building pynabled

## Default build

```bash
maturin develop
```

Builds and installs pynabled in editable mode with default features.

## Building with features

Cargo features control optional capabilities. Use `--features` with maturin:

```bash
# OpenBLAS-backed LAPACK (faster decompositions)
maturin develop --features openblas-system

# Parallel CPU kernels
maturin develop --features accelerator-rayon

# Combined
maturin develop --features openblas-system,accelerator-rayon

# PyArrow/Arrow interop (requires arrow feature)
maturin develop --features arrow
```

## pip install from source

For source installs via `pip install .`, pass features via `MATURIN_PEP517_ARGS`:

```bash
MATURIN_PEP517_ARGS="--features openblas-system" pip install .
```

**Note:** Python extras (`pip install pynabled[openblas]`) do not automatically map to Cargo features. You must pass `--features` explicitly when building from source.

## Optional dependencies

- `dev`: pytest (for tests)
- `openblas`: build flag only; use `--features openblas-system` with maturin
- `accelerator`: build flag only; use `--features accelerator-rayon` with maturin
- `arrow`: pyarrow (for Arrow interop when built with `--features arrow`)

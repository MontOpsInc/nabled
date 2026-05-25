# Feature Matrix — `nabled` 0.1+

The `nabled` facade is now a thin re-export crate with per-domain opt-in
features. This file is the single source of truth for which feature pulls
which workspace crate / module / domain.

## Default

```toml
[dependencies]
nabled = "0.1"
```

→ enables only the `linalg` feature. `nabled::linalg::*` is available; no
Physical AI or ML modules are compiled.

## Per-domain features

| Cargo feature | Re-exports / enables                               | Workspace crate(s) pulled in |
| ------------- | -------------------------------------------------- | ---------------------------- |
| `linalg`      | `nabled::linalg::*`                                | `nabled-linalg`              |
| `geometry`    | (alias for `linalg`; geometry lives under linalg)  | `nabled-linalg`              |
| `signal`      | `nabled::signal::*`                                | `nabled-linalg` (signal)     |
| `ml`          | `nabled::ml::*`, `nabled::optimization::*`         | `nabled-ml`                  |
| `model`       | `nabled::model::*`                                 | `nabled-model`               |
| `kinematics`  | `nabled::kinematics::*`                            | `nabled-kinematics`, `model` |
| `dynamics`    | `nabled::dynamics::*` (incl. `dynamics::tree::*`)  | `nabled-dynamics`, `kinematics`, `model` |
| `control`     | `nabled::control::*`                               | `nabled-control`             |
| `sensor`      | `nabled::sensor::*`                                | `nabled-sensor`              |
| `sim`         | `nabled::sim::*` (full sim/control/sensor/ml/etc.) | `nabled-sim` + all PAI deps  |
| `physical-ai` | umbrella: enables all of the above                 | full Physical AI vertical    |

## Backend / provider features (orthogonal)

These do not change the public API surface; they select which numerical
provider backs the linalg/decomposition layer. They can be combined with any
of the per-domain features above.

| Cargo feature       | What it enables                                              |
| ------------------- | ------------------------------------------------------------ |
| `blas`              | Internal BLAS-style helper paths                             |
| `lapack-provider`   | Provider-backed decompositions via `ndarray-linalg`          |
| `openblas-system`   | `lapack-provider` against system OpenBLAS                    |
| `netlib-system`     | `lapack-provider` against system Netlib                      |
| `openblas-static`   | `lapack-provider` against vendored OpenBLAS                  |
| `netlib-static`     | `lapack-provider` against vendored Netlib                    |
| `magma-system`      | MAGMA-backed dense path (`lapack-provider` + system MAGMA)   |
| `accelerator-rayon` | CPU-parallel accelerator backend                             |
| `accelerator-wgpu`  | WGPU / GPU accelerator backend                               |

## Interop features

| Cargo feature | Enables                                                          |
| ------------- | ---------------------------------------------------------------- |
| `arrow`       | Arrow extension types over ndarray (`nabled::arrow_interop`)     |
| `test-utils`  | Reusable fixture loaders behind `cfg(feature = "test-utils")`    |

## Migration from `nabled = "0.0.x"` defaults

Before, `default-features = true` implicitly pulled in the full Physical AI
stack (kinematics / dynamics / control / sensor / sim / ml / model). That is
no longer the case.

To restore the old behavior, depend on `nabled` with the `physical-ai`
feature explicitly:

```toml
[dependencies]
nabled = { version = "0.1", features = ["physical-ai"] }
```

Or pick only the domains you actually use, e.g. just kinematics + dynamics:

```toml
[dependencies]
nabled = { version = "0.1", features = ["kinematics", "dynamics"] }
```

This change is intentional pre-1.0 to keep the default surface lean and to
avoid forcing every downstream consumer to pull `ndarray-linalg`, `pyo3`,
`numpy`, and the Physical AI crates when all they wanted was `Array2::svd`.

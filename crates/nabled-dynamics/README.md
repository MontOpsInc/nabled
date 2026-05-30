# nabled-dynamics

Rigid-body dynamics for the nabled Physical AI stack.

`nabled-dynamics` implements inverse dynamics (RNEA), composite rigid-body inertia (CRBA),
and forward dynamics (ABA and related methods) for serial chains and tree models routed
through `nabled-model` branch specs. Spatial algebra utilities live in `spatial`.

## Install

```toml
[dependencies]
nabled-dynamics = "0.0.10"
```

## Key modules

1. `spatial`: spatial vectors, transforms, and inertia primitives.
2. `config`: `DynamicsConfig`, `ForwardDynamicsMethod`.
3. `rnea`, `crba`, `fd`, `id`: serial-chain dynamics kernels.
4. `tree`: `rnea_tree`, `mass_matrix_tree`, `forward_dynamics_tree` (+ `_into` variants).

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg`, `nabled-kinematics`, `nabled-model`.
- **Used by:** `nabled-sim`, facade `nabled` (`physical-ai`).

## Optional features

1. `blas`, `lapack-provider`: forwarded to `nabled-linalg` and `nabled-kinematics`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.

```toml
[dependencies]
nabled-dynamics = { version = "0.0.10", features = ["openblas-system"] }
```

## Example

```rust
use nabled_dynamics::DynamicsConfig;
use nabled_model::fixture::load_planar2r_json;

let fixture = load_planar2r_json()?;
let _model = fixture.to_robot_model::<f64>()?;
let _chain = fixture.to_chain_spec::<f64>()?;
let _config = DynamicsConfig::default();
```

## Docs

1. API docs: <https://docs.rs/nabled-dynamics>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Facade feature: `nabled` with `physical-ai`

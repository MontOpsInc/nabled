# nabled-model

Robot model representation and ingest for the nabled Physical AI stack.

`nabled-model` builds serial and tree robot models from URDF, Denavit–Hartenberg parameters,
and test fixtures. It converts models to `nabled-kinematics::ChainSpec` for FK/IK and to
branch specs for `nabled-dynamics` tree algorithms.

## Install

```toml
[dependencies]
nabled-model = "0.0.11"
```

## Key modules

1. `joint`, `link`, `robot`: core model types and `RobotModel`.
2. `origin`, `tree_model`: URDF joint origins and tree kinematics trait impl.
3. `dh`: `to_chain_spec`, `extract_chain_spec`, dynamics branch extraction.
4. `urdf`: parse URDF strings and files into `RobotModel`.
5. `fixture`: JSON fixtures for planar, 6-DOF, and branched test models.

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg`, `nabled-kinematics`.
- **Used by:** `nabled-dynamics`, `nabled-sim`, facade `nabled` (`physical-ai`).

## Optional features

1. `blas`, `lapack-provider`: forwarded to `nabled-linalg` and `nabled-kinematics`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.

```toml
[dependencies]
nabled-model = { version = "0.0.11", features = ["openblas-system"] }
```

## Example

```rust
use nabled_model::urdf::from_urdf_str;

let model = from_urdf_str::<f64>(include_str!("robot.urdf"))?;
println!("DOF: {}", model.dof());
```

## Docs

1. API docs: <https://docs.rs/nabled-model>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Capability registry: `docs/PHYSICAL_AI_CAPABILITY_REGISTRY.md`
4. Python bindings: `pynabled.physical_ai` (when enabled)

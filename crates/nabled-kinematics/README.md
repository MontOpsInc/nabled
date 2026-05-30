# nabled-kinematics

Ndarray-native robot kinematics for the nabled Physical AI stack.

`nabled-kinematics` provides serial-chain and tree forward kinematics, geometric
Jacobians, and damped least-squares inverse kinematics over DH chains and URDF-derived
tree models. Geometry primitives (`Transform3`, `Rotation3`, SE(3)/SO(3) helpers) come
from `nabled-linalg::geometry`.

## Install

```toml
[dependencies]
nabled-kinematics = "0.0.10"
```

## Key modules

1. `chain`: `ChainSpec`, DH conventions, joint types and limits.
2. `fk`: serial forward kinematics (`fk`, `fk_view`, `fk_into`).
3. `jacobian`: geometric Jacobians for serial chains and trees.
4. `ik`: DLS inverse kinematics with optional joint limits and tree IK.
5. `tree`: tree FK/Jacobians via `KinematicTreeModel` (implemented by `nabled-model`).

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg` (geometry).
- **Used by:** `nabled-model`, `nabled-dynamics`, `nabled-sim`, facade `nabled` (`physical-ai`).

## Optional features

1. `blas`: enables `ndarray/blas` via `nabled-linalg`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`: LAPACK provider paths.
3. `magma-system`: NVIDIA MAGMA provider-backed paths.

```toml
[dependencies]
nabled-kinematics = { version = "0.0.10", features = ["openblas-system"] }
```

## Example

```rust
use nabled_kinematics::{fk, ChainSpec, DhConvention, JointType};
use nabled_model::fixture::load_planar2r_json;
use ndarray::arr1;

let fixture = load_planar2r_json()?;
let chain = fixture.to_chain_spec::<f64>()?;
let q = arr1(&[0.1, -0.2]);
let pose = fk(&chain, &q)?;
let _ = pose.translation;
```

## Docs

1. API docs: <https://docs.rs/nabled-kinematics>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Facade feature: `nabled` with `physical-ai`
4. Python bindings: `pynabled.physical_ai` (when enabled)

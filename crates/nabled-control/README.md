# nabled-control

LTI control synthesis for the nabled Physical AI stack.

`nabled-control` composes `nabled-linalg` decompositions (Sylvester, Lyapunov, eigen) into
discrete-time algebraic Riccati solvers, LQR design, gramians, pole placement, and observer
design—without reimplementing core linear algebra.

## Install

```toml
[dependencies]
nabled-control = "0.0.11"
```

## Key modules

1. `dare`: DARE residual and iterative solvers.
2. `lqr`: discrete-time LQR gain computation.
3. `gramian`: controllability and observability gramians.
4. `pole`: pole placement routines.
5. `observer`: Luenberger-style observer design.

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg`.
- **Used by:** `nabled-sim`, facade `nabled` (`physical-ai`).
- **Independent of:** `nabled-sensor` (estimation stays in the sensor crate).

## Optional features

1. `blas`: enables `ndarray/blas` via `nabled-linalg`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.

```toml
[dependencies]
nabled-control = { version = "0.0.11", features = ["openblas-system"] }
```

## Example

```rust
use nabled_control::lqr::discrete_lqr;
use ndarray::arr2;

let a = arr2(&[[1.0, 0.1], [0.0, 1.0]]);
let b = arr2(&[[0.0], [0.1]]);
let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let r = arr2(&[[1.0]]);
let result = discrete_lqr(&a, &b, &q, &r)?;
let _gain = result.gain;
```

## Docs

1. API docs: <https://docs.rs/nabled-control>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Facade feature: `nabled` with `physical-ai`

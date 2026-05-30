# nabled-sensor

Sensor models and state estimation for the nabled Physical AI stack.

`nabled-sensor` provides linear Kalman filtering, extended Kalman filter scaffolding,
pinhole camera projection, and strapdown IMU integration over ndarray types. Estimation
routines compose `nabled-linalg` linear algebra and stay independent of `nabled-control`.

## Install

```toml
[dependencies]
nabled-sensor = "0.0.10"
```

## Key modules

1. `kalman`: linear Kalman predict/update steps.
2. `ekf`: `EkModel` trait and EKF helpers.
3. `camera`: `PinholeIntrinsics` and projection utilities.
4. `imu`: strapdown IMU propagation helpers.

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg`.
- **Used by:** `nabled-sim`, facade `nabled` (`physical-ai`).

## Optional features

1. `blas`, `lapack-provider`: forwarded to `nabled-linalg`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.

```toml
[dependencies]
nabled-sensor = { version = "0.0.10", features = ["openblas-system"] }
```

## Example

```rust
use nabled_sensor::kalman::{predict, KalmanState};
use ndarray::arr2;

let state = KalmanState {
    mean: ndarray::arr1(&[0.0, 0.0]),
    covariance: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
};
let f = arr2(&[[1.0, 0.1], [0.0, 1.0]]);
let q = arr2(&[[0.01, 0.0], [0.0, 0.01]]);
let predicted = predict(&state, &f.view(), &q.view())?;
let _ = predicted.mean;
```

## Docs

1. API docs: <https://docs.rs/nabled-sensor>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Facade feature: `nabled` with `physical-ai`

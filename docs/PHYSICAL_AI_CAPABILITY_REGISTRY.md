# Physical AI Capability Registry

Last updated: 2026-05-23

Single-owner registry for Physical AI public APIs. Domain crates orchestrate; shared math lives in horizontal layers.

## Locked Boundary Decisions

| Decision | Owner | Notes |
|---|---|---|
| `ChainSpec` | `nabled-kinematics::chain` | model converts via `to_chain_spec()` |
| Column lag/shift | `nabled-ml::stats::lag` | `lag_view`, `shift_columns_into` |
| Spectral/autocorr | `nabled-linalg::signal` | windows, autocorr, cross-corr, FFT |
| Pose error | `kinematics::pose_error*` → `geometry::se3::log` | no local SE(3) log |
| DARE/LQR | `nabled-control` | composes sylvester/lyapunov/eigen |
| Kalman | `nabled-sensor` | no `nabled-control` dep |
| Dynamics FK | `nabled-kinematics` | dynamics calls kinematics, no reimplementation |
| Quat/SO3/SE3 | `nabled-linalg::geometry` | compose-down rule |

## Function Ownership

### `nabled-linalg::geometry`

| Function | Owner |
|---|---|
| `quat::*` | `geometry::quat` |
| `so3::*` | `geometry::so3` |
| `se3::*` | `geometry::se3` |
| `twist::*` | `geometry::twist` |

### `nabled-ml::stats`

| Function | Owner |
|---|---|
| `online::*` | `stats::online` |
| `ewma::*` | `stats::ewma` |
| `rolling::*` | `stats::rolling` |
| `lag::*` | `stats::lag` |

### `nabled-linalg::signal`

| Function | Owner |
|---|---|
| `window::*` | `signal::window` |
| `correlation::*` | `signal::correlation` |
| `fft::*` (feature) | `signal::fft` |

### `nabled-kinematics`

| Function | Owner |
|---|---|
| `chain::*` | `kinematics::chain` |
| `fk::*` | `kinematics::fk` |
| `jacobian::*` | `kinematics::jacobian` |
| `ik::*`, `IkResult`, `IkWorkspace`, `JointLimits` | `kinematics::ik`, `kinematics::chain` |

### `nabled-model`

| Function | Owner |
|---|---|
| `joint::*`, `link::*`, `robot::*` | `nabled-model` |
| `dh::to_chain_spec` | `nabled-model::dh` |
| `urdf::*`, `fixture::load_planar2r_json` | `nabled-model` |

### `nabled-dynamics`

| Function | Owner |
|---|---|
| `spatial::*`, `config::DynamicsConfig`, `rnea::*`, `crba::*`, `fd::*`, `id::*` | `nabled-dynamics` |

### `nabled-control`

| Function | Owner |
|---|---|
| `dare::dare_residual*`, `lqr::*`, `gramian::*`, `pole::*`, `observer::*` | `nabled-control` |

### `nabled-sensor`

| Function | Owner |
|---|---|
| `kalman::*`, `ekf::EkModel`, `camera::PinholeIntrinsics`, `imu::strapdown_*` | `nabled-sensor` |

### Deferred

| Surface | Status |
|---|---|
| `pynabled` Physical AI bindings | deferred |

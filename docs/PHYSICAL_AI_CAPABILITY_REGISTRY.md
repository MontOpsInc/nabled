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
| `tree::*`, `KinematicTreeModel` | `kinematics::tree` (URDF-origin tree FK/Jacobian) |
| `jacobian::*` | `kinematics::jacobian` |
| `ik::*`, `IkResult`, `IkWorkspace`, `JointLimits` | `kinematics::ik`, `kinematics::chain` |
| `ik::inverse_kinematics_tree_dls*` | `kinematics::ik` (tree DLS; full-model `q` in actuated order) |

### `nabled-model`

| Function | Owner |
|---|---|
| `joint::*`, `link::*`, `robot::*` | `nabled-model` |
| `origin::*`, `tree_model::*` | `nabled-model` (URDF joint origins, tree trait impl) |
| `dh::to_chain_spec`, `dh::extract_chain_spec` | `nabled-model::dh` |
| `dh::extract_chain_spec_for_dynamics`, `DynamicsBranchSpec` | `nabled-model::dh` (per-branch RNEA/FD; whole-tree RNEA out of scope) |
| `urdf::*`, `fixture::load_planar2r_json` | `nabled-model` |

### `nabled-dynamics`

| Function | Owner |
|---|---|
| `spatial::*`, `config::DynamicsConfig`, `rnea::*`, `crba::*`, `fd::*`, `id::*` | `nabled-dynamics` |
| `tree::{rnea_tree, mass_matrix_tree, forward_dynamics_tree}` (+ `_into`) | `nabled-dynamics::tree` (branch-routed via `extract_chain_spec_for_dynamics`; whole-tree RNEA still out of scope) |

### `nabled-control`

| Function | Owner |
|---|---|
| `dare::dare_residual*`, `lqr::*`, `gramian::*`, `pole::*`, `observer::*` | `nabled-control` |

### `nabled-sensor`

| Function | Owner |
|---|---|
| `kalman::*`, `ekf::EkModel`, `camera::PinholeIntrinsics`, `imu::strapdown_*` | `nabled-sensor` |

### `nabled-sim`

Cross-crate orchestration (compose-down only — no algorithm reimplementation).

| Module | Composes |
|---|---|
| `context::RobotContext` | `nabled-model` + serial `ChainSpec` validation, `extract_chain_spec_for_dynamics` |
| `sim::semi_implicit_step` | `nabled-dynamics::forward_dynamics`, `nabled-kinematics::fk` |
| `manipulation::TrajectoryIk` | Serial `inverse_kinematics_dls_with_limits`, `pose_error` |
| `manipulation::TrajectoryTreeIk` | `inverse_kinematics_tree_dls_with_limits`, tree FK verify |
| `control_loop::ClosedLoopStep` | `discrete_lqr`, `luenberger_gain` (no sensor dep) |
| `estimation::EstimationPipeline` | `ekf_predict`/`ekf_update`, `InnovationMonitor` → `rolling_covariance` (no control dep) |
| `pipeline::PhysicalAiPipeline` | Sim torque log + stats (S24) |

Docs: `docs/PHYSICAL_AI_ORCHESTRATOR.md`. Facade: `nabled::sim`.

### `pynabled` Physical AI (post-0.0.8)

| Module | Rust source | Python surface |
|---|---|---|
| `pynabled.geometry` | `nabled::linalg::geometry` | quat, SO3, SE3, `Transform3` |
| `pynabled.kinematics` | `nabled::kinematics` | FK, Jacobian, DLS IK |
| `pynabled.model` | `nabled::model` | URDF ingest, fixtures, `to_chain_spec` |
| `pynabled.dynamics` | `nabled::dynamics` | RNEA, mass matrix, forward dynamics |
| `pynabled.control` | `nabled::control` | LQR, DARE, pole placement, observer |
| `pynabled.sensor` | `nabled::sensor` | Kalman, EKF callbacks, camera, IMU |
| `pynabled.signal` | `nabled::linalg::signal` | FFT/autocorr (feature `signal`) |

Parity matrix: `docs/PYNABLED_PHYSICAL_AI_PARITY.md`.

### Tree dynamics and IK scope

| Surface | Status | Notes |
|---|---|---|
| Tree FK/Jacobian | implemented | `kinematics::tree::*`, URDF-origin models |
| Branched-tree IK | implemented | `inverse_kinematics_tree_dls*`; `q` in `actuated_indices()` order |
| Branch-routed tree RNEA / mass / FD | implemented | `dynamics::tree::{rnea_tree, mass_matrix_tree, forward_dynamics_tree}` (+ `_into`); per-branch via serial RNEA/CRBA/FD with scatter to full-model actuated ordering. See `D-MOD-3`. |
| Whole-tree coupled RNEA | out of scope | compose multiple branch calls or use `extract_chain_spec_for_dynamics` + branch `q` slice directly |

### URDF / DH ingestion

| Surface | Status | Notes |
|---|---|---|
| URDF parsing | implemented | `model::urdf::from_urdf_*`; never synthesizes DH parameters. `BodySpec::dh_params = None` for URDF-derived bodies. |
| URDF → tree FK / Jacobian / IK | supported | use `kinematics::tree::*` and `kinematics::ik::inverse_kinematics_tree_dls*`. |
| URDF → DH `ChainSpec` | not supported | `model::dh::to_chain_spec` / `extract_chain_spec` / `extract_chain_spec_for_dynamics` fail loudly when DH parameters are missing on the requested branch. See `D-MOD-2`. |
| Programmatic DH builders | supported | callers can set `BodySpec::dh_params = Some(DhParams { … })` to opt into DH-based serial dynamics. |

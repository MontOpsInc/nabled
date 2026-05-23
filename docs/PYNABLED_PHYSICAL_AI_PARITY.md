# pynabled Physical AI Parity Matrix

Last updated: 2026-05-23

Maps Rust `nabled::{kinematics,model,dynamics,control,sensor,linalg::geometry,linalg::signal}` to Python modules under `pynabled.*`.

## Module Map

| Python module | Rust source | Feature flag | Notes |
|---|---|---|---|
| `pynabled.geometry` | `nabled::linalg::geometry` | default | quat, SO3, SE3 compose/log/exp, `Transform3` |
| `pynabled.kinematics` | `nabled::kinematics` | default | FK, Jacobian, DLS IK (`IkConfig`, `IkResult`) |
| `pynabled.model` | `nabled::model` | default | URDF ingest, fixtures, `to_chain_spec` |
| `pynabled.dynamics` | `nabled::dynamics` | default | RNEA, mass matrix, forward dynamics, `DynamicsConfig` |
| `pynabled.control` | `nabled::control` | default | LQR, DARE, gramian, pole placement, observer |
| `pynabled.sensor` | `nabled::sensor` | default | Kalman, EKF (Python callbacks), camera, IMU |
| `pynabled.signal` | `nabled::linalg::signal` | `signal` | rfft/irfft, autocorrelation, dominant frequency |

## Integration Scenarios

| Scenario | Rust test | Python test | Status |
|---|---|---|---|
| S1 | `s1_planar_2r_fk_jacobian` | `test_s1_planar_2r_fk_jacobian` | Parity |
| S2 | `s2_six_dof_fk_jacobian` | `test_s2_six_dof_fk_jacobian` | Parity |
| S3 | `s3_dls_ik_to_target_pose` | `test_s3_dls_ik_to_target_pose` | Parity |
| S4 | `s4_urdf_model_fk` | `test_s4_urdf_model_fk` | Parity |
| S5–S21 | `physical_ai_integration.rs` | `test_physical_ai_integration.py` | Parity |
| S22 | `s22_y_branch_tree_fk` | — | Rust-only (tree FK) |

Fixtures: `crates/nabled/tests/fixtures/physical_ai/`.

## Allocation Policy

Follow `docs/PYNABLED_ARCHITECTURE.md`:

1. View-first ingress via `real_array1` / `real_array2` helpers.
2. Explicit `out=` / `*_into` for hot paths (dynamics, Kalman, signal FFT).
3. Result carriers (`Transform3`, `KalmanState`, `LqrResult`, `IkResult`) for structured returns.
4. EKF model callbacks require Python callables; transient NumPy carriers are unavoidable per callback contract.

## Release Sequencing

1. Ship PyPI `0.0.8` first (existing `N-PY-*` gate).
2. Bump to `0.0.9` when Physical AI Python surface is validated post-release.
3. Run `just python-quality` (includes `test_physical_ai_integration.py`).

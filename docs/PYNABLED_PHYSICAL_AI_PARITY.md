# pynabled Physical AI Parity Matrix

Last updated: 2026-05-25

Maps Rust `nabled::{kinematics,model,dynamics,control,sensor,linalg::geometry,linalg::signal}` to Python modules under `pynabled.*`.

## Module Map

| Python module | Rust source | `nabled` feature | Notes |
|---|---|---|---|
| `pynabled.geometry` | `nabled::linalg::geometry` | `linalg` (default) | quat, SO3, SE3 compose/log/exp, `Transform3`, `out=` on hot SE3 paths |
| `pynabled.kinematics` | `nabled::kinematics` | `physical-ai` (`kinematics`) | serial + tree FK/Jacobian via Rust `_view` paths, DLS IK (`IkConfig`, `IkResult`, `IkWorkspace`) |
| `pynabled.model` | `nabled::model` | `physical-ai` (`model`) | URDF ingest (DH-free), fixtures, `to_chain_spec`, `extract_chain_spec` — both raise loudly on URDF-derived models |
| `pynabled.dynamics` | `nabled::dynamics` | `physical-ai` (`dynamics`) | RNEA, mass matrix, forward dynamics, `DynamicsConfig`, `out=`. Branch-routed tree variants `rnea_tree`/`mass_matrix_tree`/`forward_dynamics_tree` with `out=` (D-MOD-3) |
| `pynabled.control` | `nabled::control` | `physical-ai` (`control`) | LQR, DARE (incl. `dare_residual_norm` view), gramian, pole placement, observer; view-first ingress (D-MOD-4) |
| `pynabled.sensor` | `nabled::sensor` | `physical-ai` (`sensor`) | Kalman with `out=`, EKF (Python callbacks), camera, IMU `strapdown_predict` view + `_into` (D-MOD-4) |
| `pynabled.signal` | `nabled::linalg::signal` | `signal` | rfft/irfft with `out=`, autocorrelation, dominant frequency |

`pynabled` enables `nabled`'s `physical-ai` umbrella feature by default for
its source builds (see `crates/pynabled/Cargo.toml`).

## Integration Scenarios

| Scenario | Rust test | Python test | Status |
|---|---|---|---|
| S1 | `s1_planar_2r_fk_jacobian` | `test_s1_planar_2r_fk_jacobian` | Parity |
| S2 | `s2_six_dof_fk_jacobian` | `test_s2_six_dof_fk_jacobian` | Parity |
| S3 | `s3_dls_ik_to_target_pose` | `test_s3_dls_ik_to_target_pose` | Parity |
| S4 (rerouted) | `s4_urdf_model_tree_fk` | `test_s4_urdf_model_fk` | Parity — URDF-derived models verify pose via tree FK only; `to_chain_spec` raises (D-MOD-2) |
| S5–S21 | `physical_ai_integration.rs` | `test_physical_ai_integration.py` | Parity |
| S22 | `s22_y_branch_tree_fk` | `test_s22_y_branch_tree_fk` | Parity — `extract_chain_spec` on URDF model now raises |
| S23 | `s23_y_branch_tree_ik` | `test_s23_y_branch_tree_ik` | Parity |
| S26 | `s26_rnea_mass_matrix_tree_y_branch` | `test_s26_rnea_tree_matches_serial_rnea_on_planar2r` | Parity — branch RNEA + branch mass matrix (D-MOD-3) |
| S27 | `s27_forward_dynamics_tree_round_trip` | `test_s27_forward_dynamics_tree_round_trip_planar2r` | Parity — branch FD round-trip through branch RNEA |

Fixtures: `crates/nabled/tests/fixtures/physical_ai/`.

## Allocation Policy

Follow `docs/PYNABLED_ARCHITECTURE.md`:

1. View-first ingress via `real_array1` / `real_array2` helpers — every
   `physical_ai` Python entry point that has a Rust `_view` variant now
   calls it directly via `PyReadonlyArrayN::as_array()` (no
   intermediate `.to_owned()`) (D-MOD-4).
2. Explicit `out=` / `*_into` for hot paths (dynamics including
   `rnea_tree` / `forward_dynamics_tree` / `mass_matrix_tree`, Kalman,
   signal FFT, SE3 log/compose, IK workspace, IMU strapdown predict).
3. Result carriers (`Transform3`, `KalmanState`, `LqrResult`, `IkResult`) for structured returns.
4. EKF model callbacks require Python callables; transient NumPy carriers are unavoidable per callback contract.

## Release Sequencing

1. PyPI `0.0.8` shipped first (existing `N-PY-*` gate).
2. PyPI `0.0.9` carries Physical AI depth: tree FK/IK, `IkWorkspace`, S22–S23 parity, allocation-control on hot paths.
3. Run `just python-quality` (includes `test_physical_ai_integration.py` S1–S23).

## Coverage Snapshot

Python Physical AI submodules (`geometry`, `kinematics`, `model`, `dynamics`, `control`, `sensor`, `signal`) are measured in `scripts/python_quality_gate.sh` after the Physical AI pytest slice; target ≥90% line coverage on those modules.

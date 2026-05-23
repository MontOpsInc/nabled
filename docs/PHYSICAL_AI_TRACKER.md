# Physical AI Execution Tracker

Last updated: 2026-05-23

## Purpose

Track Physical AI domain scaffolding and implementation across multitask rounds.
Operational sequencing for the broader workspace lives in `docs/EXECUTION_TRACKER.md`.

## Status Legend

- **Done**: landed and validated.
- **Next**: active work item.
- **Needed**: not started or blocked.

## Round Map

| Round | Scope | Gate |
|---|---|---|
| 0 | Platform contract, workspace, crate shells, integration skeleton | `cargo check --workspace` |
| 1 | geometry, stats rolling/lag, kinematics FK/Jacobian | S1, S2, S8 |
| 2 | IK, model/URDF, signal window/correlation | S3, S4 |
| 3 | dynamics, control DARE/LQR/gramians/pole/observer, sensor Kalman/EKF/camera/IMU | S5–S21 |
| 4 | Doc hardening, coverage, capability matrix | `just checks` |

## PAI Items

| ID | Item | Status | Notes |
|---|---|---|---|
| PAI-0 | Platform contract + workspace wiring | Done | docs, crate shells, facade re-exports, integration skeleton |
| PAI-1 | `nabled-linalg::geometry` full API | Done | quat/SO3/SE3/twist + unit tests |
| PAI-2 | `nabled-ml::stats` online/ewma/rolling/lag | Done | lag_view/shift only; no spectral ops |
| PAI-3 | `nabled-kinematics` chain/fk/jacobian | Done | `ChainSpec` owner; FK/Jacobian validated S1/S2 |
| PAI-4 | `nabled-kinematics::ik` true DLS | Done | `IkResult`, limits, cold-start S3, bench |
| PAI-5 | `nabled-model` URDF v2 + fixture loader | Done | quick-xml tree, inertials/limits/axis; S4 parity |
| PAI-6 | `nabled-linalg::signal` RealFft pipeline | Done | `RfftSpectrum`, round-trip, windowed_rfft; S12–S14 |
| PAI-7 | `nabled-dynamics` RNEA/CRBA/FD | Done | spatial 6×6, `DynamicsConfig`; S5/S5b pass |
| PAI-8 | `nabled-control` pole/observer/gramians | Done | Ackermann, Luenberger, discrete gramians, `dare_residual`; S9/S10/S15/S16/S21 |
| PAI-9 | `nabled-sensor` Kalman/EKF/camera/IMU | Done | EKF, `PinholeIntrinsics`, strapdown; S7/S11/S17–S19 |
| PAI-10 | Integration scenarios S1–S21 | Done | 22 pass, 0 ignored (`signal` feature for S12–S14) |
| PAI-11 | Spatial inertia + joint subspaces | Done | MT-5: `spatial_inertia_6x6`, `joint_motion_subspace`, gravity vector |
| PAI-12 | JSON fixture loader shared | Done | `load_planar2r_json`, dynamics cases in `2r_planar.json` |
| PAI-13 | Capability matrix + registry sync | Done | Physical AI rows → Implemented |

## Locked Boundaries

See `docs/PHYSICAL_AI_CAPABILITY_REGISTRY.md` for single-owner function table.

## Validation Snapshot (2026-05-23)

- `cargo test --workspace --lib`: pass
- `cargo test -p nabled --test physical_ai_integration --features signal`: 22 passed, 0 ignored
- `cargo +stable clippy --workspace --no-default-features --features "openblas-system accelerator-rayon accelerator-wgpu signal" --all-targets -- -D warnings`: run locally with OpenBLAS link env
- `just checks`: clippy/fmt pass when OpenBLAS available; provider/coverage gates environment-dependent

## Remaining Gaps

1. Full `just checks` provider/coverage gates require OpenBLAS or `netlib-static` link environment.
2. Multi-DOF branched-tree FK beyond serial `ChainSpec` extraction (tree API exists; FK remains serial).
3. Optional dynamics/kinematics bench CI wiring when GPU/OpenBLAS runners available.

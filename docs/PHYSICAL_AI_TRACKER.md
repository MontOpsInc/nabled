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
| 3 | dynamics stubs, control DARE/LQR, sensor Kalman | S6, S7 |
| 4 | Doc hardening, coverage, capability matrix | `just checks` |

## PAI Items

| ID | Item | Status | Notes |
|---|---|---|---|
| PAI-0 | Platform contract + workspace wiring | Done | docs, crate shells, facade re-exports, integration skeleton |
| PAI-1 | `nabled-linalg::geometry` full API | Done | quat/SO3/SE3/twist + unit tests |
| PAI-2 | `nabled-ml::stats` online/ewma/rolling/lag | Done | lag_view/shift only; no spectral ops |
| PAI-3 | `nabled-kinematics` chain/fk/jacobian | Done | `ChainSpec` owner; FK/Jacobian validated S1/S2 |
| PAI-4 | `nabled-kinematics::ik` DLS | Done | BFGS-backed IK; `pose_error` wraps `geometry::se3::log`; S3 pass |
| PAI-5 | `nabled-model` + URDF minimal | Done | serial revolute/prismatic; S4 pass |
| PAI-6 | `nabled-linalg::signal` window/correlation | Done | FFT stub behind feature |
| PAI-7 | `nabled-dynamics` scaffold | Done | RNEA/CRBA/FD stubs; S5 ignored pending reference |
| PAI-8 | `nabled-control` DARE/LQR | Done | no dynamics dep; S6 pass |
| PAI-9 | `nabled-sensor` linear Kalman | Done | no control dep; S7 pass |
| PAI-10 | Integration scenarios S1–S8 | Done | 7 pass, 1 ignored (S5) |

## Locked Boundaries

See `docs/PHYSICAL_AI_CAPABILITY_REGISTRY.md` for single-owner function table.

## Validation Snapshot (2026-05-23)

- `cargo check --workspace`: pass
- `cargo test --workspace --lib`: pass
- `cargo test -p nabled --test physical_ai_integration`: 7 passed, 1 ignored
- `cargo +stable clippy --workspace --no-default-features --features "openblas-system accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings`: pass
- `just checks`: clippy/fmt pass; fails at `test-provider` when system OpenBLAS is unavailable (linker)

## Remaining Gaps

1. S5 RNEA reference implementation and test body.
2. IK from arbitrary initialization (S3 uses warm start via `q_init = q_target`).
3. Full `just checks` provider/coverage gates require OpenBLAS or `netlib-static` link environment.
4. Dynamics/control/sensor stubs beyond landed DARE/LQR/Kalman paths (gramians, EKF, camera, IMU).

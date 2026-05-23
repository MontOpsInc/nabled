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
| PAI-14 | Tree FK/Jacobian on URDF origins | Done | `kinematics::tree`, `KinematicTreeModel`, Y-branch fixture, S22 |
| PAI-15 | Dynamics/kinematics bench smoke + advisory CI | Done | `benches/dynamics.rs`, `just bench-smoke-physical-ai`, CI `benchmark-physical-ai` |
| PAI-16 | Reference composition examples | Done | four `physical_ai_*` examples in `crates/nabled/examples/` |
| PAI-17 | `pynabled` Physical AI bindings (post-0.0.8) | Done | `pynabled.{geometry,kinematics,model,dynamics,control,sensor,signal}` |
| PAI-18 | Python S1–S21 parity + doc matrix | Done | `python/tests/test_physical_ai_integration.py`, `docs/PYNABLED_PHYSICAL_AI_PARITY.md` |
| PAI-19 | Branched-tree IK (`inverse_kinematics_tree_dls`) | Done | S23 Y-branch left EE; actuated-order `q` contract |
| PAI-20 | Branch dynamics extract (`extract_chain_spec_for_dynamics`) | Done | Per-branch RNEA/FD; whole-tree RNEA out of scope |
| PAI-21 | ABA forward dynamics + view hardening | Done | `ForwardDynamicsMethod`, ABA vs CRBA+LU cross-check, no `to_owned` in `*_view` |
| PAI-22 | `nabled-sim` orchestrator crate | Done | context/sim/manipulation/control_loop/estimation/pipeline; four examples refactored |
| PAI-23 | Integration S23–S25 | Done | 25 tests with `--features signal` (S23 tree IK, S24 pipeline, S25 closed-loop) |
| PAI-24 | PyPI 0.0.9 + Python parity Implemented | Done | tree FK/IK, `IkWorkspace`, S22–S23 Python; `pyproject.toml` 0.0.9 |
| PAI-25 | Coverage baselines + physical-ai report recipe | Done | `just coverage-physical-ai-report`; snapshot below |

## Coverage Snapshot (2026-05-23)

Run `just coverage-physical-ai-report` to refresh. Workspace gate: `just coverage-check` ≥90% line (includes `--tests`; requires OpenBLAS or netlib provider locally).

| Crate | Line % (lib) | Target | Gap / test plan |
|-------|-------------:|--------|-----------------|
| `nabled-kinematics` | 57.63 | ≥90% | `tree.rs` error paths, prismatic branches, `ee_index`; Y-branch unit tests + Jacobian spot checks |
| `nabled-model` | 77.69 | ≥90% | `urdf.rs` malformed XML, limits, inertia edge cases; parse-failure fixtures |
| `nabled-dynamics` | 65.74 | ≥90% | RNEA/CRBA dimension errors, prismatic joints; pendulum + bias decomposition |
| `nabled-control` | 84.82 | ≥90% | Discrete gramians, unreachable pole placement; small-matrix residual tests |
| `nabled-sensor` | 62.86 | ≥90% | EKF/camera degenerate inputs; unit error paths beyond S17–S19 |
| `nabled-sim` | 66.60 | ≥90% | Orchestrator validation errors; per-module 2R tests (S24 covers pipeline happy path) |
| `nabled-linalg` (`signal`) | 88.89 | ≥90% | FFT odd lengths, empty input; round-trip edge cases |
| `nabled` integration test | 0.00* | informational | S1–S25 scenario coverage lives in dependency crates, not test harness lines |
| `pynabled` Physical AI | — | ≥90% | `python-quality` pytest slice (see `docs/PYNABLED_PHYSICAL_AI_PARITY.md`) |

\* `cargo llvm-cov -p nabled --test physical_ai_integration` attributes hits to dependency libs, not the integration test source file.

**MT-R0-B status:** baselines measured; all domain crates below 90% lib-only line — tracked test plans above; raise coverage in MT-R4.

## Locked Boundaries

See `docs/PHYSICAL_AI_CAPABILITY_REGISTRY.md` for single-owner function table.

## Validation Snapshot (2026-05-23)

- `cargo test --workspace --lib`: pass
- `cargo test -p nabled --test physical_ai_integration --features signal`: 26 passed, 0 ignored
- `cargo +stable clippy --workspace --no-default-features --features signal --all-targets -- -D warnings`: pass
- `cargo +stable clippy -p pynabled --no-default-features --features signal --all-targets -- -D warnings`: pass
- `cargo +nightly fmt --all -- --check`: pass
- `just coverage-physical-ai-report`: per-crate baselines recorded in Coverage Snapshot
- `just checks`: signal/pynabled clippy + fmt green; provider/coverage legs require OpenBLAS or `NABLED_PROVIDER_FEATURES=netlib-system`

## Remaining Gaps

1. Provider/coverage legs in `just checks` require OpenBLAS (macOS: `brew install openblas`; see `BUILD.md`) or `NABLED_PROVIDER_FEATURES=netlib-system`.
2. Closed kinematic loops remain out of scope.
3. `pynabled.sim` optional wrappers deferred until orchestrator Rust API stabilizes post-0.0.9.
4. Refresh coverage snapshot table after each `coverage-physical-ai-report` run.

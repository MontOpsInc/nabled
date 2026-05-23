# Physical AI Orchestrator (`nabled-sim`)

Last updated: 2026-05-23

## Role

`nabled-sim` is the **composition layer** for Physical AI workflows spanning model, kinematics, dynamics, control, sensor, and horizontal stats/signal helpers. It **compose-down only** — algorithms stay in owner crates (see `docs/PHYSICAL_AI_CAPABILITY_REGISTRY.md`).

Facade re-export: `nabled::sim` → `nabled_sim::*`.

## Module map

| Module | Types | Composes |
|--------|-------|----------|
| `context` | `RobotContext` | Validates `model.dof() == chain.num_joints()` for serial paths |
| `sim` | `SimState`, `SimConfig`, `semi_implicit_step` | `forward_dynamics` + semi-implicit Euler; optional EE pose log |
| `manipulation` | `TrajectoryIk`, `TrajectoryTreeIk` | Batch DLS IK + FK verify; tree variant uses `inverse_kinematics_tree_dls` |
| `control_loop` | `ClosedLoopStep`, `ClosedLoopPlant` | `discrete_lqr` + `luenberger_gain` (no `nabled-sensor` import) |
| `estimation` | `EstimationPipeline`, `InnovationMonitor` | EKF predict/update; optional `rolling_covariance` on innovations |
| `pipeline` | `PhysicalAiPipeline`, `TorqueLog` | Sim N steps → torque log → rolling stats |

## Boundary rules

1. **Sensor ⊥ control** at crate level: `nabled-sensor` and `nabled-control` do not depend on each other; orchestrator sequences both only in application pipelines.
2. **Tree vs serial**: tree FK/IK use `KinematicTreeModel`; branch dynamics use `extract_chain_spec_for_dynamics`.
3. **Closed loops**: kinematic loops remain out of scope; whole-tree RNEA is out of scope — per-branch extraction is supported.
4. **Allocation**: prefer inner crate `*_into` / workspace types when exposed (`IkWorkspace`, FD buffers).

## Workflow (sim + stats)

```mermaid
flowchart LR
  ctx[RobotContext::validate]
  sim[semi_implicit_step]
  log[TorqueLog]
  stats[rolling_covariance]

  ctx --> sim --> log --> stats
```

Integration scenario **S24** exercises sim → torque log → rolling covariance.

## Examples

All four `physical_ai_*` examples in `crates/nabled/examples/` use orchestrator modules:

| Example | Module |
|---------|--------|
| `physical_ai_planar2r_sim` | `sim` + `context` |
| `physical_ai_trajectory_ik` | `manipulation::TrajectoryIk` |
| `physical_ai_lqr_observer` | `control_loop::ClosedLoopStep` |
| `physical_ai_ekf_fusion` | `estimation::EstimationPipeline` |

## Python (optional, post-0.0.9)

Thin `pynabled.sim` wrappers planned once Rust API stabilizes; not required for 0.0.9 core parity.

## Tests

- Unit tests per module on 2R / 6-DOF fixtures (`crates/nabled-sim/src/*/tests`).
- **S24**: orchestrator sim + rolling torque covariance (`physical_ai_integration`).
- **S25** (optional): closed-loop double integrator via `control_loop`.

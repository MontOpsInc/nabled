# nabled-sim

Cross-crate orchestration for Physical AI workflows in the nabled stack.

`nabled-sim` composes domain crates (`nabled-model`, `nabled-kinematics`, `nabled-dynamics`,
`nabled-control`, `nabled-sensor`) and horizontal layers (`nabled-ml::stats`, optional
`nabled-linalg::signal`) without reimplementing their algorithms. Use it for validated
robot context, simulation steps, batch IK, closed-loop control, and estimation pipelines.

## Install

```toml
[dependencies]
nabled-sim = "0.0.10"
```

## Key modules

1. `context`: `RobotContext` — model validation and chain/dynamics extraction.
2. `sim`: semi-implicit integration calling dynamics FK/FD.
3. `manipulation`: batch IK and pose-driven workflows.
4. `control_loop`: closed-loop control wiring.
5. `estimation`: filter pipelines over sensor models.
6. `pipeline`: higher-level workflow composition.

## Crate graph

- **Depends on:** all Physical AI domain crates plus `nabled-core`, `nabled-linalg`, `nabled-ml`.
- **Used by:** facade `nabled` (`physical-ai`), `pynabled.physical_ai`.

## Optional features

1. `blas`, `lapack-provider`: forwarded across dependent crates.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.
3. `signal`: enables `nabled-linalg/signal` for filter/signal paths.

```toml
[dependencies]
nabled-sim = { version = "0.0.10", features = ["openblas-system", "signal"] }
```

## Example

```rust
use nabled_dynamics::DynamicsConfig;
use nabled_model::fixture::load_planar2r_json;
use nabled_sim::context::RobotContext;

let fixture = load_planar2r_json()?;
let model = fixture.to_robot_model::<f64>()?;
let chain = fixture.to_chain_spec::<f64>()?;
let ctx = RobotContext::new(model, chain, DynamicsConfig::default());
ctx.validate()?;
```

## Docs

1. API docs: <https://docs.rs/nabled-sim>
2. Orchestrator design: `docs/PHYSICAL_AI_ORCHESTRATOR.md`
3. Workspace repo: <https://github.com/MontOpsInc/nabled>
4. Facade feature: `nabled` with `physical-ai`

# nabled-sim

Cross-crate orchestration layer for Physical AI workflows in the nabled stack.

`nabled-sim` composes domain crates (`nabled-model`, `nabled-kinematics`, `nabled-dynamics`,
`nabled-control`, `nabled-sensor`) and horizontal layers (`nabled-ml::stats`, optional
`nabled-linalg::signal`) without reimplementing their algorithms.

See `docs/PHYSICAL_AI_ORCHESTRATOR.md` for module boundaries and workflow diagrams.

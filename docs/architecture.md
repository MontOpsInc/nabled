# Architecture

## Objective

Build a high-performance, ndarray-native numerical stack with clean composition semantics.

## Workspace Plan

```
.
├── Cargo.toml                 # workspace root
├── crates/
│   ├── nabled-core/           # shared types, traits, numerics, errors
│   ├── nabled-linalg/         # decompositions, solves, matrix functions
│   ├── nabled-ml/             # pca, regression, iterative, jacobian
│   └── nabled/                # facade/re-export crate
└── docs/
```

## Current Reality

This document describes the target architecture. The repository is currently in transition.

1. The root manifest is a virtual workspace manifest.
2. Domain implementation now lives under `crates/nabled-linalg` and `crates/nabled-ml`.
3. Facade exports and binary/report tooling live under `crates/nabled/src/`.
4. `docs/STATUS.md` is the authoritative migration-state snapshot.

## Layer Responsibilities

1. `nabled-core`
   1. Numeric trait bounds and shared error categories.
   2. Array validation helpers and shape checks.
   3. Common utilities that do not pull higher-level algorithm dependencies.
2. `nabled-linalg`
   1. Linear algebra kernels and decompositions.
   2. Deterministic algorithm behavior and stability checks.
   3. Matrix/vector operations that are domain primitives for ML.
3. `nabled-ml`
   1. Pipeline-oriented algorithms built from `nabled-linalg`.
   2. Feature engineering and statistical routines.
   3. APIs for common ML workflows over vectors/matrices.
4. `nabled`
   1. Stable user-facing API surface.
   2. Re-exports from `core`, `linalg`, `ml`.

## API Design Rules

1. Favor view-based inputs where possible.
2. Offer allocation-optional variants for hot paths.
3. Keep error semantics explicit and stable.
4. Avoid backend-specific leakiness in public API signatures.

## Performance Rules

1. No implicit copy-heavy conversion in compute kernels.
2. Keep data in ndarray-native structures through entire call chains.
3. Benchmark algorithm kernels independently from orchestration code.

# Decisions

## Locked Decisions

1. Canonical compute substrate is `ndarray`.
2. Public APIs are pure numerical APIs over ndarray types.
3. `nabled` does not depend on Arrow types.
4. Workspace structure is required for long-term scale.
5. No hidden data conversion in hot compute paths.
6. Quality gates remain strict: pedantic linting, CI parity, and coverage >= 90%.

## API Purity Model

1. A function accepts ndarray inputs and returns ndarray outputs (or scalar/error outputs).
2. Any additional controls are explicit function arguments.
3. No assumptions about calling context (database, Arrow, SQL, transport layer).

## Data Types

1. Vector: `Array1<T>`, `ArrayView1<'a, T>`, `ArrayViewMut1<'a, T>`.
2. Matrix: `Array2<T>`, `ArrayView2<'a, T>`, `ArrayViewMut2<'a, T>`.
3. Higher-rank tensors (future): `ArrayD<T>` and fixed-rank aliases as needed.
4. Complex support: `num_complex::Complex32/Complex64` where algorithms are mathematically valid.

## Near-Term Non-Goals

1. Cross-library interop adapters.
2. Python bindings.
3. Arrow integration inside `nabled`.

These are deferred until the ndarray-first core is complete and stable.

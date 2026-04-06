# Pynabled Parity Matrix

Last updated: 2026-04-05

## Purpose

This document is the authoritative parity target for `feat/pynabled-bindings`.

It answers four questions:

1. What Rust `nabled` / `nabled::arrow` surface is in scope for Python release?
2. What does `pynabled` expose today?
3. Where is parity partial or missing?
4. What must be true before `pynabled` can be called production-ready with no unapproved gaps?

Operational sequencing still lives in `docs/EXECUTION_TRACKER.md`.

Boundary architecture and carrier/copy/result contracts live in `docs/PYNABLED_ARCHITECTURE.md`.

## Source Of Truth

Use code, not memory:

1. Rust facade/domain truth:
   - `crates/nabled/src/lib.rs`
   - `crates/nabled/src/arrow.rs`
   - `crates/nabled-linalg/src/`
   - `crates/nabled-ml/src/`
2. Python binding truth:
   - `crates/pynabled/src/lib.rs`
   - `crates/pynabled/src/linalg/`
   - `crates/pynabled/src/ml/`
   - `crates/pynabled/src/sparse/`
   - `crates/pynabled/src/arrow.rs`
   - `python/pynabled/__init__.py`
3. Build/feature truth:
   - `crates/pynabled/Cargo.toml`
   - `pyproject.toml`
   - `BUILD.md`

## Parity Rules

1. Python release scope is parity with the current admitted Rust facade surface, not parity with an older branch snapshot.
2. “No gaps / no compromises” means no silent omission of a Rust capability family that is in admitted scope.
3. `_view` helpers do not need 1:1 Python names, but Python must preserve the same zero-copy or explicit-copy contract.
4. Rust `*_into` / workspace hot-path semantics do not need identical names, but Python must expose equivalent allocation-control semantics where they materially affect performance.
5. Provider/backend/build behavior must be exposed truthfully. Hidden or undocumented feature loss is not acceptable for release.
6. Arrow/PyArrow interop must follow the same canonical `ndarrow` carrier policy as `nabled::arrow`.

## Status Legend

- `Full`: current Python surface is sufficient for release parity in that row.
- `Partial`: Python exposes meaningful functionality, but release parity is not satisfied.
- `Missing`: Python exposes none of the admitted Rust capability family.

## Domain Parity Matrix

| Domain | Rust admitted release surface | Current `pynabled` surface | Status | Release requirement |
|---|---|---|---|---|
| `vector` | Real dot/norm/cosine/distance, workspace-backed pairwise paths, and batched dot/norm/cosine/distance/normalize; admitted complex surface covers Hermitian dot, norm, cosine, and the corresponding batched rows | `dot`, `l2_norm`, `cosine_similarity`, `cosine_distance`, `pairwise_l2_distance`, `pairwise_cosine_similarity`, `pairwise_cosine_distance`, and batched dot/norm/cosine/distance/normalize now accept both `float32` and `float64`, while the admitted complex batch/vector rows (`dot`, `l2_norm`, `cosine_similarity`, `batched_dot`, `batched_l2_norm`, `batched_cosine_similarity`, `batched_normalize`) also accept `complex128` | Partial | Add Python allocation-control/workspace equivalents for heavy pairwise/batched paths and keep any future Rust-admitted complex distance rows in parity if the Rust surface expands |
| `matrix` | Real and complex `matvec` / `matmat` plus real batched row matvec, batched matmat, and broadcast-left/right batched matmat, with `*_into` and backend-dispatched hot paths | `matvec` and `matmat` now accept `complex128` as well as real `float32` / `float64`, and the admitted real batch kernel surface now includes `batched_row_matvec`, `batched_matmat`, `batched_matmat_broadcast_right`, and `batched_matmat_broadcast_left` | Partial | Add Python-visible allocation-control/back-end contract for hot kernels; current Rust batch-kernel admission is real-only, so this row no longer tracks a missing Python complex-batch surface |
| `svd` | Real and complex decomposition, toleranced/truncated variants, pseudo-inverse, null space, reconstruction, rank, condition number, view-first execution | Real decomposition/truncated/pinv/reconstruct/rank/condition/null-space now accept both `float32` and `float64`; direct complex parity now covers `svd_decompose`, reconstruction, rank, and condition number with typed `SvdResult` objects | Partial | Add the remaining admitted complex/configurable rows (`truncated`, pseudo-inverse, null-space, richer toleranced/config entrypoints) |
| `qr` | Real and complex full/reduced/pivoted QR, least-squares, reconstruction, condition number | `qr_decompose` now accepts `complex128`; real decomposition + least-squares accept both `float32` and `float64` and return typed `QrResult` objects | Partial | Add reduced/pivoted/reconstruct/condition surfaces and the remaining admitted complex QR breadth |
| `lu` | Real and complex decompose/solve/inverse/determinant/log-determinant plus mixed-precision solve helpers | Real decompose/solve/inverse/determinant now accept both `float32` and `float64`, with decomposition returning typed `LuResult` objects; direct complex parity now covers solve / inverse / determinant | Partial | Add complex decomposition, log-determinant, and mixed-precision policy |
| `cholesky` | Real and complex decompose/solve/inverse with view and `*_into` coverage | `cholesky_decompose`, `cholesky_solve`, and `cholesky_inverse` now accept `complex128` as well as real `float32` / `float64`, with decomposition returning typed `CholeskyResult` objects | Partial | Add allocation-control semantics for repeated solves and any remaining admitted complex/reuse rows |
| `eigen` | Real symmetric/generalized/non-symmetric, balancing, left/right bi-eigen, complex non-symmetric | Real symmetric/generalized/non-symmetric now accept both `float32` and `float64`, return typed result objects, real nonsymmetric Python egress preserves complex arrays directly, and non-symmetric direct complex parity now exists for `complex128` inputs | Partial | Add balancing, bi-eigen, and the remaining admitted symmetric/generalized complex/eigen metadata surfaces |
| `schur` | Real and complex Schur decomposition | `schur_compute` now accepts `complex128` as well as real `float32` / `float64` and returns typed `SchurResult` objects | Partial | Add any remaining admitted complex config/reuse semantics |
| `polar` | Real and complex polar decomposition | `polar_compute` now accepts `complex128` as well as real `float32` / `float64` and returns typed `PolarResult` objects | Partial | Add any remaining admitted complex config/reuse semantics |
| `sylvester` / `lyapunov` | Real and complex solves, mixed-precision variants, `*_into`, reusable workspace semantics | `sylvester_solve` and `lyapunov_solve` now accept `complex128` as well as real `float32` / `float64` | Partial | Add mixed-precision coverage and Python allocation-control/workspace equivalent |
| `matrix_functions` | Real and complex `exp`, `log`, `power`, `sign` families with `*_into` and workspace-backed variants | Direct complex parity now covers admitted `complex128` rows for `matrix_exp`, `matrix_exp_eigen`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_power`, and `matrix_sign`; real rows still accept both `float32` and `float64` and preserve real output dtype | Partial | Add the remaining admitted complex/configurable rows and equivalent allocation-control semantics |
| `orthogonalization` | Modified Gram-Schmidt over real and complex matrices plus real classic Gram-Schmidt | `gram_schmidt` now accepts `complex128`, and both `gram_schmidt` / `gram_schmidt_classic` accept real `float32` and `float64` | Full | Hold the current modified/classic Gram-Schmidt contract stable unless Rust admission expands |
| `batched` | Real batched QR/SVD/LU/Cholesky/symmetric eigen over matrix stacks with canonical result structs | Real batched QR/SVD/LU/Cholesky/symmetric eigen now accept both `float32` and `float64` and return lists of typed result objects | Full | Hold the typed result-object contract stable unless Rust later admits complex batched decomposition rows |
| `sparse` | CSR/CSC/COO formats, conversion, sparse-dense and sparse-sparse products, iterative solvers, direct sparse LU, preconditioners, factorization reuse | `pynabled` now exposes first-class `CsrMatrix`, `CscMatrix`, and `CooMatrix` carriers with SciPy-compatible ingress, explicit `CSR -> CSC`, `CSC -> CSR`, and `COO -> CSR` conversions, native CSC matvec, sparse-dense matmat, sparse-sparse matmat returning canonical `CsrMatrix`, direct matrix/top-level Jacobi, Gauss-Seidel, conjugate-gradient, Jacobi-PCG, direct `BiCGSTAB`, and IC0-PCG solve entrypoints, reusable `JacobiPreconditioner`, reusable `ILU0` / `ILUT` / `ILUK` / `IC0` / `ILDL0` factors, direct sparse LU reuse including multi-RHS solves, reusable GMRES / `BiCGSTAB` solve paths over the landed `ILU0` / `ILUT` / `ILUK` / `ILDL0` factors for both single-RHS and multi-RHS workloads, and reusable IC0-backed `pcg_solve(...)` on the `IC0Factorization` object itself | Partial | Add the remaining sparse solver/config breadth (now mainly the still-unbound direct config/convenience rows around ILUT/ILUK-backed workflows), keep non-CSR carrier behavior explicit where normalization to CSR still occurs, and finish any remaining release-grade sparse docs/result contracts |
| `tensor` | Cube kernels, higher-rank last-axis ops, axis permutation/contraction, N-D batched matmul, real/complex breadth, HOSVD/HOOI/Tucker/CP/TT/einsum families | Direct NumPy tensor kernels and decomposition/network breadth now accept both `float32` and `float64`, basic tensor kernels/einsum have complex bindings, and tensor result objects preserve dtype across HOSVD / HOOI / Tucker / CP / TT workflows | Full | Hold the direct NumPy tensor contract stable; Arrow-native tensor carriers remain tracked under the `arrow` row |
| `iterative` | Real and complex dense `conjugate_gradient` / `gmres` with config + view-first paths | Real and complex `conjugate_gradient`, `gmres`; real dense solves now accept `float32` and `float64` | Partial | Add richer config/result exposure and close remaining non-dense / higher-level `f32` gaps around iterative-adjacent workflows |
| `jacobian` | Numerical Jacobian, central Jacobian, gradient, Hessian | Forward/central Jacobian, gradient, Hessian for Python callables now accept both `float32` and `float64`; callback ingress/egress preserves real dtype, and the shared `float32` finite-difference default step is now stable enough for Hessian-capable workflows | Partial | Document callback/result contracts for production API and keep the `float32` numerical contract explicit |
| `optimization` | Line search, gradient descent, Adam, momentum, RMSProp, projected GD, SGD, BFGS, real + complex | Real and complex line search / gradient descent / Adam / momentum / RMSProp / projected GD / SGD / BFGS via Python callables; real callable flows now accept both `float32` and `float64`, preserve dtype end-to-end, and use `float32`-appropriate default convergence/epsilon policy instead of raw `float64` defaults | Partial | Stabilize callback/config ergonomics and document optimizer contracts |
| `pca` | Real and complex `compute`, `transform`, `inverse_transform` | Real and complex `compute_pca`, `pca_transform`, `pca_inverse_transform`; real flows now accept `float32` and `float64`, and PCA now returns typed `PcaResult` objects used directly by transform/inverse-transform helpers | Partial | Keep the result-object contract stable while remaining numerics parity lands |
| `regression` | Real and complex linear regression with canonical result structs | Real and complex `linear_regression`; real flows now accept `float32` and `float64`, and regression now returns typed `RegressionResult` objects including coefficients, fitted values, residuals, and `r_squared` | Partial | Keep the result-object contract stable while remaining numerics parity lands |
| `stats` | Real and complex means/centering/covariance/correlation | Real and complex means/centering/covariance/correlation; real flows now accept `float32` and `float64` | Partial | Document complex-stat contracts clearly and preserve dtype/copy semantics consistently in the remaining higher-level rows |
| `arrow` / `PyArrow` | Full admitted `nabled::arrow` surface across dense, sparse, tensor, ML, and Arrow-admitted result contracts | Current bridge still only exposes `arrow_dot`, `arrow_l2_norm`, and `arrow_svd_decompose`, but those rows now accept both `float32` and `float64`; `arrow_svd_decompose` returns typed `SvdResult` objects and preserves real dtype on current NumPy egress | Partial | Reach full admitted `nabled::arrow` parity using canonical `ndarrow` carriers and explicit zero-copy contracts |

## Cross-Cutting Parity Matrix

| Axis | Rust baseline | Current Python state | Status | Release requirement |
|---|---|---|---|---|
| Real dtype breadth | Broad `f32` + `f64` across linalg/ml/tensor domains | The currently exposed real-valued Python surface now admits both `float32` and `float64` across dense, sparse baseline, ML, direct NumPy tensor, and current Arrow rows, with dtype-preserving array outputs on the current result-bearing paths | Full | Maintain this contract on every newly bound Python row and keep mixed real dtypes failing explicitly |
| Complex breadth | Broad complex parity across dense, tensor, and ML/stat domains; limited to admitted Rust APIs | Complex ML/tensor coverage is broad, direct dense linalg/decomposition/function parity is meaningful across the core vector/matrix/decomposition surface, and admitted complex batched vector kernels are now exposed; remaining gaps are concentrated in configurable/reusable dense rows and the still-unbound admitted complex families | Partial | Expose the remaining admitted complex release surface or explicitly narrow Rust admission first |
| View / copy contract | Rust is view-first across hot paths | View-based dense Python APIs now admit strided / Fortran-order NumPy inputs, but higher-level result/tensor/sparse paths still materialize owned arrays in places | Partial | Preserve zero-copy ingress where possible and make unavoidable copies explicit and documented |
| Allocation control | Rust exposes `*_into` and reusable workspace/result helpers in hot domains | No `out=` / workspace / reusable-plan equivalent | Missing | Add Python allocation-control semantics for performance-critical paths |
| Result object fidelity | Rust returns typed result structs for decompositions and higher-level workflows | Python now returns typed result objects across current decomposition, batched, PCA/regression, tensor, Arrow SVD, and reusable sparse factorization/preconditioner workflows; sparse iterative reuse now also stays attached to those reusable factor objects, and the remaining fidelity gaps are concentrated in config/result-rich iterative workflows outside the landed sparse reuse paths | Partial | Preserve release-relevant metadata and finish the remaining config/result-rich solver workflows |
| Sparse data carriers | Rust uses typed sparse matrix structs and admitted Arrow sparse carriers | Python now has first-class `CsrMatrix`, `CscMatrix`, and `CooMatrix` carriers with preserved `int32` / `int64` indices, explicit normalization controls, SciPy-compatible ingress, and reusable sparse factorization/preconditioner objects round-tripping back through canonical carriers; some non-CSR operations still normalize explicitly to CSR when Rust only admits the CSR form for that kernel | Partial | Keep the multi-carrier sparse contract stable, document/limit non-CSR normalization points explicitly, and align any remaining reusable sparse result objects with the canonical carriers |
| Arrow carriers | Rust uses canonical `ndarrow` carriers across admitted Arrow workflows | Python Arrow bridge now handles `float32` / `float64` primitive arrays and fixed-size-list matrices for the currently exposed rows, but it is still only a small subset and not yet rebuilt around the full canonical `ndarrow` carrier set | Partial | Use the same canonical `ndarrow` carrier set and explicit ingress/egress contracts as Rust |
| PyArrow egress | Rust Arrow facade can stay Arrow-native where natural | Python `arrow_svd_decompose` converts results to NumPy | Partial | Keep Arrow-native egress where the Rust facade already defines an Arrow-native contract |
| Provider feature exposure | Rust facade admits `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system` | Python exposes only `openblas-system` | Partial | Expose/document truthful source-build paths for admitted provider features |
| Backend feature exposure | Rust facade admits `accelerator-rayon` and `accelerator-wgpu` | Python exposes only `accelerator-rayon` | Partial | Expose/document truthful backend build paths for admitted backends |
| Feature UX truthfulness | Rust feature names and behavior are explicit | Python extras do not fully map to Cargo features today | Partial | Ensure Python install/build UX matches actual Cargo feature behavior |

## Release-Blocking Gaps Locked By This Matrix

These are the gaps that directly drive `N-PY-003..N-PY-008`:

1. Sparse breadth is now materially narrower: remaining work is concentrated in the still-unbound sparse solver/config/result rows beyond the landed CSR/CSC/COO carriers, direct iterative surface, and reusable factor workflows.
2. Remaining dense parity work is now narrower: Python still needs the remaining admitted configurable/reusable dense rows plus any still-unbound admitted complex families, but the primitive vector/matrix breadth is materially closer to parity.
3. Missing Arrow/PyArrow breadth: nearly the full admitted `nabled::arrow` surface.
4. Missing allocation-control semantics for hot Python workflows.
5. Missing truthful provider/backend feature exposure.
6. Remaining result-fidelity gaps are now narrower: config/result-rich iterative workflows still need production-grade result contracts.

## Implementation Order Derived From This Matrix

1. `N-PY-003`: continue from the landed ML + tensor parity passes:
   - remaining admitted dense configurable/reusable gaps and still-unbound complex rows
   - remaining sparse solver/config breadth beyond the landed CSR/CSC/COO carriers, direct iterative surface, and reusable factor workflows
   - remaining result-fidelity gaps in higher-level workflows
2. `N-PY-004`: expand Arrow/PyArrow to the full admitted `nabled::arrow` surface.
3. `N-PY-005`: make provider/backend exposure truthful and usable from Python source builds.
4. `N-PY-007`: add allocation-control semantics and eliminate avoidable copy/layout regressions.
5. `N-PY-006` and `N-PY-008`: lock the release with tests, coverage, docs, and supply-chain hardening.

## Definition Of Done For This Document

When updating this matrix:

1. Compare against current Rust code, not older audit notes.
2. Record release expectations explicitly instead of implying them.
3. Update `docs/EXECUTION_TRACKER.md` in the same change set when a row moves materially.

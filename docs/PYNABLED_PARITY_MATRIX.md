# Pynabled Parity Matrix

Last updated: 2026-04-03

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
| `vector` | Real and complex dot/norm/cosine/distance plus batched dot/norm/cosine/distance/normalize and workspace-backed pairwise paths | Real `dot`, `l2_norm`, `cosine_similarity`, `pairwise_l2_distance`, `pairwise_cosine_similarity` | Partial | Add `f32` + complex parity, batched vector APIs, normalization/distance breadth, and an allocation-control story for heavy pairwise/batched paths |
| `matrix` | Real and complex `matvec`, `matmat`, batched row matvec, batched matmat, broadcasted batch matmat, `*_into`, backend-dispatched hot paths | Real `matvec`, `matmat`, `batched_row_matvec`, `batched_matmat` | Partial | Add `f32` + complex parity, missing broadcasted batch families, and Python-visible allocation-control/back-end contract for hot kernels |
| `svd` | Real and complex decomposition, toleranced/truncated variants, pseudo-inverse, null space, reconstruction, rank, condition number, view-first execution | Real decomposition/truncated/pinv/reconstruct/rank/condition/null-space only | Partial | Add `f32` + complex parity, toleranced/configurable entrypoints, and preserve canonical result metadata/contracts |
| `qr` | Real and complex full/reduced/pivoted QR, least-squares, reconstruction, condition number | Real decomposition + least-squares only | Partial | Add reduced/pivoted/reconstruct/condition surfaces and complex parity |
| `lu` | Real and complex decompose/solve/inverse/determinant/log-determinant plus mixed-precision solve helpers | Real decompose/solve/inverse/determinant only | Partial | Add `f32` + complex parity, log-determinant, mixed-precision policy, and truthful result metadata |
| `cholesky` | Real and complex decompose/solve/inverse with view and `*_into` coverage | Real decompose/solve/inverse only | Partial | Add `f32` + complex parity and allocation-control semantics for repeated solves |
| `eigen` | Real symmetric/generalized/non-symmetric, balancing, left/right bi-eigen, complex non-symmetric | Real symmetric/generalized/non-symmetric only | Partial | Add balancing, bi-eigen, `f32` breadth where admitted, and complex non-symmetric parity |
| `schur` | Real and complex Schur decomposition | Real Schur only | Partial | Add complex parity |
| `polar` | Real and complex polar decomposition | Real polar only | Partial | Add complex parity |
| `sylvester` / `lyapunov` | Real and complex solves, mixed-precision variants, `*_into`, reusable workspace semantics | Real `sylvester_solve`, `lyapunov_solve` only | Partial | Add complex parity, mixed-precision coverage, and Python allocation-control/workspace equivalent |
| `matrix_functions` | Real and complex `exp`, `log`, `power`, `sign` families with `*_into` and workspace-backed variants | Real `matrix_exp`, `matrix_exp_eigen`, `matrix_log_taylor`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_power`, `matrix_sign` | Partial | Add complex parity, `f32` breadth, and equivalent allocation-control semantics |
| `orthogonalization` | Real and complex Gram-Schmidt variants | Real `gram_schmidt`, `gram_schmidt_classic` only | Partial | Add complex parity |
| `batched` | Batched QR/SVD/LU/Cholesky/symmetric eigen over matrix stacks with canonical result structs | Real batched QR/SVD/LU/Cholesky/symmetric eigen only | Partial | Add admitted dtype/complex breadth and preserve canonical result semantics |
| `sparse` | CSR/CSC/COO formats, conversion, sparse-dense and sparse-sparse products, iterative solvers, direct sparse LU, preconditioners, factorization reuse | Raw-buffer CSR `matvec`, `jacobi_solve`, `pcg_solve` only | Partial | Add admitted sparse format breadth, sparse products, iterative breadth, preconditioners/factorization reuse, direct sparse LU, and a first-class Python sparse carrier contract |
| `tensor` | Cube kernels, higher-rank last-axis ops, axis permutation/contraction, N-D batched matmul, real/complex breadth, HOSVD/HOOI/Tucker/CP/TT/einsum families | Real tensor decomposition/network breadth is now broadly bound (`hosvd3`, `hosvd_nd`, `hooi_nd`, Tucker helpers, rank-3/N-D `cp_als` with diagnostics/reporting, TT-SVD/algebra/reconstruction) and basic tensor kernels/einsum now have complex bindings | Partial | Close remaining `f32` breadth and stabilize release-grade result-object contracts for tensor decomposition/network outputs |
| `iterative` | Real and complex dense `conjugate_gradient` / `gmres` with config + view-first paths | Real and complex `conjugate_gradient`, `gmres` | Partial | Add `f32` breadth and richer config/result exposure |
| `jacobian` | Numerical Jacobian, central Jacobian, gradient, Hessian | Forward/central Jacobian, gradient, Hessian for Python callables | Partial | Add `f32` breadth and document callback/result contracts for production API |
| `optimization` | Line search, gradient descent, Adam, momentum, RMSProp, projected GD, SGD, BFGS, real + complex | Real and complex line search / gradient descent / Adam / momentum / RMSProp / projected GD / SGD / BFGS via Python callables | Partial | Add `f32` breadth, stabilize callback/config ergonomics, and document optimizer contracts |
| `pca` | Real and complex `compute`, `transform`, `inverse_transform` | Real and complex `compute_pca`, `pca_transform`, `pca_inverse_transform` | Partial | Preserve release-grade result-object fidelity and add `f32` breadth |
| `regression` | Real and complex linear regression with canonical result structs | Real and complex `linear_regression` | Partial | Preserve canonical result metadata beyond coefficients / `r_squared` and add `f32` breadth |
| `stats` | Real and complex means/centering/covariance/correlation | Real and complex means/centering/covariance/correlation | Partial | Add `f32` breadth and document complex-stat contracts clearly |
| `arrow` / `PyArrow` | Full admitted `nabled::arrow` surface across dense, sparse, tensor, ML, and Arrow-admitted result contracts | `arrow_dot`, `arrow_l2_norm`, `arrow_svd_decompose` only | Partial | Reach full admitted `nabled::arrow` parity using canonical `ndarrow` carriers and explicit zero-copy contracts |

## Cross-Cutting Parity Matrix

| Axis | Rust baseline | Current Python state | Status | Release requirement |
|---|---|---|---|---|
| Real dtype breadth | Broad `f32` + `f64` across linalg/ml/tensor domains | Nearly all bindings are `f64`-only | Partial | Expose admitted `f32` release surface wherever Rust already admits it |
| Complex breadth | Broad complex parity across dense, tensor, and ML/stat domains; limited to admitted Rust APIs | Meaningful complex ML/tensor coverage now exists, but dense linalg/decomposition/function parity is still materially incomplete | Partial | Expose admitted complex release surface or explicitly narrow Rust admission first |
| View / copy contract | Rust is view-first across hot paths | View-based dense Python APIs now admit strided / Fortran-order NumPy inputs, but higher-level result/tensor/sparse paths still materialize owned arrays in places | Partial | Preserve zero-copy ingress where possible and make unavoidable copies explicit and documented |
| Allocation control | Rust exposes `*_into` and reusable workspace/result helpers in hot domains | No `out=` / workspace / reusable-plan equivalent | Missing | Add Python allocation-control semantics for performance-critical paths |
| Result object fidelity | Rust returns typed result structs for decompositions and higher-level workflows | Python mostly flattens results to tuples/arrays | Partial | Preserve release-relevant metadata and make result contracts stable/documented |
| Sparse data carriers | Rust uses typed sparse matrix structs and admitted Arrow sparse carriers | Python uses raw CSR buffers only | Partial | Add a first-class Python sparse carrier story (SciPy-compatible object or typed wrapper) with zero-copy where feasible |
| Arrow carriers | Rust uses canonical `ndarrow` carriers across admitted Arrow workflows | Python Arrow bridge only handles `float64` primitive arrays and fixed-size-list matrices | Partial | Use the same canonical `ndarrow` carrier set and explicit ingress/egress contracts as Rust |
| PyArrow egress | Rust Arrow facade can stay Arrow-native where natural | Python `arrow_svd_decompose` converts results to NumPy | Partial | Keep Arrow-native egress where the Rust facade already defines an Arrow-native contract |
| Provider feature exposure | Rust facade admits `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system` | Python exposes only `openblas-system` | Partial | Expose/document truthful source-build paths for admitted provider features |
| Backend feature exposure | Rust facade admits `accelerator-rayon` and `accelerator-wgpu` | Python exposes only `accelerator-rayon` | Partial | Expose/document truthful backend build paths for admitted backends |
| Feature UX truthfulness | Rust feature names and behavior are explicit | Python extras do not fully map to Cargo features today | Partial | Ensure Python install/build UX matches actual Cargo feature behavior |

## Release-Blocking Gaps Locked By This Matrix

These are the gaps that directly drive `N-PY-003..N-PY-008`:

1. Missing cross-domain numerics: broad `f32` parity remains largely absent.
2. Missing sparse breadth: formats, products, solver/preconditioner depth, reuse paths.
3. Missing dense/linalg complex breadth outside the landed ML/tensor surfaces.
4. Missing Arrow/PyArrow breadth: nearly the full admitted `nabled::arrow` surface.
5. Missing allocation-control semantics for hot Python workflows.
6. Missing truthful provider/backend feature exposure.
7. High-level ML/tensor result fidelity still needs production-grade result-object contracts where Python currently flattens rich Rust structs.

## Implementation Order Derived From This Matrix

1. `N-PY-003`: continue from the landed ML + tensor parity passes:
   - broad `f32` breadth
   - sparse breadth
   - remaining dense complex parity
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

# pynabled

Python bindings for **[nabled](https://github.com/MontOpsInc/nabled)** — ndarray-style linear algebra and ML-oriented routines (SVD, QR, LU, Cholesky, eigen, PCA, sparse solvers, tensors, and more).

## Install from PyPI

```bash
pip install pynabled
```

Requires Python 3.10+ and NumPy.

## Install from source (development)

```bash
pip install maturin numpy
maturin develop
```

Optional provider/backend/Arrow support is a source-build workflow. `pynabled` now exposes the same
Cargo feature names as the Rust facade (`openblas-system`, `openblas-static`, `netlib-system`,
`netlib-static`, `magma-system`, `accelerator-rayon`, `accelerator-wgpu`, `arrow`); build them
explicitly with maturin or `MATURIN_PEP517_ARGS`, then inspect the installed extension with
`pynabled.build_features()`. See **[BUILD.md](../BUILD.md)** in the repository root.

## Quick example

```python
import numpy as np
import pynabled

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
result = pynabled.svd_decompose(a)
print("singular values:", result.singular_values)
```

`numpy.ndarray` is the canonical CPU-array carrier for `pynabled`. Borrowed NumPy views are used
where the Rust API admits them, including non-C-contiguous inputs for the view-based dense paths.
Current real-valued vector/matrix/decomposition/function/batched/statistics/regression/PCA/
iterative/callable-ML/tensor bindings all accept both `float32` and `float64` under the same
public function names. The current Arrow slice now also exposes the admitted real dense/
decomposition/matrix-function/PCA/regression rows plus canonical complex dense Arrow carriers
across vector/matrix/statistics/orthogonalization/triangular/decomposition/matrix-function/PCA/
regression workflows, callback-driven iterative/Jacobian/optimization rows over PyArrow carriers,
canonical `ndarrow.csr_matrix` / `ndarrow.csr_matrix_batch` sparse carriers with direct sparse
solve/product/reuse workflows, fixed-shape-tensor batched QR/SVD/LU/Cholesky/symmetric-eigen
result wrappers, and canonical fixed-shape / variable-shape tensor workflows across last-axis ops,
permutation/contraction, batched matmul, cube kernels, einsum, CP-ALS, HOSVD/HOOI/Tucker, and TT
helpers over PyArrow/`ndarrow`. Arrow-native outputs stay Arrow-native where the Rust Arrow facade
already does so, while ndarray-native decomposition/PCA/regression/tensor rows reuse the same
typed Python result objects as the NumPy-facing API. Mixed real dtypes are rejected explicitly
instead of being silently cast, complex Arrow ingress follows canonical `ndarrow.complex64`
storage/field contracts rather than silently materializing NumPy buffers, and sparse/tensor Arrow
ingress now matches `ndarrow`'s field/storage metadata contract explicitly instead of depending on
default PyArrow extension-field serialization.
Callable-driven `jacobian` / `optimization` workflows preserve the caller's real dtype end-to-end,
and their `float32` defaults use `float32`-appropriate finite-difference / convergence settings
instead of raw `float64` thresholds. Some higher-level wrappers still materialize owned arrays
internally where the current Rust API shape requires it.
The direct NumPy tensor primitive kernels now also accept `out=` so callers can reuse output
buffers for cube kernels, last-axis reductions/normalization, permutation, contraction, and
batched matmul. `out` must already have the expected dtype/rank/shape contract for the selected
kernel, writable Fortran-order outputs are accepted, and aliasing an input as `out` fails
explicitly instead of silently copying around the overlap.
The direct NumPy vector/matrix hot kernels now follow the same pattern: pairwise/batched vector
rows and direct/batched matrix kernels accept `out=` for caller-provided result reuse under the
same public names. Writable Fortran-order 2D/3D outputs are accepted where rank permits, and
aliasing an input as `out` fails explicitly instead of silently materializing around the overlap.
The currently admitted higher-level dense helpers now follow the same explicit reuse contract where
the Rust core already exposes `*_into`: `svd_pseudo_inverse`, `svd_reconstruct_matrix`,
`matrix_exp`, `matrix_exp_eigen`, `matrix_log_taylor`, `matrix_log_eigen`, `matrix_log_svd`,
`matrix_power`, `matrix_sign`, `sylvester_solve`, `lyapunov_solve`, `pca_transform`, and
`pca_inverse_transform` all accept `out=` with the same writeability/shape/dtype requirements.
`compute_pca(...)`, `compute_pca_complex(...)`, `linear_regression(...)`, and
`linear_regression_complex(...)` now also accept typed `out=` result buffers
(`PcaResult` / `RegressionResult`) under the existing public names so repeated ML workflows can
reuse result storage instead of forcing fresh Python allocations on every call.
`svd_pseudo_inverse(...)` can also consume a previously computed `SvdResult` directly, so repeated
pseudo-inverse workflows can reuse decomposition factors instead of recomputing SVD from the
original matrix each time.
`svd_condition_number(...)` and `svd_rank(...)` now read singular values directly instead of
rebuilding owned intermediary SVD objects.
Tensor reconstruction/projection/contraction helpers now also reuse caller-provided outputs where
the Rust core already has truthful `*_into` coverage: `tensor_hosvd_nd_reconstruct`,
`tensor_hosvd3_reconstruct`, `tensor_tucker_project`, `tensor_tucker_expand`,
`tensor_einsum`, `tensor_einsum_complex`, `tensor_cp_als3_reconstruct`,
`tensor_cp_als_nd_reconstruct`, and `tensor_tt_svd_reconstruct` all accept `out=` under the same
dtype/rank/shape contract instead of always materializing fresh tensors.
`qr_reconstruct_matrix(...)` now follows the same Rust-backed `out=` contract for both direct and
pivoted QR results, and `CholeskyResult` can now be passed back into `cholesky_solve(...)` /
`cholesky_inverse(...)` for repeated factor reuse instead of re-factorizing the original matrix.
`LuResult` now follows that same typed factor-reuse contract for real LU workflows:
`lu_solve(...)`, `lu_inverse(...)`, `lu_determinant(...)`, and `lu_log_determinant(...)` all
accept the factor result directly, and the solve/inverse rows also accept `out=` under the
existing public names.

Where the Rust core already exposes reusable scratch/workspace objects, the Python API now keeps
that contract visible too. `PairwiseCosineWorkspace`, `MatrixFunctionWorkspace`, and
`SylvesterWorkspace` can be passed back into the existing public functions through `workspace=`
for repeated workloads that would otherwise keep reallocating scratch buffers. `matrix_exp_eigen`
now follows that same matrix-function workspace contract, and Schur decomposition exposes
`SchurWorkspace` plus reusable `SchurResult` buffers through
`schur_compute(..., out=..., workspace=...)`:

```python
import numpy as np
import pynabled

workspace = pynabled.MatrixFunctionWorkspace(np.float64)
out = np.empty_like(a)

pynabled.matrix_exp(a, out=out, workspace=workspace)
```

The iterative and callable-driven ML surface now uses typed config objects instead of exposing raw
parameter shims as the production contract. `conjugate_gradient(...)` / `gmres(...)` accept
`IterativeConfig`, Jacobian helpers accept `JacobianConfig`, and optimizer/line-search helpers use
`LineSearchConfig`, `GradientDescentConfig`, `AdamConfig`, `MomentumConfig`, `RMSPropConfig`,
`ProjectedGradientConfig`, and `BFGSConfig`. Passing both `config=` and explicit tuning kwargs is
rejected instead of silently picking one side.
The dense iterative solve rows now also expose Rust-backed `out=` reuse under the existing public
names for both real and complex workflows, and the complex iterative bindings now follow the same
view-first NumPy ingress contract as the real rows, including Fortran-order / strided inputs.
Complex PCA/regression/stats rows now have that same strided-input proof and now all borrow
through the shared helper-based view-first boundary instead of bespoke typed-array paths. PCA
transform/inverse no longer rebuild owned temporary component / mean state just to project or
reconstruct.

Callback-driven Jacobian and optimizer helpers remain convenience-oriented APIs. They are
production-supported, but each objective/gradient evaluation crosses back into Python, so they are
not the same performance contract as the array-in/array-out kernels whose hot loops stay in Rust.

Current dense primitive breadth is also wider: vector APIs now include cosine distance,
pairwise cosine distance, and row-wise batched dot/norm/cosine/distance/normalize helpers, while
matrix APIs now include broadcast-left/right batched matmat alongside the earlier direct and
batched kernels.

The dense result-bearing surface is also less tuple-oriented now: QR exposes reduced/pivoted
decomposition plus typed reconstruction/condition helpers, LU exposes signed log-determinant for
real matrices, and non-symmetric eigen exposes explicit balancing plus matched left/right
bi-eigen result objects.
Provider-bound mixed-precision refinement helpers are now visible too: `lu_solve_mixed(...)`,
`sylvester_solve_mixed(...)`, and `lyapunov_solve_mixed(...)` return typed result objects with
`solution` plus `refinement_iterations`. Those rows require a source build with `magma-system`
and intentionally admit only the truthful mixed-provider dtypes (`float64` / `complex128`)
instead of silently downcasting `float32`.

Where the admitted Rust surface already has direct complex kernels, the current Python dense API
now also accepts `complex128` across the main vector/matrix/decomposition/function families
(`dot`, `l2_norm`, `cosine_similarity`, `matvec`, `matmat`, `svd`, `qr`, `lu` solve/inverse/
determinant, `cholesky`, non-symmetric `eigen`, `schur`, `polar`, `sylvester`, `lyapunov`,
admitted complex `matrix_functions`, and `gram_schmidt`). Unsupported rows fail explicitly rather
than silently casting or dropping back to a different numerical contract.

For the current admitted Rust batch surface, complex support also extends to the batch-vector
helpers (`batched_dot`, `batched_l2_norm`, `batched_cosine_similarity`, `batched_normalize`).

Structured decomposition / ML / tensor workflows now return typed Python result objects with named
fields instead of anonymous tuples. For example, `svd_decompose(...)` returns `pynabled.SvdResult`,
`compute_pca(...)` returns `pynabled.PcaResult`, and tensor decomposition helpers return
corresponding `Hosvd*` / `CpAls*` / `TensorTrainResult` objects.

Sparse CSR workflows use `pynabled.CsrMatrix` as the canonical Python carrier. SciPy-compatible
objects can be normalized into that carrier explicitly with `CsrMatrix.from_scipy(...)` or passed
to the public sparse wrappers directly. The current CSR carrier preserves `int32` / `int64` index
buffers and `float32` / `float64` data rather than normalizing everything to one dtype. Explicit
normalization is available through `dtype=` / `index_dtype=` on construction plus
`CsrMatrix.astype(...)` and `CsrMatrix.with_index_dtype(...)`, while mixed sparse operand dtypes
continue to fail explicitly instead of being silently cast.

Sparse reuse paths are now first-class as well: `CsrMatrix` can build reusable
`JacobiPreconditioner`, `ILU0` / `ILUT` / `ILUK`, `IC0`, `ILDL0`, and direct
`SparseLUFactorization` objects. Those wrappers keep the reusable sparse state in Rust instead of
reconstructing factors from NumPy on every apply/solve call, while factor properties still round
trip back through canonical `CsrMatrix` carriers. `ILUTConfig` and `ILUKConfig` expose the current
configurable sparse setup profiles, and sparse LU reuse already covers both single-RHS and
multi-RHS solve workflows.

Sparse carrier breadth is now wider as well: `pynabled` also exposes first-class `CscMatrix` and
`CooMatrix` carriers with SciPy-compatible ingress plus explicit `CSR -> CSC`, `CSC -> CSR`, and
`COO -> CSR` conversions. CSC matvec stays native to the CSC carrier, sparse-sparse matmat returns
the canonical `CsrMatrix`, and the reusable `ILU0` / `ILUT` / `ILUK` / `ILDL0` factor objects now
also drive GMRES / `BiCGSTAB` solve and multi-RHS solve workflows without rebuilding the sparse
factorization each call. Where a sparse kernel only admits the CSR form today, `pynabled` keeps
that normalization explicit through the carrier methods rather than silently pretending the storage
format is interchangeable. Direct sparse iterative entrypoints now also include Gauss-Seidel,
conjugate-gradient, direct `BiCGSTAB`, and IC0-preconditioned `PCG`, and the reusable
`IC0Factorization` object now supports `pcg_solve(...)` without refactorizing the matrix on every
call. The direct preconditioned sparse convenience surface is now first-class too: Python exposes
one-shot GMRES / `BiCGSTAB` entrypoints over `ILU0`, `ILUT`, `ILUK`, and `ILDL0` both as top-level
functions and `CsrMatrix` methods, while `ILUTFactorization` / `ILUKFactorization` preserve their
selected config objects on the Python side. Those one-shot helpers still rebuild the factorization
each call, so repeated-RHS/performance-sensitive workflows should continue to use the reusable
factorization objects directly.

## Documentation

- Rust library API: [docs.rs/nabled](https://docs.rs/nabled)
- Full Python surface: `python/pynabled/__init__.py`
- Examples: `python/examples/`
- Publishing wheels to PyPI: [docs/PYPI_PUBLISH.md](https://github.com/MontOpsInc/nabled/blob/main/docs/PYPI_PUBLISH.md) (also in-repo under `docs/`)

## License

MIT OR Apache-2.0 (same as the nabled workspace).

---
name: Fix repo quality issues
overview: Fix version drift across docs, remove arrow examples, implement random_matrix, compute_svd_with_tolerance, and compute_qr_with_pivoting; remove unused proptest; keep ndarray modules using convert-through-nalgebra pattern.
todos: []
isProject: false
---

# Fix Repo Quality Issues

## Summary

Address version drift, remove arrow examples, implement placeholder/simplified functions, implement proper `compute_qr_with_pivoting` via nalgebra's ColPivQR, and remove unused dev-dependencies. Preserve the existing module-coupling pattern (ndarray backends delegate to nalgebra via [utils](src/utils.rs) conversions).

---

## 1. Fix Version Drift

Align all version references to **0.3.0** (current [Cargo.toml](Cargo.toml) version).


| File                             | Change                                                            |
| -------------------------------- | ----------------------------------------------------------------- |
| [src/lib.rs](src/lib.rs) line 12 | `rust-linalg = "0.1.0"` → `rust-linalg = "0.3.0"` in doc comment  |
| [README.md](README.md) line 26   | `rust-linalg = "0.2.0"` → `rust-linalg = "0.3.0"` in Installation |
| [CHANGELOG.md](CHANGELOG.md)     | Update `[Unreleased]` link to compare v0.3.0...HEAD (line 136)    |


---

## 2. Remove Arrow Examples

Delete the three arrow example files (examples remain buildable only when arrow feature is enabled; removing them avoids default-build failures):

- [examples/arrow_svd_example.rs](examples/arrow_svd_example.rs)
- [examples/arrow_qr_example.rs](examples/arrow_qr_example.rs)
- [examples/arrow_full_example.rs](examples/arrow_full_example.rs)

Do **not** remove:

- `src/arrow/`* (arrow module)
- `tests/arrow_integration_tests.rs` (feature-gated)

---

## 3. Implement random_matrix

In [src/utils.rs](src/utils.rs), replace the placeholder (lines 41-46) with an implementation using `ndarray_rand` and `rand`.

**Approach:** Use `ndarray_rand::rand_distr::Uniform` with `Array2::random()`. For `f32`/`f64`, use `Uniform::new(0.0, 1.0)` (or similar). The current `random_matrix<T: Float>` signature restricts to `Float`; `ndarray_rand::RandomExt` typically works with `f32` and `f64`. Add trait bounds as needed (e.g. `SampleUniform`, `Copy`).

**Implementation sketch:**

```rust
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn random_matrix<T>(rows: usize, cols: usize) -> Array2<T>
where
    T: Float + rand::distr::uniform::SampleUniform,
{
    Array2::random((rows, cols), Uniform::new(T::zero(), T::one()))
}
```

Update the doc comment to describe the uniform [0, 1) distribution. Handle `T::zero()` and `T::one()` for generic `Float`; if `SampleUniform` is awkward for generics, consider `random_matrix_f64`/`random_matrix_f32` or a sealed trait. Fallback: keep generic but only implement for `f64` via a dedicated function and deprecate/wrap the generic one.

---

## 4. Implement compute_svd_with_tolerance

In [src/svd.rs](src/svd.rs) (lines 86-97), the function currently ignores `tolerance` and delegates to `compute_svd`.

**Implementation:** After computing the full SVD via `compute_svd`, zero out singular values smaller than `tolerance` in the returned `NalgebraSVD`. This matches typical use (e.g. rank determination, low-rank approximation). Clamp singular values: `if sv < tolerance { T::zero() } else { sv }`.

**Signature change:** Use the `tolerance` parameter instead of `_tolerance`.

**ndarray_svd:** There is no `compute_svd_with_tolerance` in ndarray_svd. For consistent coupling, add `ndarray_svd::compute_svd_with_tolerance` that converts to nalgebra, calls `nalgebra_svd::compute_svd_with_tolerance`, then converts the result back to ndarray (same pattern as `compute_svd`).

---

## 5. Implement compute_qr_with_pivoting

In [src/qr.rs](src/qr.rs) (lines 249-276), replace the stub that uses plain `matrix.qr()` with nalgebra's `ColPivQR`.

**Implementation:**

- Use `nalgebra::linalg::ColPivQR::new(matrix.clone())` (or `matrix.col_piv_qr()` if available).
- Extract `q`, `r`, and permutation via `.q()`, `.r()`, `.p()`.
- Convert `PermutationSequence` to `Option<DMatrix<T>>` for `QRResult.p`: create `DMatrix::identity(n, n)` and apply the permutation to obtain the permutation matrix (e.g. apply to columns of identity).
- Compute `rank` via existing `determine_rank(&r, config.rank_tolerance)`.
- Return `QRResult { q, r, p: Some(p_matrix), rank }`.

**ndarray_qr:** Already calls `nalgebra_qr::compute_qr_with_pivoting` (line 488). No change needed; it will pick up the fix.

---

## 6. Remove Unused Dev-Dependencies

In [Cargo.toml](Cargo.toml), remove:

- `proptest = "1.0"` (not used in any test file)

Keep `criterion = "0.5"` (used in [benches/svd_benchmarks.rs](benches/svd_benchmarks.rs)).

---

## 7. Keep Module Coupling Consistent

**Current pattern:** ndarray modules (`ndarray_svd`, `ndarray_qr`, `ndarray_cholesky`, etc.) convert input to nalgebra via `crate::utils::ndarray_to_nalgebra`, call the nalgebra implementation, then convert results back via `nalgebra_to_ndarray`.

**Actions:**

- Add `ndarray_svd::compute_svd_with_tolerance` (per section 4) using this same pattern.
- `random_matrix` lives in utils and returns `Array2`; it does not add new coupling.
- No new direct nalgebra-only or ndarray-only code paths that bypass this pattern.

---

## File Checklist


| File                         | Action                                                    |
| ---------------------------- | --------------------------------------------------------- |
| [Cargo.toml](Cargo.toml)     | Remove proptest                                           |
| [src/lib.rs](src/lib.rs)     | Fix version in doc                                        |
| [README.md](README.md)       | Fix version                                               |
| [CHANGELOG.md](CHANGELOG.md) | Fix Unreleased link                                       |
| [src/utils.rs](src/utils.rs) | Implement random_matrix                                   |
| [src/svd.rs](src/svd.rs)     | Implement compute_svd_with_tolerance (nalgebra + ndarray) |
| [src/qr.rs](src/qr.rs)       | Implement compute_qr_with_pivoting via ColPivQR           |
| examples/arrow_*.rs          | Delete 3 files                                            |


---

## Testing

- Run `cargo test --lib --tests` to ensure all tests pass.
- Run `cargo build` (default, no features) to confirm examples no longer break the build.
- If arrow tests are needed: `cargo test --features arrow`.


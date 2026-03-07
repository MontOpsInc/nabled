//! Internal ndarray-native helpers used across domain modules.

#[cfg(feature = "magma-system")]
use std::sync::OnceLock;
#[cfg(feature = "magma-system")]
#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

use nabled_core::scalar::NabledReal;
#[cfg(any(not(feature = "lapack-provider"), feature = "magma-system"))]
use ndarray::ArrayView1;
use ndarray::{Array1, Array2, ArrayView2};

pub(crate) const DEFAULT_TOLERANCE: f64 = 1.0e-12;
pub(crate) type LuDecomposition<T> = (Array2<T>, Array2<T>, Vec<usize>, i8);

/// Shared tolerance/iteration policy for dense ndarray kernels.
pub(crate) struct DenseKernelPolicy;

#[cfg(all(feature = "magma-system", test))]
static MAGMA_VERIFY_FORCE_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

impl DenseKernelPolicy {
    pub(crate) const BASE_TOLERANCE: f64 = DEFAULT_TOLERANCE;
    pub(crate) const JACOBI_MAX_ITERATIONS: usize = 256;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_COUNT: usize = 32;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_COUNT_FLOOR: usize = 8;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_DIM: usize = 32;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_DIM_FLOOR: usize = 16;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_WORK: usize = 524_288;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_BATCH_MIN_DECOMPOSITION_WORK_FLOOR: usize = 8_192;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_MIN_DECOMPOSITION_DIM: usize = 128;
    #[cfg(feature = "magma-system")]
    pub(crate) const MAGMA_MIN_DECOMPOSITION_DIM_FLOOR: usize = 16;
    pub(crate) const MATRIX_FUNCTION_SERIES_TERMS: usize = 128;
    #[cfg(not(feature = "lapack-provider"))]
    pub(crate) const POLAR_MAX_ITERATIONS: usize = 64;
    pub(crate) const QR_MAX_ITERATIONS: usize = 100;
    pub(crate) const SCHUR_MIN_ITERATIONS: usize = 128;

    #[must_use]
    pub(crate) fn rank_tolerance(requested: f64) -> f64 { requested.max(Self::BASE_TOLERANCE) }

    #[must_use]
    pub(crate) fn taylor_tolerance(requested: f64) -> f64 { requested.max(Self::BASE_TOLERANCE) }

    #[must_use]
    pub(crate) fn schur_iterations(requested: usize) -> usize {
        requested.max(Self::SCHUR_MIN_ITERATIONS)
    }

    #[cfg(not(feature = "lapack-provider"))]
    #[must_use]
    pub(crate) fn polar_convergence_tolerance() -> f64 { Self::BASE_TOLERANCE.sqrt() }

    #[cfg(feature = "magma-system")]
    fn env_positive_usize(name: &str) -> Option<usize> {
        let raw = std::env::var(name).ok()?;
        let parsed = raw.parse::<usize>().ok()?;
        (parsed > 0).then_some(parsed)
    }

    #[cfg(feature = "magma-system")]
    fn env_truthy(name: &str) -> bool {
        std::env::var(name).ok().is_some_and(|raw| {
            let value = raw.trim();
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
        })
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_strict_mode() -> bool {
        static VALUE: OnceLock<bool> = OnceLock::new();
        *VALUE.get_or_init(|| Self::env_truthy("NABLED_MAGMA_STRICT"))
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_verify_force_mode() -> bool {
        static VALUE: OnceLock<bool> = OnceLock::new();
        #[cfg(test)]
        {
            match MAGMA_VERIFY_FORCE_OVERRIDE.load(Ordering::Relaxed) {
                0 => return false,
                1 => return true,
                _ => {}
            }
        }
        *VALUE.get_or_init(|| Self::env_truthy("NABLED_MAGMA_VERIFY_FORCE"))
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_fail_fast_mode() -> bool { Self::magma_strict_mode() }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_min_decomposition_dim() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_MIN_DECOMPOSITION_DIM")
                .unwrap_or(Self::MAGMA_MIN_DECOMPOSITION_DIM)
                .max(Self::MAGMA_MIN_DECOMPOSITION_DIM_FLOOR)
        })
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_batch_min_decomposition_count() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_COUNT")
                .unwrap_or(Self::MAGMA_BATCH_MIN_DECOMPOSITION_COUNT)
                .max(Self::MAGMA_BATCH_MIN_DECOMPOSITION_COUNT_FLOOR)
        })
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_batch_min_decomposition_dim() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_DIM")
                .unwrap_or(Self::MAGMA_BATCH_MIN_DECOMPOSITION_DIM)
                .max(Self::MAGMA_BATCH_MIN_DECOMPOSITION_DIM_FLOOR)
        })
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn magma_batch_min_decomposition_work() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_WORK")
                .unwrap_or(Self::MAGMA_BATCH_MIN_DECOMPOSITION_WORK)
                .max(Self::MAGMA_BATCH_MIN_DECOMPOSITION_WORK_FLOOR)
        })
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn prefer_magma_decomposition(rows: usize, cols: usize) -> bool {
        if Self::magma_verify_force_mode() {
            return rows > 0 && cols > 0;
        }
        rows.min(cols) >= Self::magma_min_decomposition_dim()
    }

    #[cfg(feature = "magma-system")]
    #[must_use]
    pub(crate) fn prefer_magma_batched_decomposition(
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> bool {
        if batch == 0 || rows == 0 || cols == 0 {
            return false;
        }
        if Self::magma_verify_force_mode() {
            return true;
        }

        let min_dim = rows.min(cols);
        if min_dim >= Self::magma_min_decomposition_dim() {
            // A few very large matrices are still good MAGMA candidates.
            return true;
        }

        let batch_count_ok = batch >= Self::magma_batch_min_decomposition_count();
        let batch_dim_ok = min_dim >= Self::magma_batch_min_decomposition_dim();
        let work = batch.saturating_mul(rows).saturating_mul(cols);
        let batch_work_ok = work >= Self::magma_batch_min_decomposition_work();

        batch_count_ok && batch_dim_ok && batch_work_ok
    }

    #[cfg(all(feature = "magma-system", test))]
    pub(crate) fn set_magma_verify_force_override(value: Option<bool>) {
        let encoded = match value {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };
        MAGMA_VERIFY_FORCE_OVERRIDE.store(encoded, Ordering::Relaxed);
    }
}

pub(crate) fn validate_square_non_empty<T>(matrix: &Array2<T>) -> Result<(), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    Ok(())
}

pub(crate) fn validate_finite<T: NabledReal>(matrix: &Array2<T>) -> Result<(), &'static str> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok(())
}

pub(crate) fn identity<T: NabledReal>(n: usize) -> Array2<T> {
    let mut id = Array2::<T>::zeros((n, n));
    for i in 0..n {
        id[[i, i]] = T::one();
    }
    id
}

pub(crate) fn is_symmetric<T: NabledReal>(matrix: &ArrayView2<'_, T>, tolerance: T) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

pub(crate) fn lu_decompose<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<LuDecomposition<T>, &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }

    let n = matrix.nrows();
    let mut l = Array2::<T>::zeros((n, n));
    let mut u = matrix.to_owned();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut sign = 1_i8;

    for i in 0..n {
        l[[i, i]] = T::one();
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_value = u[[k, k]].abs();
        for row in (k + 1)..n {
            let candidate = u[[row, k]].abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = row;
            }
        }

        if pivot_value <= T::from_f64(DEFAULT_TOLERANCE).unwrap_or(T::epsilon()) {
            return Err("singular");
        }

        if pivot_row != k {
            for col in 0..n {
                let tmp = u[[k, col]];
                u[[k, col]] = u[[pivot_row, col]];
                u[[pivot_row, col]] = tmp;
            }
            for col in 0..k {
                let tmp = l[[k, col]];
                l[[k, col]] = l[[pivot_row, col]];
                l[[pivot_row, col]] = tmp;
            }
            pivots.swap(k, pivot_row);
            sign = -sign;
        }

        for row in (k + 1)..n {
            let factor = u[[row, k]] / u[[k, k]];
            l[[row, k]] = factor;
            for col in k..n {
                u[[row, col]] = u[[row, col]] - factor * u[[k, col]];
            }
        }
    }

    Ok((l, u, pivots, sign))
}

#[allow(clippy::many_single_char_names)]
#[cfg(any(not(feature = "lapack-provider"), feature = "magma-system"))]
pub(crate) fn lu_solve<T: NabledReal>(
    l: &Array2<T>,
    u: &Array2<T>,
    pivots: &[usize],
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, &'static str> {
    let n = l.nrows();
    if rhs.len() != n || u.nrows() != n || u.ncols() != n || l.ncols() != n || pivots.len() != n {
        return Err("bad_dimensions");
    }

    let mut pb = Array1::<T>::zeros(n);
    for i in 0..n {
        pb[i] = rhs[pivots[i]];
    }

    let mut y = Array1::<T>::zeros(n);
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum;
    }

    let mut x = Array1::<T>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[[i, j]] * x[j];
        }
        let diagonal = u[[i, i]];
        if diagonal.abs() <= T::from_f64(DEFAULT_TOLERANCE).unwrap_or(T::epsilon()) {
            return Err("singular");
        }
        x[i] = sum / diagonal;
    }

    Ok(x)
}

#[allow(clippy::many_single_char_names)]
#[cfg(any(not(feature = "lapack-provider"), feature = "magma-system"))]
pub(crate) fn inverse_from_lu<T: NabledReal>(
    l: &Array2<T>,
    u: &Array2<T>,
    pivots: &[usize],
) -> Result<Array2<T>, &'static str> {
    let n = l.nrows();
    let mut inverse = Array2::<T>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<T>::zeros(n);
        e[col] = T::one();
        let x = lu_solve(l, u, pivots, &e.view())?;
        for row in 0..n {
            inverse[[row, col]] = x[row];
        }
    }
    Ok(inverse)
}

pub(crate) fn qr_gram_schmidt<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> (Array2<T>, Array2<T>, usize) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();

    let mut q = Array2::<T>::zeros((rows, cols));
    let mut r = Array2::<T>::zeros((cols, cols));
    let mut rank = 0_usize;
    let mut v = Array1::<T>::zeros(rows);

    for j in 0..cols {
        for row in 0..rows {
            v[row] = matrix[[row, j]];
        }
        for i in 0..j {
            let q_col = q.column(i);
            let mut projection = T::zero();
            for row in 0..rows {
                projection += q_col[row] * v[row];
            }
            r[[i, j]] = projection;
            for row in 0..rows {
                v[row] -= projection * q_col[row];
            }
        }

        let norm =
            v.iter().map(|value| *value * *value).fold(T::zero(), |sum, value| sum + value).sqrt();
        r[[j, j]] = norm;
        if norm > tolerance {
            rank += 1;
            for row in 0..rows {
                q[[row, j]] = v[row] / norm;
            }
        }
    }

    (q, r, rank)
}

#[allow(clippy::many_single_char_names)]
pub(crate) fn jacobi_eigen_symmetric<T: NabledReal>(
    matrix: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<(Array1<T>, Array2<T>), &'static str> {
    validate_square_non_empty(matrix)?;
    validate_finite(matrix)?;
    let base_tolerance = T::from_f64(DEFAULT_TOLERANCE).unwrap_or(T::epsilon());
    if !is_symmetric(&matrix.view(), tolerance.max(base_tolerance)) {
        return Err("not_symmetric");
    }

    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut eigenvectors = identity(n);
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap_or(one + one);

    for _ in 0..max_iterations {
        let mut p = 0_usize;
        let mut q = 1_usize;
        let mut max_off_diag = zero;

        for i in 0..n {
            for j in (i + 1)..n {
                let value = a[[i, j]].abs();
                if value > max_off_diag {
                    max_off_diag = value;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off_diag <= tolerance {
            let mut eigenvalues = Array1::<T>::zeros(n);
            for i in 0..n {
                eigenvalues[i] = a[[i, i]];
            }
            return Ok((eigenvalues, eigenvectors));
        }

        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        if apq.abs() <= tolerance.max(base_tolerance) {
            continue;
        }

        // Stable Jacobi rotation parameters.
        let tau = (aqq - app) / (two * apq);
        let t = if tau >= zero {
            one / (tau + (one + tau * tau).sqrt())
        } else {
            -one / (-tau + (one + tau * tau).sqrt())
        };
        let c = one / (one + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let akp = a[[k, p]];
                let akq = a[[k, q]];
                a[[k, p]] = c * akp - s * akq;
                a[[p, k]] = a[[k, p]];
                a[[k, q]] = s * akp + c * akq;
                a[[q, k]] = a[[k, q]];
            }
        }

        a[[p, p]] = app - t * apq;
        a[[q, q]] = aqq + t * apq;
        a[[p, q]] = zero;
        a[[q, p]] = zero;

        for k in 0..n {
            let vkp = eigenvectors[[k, p]];
            let vkq = eigenvectors[[k, q]];
            eigenvectors[[k, p]] = c * vkp - s * vkq;
            eigenvectors[[k, q]] = s * vkp + c * vkq;
        }
    }

    Err("convergence")
}

pub(crate) fn sort_eigenpairs_desc<T: NabledReal>(
    eigenvalues: &Array1<T>,
    eigenvectors: &Array2<T>,
) -> (Array1<T>, Array2<T>) {
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&left, &right| {
        eigenvalues[right].partial_cmp(&eigenvalues[left]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_values = Array1::<T>::zeros(n);
    let mut sorted_vectors = Array2::<T>::zeros((n, n));

    for (new_col, &old_col) in indices.iter().enumerate() {
        sorted_values[new_col] = eigenvalues[old_col];
        for row in 0..n {
            sorted_vectors[[row, new_col]] = eigenvectors[[row, old_col]];
        }
    }

    (sorted_values, sorted_vectors)
}

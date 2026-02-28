//! Internal ndarray-native helpers used across domain modules.

use ndarray::{Array1, Array2};

pub(crate) const DEFAULT_TOLERANCE: f64 = 1.0e-12;
pub(crate) type LuDecomposition = (Array2<f64>, Array2<f64>, Vec<usize>, i8);

#[must_use]
pub(crate) fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or(f64::from(u32::MAX), f64::from)
}

pub(crate) fn validate_square_non_empty(matrix: &Array2<f64>) -> Result<(), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    Ok(())
}

pub(crate) fn validate_finite(matrix: &Array2<f64>) -> Result<(), &'static str> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok(())
}

pub(crate) fn identity(n: usize) -> Array2<f64> {
    let mut id = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        id[[i, i]] = 1.0;
    }
    id
}

pub(crate) fn is_symmetric(matrix: &Array2<f64>, tolerance: f64) -> bool {
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

pub(crate) fn lu_decompose(matrix: &Array2<f64>) -> Result<LuDecomposition, &'static str> {
    validate_square_non_empty(matrix)?;
    validate_finite(matrix)?;

    let n = matrix.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut u = matrix.clone();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut sign = 1_i8;

    for i in 0..n {
        l[[i, i]] = 1.0;
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

        if pivot_value <= DEFAULT_TOLERANCE {
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
                u[[row, col]] -= factor * u[[k, col]];
            }
        }
    }

    Ok((l, u, pivots, sign))
}

#[allow(clippy::many_single_char_names)]
#[cfg(not(feature = "openblas-system"))]
pub(crate) fn lu_solve(
    l: &Array2<f64>,
    u: &Array2<f64>,
    pivots: &[usize],
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, &'static str> {
    let n = l.nrows();
    if rhs.len() != n || u.nrows() != n || u.ncols() != n || l.ncols() != n || pivots.len() != n {
        return Err("bad_dimensions");
    }

    let mut pb = Array1::<f64>::zeros(n);
    for i in 0..n {
        pb[i] = rhs[pivots[i]];
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum;
    }

    let mut x = Array1::<f64>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[[i, j]] * x[j];
        }
        let diagonal = u[[i, i]];
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err("singular");
        }
        x[i] = sum / diagonal;
    }

    Ok(x)
}

#[allow(clippy::many_single_char_names)]
#[cfg(not(feature = "openblas-system"))]
pub(crate) fn inverse_from_lu(
    l: &Array2<f64>,
    u: &Array2<f64>,
    pivots: &[usize],
) -> Result<Array2<f64>, &'static str> {
    let n = l.nrows();
    let mut inverse = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[col] = 1.0;
        let x = lu_solve(l, u, pivots, &e)?;
        for row in 0..n {
            inverse[[row, col]] = x[row];
        }
    }
    Ok(inverse)
}

pub(crate) fn qr_gram_schmidt(
    matrix: &Array2<f64>,
    tolerance: f64,
) -> (Array2<f64>, Array2<f64>, usize) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();

    let mut q = Array2::<f64>::zeros((rows, cols));
    let mut r = Array2::<f64>::zeros((cols, cols));
    let mut rank = 0_usize;

    for j in 0..cols {
        let mut v = matrix.column(j).to_owned();
        for i in 0..j {
            let q_col = q.column(i);
            let mut projection = 0.0_f64;
            for row in 0..rows {
                projection += q_col[row] * v[row];
            }
            r[[i, j]] = projection;
            for row in 0..rows {
                v[row] -= projection * q_col[row];
            }
        }

        let norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
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
pub(crate) fn jacobi_eigen_symmetric(
    matrix: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Array1<f64>, Array2<f64>), &'static str> {
    validate_square_non_empty(matrix)?;
    validate_finite(matrix)?;
    if !is_symmetric(matrix, tolerance.max(DEFAULT_TOLERANCE)) {
        return Err("not_symmetric");
    }

    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut eigenvectors = identity(n);

    for _ in 0..max_iterations {
        let mut p = 0_usize;
        let mut q = 1_usize;
        let mut max_off_diag = 0.0_f64;

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
            let mut eigenvalues = Array1::<f64>::zeros(n);
            for i in 0..n {
                eigenvalues[i] = a[[i, i]];
            }
            return Ok((eigenvalues, eigenvectors));
        }

        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];

        if apq.abs() <= tolerance.max(DEFAULT_TOLERANCE) {
            continue;
        }

        // Stable Jacobi rotation parameters.
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
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
        a[[p, q]] = 0.0;
        a[[q, p]] = 0.0;

        for k in 0..n {
            let vkp = eigenvectors[[k, p]];
            let vkq = eigenvectors[[k, q]];
            eigenvectors[[k, p]] = c * vkp - s * vkq;
            eigenvectors[[k, q]] = s * vkp + c * vkq;
        }
    }

    Err("convergence")
}

pub(crate) fn sort_eigenpairs_desc(
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&left, &right| {
        eigenvalues[right].partial_cmp(&eigenvalues[left]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_values = Array1::<f64>::zeros(n);
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));

    for (new_col, &old_col) in indices.iter().enumerate() {
        sorted_values[new_col] = eigenvalues[old_col];
        for row in 0..n {
            sorted_vectors[[row, new_col]] = eigenvectors[[row, old_col]];
        }
    }

    (sorted_values, sorted_vectors)
}

//! QR decomposition bindings for Python.

use nabled_core::scalar::NabledReal;
use ndarray::{ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

const BASE_TOLERANCE: f64 = 1.0e-12;

fn reconstruct_into<T: NabledReal>(
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
    output: &mut ArrayViewMut2<'_, T>,
) -> Result<(), nabled_linalg::qr::QRError> {
    if q.ncols() != r.nrows() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "q.ncols() must equal r.nrows()".to_string(),
        ));
    }
    if output.dim() != (q.nrows(), r.ncols()) {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(T::zero());
    for i in 0..q.nrows() {
        for j in 0..r.ncols() {
            let mut sum = T::zero();
            for k in 0..q.ncols() {
                sum += q[[i, k]] * r[[k, j]];
            }
            output[[i, j]] = sum;
        }
    }
    Ok(())
}

fn reconstruct_complex_into(
    q: &ArrayView2<'_, Complex64>,
    r: &ArrayView2<'_, Complex64>,
    output: &mut ArrayViewMut2<'_, Complex64>,
) -> Result<(), nabled_linalg::qr::QRError> {
    if q.ncols() != r.nrows() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "q.ncols() must equal r.nrows()".to_string(),
        ));
    }
    if output.dim() != (q.nrows(), r.ncols()) {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(Complex64::new(0.0, 0.0));
    for i in 0..q.nrows() {
        for j in 0..r.ncols() {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..q.ncols() {
                sum += q[[i, k]] * r[[k, j]];
            }
            output[[i, j]] = sum;
        }
    }
    Ok(())
}

fn permutation_order<T: NabledReal>(
    permutation: &ArrayView2<'_, T>,
) -> Result<Vec<usize>, nabled_linalg::qr::QRError> {
    if permutation.nrows() != permutation.ncols() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "permutation matrix must be square".to_string(),
        ));
    }

    let tolerance = T::from_f64(BASE_TOLERANCE).unwrap_or(T::epsilon());
    let mut order = vec![usize::MAX; permutation.ncols()];
    for col in 0..permutation.ncols() {
        for row in 0..permutation.nrows() {
            if permutation[[row, col]].abs() > tolerance {
                order[col] = row;
                break;
            }
        }
        if order[col] == usize::MAX {
            return Err(nabled_linalg::qr::QRError::InvalidInput(
                "permutation matrix must contain one non-zero entry per column".to_string(),
            ));
        }
    }
    Ok(order)
}

fn complex_permutation_order(
    permutation: &ArrayView2<'_, Complex64>,
) -> Result<Vec<usize>, nabled_linalg::qr::QRError> {
    if permutation.nrows() != permutation.ncols() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "permutation matrix must be square".to_string(),
        ));
    }

    let mut order = vec![usize::MAX; permutation.ncols()];
    for col in 0..permutation.ncols() {
        for row in 0..permutation.nrows() {
            if permutation[[row, col]].norm() > BASE_TOLERANCE {
                order[col] = row;
                break;
            }
        }
        if order[col] == usize::MAX {
            return Err(nabled_linalg::qr::QRError::InvalidInput(
                "permutation matrix must contain one non-zero entry per column".to_string(),
            ));
        }
    }
    Ok(order)
}

fn reconstruct_pivoted_into<T: NabledReal>(
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
    p: &ArrayView2<'_, T>,
    output: &mut ArrayViewMut2<'_, T>,
) -> Result<(), nabled_linalg::qr::QRError> {
    let order = permutation_order(p)?;
    if q.ncols() != r.nrows() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "q.ncols() must equal r.nrows()".to_string(),
        ));
    }
    if p.nrows() != r.ncols() || p.ncols() != r.ncols() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "permutation shape must match r column dimensions".to_string(),
        ));
    }
    if output.dim() != (q.nrows(), r.ncols()) {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(T::zero());
    for pivoted_col in 0..r.ncols() {
        let output_col = order[pivoted_col];
        for row in 0..q.nrows() {
            let mut sum = T::zero();
            for inner in 0..q.ncols() {
                sum += q[[row, inner]] * r[[inner, pivoted_col]];
            }
            output[[row, output_col]] = sum;
        }
    }
    Ok(())
}

fn reconstruct_complex_pivoted_into(
    q: &ArrayView2<'_, Complex64>,
    r: &ArrayView2<'_, Complex64>,
    p: &ArrayView2<'_, Complex64>,
    output: &mut ArrayViewMut2<'_, Complex64>,
) -> Result<(), nabled_linalg::qr::QRError> {
    let order = complex_permutation_order(p)?;
    if q.ncols() != r.nrows() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "q.ncols() must equal r.nrows()".to_string(),
        ));
    }
    if p.nrows() != r.ncols() || p.ncols() != r.ncols() {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "permutation shape must match r column dimensions".to_string(),
        ));
    }
    if output.dim() != (q.nrows(), r.ncols()) {
        return Err(nabled_linalg::qr::QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(Complex64::new(0.0, 0.0));
    for pivoted_col in 0..r.ncols() {
        let output_col = order[pivoted_col];
        for row in 0..q.nrows() {
            let mut sum = Complex64::new(0.0, 0.0);
            for inner in 0..q.ncols() {
                sum += q[[row, inner]] * r[[inner, pivoted_col]];
            }
            output[[row, output_col]] = sum;
        }
    }
    Ok(())
}

fn condition_number_from_r<T: NabledReal>(r: &ArrayView2<'_, T>) -> T {
    if r.is_empty() {
        return T::zero();
    }

    let n = r.nrows().min(r.ncols());
    let mut max_diagonal = T::zero();
    let mut min_diagonal = T::infinity();
    let tolerance = T::from_f64(BASE_TOLERANCE).unwrap_or(T::epsilon());
    for i in 0..n {
        let value = r[[i, i]].abs();
        max_diagonal = max_diagonal.max(value);
        if value > tolerance {
            min_diagonal = min_diagonal.min(value);
        }
    }

    if min_diagonal.is_finite() { max_diagonal / min_diagonal } else { T::infinity() }
}

fn qr_config_f32(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> PyResult<nabled_linalg::qr::QRConfig<f32>> {
    let mut config = nabled_linalg::qr::QRConfig::<f32>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = utils::f64_to_real(rank_tolerance, "rank_tolerance")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    Ok(config)
}

fn qr_config_f64(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> nabled_linalg::qr::QRConfig<f64> {
    let mut config = nabled_linalg::qr::QRConfig::<f64>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = rank_tolerance;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    config
}

fn qr_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> (Py<PyAny>, Py<PyAny>, usize) {
    (
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        result.rank,
    )
}

fn qr_pivoted_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    let permutation = result
        .p
        .ok_or_else(|| PyTypeError::new_err("internal QR pivoting result missing permutation"))?;
    Ok((
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        utils::pyarray2_from_owned(py, permutation),
        result.rank,
    ))
}

/// Compute QR decomposition. Returns `(Q, R, rank)`.
#[pyfunction(name = "qr_decompose", signature = (a, rank_tolerance=None, max_iterations=None))]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::decompose_complex_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
    }
}

/// Compute reduced (economy) QR decomposition. Returns `(Q, R, rank)`.
#[pyfunction(
    name = "qr_decompose_reduced",
    signature = (a, rank_tolerance=None, max_iterations=None)
)]
pub fn decompose_reduced<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result = nabled_linalg::qr::decompose_reduced_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::decompose_reduced_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
    }
}

/// Compute QR decomposition with column pivoting. Returns `(Q, R, P, rank)`.
#[pyfunction(
    name = "qr_decompose_pivoted",
    signature = (a, rank_tolerance=None, max_iterations=None)
)]
pub fn decompose_pivoted<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, true)?;
            let result = nabled_linalg::qr::decompose_with_pivoting_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, true);
            let result = nabled_linalg::qr::decompose_with_pivoting_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, true);
            let result = nabled_linalg::qr::decompose_complex_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
    }
}

/// Solve least-squares problem `min ||Ax - b||`.
#[pyfunction(
    name = "qr_solve_least_squares",
    signature = (a, b, rank_tolerance=None, max_iterations=None)
)]
pub fn solve_least_squares<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array1(b, "b")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray1::F32(b_arr)) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result = nabled_linalg::qr::solve_least_squares_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray1::F64(b_arr)) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::solve_least_squares_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b"])),
    }
}

/// Solve least-squares problem `min ||Ax - b||` into caller-provided `output`.
#[pyfunction(
    name = "qr_solve_least_squares_into",
    signature = (a, b, output, rank_tolerance=None, max_iterations=None)
)]
pub fn solve_least_squares_into(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<()> {
    match (utils::real_array2(a, "a")?, utils::real_array1(b, "b")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray1::F32(b_arr)) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let mut out_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::qr::solve_least_squares_view_into(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray1::F64(b_arr)) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let mut out_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::qr::solve_least_squares_view_into(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b", "output"])),
    }
}

/// Solve least squares directly from precomputed QR factors.
#[pyfunction(
    name = "qr_solve_least_squares_from_factor",
    signature = (q, r, b, p=None, rank_tolerance=None)
)]
pub fn solve_least_squares_from_factor<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    p: Option<&Bound<'py, PyAny>>,
    rank_tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(q, "q")?, utils::real_array2(r, "r")?, utils::real_array1(b, "b")?) {
        (
            utils::RealReadonlyArray2::F32(q_arr),
            utils::RealReadonlyArray2::F32(r_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let permutation = match p {
                Some(value) => match utils::real_array2(value, "p")? {
                    utils::RealReadonlyArray2::F32(permutation) => Some(permutation),
                    utils::RealReadonlyArray2::F64(_) => {
                        return Err(utils::matching_real_dtype_error(&["q", "r", "b", "p"]));
                    }
                },
                None => None,
            };
            let config = qr_config_f32(rank_tolerance, None, permutation.is_some())?;
            let qr = nabled_linalg::qr::QRResult {
                q:    q_arr.as_array().to_owned(),
                r:    r_arr.as_array().to_owned(),
                p:    permutation.map(|value| value.as_array().to_owned()),
                rank: q_arr.as_array().ncols().min(r_arr.as_array().ncols()),
            };
            let result = nabled_linalg::qr::solve_least_squares_from_qr_result_view(
                &qr,
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let permutation = match p {
                Some(value) => match utils::real_array2(value, "p")? {
                    utils::RealReadonlyArray2::F64(permutation) => Some(permutation),
                    utils::RealReadonlyArray2::F32(_) => {
                        return Err(utils::matching_real_dtype_error(&["q", "r", "b", "p"]));
                    }
                },
                None => None,
            };
            let config = qr_config_f64(rank_tolerance, None, permutation.is_some());
            let qr = nabled_linalg::qr::QRResult {
                q:    q_arr.as_array().to_owned(),
                r:    r_arr.as_array().to_owned(),
                p:    permutation.map(|value| value.as_array().to_owned()),
                rank: q_arr.as_array().ncols().min(r_arr.as_array().ncols()),
            };
            let result = nabled_linalg::qr::solve_least_squares_from_qr_result_view(
                &qr,
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "r", "b"])),
    }
}

/// Solve least squares directly from precomputed QR factors into caller-provided `output`.
#[pyfunction(
    name = "qr_solve_least_squares_from_factor_into",
    signature = (q, r, b, output, p=None, rank_tolerance=None)
)]
pub fn solve_least_squares_from_factor_into(
    q: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    p: Option<&Bound<'_, PyAny>>,
    rank_tolerance: Option<f64>,
) -> PyResult<()> {
    match (utils::real_array2(q, "q")?, utils::real_array2(r, "r")?, utils::real_array1(b, "b")?) {
        (
            utils::RealReadonlyArray2::F32(q_arr),
            utils::RealReadonlyArray2::F32(r_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let permutation = match p {
                Some(value) => match utils::real_array2(value, "p")? {
                    utils::RealReadonlyArray2::F32(permutation) => Some(permutation),
                    utils::RealReadonlyArray2::F64(_) => {
                        return Err(utils::matching_real_dtype_error(&["q", "r", "b", "p"]));
                    }
                },
                None => None,
            };
            let config = qr_config_f32(rank_tolerance, None, permutation.is_some())?;
            let qr = nabled_linalg::qr::QRResult {
                q:    q_arr.as_array().to_owned(),
                r:    r_arr.as_array().to_owned(),
                p:    permutation.map(|value| value.as_array().to_owned()),
                rank: q_arr.as_array().ncols().min(r_arr.as_array().ncols()),
            };
            let mut out_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::qr::solve_least_squares_from_qr_result_view_into(
                &qr,
                &b_arr.as_array(),
                &config,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let permutation = match p {
                Some(value) => match utils::real_array2(value, "p")? {
                    utils::RealReadonlyArray2::F64(permutation) => Some(permutation),
                    utils::RealReadonlyArray2::F32(_) => {
                        return Err(utils::matching_real_dtype_error(&["q", "r", "b", "p"]));
                    }
                },
                None => None,
            };
            let config = qr_config_f64(rank_tolerance, None, permutation.is_some());
            let qr = nabled_linalg::qr::QRResult {
                q:    q_arr.as_array().to_owned(),
                r:    r_arr.as_array().to_owned(),
                p:    permutation.map(|value| value.as_array().to_owned()),
                rank: q_arr.as_array().ncols().min(r_arr.as_array().ncols()),
            };
            let mut out_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::qr::solve_least_squares_from_qr_result_view_into(
                &qr,
                &b_arr.as_array(),
                &config,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "r", "b", "output"])),
    }
}

/// Reconstruct matrix `Q * R`.
#[pyfunction(name = "qr_reconstruct_matrix")]
pub fn reconstruct_matrix<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(q, "q")?, utils::numeric_array2(r, "r")?) {
        (utils::NumericReadonlyArray2::F32(q_arr), utils::NumericReadonlyArray2::F32(r_arr)) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let mut out = ndarray::Array2::<f32>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_into(&q_view, &r_view, &mut out.view_mut()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        (utils::NumericReadonlyArray2::F64(q_arr), utils::NumericReadonlyArray2::F64(r_arr)) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let mut out = ndarray::Array2::<f64>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_into(&q_view, &r_view, &mut out.view_mut()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        (utils::NumericReadonlyArray2::C64(q_arr), utils::NumericReadonlyArray2::C64(r_arr)) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let mut out = ndarray::Array2::<Complex64>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_complex_into(&q_view, &r_view, &mut out.view_mut()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["q", "r"])),
    }
}

/// Reconstruct matrix `Q * R` into caller-provided output.
#[pyfunction(name = "qr_reconstruct_matrix_into")]
pub fn reconstruct_matrix_into(
    q: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(q, "q")?, utils::numeric_array2(r, "r")?) {
        (utils::NumericReadonlyArray2::F32(q_arr), utils::NumericReadonlyArray2::F32(r_arr)) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            reconstruct_into(&q_arr.as_array(), &r_arr.as_array(), &mut out.as_array_mut())
                .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::F64(q_arr), utils::NumericReadonlyArray2::F64(r_arr)) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            reconstruct_into(&q_arr.as_array(), &r_arr.as_array(), &mut out.as_array_mut())
                .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(q_arr), utils::NumericReadonlyArray2::C64(r_arr)) => {
            let mut out = utils::output_array2::<Complex64>(output, "output", "complex128")?;
            reconstruct_complex_into(&q_arr.as_array(), &r_arr.as_array(), &mut out.as_array_mut())
                .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["q", "r", "output"])),
    }
}

/// Reconstruct original matrix from a pivoted QR result.
#[pyfunction(name = "qr_reconstruct_matrix_pivoted")]
pub fn reconstruct_matrix_pivoted<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
    p: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(q, "q")?,
        utils::numeric_array2(r, "r")?,
        utils::numeric_array2(p, "p")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(q_arr),
            utils::NumericReadonlyArray2::F32(r_arr),
            utils::NumericReadonlyArray2::F32(p_arr),
        ) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let p_view = p_arr.as_array();
            let mut out = ndarray::Array2::<f32>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_pivoted_into(&q_view, &r_view, &p_view, &mut out.view_mut())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        (
            utils::NumericReadonlyArray2::F64(q_arr),
            utils::NumericReadonlyArray2::F64(r_arr),
            utils::NumericReadonlyArray2::F64(p_arr),
        ) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let p_view = p_arr.as_array();
            let mut out = ndarray::Array2::<f64>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_pivoted_into(&q_view, &r_view, &p_view, &mut out.view_mut())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        (
            utils::NumericReadonlyArray2::C64(q_arr),
            utils::NumericReadonlyArray2::C64(r_arr),
            utils::NumericReadonlyArray2::C64(p_arr),
        ) => {
            let q_view = q_arr.as_array();
            let r_view = r_arr.as_array();
            let p_view = p_arr.as_array();
            let mut out = ndarray::Array2::<Complex64>::zeros((q_view.nrows(), r_view.ncols()));
            reconstruct_complex_pivoted_into(&q_view, &r_view, &p_view, &mut out.view_mut())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, out))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["q", "r", "p"])),
    }
}

/// Reconstruct original matrix from a pivoted QR result into caller-provided output.
#[pyfunction(name = "qr_reconstruct_matrix_pivoted_into")]
pub fn reconstruct_matrix_pivoted_into(
    q: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(q, "q")?,
        utils::numeric_array2(r, "r")?,
        utils::numeric_array2(p, "p")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(q_arr),
            utils::NumericReadonlyArray2::F32(r_arr),
            utils::NumericReadonlyArray2::F32(p_arr),
        ) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            reconstruct_pivoted_into(
                &q_arr.as_array(),
                &r_arr.as_array(),
                &p_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(q_arr),
            utils::NumericReadonlyArray2::F64(r_arr),
            utils::NumericReadonlyArray2::F64(p_arr),
        ) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            reconstruct_pivoted_into(
                &q_arr.as_array(),
                &r_arr.as_array(),
                &p_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(q_arr),
            utils::NumericReadonlyArray2::C64(r_arr),
            utils::NumericReadonlyArray2::C64(p_arr),
        ) => {
            let mut out = utils::output_array2::<Complex64>(output, "output", "complex128")?;
            reconstruct_complex_pivoted_into(
                &q_arr.as_array(),
                &r_arr.as_array(),
                &p_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["q", "r", "p", "output"])),
    }
}

/// Estimate condition number from the `R` diagonal.
#[pyfunction(name = "qr_condition_number")]
pub fn condition_number(py: Python<'_>, r: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(r, "r")? {
        utils::RealReadonlyArray2::F32(r_arr) => {
            Ok(utils::py_float(py, condition_number_from_r(&r_arr.as_array()).into()))
        }
        utils::RealReadonlyArray2::F64(r_arr) => {
            Ok(utils::py_float(py, condition_number_from_r(&r_arr.as_array())))
        }
    }
}

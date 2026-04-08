//! Matrix function bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

const DEFAULT_MAX_TERMS: usize = 64;

const DEFAULT_TOLERANCE: f64 = 1e-14;

/// Matrix exponential via Taylor series.
#[pyfunction(name = "matrix_exp")]
pub fn matrix_exp<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_complex_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix exponential via Taylor series into `output`.
#[pyfunction(name = "matrix_exp_into", signature = (matrix, output, max_terms=None, tolerance=None))]
pub fn matrix_exp_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<()> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_exp_complex_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix exponential via eigendecomposition.
#[pyfunction(name = "matrix_exp_eigen")]
pub fn matrix_exp_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::matrix_functions::matrix_exp_eigen_complex_view(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix log via Taylor series.
#[pyfunction(name = "matrix_log_taylor")]
pub fn matrix_log_taylor<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via Taylor series into `output`.
#[pyfunction(
    name = "matrix_log_taylor_into",
    signature = (matrix, output, max_terms=None, tolerance=None)
)]
pub fn matrix_log_taylor_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<()> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via eigendecomposition.
#[pyfunction(name = "matrix_log_eigen")]
pub fn matrix_log_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via eigendecomposition into `output`.
#[pyfunction(name = "matrix_log_eigen_into")]
pub fn matrix_log_eigen_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via SVD.
#[pyfunction(name = "matrix_log_svd")]
pub fn matrix_log_svd<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via SVD into `output`.
#[pyfunction(name = "matrix_log_svd_into")]
pub fn matrix_log_svd_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_log_svd_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix power A^p.
#[pyfunction(name = "matrix_power")]
pub fn matrix_power<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    power: f64,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let power = utils::f64_to_f32(power, "power")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_complex_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix power into `output`.
#[pyfunction(name = "matrix_power_into")]
pub fn matrix_power_into(
    matrix: &Bound<'_, PyAny>,
    power: f64,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let power = utils::f64_to_f32(power, "power")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_power_complex_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix sign function.
#[pyfunction(name = "matrix_sign")]
pub fn matrix_sign<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix sign function into `output`.
#[pyfunction(name = "matrix_sign_into")]
pub fn matrix_sign_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_sign_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_sign_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_sign_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

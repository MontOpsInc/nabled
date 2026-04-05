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
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let result =
                nabled_linalg::matrix_functions::matrix_exp_view(&arr.as_array(), terms, tol)
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let result =
                nabled_linalg::matrix_functions::matrix_exp_view(&arr.as_array(), terms, tol)
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix exponential via eigendecomposition.
#[pyfunction(name = "matrix_exp_eigen")]
pub fn matrix_exp_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
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
            let result = nabled_linalg::matrix_functions::matrix_log_taylor_view(
                &arr.as_array(),
                terms,
                tol,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let result = nabled_linalg::matrix_functions::matrix_log_taylor_view(
                &arr.as_array(),
                terms,
                tol,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix log via eigendecomposition.
#[pyfunction(name = "matrix_log_eigen")]
pub fn matrix_log_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_log_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_log_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix log via SVD.
#[pyfunction(name = "matrix_log_svd")]
pub fn matrix_log_svd<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_log_svd_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_log_svd_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
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
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let power = utils::f64_to_f32(power, "power")?;
            let result = nabled_linalg::matrix_functions::matrix_power_view(&arr.as_array(), power)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_power_view(&arr.as_array(), power)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix sign function.
#[pyfunction(name = "matrix_sign")]
pub fn matrix_sign<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_sign_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_sign_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

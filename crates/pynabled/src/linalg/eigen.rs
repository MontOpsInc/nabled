//! Eigenvalue decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn nonsymmetric_config_f32(
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> PyResult<nabled_linalg::eigen::NonsymmetricEigenConfig<f32>> {
    let mut config =
        nabled_linalg::eigen::NonsymmetricEigenConfig::<f32> { balance, ..Default::default() };
    if let Some(balance_max_iterations) = balance_max_iterations {
        config.balance_max_iterations = balance_max_iterations;
    }
    if let Some(balance_tolerance) = balance_tolerance {
        config.balance_tolerance = utils::f64_to_real(balance_tolerance, "balance_tolerance")?;
    }
    Ok(config)
}

fn nonsymmetric_config_f64(
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> nabled_linalg::eigen::NonsymmetricEigenConfig<f64> {
    let mut config =
        nabled_linalg::eigen::NonsymmetricEigenConfig::<f64> { balance, ..Default::default() };
    if let Some(balance_max_iterations) = balance_max_iterations {
        config.balance_max_iterations = balance_max_iterations;
    }
    if let Some(balance_tolerance) = balance_tolerance {
        config.balance_tolerance = balance_tolerance;
    }
    config
}

/// Symmetric eigenvalue decomposition. Returns `(eigenvalues, eigenvectors)`.
#[pyfunction(name = "eigen_symmetric")]
pub fn symmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::eigen::symmetric_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::eigen::symmetric_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
    }
}

/// Generalized eigenvalue decomposition. Returns `(eigenvalues, eigenvectors)`.
#[pyfunction(name = "eigen_generalized")]
pub fn generalized<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match (utils::real_array2(matrix_a, "matrix_a")?, utils::real_array2(matrix_b, "matrix_b")?) {
        (utils::RealReadonlyArray2::F32(a), utils::RealReadonlyArray2::F32(b)) => {
            let result = nabled_linalg::eigen::generalized_view(&a.as_array(), &b.as_array())
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        (utils::RealReadonlyArray2::F64(a), utils::RealReadonlyArray2::F64(b)) => {
            let result = nabled_linalg::eigen::generalized_view(&a.as_array(), &b.as_array())
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// Non-symmetric eigenvalue decomposition. Returns `(eigenvalues, schur_vectors)`.
#[pyfunction(name = "eigen_nonsymmetric")]
pub fn nonsymmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::eigen::nonsymmetric_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.schur_vectors),
            ))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::eigen::nonsymmetric_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.schur_vectors),
            ))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::eigen::nonsymmetric_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.schur_vectors),
            ))
        }
    }
}

/// Balance a real non-symmetric matrix. Returns `(balanced_matrix, balancing_diagonal)`.
#[pyfunction(
    name = "eigen_balance_nonsymmetric",
    signature = (matrix, balance=true, balance_max_iterations=None, balance_tolerance=None)
)]
pub fn balance_nonsymmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let config =
                nonsymmetric_config_f32(balance, balance_max_iterations, balance_tolerance)?;
            let (balanced_matrix, balancing_diagonal) =
                nabled_linalg::eigen::balance_nonsymmetric_view(&arr.as_array(), &config)
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, balanced_matrix),
                utils::pyarray1_from_owned(py, balancing_diagonal),
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let config =
                nonsymmetric_config_f64(balance, balance_max_iterations, balance_tolerance);
            let (balanced_matrix, balancing_diagonal) =
                nabled_linalg::eigen::balance_nonsymmetric_view(&arr.as_array(), &config)
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, balanced_matrix),
                utils::pyarray1_from_owned(py, balancing_diagonal),
            ))
        }
    }
}

/// Non-symmetric eigen decomposition with matched left/right eigenvectors.
#[pyfunction(
    name = "eigen_nonsymmetric_bi",
    signature = (matrix, balance=true, balance_max_iterations=None, balance_tolerance=None)
)]
pub fn nonsymmetric_bi<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let config =
                nonsymmetric_config_f32(balance, balance_max_iterations, balance_tolerance)?;
            let result = nabled_linalg::eigen::nonsymmetric_bi_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.right_eigenvectors),
                utils::pyarray2_from_owned(py, result.left_eigenvectors),
                utils::pyarray1_from_owned(py, result.balancing_diagonal),
                utils::pyarray2_from_owned(py, result.balanced_matrix),
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let config =
                nonsymmetric_config_f64(balance, balance_max_iterations, balance_tolerance);
            let result = nabled_linalg::eigen::nonsymmetric_bi_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.right_eigenvectors),
                utils::pyarray2_from_owned(py, result.left_eigenvectors),
                utils::pyarray1_from_owned(py, result.balancing_diagonal),
                utils::pyarray2_from_owned(py, result.balanced_matrix),
            ))
        }
    }
}

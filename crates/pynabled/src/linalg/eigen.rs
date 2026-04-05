//! Eigenvalue decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Symmetric eigenvalue decomposition. Returns (eigenvalues, eigenvectors).
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

/// Generalized eigenvalue decomposition. Returns (eigenvalues, eigenvectors).
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

/// Non-symmetric eigenvalue decomposition. Returns (eigenvalues, schur_vectors).
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

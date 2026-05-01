//! Batched decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Batched QR decomposition. Returns list of (Q, R, rank) tuples.
#[pyfunction(name = "batched_qr")]
pub fn qr<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, usize)>> {
    match utils::real_array3(matrices, "matrices")? {
        utils::RealReadonlyArray3::F32(arr) => {
            let config = nabled_linalg::qr::QRConfig::<f32>::default();
            let results =
                nabled_linalg::batched::qr_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray2_from_owned(py, r.q),
                        utils::pyarray2_from_owned(py, r.r),
                        r.rank,
                    )
                })
                .collect())
        }
        utils::RealReadonlyArray3::F64(arr) => {
            let config = nabled_linalg::qr::QRConfig::<f64>::default();
            let results =
                nabled_linalg::batched::qr_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray2_from_owned(py, r.q),
                        utils::pyarray2_from_owned(py, r.r),
                        r.rank,
                    )
                })
                .collect())
        }
    }
}

/// Batched SVD. Returns list of (U, singular_values, Vt) tuples.
#[pyfunction(name = "batched_svd")]
pub fn svd<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>> {
    match utils::real_array3(matrices, "matrices")? {
        utils::RealReadonlyArray3::F32(arr) => {
            let results = nabled_linalg::batched::svd_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray2_from_owned(py, r.u),
                        utils::pyarray1_from_owned(py, r.singular_values),
                        utils::pyarray2_from_owned(py, r.vt),
                    )
                })
                .collect())
        }
        utils::RealReadonlyArray3::F64(arr) => {
            let results = nabled_linalg::batched::svd_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray2_from_owned(py, r.u),
                        utils::pyarray1_from_owned(py, r.singular_values),
                        utils::pyarray2_from_owned(py, r.vt),
                    )
                })
                .collect())
        }
    }
}

/// Batched LU decomposition. Returns list of `(L, U, pivots, permutation_sign)` tuples.
#[pyfunction(name = "batched_lu")]
pub fn lu<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i8)>> {
    match utils::real_array3(matrices, "matrices")? {
        utils::RealReadonlyArray3::F32(arr) => {
            let results = nabled_linalg::batched::lu_view_with_metadata(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|(result, pivots, permutation_sign)| {
                    (
                        utils::pyarray2_from_owned(py, result.l),
                        utils::pyarray2_from_owned(py, result.u),
                        utils::pyarray1_from_owned(
                            py,
                            utils::usize_array1_to_i64(pivots, "pivots")
                                .expect("usize pivot indices should fit in Python int64 arrays"),
                        ),
                        permutation_sign,
                    )
                })
                .collect())
        }
        utils::RealReadonlyArray3::F64(arr) => {
            let results = nabled_linalg::batched::lu_view_with_metadata(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|(result, pivots, permutation_sign)| {
                    (
                        utils::pyarray2_from_owned(py, result.l),
                        utils::pyarray2_from_owned(py, result.u),
                        utils::pyarray1_from_owned(
                            py,
                            utils::usize_array1_to_i64(pivots, "pivots")
                                .expect("usize pivot indices should fit in Python int64 arrays"),
                        ),
                        permutation_sign,
                    )
                })
                .collect())
        }
    }
}

/// Batched Cholesky decomposition. Returns list of L matrices.
#[pyfunction(name = "batched_cholesky")]
pub fn cholesky<'py>(py: Python<'py>, matrices: &Bound<'py, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
    match utils::real_array3(matrices, "matrices")? {
        utils::RealReadonlyArray3::F32(arr) => {
            let results =
                nabled_linalg::batched::cholesky_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results.into_iter().map(|r| utils::pyarray2_from_owned(py, r.l)).collect())
        }
        utils::RealReadonlyArray3::F64(arr) => {
            let results =
                nabled_linalg::batched::cholesky_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results.into_iter().map(|r| utils::pyarray2_from_owned(py, r.l)).collect())
        }
    }
}

/// Batched symmetric eigendecomposition. Returns list of (eigenvalues, eigenvectors) tuples.
#[pyfunction(name = "batched_symmetric_eigen")]
pub fn symmetric_eigen<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
    match utils::real_array3(matrices, "matrices")? {
        utils::RealReadonlyArray3::F32(arr) => {
            let results =
                nabled_linalg::batched::symmetric_eigen_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray1_from_owned(py, r.eigenvalues),
                        utils::pyarray2_from_owned(py, r.eigenvectors),
                    )
                })
                .collect())
        }
        utils::RealReadonlyArray3::F64(arr) => {
            let results =
                nabled_linalg::batched::symmetric_eigen_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(results
                .into_iter()
                .map(|r| {
                    (
                        utils::pyarray1_from_owned(py, r.eigenvalues),
                        utils::pyarray2_from_owned(py, r.eigenvectors),
                    )
                })
                .collect())
        }
    }
}

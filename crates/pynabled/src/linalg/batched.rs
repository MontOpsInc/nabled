//! Batched decomposition bindings for Python.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Batched QR decomposition. Returns list of (Q, R, rank) tuples.
#[pyfunction(name = "batched_qr")]
pub fn qr<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Vec<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, usize)>> {
    utils::require_contiguous(matrices)?;
    let arr = matrices.readonly();
    let config = nabled_linalg::qr::QRConfig::<f64>::default();
    let results = nabled_linalg::batched::qr_view(&arr.as_array(), &config).map_err(to_py_err)?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push((
            PyArray2::from_owned_array(py, r.q).unbind(),
            PyArray2::from_owned_array(py, r.r).unbind(),
            r.rank,
        ));
    }
    Ok(out)
}

/// Batched SVD. Returns list of (U, singular_values, Vt) tuples.
#[pyfunction(name = "batched_svd")]
pub fn svd<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Vec<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)>> {
    utils::require_contiguous(matrices)?;
    let arr = matrices.readonly();
    let results = nabled_linalg::batched::svd_view(&arr.as_array()).map_err(to_py_err)?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push((
            PyArray2::from_owned_array(py, r.u).unbind(),
            PyArray1::from_owned_array(py, r.singular_values).unbind(),
            PyArray2::from_owned_array(py, r.vt).unbind(),
        ));
    }
    Ok(out)
}

/// Batched LU decomposition. Returns list of (L, U) tuples.
#[pyfunction(name = "batched_lu")]
pub fn lu<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Vec<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)>> {
    utils::require_contiguous(matrices)?;
    let arr = matrices.readonly();
    let results = nabled_linalg::batched::lu_view(&arr.as_array()).map_err(to_py_err)?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push((
            PyArray2::from_owned_array(py, r.l).unbind(),
            PyArray2::from_owned_array(py, r.u).unbind(),
        ));
    }
    Ok(out)
}

/// Batched Cholesky decomposition. Returns list of L matrices.
#[pyfunction(name = "batched_cholesky")]
pub fn cholesky<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    utils::require_contiguous(matrices)?;
    let arr = matrices.readonly();
    let results = nabled_linalg::batched::cholesky_view(&arr.as_array()).map_err(to_py_err)?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(PyArray2::from_owned_array(py, r.l).unbind());
    }
    Ok(out)
}

/// Batched symmetric eigendecomposition. Returns list of (eigenvalues, eigenvectors) tuples.
#[pyfunction(name = "batched_symmetric_eigen")]
pub fn symmetric_eigen<'py>(
    py: Python<'py>,
    matrices: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Vec<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)>> {
    utils::require_contiguous(matrices)?;
    let arr = matrices.readonly();
    let results =
        nabled_linalg::batched::symmetric_eigen_view(&arr.as_array()).map_err(to_py_err)?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push((
            PyArray1::from_owned_array(py, r.eigenvalues).unbind(),
            PyArray2::from_owned_array(py, r.eigenvectors).unbind(),
        ));
    }
    Ok(out)
}

//! CSR sparse matrix bindings for Python.
//!
//! Accept (nrows, ncols, indptr, indices, data) compatible with scipy.sparse.csr_matrix.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

fn i64_to_usize(x: i64) -> PyResult<usize> {
    usize::try_from(x)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("index out of range for usize"))
}

/// Sparse matrix-vector product. CSR format: (nrows, ncols, indptr, indices, data).
#[pyfunction(name = "sparse_matvec")]
pub fn matvec<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyArray1<i64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    data: &Bound<'py, PyArray1<f64>>,
    vector: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let indptr_arr = indptr.readonly();
    let indices_arr = indices.readonly();
    let data_arr = data.readonly();
    let v = vector.readonly();

    let indptr_vec: Vec<usize> = indptr_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let indices_vec: Vec<usize> = indices_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let data_vec: Vec<f64> = data_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_vec();

    let csr =
        nabled_linalg::sparse::CsrMatrix::new(nrows, ncols, indptr_vec, indices_vec, data_vec)
            .map_err(to_py_err)?;

    let result =
        nabled_linalg::sparse::matvec(&csr, &v.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Jacobi iterative solve for sparse Ax = b.
#[pyfunction(name = "sparse_jacobi_solve")]
pub fn jacobi_solve<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyArray1<i64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    data: &Bound<'py, PyArray1<f64>>,
    rhs: &Bound<'py, PyArray1<f64>>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let indptr_arr = indptr.readonly();
    let indices_arr = indices.readonly();
    let data_arr = data.readonly();
    let r = rhs.readonly();

    let indptr_vec: Vec<usize> = indptr_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let indices_vec: Vec<usize> = indices_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let data_vec: Vec<f64> = data_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_vec();

    let csr =
        nabled_linalg::sparse::CsrMatrix::new(nrows, ncols, indptr_vec, indices_vec, data_vec)
            .map_err(to_py_err)?;

    let tol = tolerance.unwrap_or(1e-10);
    let max_it = max_iterations.unwrap_or(5000);

    let result = nabled_linalg::sparse::jacobi_solve(&csr, &r.as_array().to_owned(), tol, max_it)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// PCG (preconditioned conjugate gradient) solve for symmetric positive definite sparse Ax = b.
#[pyfunction(name = "sparse_pcg_solve")]
pub fn pcg_solve<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyArray1<i64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    data: &Bound<'py, PyArray1<f64>>,
    rhs: &Bound<'py, PyArray1<f64>>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let indptr_arr = indptr.readonly();
    let indices_arr = indices.readonly();
    let data_arr = data.readonly();
    let r = rhs.readonly();

    let indptr_vec: Vec<usize> = indptr_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let indices_vec: Vec<usize> = indices_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .iter()
        .map(|&x| i64_to_usize(x))
        .collect::<PyResult<Vec<_>>>()?;
    let data_vec: Vec<f64> = data_arr
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .to_vec();

    let csr =
        nabled_linalg::sparse::CsrMatrix::new(nrows, ncols, indptr_vec, indices_vec, data_vec)
            .map_err(to_py_err)?;

    let tol = tolerance.unwrap_or(1e-10);
    let max_it = max_iterations.unwrap_or(5000);

    let result = nabled_linalg::sparse::pcg_solve(&csr, &r.as_array().to_owned(), tol, max_it)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

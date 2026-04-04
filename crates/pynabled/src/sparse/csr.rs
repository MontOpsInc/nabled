//! Raw CSR sparse matrix bindings for Python.
//!
//! Public Python-facing sparse carrier ergonomics live in `python/pynabled/sparse.py`.

use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::to_py_err;

type PyCsrParts = (usize, usize, Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>);

fn py_value_error(message: impl ToString) -> PyErr { PyValueError::new_err(message.to_string()) }

fn csr_view_from_slices<'a>(
    nrows: usize,
    ncols: usize,
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f64],
) -> PyResult<nabled_linalg::sparse::CsrMatrixView<'a, i64, f64, i64>> {
    nabled_linalg::sparse::CsrMatrixView::new(nrows, ncols, indptr, indices, data)
        .map_err(to_py_err)
}

fn usize_vec_to_i64(values: Vec<usize>) -> PyResult<Vec<i64>> {
    values
        .into_iter()
        .map(|value| i64::try_from(value).map_err(|_| py_value_error("index out of range for i64")))
        .collect()
}

fn py_csr_parts(
    py: Python<'_>,
    matrix: nabled_linalg::sparse::CsrMatrix<f64>,
) -> PyResult<PyCsrParts> {
    let indptr = Array1::from_vec(usize_vec_to_i64(matrix.indptr)?);
    let indices = Array1::from_vec(usize_vec_to_i64(matrix.indices)?);
    Ok((
        matrix.nrows,
        matrix.ncols,
        PyArray1::from_owned_array(py, indptr).unbind(),
        PyArray1::from_owned_array(py, indices).unbind(),
        PyArray1::from_owned_array(py, Array1::from_vec(matrix.data)).unbind(),
    ))
}

/// Sparse matrix-vector product over raw CSR components.
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
    let vector_arr = vector.readonly();
    let matrix = csr_view_from_slices(
        nrows,
        ncols,
        indptr_arr.as_slice().map_err(py_value_error)?,
        indices_arr.as_slice().map_err(py_value_error)?,
        data_arr.as_slice().map_err(py_value_error)?,
    )?;
    let result =
        nabled_linalg::sparse::matvec_view(&matrix, &vector_arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Sparse-dense matrix multiplication `A @ B` over raw CSR components.
#[pyfunction(name = "sparse_matmat_dense")]
pub fn matmat_dense<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyArray1<i64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    data: &Bound<'py, PyArray1<f64>>,
    dense: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let indptr_arr = indptr.readonly();
    let indices_arr = indices.readonly();
    let data_arr = data.readonly();
    let dense_arr = dense.readonly();
    let matrix = csr_view_from_slices(
        nrows,
        ncols,
        indptr_arr.as_slice().map_err(py_value_error)?,
        indices_arr.as_slice().map_err(py_value_error)?,
        data_arr.as_slice().map_err(py_value_error)?,
    )?;
    let result = nabled_linalg::sparse::matmat_dense_view(&matrix, &dense_arr.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Transpose a CSR matrix from raw CSR components.
#[pyfunction(name = "sparse_transpose")]
pub fn transpose<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyArray1<i64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    data: &Bound<'py, PyArray1<f64>>,
) -> PyResult<PyCsrParts> {
    let indptr_arr = indptr.readonly();
    let indices_arr = indices.readonly();
    let data_arr = data.readonly();
    let matrix = csr_view_from_slices(
        nrows,
        ncols,
        indptr_arr.as_slice().map_err(py_value_error)?,
        indices_arr.as_slice().map_err(py_value_error)?,
        data_arr.as_slice().map_err(py_value_error)?,
    )?;
    let result = nabled_linalg::sparse::transpose_view(&matrix).map_err(to_py_err)?;
    py_csr_parts(py, result)
}

/// Jacobi iterative solve over raw CSR components.
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
    let rhs_arr = rhs.readonly();
    let matrix = csr_view_from_slices(
        nrows,
        ncols,
        indptr_arr.as_slice().map_err(py_value_error)?,
        indices_arr.as_slice().map_err(py_value_error)?,
        data_arr.as_slice().map_err(py_value_error)?,
    )?;
    let tol = tolerance.unwrap_or(1e-10);
    let max_it = max_iterations.unwrap_or(5000);
    let result =
        nabled_linalg::sparse::jacobi_solve_view(&matrix, &rhs_arr.as_array(), tol, max_it)
            .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// PCG solve over raw CSR components.
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
    let rhs_arr = rhs.readonly();
    let matrix = csr_view_from_slices(
        nrows,
        ncols,
        indptr_arr.as_slice().map_err(py_value_error)?,
        indices_arr.as_slice().map_err(py_value_error)?,
        data_arr.as_slice().map_err(py_value_error)?,
    )?;
    let tol = tolerance.unwrap_or(1e-10);
    let max_it = max_iterations.unwrap_or(5000);
    let result = nabled_linalg::sparse::pcg_solve_view(&matrix, &rhs_arr.as_array(), tol, max_it)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

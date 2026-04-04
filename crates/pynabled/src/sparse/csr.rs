//! Raw CSR sparse matrix bindings for Python.
//!
//! Public Python-facing sparse carrier ergonomics live in `python/pynabled/sparse.py`.

use ndarray::Array1;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

type PyCsrParts = (usize, usize, Py<PyAny>, Py<PyAny>, Py<PyAny>);

fn py_value_error(message: impl ToString) -> PyErr { PyValueError::new_err(message.to_string()) }

#[allow(clippy::cast_possible_truncation)]
fn tolerance_f32(tolerance: Option<f64>, default: f32) -> f32 {
    tolerance.map_or(default, |value| value as f32)
}

fn csr_view_from_slices_f32<'a>(
    nrows: usize,
    ncols: usize,
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f32],
) -> PyResult<nabled_linalg::sparse::CsrMatrixView<'a, i64, f32, i64>> {
    nabled_linalg::sparse::CsrMatrixView::new(nrows, ncols, indptr, indices, data)
        .map_err(to_py_err)
}

fn csr_view_from_slices_f64<'a>(
    nrows: usize,
    ncols: usize,
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f64],
) -> PyResult<nabled_linalg::sparse::CsrMatrixView<'a, i64, f64, i64>> {
    nabled_linalg::sparse::CsrMatrixView::new(nrows, ncols, indptr, indices, data)
        .map_err(to_py_err)
}

fn usize_vec_to_i32(values: Vec<usize>) -> PyResult<Vec<i32>> {
    values
        .into_iter()
        .map(|value| i32::try_from(value).map_err(|_| py_value_error("index out of range for i32")))
        .collect()
}

fn usize_vec_to_i64(values: Vec<usize>) -> PyResult<Vec<i64>> {
    values
        .into_iter()
        .map(|value| i64::try_from(value).map_err(|_| py_value_error("index out of range for i64")))
        .collect()
}

fn py_csr_parts_i32_f32(
    py: Python<'_>,
    matrix: nabled_linalg::sparse::CsrMatrix<f32>,
) -> PyResult<PyCsrParts> {
    let indptr = Array1::from_vec(usize_vec_to_i32(matrix.indptr)?);
    let indices = Array1::from_vec(usize_vec_to_i32(matrix.indices)?);
    Ok((
        matrix.nrows,
        matrix.ncols,
        PyArray1::from_owned_array(py, indptr).into_any().unbind(),
        PyArray1::from_owned_array(py, indices).into_any().unbind(),
        utils::pyarray1_from_owned(py, Array1::from_vec(matrix.data)),
    ))
}

fn py_csr_parts_f32(
    py: Python<'_>,
    matrix: nabled_linalg::sparse::CsrMatrix<f32>,
) -> PyResult<PyCsrParts> {
    let indptr = Array1::from_vec(usize_vec_to_i64(matrix.indptr)?);
    let indices = Array1::from_vec(usize_vec_to_i64(matrix.indices)?);
    Ok((
        matrix.nrows,
        matrix.ncols,
        PyArray1::from_owned_array(py, indptr).into_any().unbind(),
        PyArray1::from_owned_array(py, indices).into_any().unbind(),
        utils::pyarray1_from_owned(py, Array1::from_vec(matrix.data)),
    ))
}

fn py_csr_parts_i32_f64(
    py: Python<'_>,
    matrix: nabled_linalg::sparse::CsrMatrix<f64>,
) -> PyResult<PyCsrParts> {
    let indptr = Array1::from_vec(usize_vec_to_i32(matrix.indptr)?);
    let indices = Array1::from_vec(usize_vec_to_i32(matrix.indices)?);
    Ok((
        matrix.nrows,
        matrix.ncols,
        PyArray1::from_owned_array(py, indptr).into_any().unbind(),
        PyArray1::from_owned_array(py, indices).into_any().unbind(),
        utils::pyarray1_from_owned(py, Array1::from_vec(matrix.data)),
    ))
}

fn py_csr_parts_f64(
    py: Python<'_>,
    matrix: nabled_linalg::sparse::CsrMatrix<f64>,
) -> PyResult<PyCsrParts> {
    let indptr = Array1::from_vec(usize_vec_to_i64(matrix.indptr)?);
    let indices = Array1::from_vec(usize_vec_to_i64(matrix.indices)?);
    Ok((
        matrix.nrows,
        matrix.ncols,
        PyArray1::from_owned_array(py, indptr).into_any().unbind(),
        PyArray1::from_owned_array(py, indices).into_any().unbind(),
        utils::pyarray1_from_owned(py, Array1::from_vec(matrix.data)),
    ))
}

/// Sparse matrix-vector product over raw CSR components.
#[pyfunction(name = "sparse_matvec")]
pub fn matvec<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    vector: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::index_array1(indptr, "indptr")?,
        utils::index_array1(indices, "indices")?,
        utils::real_array1(data, "data")?,
        utils::real_array1(vector, "vector")?,
    ) {
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(vector_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::matvec_view(&matrix, &vector_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(vector_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::matvec_view(&matrix, &vector_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(vector_arr),
        ) => {
            let matrix = csr_view_from_slices_f32(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::matvec_view(&matrix, &vector_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(vector_arr),
        ) => {
            let matrix = csr_view_from_slices_f64(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::matvec_view(&matrix, &vector_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _, _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _, _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
        _ => Err(utils::matching_real_dtype_error(&["data", "vector"])),
    }
}

/// Sparse-dense matrix multiplication `A @ B` over raw CSR components.
#[pyfunction(name = "sparse_matmat_dense")]
pub fn matmat_dense<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    dense: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::index_array1(indptr, "indptr")?,
        utils::index_array1(indices, "indices")?,
        utils::real_array1(data, "data")?,
        utils::real_array2(dense, "dense")?,
    ) {
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray2::F32(dense_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::matmat_dense_view(&matrix, &dense_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray2::F64(dense_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::matmat_dense_view(&matrix, &dense_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray2::F32(dense_arr),
        ) => {
            let matrix = csr_view_from_slices_f32(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::matmat_dense_view(&matrix, &dense_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray2::F64(dense_arr),
        ) => {
            let matrix = csr_view_from_slices_f64(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::matmat_dense_view(&matrix, &dense_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _, _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _, _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
        _ => Err(utils::matching_real_dtype_error(&["data", "dense"])),
    }
}

/// Transpose a CSR matrix from raw CSR components.
#[pyfunction(name = "sparse_transpose")]
pub fn transpose<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<PyCsrParts> {
    match (
        utils::index_array1(indptr, "indptr")?,
        utils::index_array1(indices, "indices")?,
        utils::real_array1(data, "data")?,
    ) {
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::transpose_view(&matrix).map_err(to_py_err)?;
            py_csr_parts_i32_f32(py, result)
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::transpose_view(&matrix).map_err(to_py_err)?;
            py_csr_parts_i32_f64(py, result)
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
        ) => {
            let matrix = csr_view_from_slices_f32(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::transpose_view(&matrix).map_err(to_py_err)?;
            py_csr_parts_f32(py, result)
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
        ) => {
            let matrix = csr_view_from_slices_f64(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::transpose_view(&matrix).map_err(to_py_err)?;
            py_csr_parts_f64(py, result)
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Jacobi iterative solve over raw CSR components.
#[pyfunction(name = "sparse_jacobi_solve")]
pub fn jacobi_solve<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::index_array1(indptr, "indptr")?,
        utils::index_array1(indices, "indices")?,
        utils::real_array1(data, "data")?,
        utils::real_array1(rhs, "rhs")?,
    ) {
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(rhs_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::jacobi_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance_f32(tolerance, 1e-6_f32),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(rhs_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::jacobi_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance.unwrap_or(1e-10),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(rhs_arr),
        ) => {
            let matrix = csr_view_from_slices_f32(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::jacobi_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance_f32(tolerance, 1e-6_f32),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(rhs_arr),
        ) => {
            let matrix = csr_view_from_slices_f64(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::jacobi_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance.unwrap_or(1e-10),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _, _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _, _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
        _ => Err(utils::matching_real_dtype_error(&["data", "rhs"])),
    }
}

/// PCG solve over raw CSR components.
#[pyfunction(name = "sparse_pcg_solve")]
pub fn pcg_solve<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::index_array1(indptr, "indptr")?,
        utils::index_array1(indices, "indices")?,
        utils::real_array1(data, "data")?,
        utils::real_array1(rhs, "rhs")?,
    ) {
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(rhs_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::pcg_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance_f32(tolerance, 1e-6_f32),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(rhs_arr),
        ) => {
            let matrix = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )
            .map_err(to_py_err)?;
            let result = nabled_linalg::sparse::pcg_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance.unwrap_or(1e-10),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
            utils::RealReadonlyArray1::F32(rhs_arr),
        ) => {
            let matrix = csr_view_from_slices_f32(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::pcg_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance_f32(tolerance, 1e-6_f32),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
            utils::RealReadonlyArray1::F64(rhs_arr),
        ) => {
            let matrix = csr_view_from_slices_f64(
                nrows,
                ncols,
                indptr_arr.as_slice().map_err(py_value_error)?,
                indices_arr.as_slice().map_err(py_value_error)?,
                data_arr.as_slice().map_err(py_value_error)?,
            )?;
            let result = nabled_linalg::sparse::pcg_solve_view(
                &matrix,
                &rhs_arr.as_array(),
                tolerance.unwrap_or(1e-10),
                max_iterations.unwrap_or(5000),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _, _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _, _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
        _ => Err(utils::matching_real_dtype_error(&["data", "rhs"])),
    }
}

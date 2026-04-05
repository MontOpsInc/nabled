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

#[expect(clippy::cast_possible_truncation)]
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
#[expect(clippy::too_many_lines)]
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
#[expect(clippy::too_many_lines)]
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

#[derive(Clone, Copy)]
enum StoredIndexDtype {
    I32,
    I64,
}

fn py_csr_parts_ref_f32(
    py: Python<'_>,
    matrix: &nabled_linalg::sparse::CsrMatrix<f32>,
    index_dtype: StoredIndexDtype,
) -> PyResult<PyCsrParts> {
    match index_dtype {
        StoredIndexDtype::I32 => py_csr_parts_i32_f32(py, matrix.clone()),
        StoredIndexDtype::I64 => py_csr_parts_f32(py, matrix.clone()),
    }
}

fn py_csr_parts_ref_f64(
    py: Python<'_>,
    matrix: &nabled_linalg::sparse::CsrMatrix<f64>,
    index_dtype: StoredIndexDtype,
) -> PyResult<PyCsrParts> {
    match index_dtype {
        StoredIndexDtype::I32 => py_csr_parts_i32_f64(py, matrix.clone()),
        StoredIndexDtype::I64 => py_csr_parts_f64(py, matrix.clone()),
    }
}

fn py_index_array(
    py: Python<'_>,
    values: &[usize],
    index_dtype: StoredIndexDtype,
) -> PyResult<Py<PyAny>> {
    match index_dtype {
        StoredIndexDtype::I32 => {
            let values = Array1::from_vec(usize_vec_to_i32(values.to_vec())?);
            Ok(PyArray1::from_owned_array(py, values).into_any().unbind())
        }
        StoredIndexDtype::I64 => {
            let values = Array1::from_vec(usize_vec_to_i64(values.to_vec())?);
            Ok(PyArray1::from_owned_array(py, values).into_any().unbind())
        }
    }
}

fn owned_csr_from_slices<
    T,
    R: nabled_linalg::sparse::CsrIndex,
    C: nabled_linalg::sparse::CsrIndex,
>(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[R],
    col_indices: &[C],
    values: &[T],
) -> PyResult<nabled_linalg::sparse::CsrMatrix<T>>
where
    T: nabled_core::scalar::NabledReal + Clone,
{
    let indptr = row_ptrs
        .iter()
        .copied()
        .map(nabled_linalg::sparse::CsrIndex::to_usize)
        .collect::<Result<Vec<_>, _>>()
        .map_err(to_py_err)?;
    let indices = col_indices
        .iter()
        .copied()
        .map(nabled_linalg::sparse::CsrIndex::to_usize)
        .collect::<Result<Vec<_>, _>>()
        .map_err(to_py_err)?;
    nabled_linalg::sparse::CsrMatrix::new(nrows, ncols, indptr, indices, values.to_vec())
        .map_err(to_py_err)
}

enum PyJacobiPreconditionerInner {
    F32(nabled_linalg::sparse::JacobiPreconditioner<f32>),
    F64(nabled_linalg::sparse::JacobiPreconditioner<f64>),
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseJacobiPreconditioner")]
pub(crate) struct PyJacobiPreconditioner {
    inner: PyJacobiPreconditionerInner,
}

#[pymethods]
impl PyJacobiPreconditioner {
    #[getter]
    fn inverse_diagonal(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.inner {
            PyJacobiPreconditionerInner::F32(preconditioner) => {
                utils::pyarray1_from_owned(py, preconditioner.inverse_diagonal.clone())
            }
            PyJacobiPreconditionerInner::F64(preconditioner) => {
                utils::pyarray1_from_owned(py, preconditioner.inverse_diagonal.clone())
            }
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyJacobiPreconditionerInner::F32(preconditioner),
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_jacobi_preconditioner(
                    preconditioner,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyJacobiPreconditionerInner::F64(preconditioner),
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_jacobi_preconditioner(
                    preconditioner,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["preconditioner", "rhs"])),
        }
    }
}

enum PyIlu0FactorizationInner {
    F32 {
        factorization: nabled_linalg::sparse::ILU0Factorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        factorization: nabled_linalg::sparse::ILU0Factorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseILU0Factorization")]
pub(crate) struct PyIlu0Factorization {
    inner: PyIlu0FactorizationInner,
}

#[pymethods]
impl PyIlu0Factorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlu0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PyIlu0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn u_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlu0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.u, *index_dtype)
            }
            PyIlu0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.u, *index_dtype)
            }
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyIlu0FactorizationInner::F32 { factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ilu0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyIlu0FactorizationInner::F64 { factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ilu0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

enum PyIlutFactorizationInner {
    F32 {
        factorization: nabled_linalg::sparse::ILUTFactorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        factorization: nabled_linalg::sparse::ILUTFactorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseILUTFactorization")]
pub(crate) struct PyIlutFactorization {
    inner: PyIlutFactorizationInner,
}

#[pymethods]
impl PyIlutFactorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlutFactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PyIlutFactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn u_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlutFactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.u, *index_dtype)
            }
            PyIlutFactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.u, *index_dtype)
            }
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyIlutFactorizationInner::F32 { factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ilut_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyIlutFactorizationInner::F64 { factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ilut_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

enum PyIlukFactorizationInner {
    F32 {
        factorization: nabled_linalg::sparse::ILUKFactorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        factorization: nabled_linalg::sparse::ILUKFactorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseILUKFactorization")]
pub(crate) struct PyIlukFactorization {
    inner: PyIlukFactorizationInner,
}

#[pymethods]
impl PyIlukFactorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlukFactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PyIlukFactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn u_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIlukFactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.u, *index_dtype)
            }
            PyIlukFactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.u, *index_dtype)
            }
        }
    }

    #[getter]
    fn level_of_fill(&self) -> usize {
        match &self.inner {
            PyIlukFactorizationInner::F32 { factorization, .. } => factorization.level_of_fill,
            PyIlukFactorizationInner::F64 { factorization, .. } => factorization.level_of_fill,
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyIlukFactorizationInner::F32 { factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_iluk_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyIlukFactorizationInner::F64 { factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_iluk_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

enum PyIc0FactorizationInner {
    F32 {
        factorization: nabled_linalg::sparse::IC0Factorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        factorization: nabled_linalg::sparse::IC0Factorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseIC0Factorization")]
pub(crate) struct PyIc0Factorization {
    inner: PyIc0FactorizationInner,
}

#[pymethods]
impl PyIc0Factorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIc0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PyIc0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn l_transpose_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIc0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l_transpose, *index_dtype)
            }
            PyIc0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l_transpose, *index_dtype)
            }
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyIc0FactorizationInner::F32 { factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ic0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyIc0FactorizationInner::F64 { factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ic0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

enum PyIldl0FactorizationInner {
    F32 {
        factorization: nabled_linalg::sparse::ILDL0Factorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        factorization: nabled_linalg::sparse::ILDL0Factorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseILDL0Factorization")]
pub(crate) struct PyIldl0Factorization {
    inner: PyIldl0FactorizationInner,
}

#[pymethods]
impl PyIldl0Factorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIldl0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PyIldl0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn l_transpose_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PyIldl0FactorizationInner::F32 { factorization, index_dtype } => {
                py_csr_parts_ref_f32(py, &factorization.l_transpose, *index_dtype)
            }
            PyIldl0FactorizationInner::F64 { factorization, index_dtype } => {
                py_csr_parts_ref_f64(py, &factorization.l_transpose, *index_dtype)
            }
        }
    }

    #[getter]
    fn d(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.inner {
            PyIldl0FactorizationInner::F32 { factorization, .. } => {
                utils::pyarray1_from_owned(py, factorization.d.clone())
            }
            PyIldl0FactorizationInner::F64 { factorization, .. } => {
                utils::pyarray1_from_owned(py, factorization.d.clone())
            }
        }
    }

    fn apply<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PyIldl0FactorizationInner::F32 { factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ildl0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PyIldl0FactorizationInner::F64 { factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::apply_ildl0_preconditioner(
                    factorization,
                    &rhs_arr.as_array(),
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

enum PySparseLuFactorizationInner {
    F32 {
        matrix:        nabled_linalg::sparse::CsrMatrix<f32>,
        factorization: nabled_linalg::sparse::SparseLUFactorization<f32>,
        index_dtype:   StoredIndexDtype,
    },
    F64 {
        matrix:        nabled_linalg::sparse::CsrMatrix<f64>,
        factorization: nabled_linalg::sparse::SparseLUFactorization<f64>,
        index_dtype:   StoredIndexDtype,
    },
}

#[pyclass(module = "pynabled._pynabled", name = "_SparseLUFactorization")]
pub(crate) struct PySparseLuFactorization {
    inner: PySparseLuFactorizationInner,
}

#[pymethods]
impl PySparseLuFactorization {
    fn l_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PySparseLuFactorizationInner::F32 { factorization, index_dtype, .. } => {
                py_csr_parts_ref_f32(py, &factorization.l, *index_dtype)
            }
            PySparseLuFactorizationInner::F64 { factorization, index_dtype, .. } => {
                py_csr_parts_ref_f64(py, &factorization.l, *index_dtype)
            }
        }
    }

    fn u_parts(&self, py: Python<'_>) -> PyResult<PyCsrParts> {
        match &self.inner {
            PySparseLuFactorizationInner::F32 { factorization, index_dtype, .. } => {
                py_csr_parts_ref_f32(py, &factorization.u, *index_dtype)
            }
            PySparseLuFactorizationInner::F64 { factorization, index_dtype, .. } => {
                py_csr_parts_ref_f64(py, &factorization.u, *index_dtype)
            }
        }
    }

    #[getter]
    fn permutation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            PySparseLuFactorizationInner::F32 { factorization, index_dtype, .. } => {
                py_index_array(py, &factorization.permutation, *index_dtype)
            }
            PySparseLuFactorizationInner::F64 { factorization, index_dtype, .. } => {
                py_index_array(py, &factorization.permutation, *index_dtype)
            }
        }
    }

    fn solve<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array1(rhs, "rhs")?) {
            (
                PySparseLuFactorizationInner::F32 { matrix, factorization, .. },
                utils::RealReadonlyArray1::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::sparse_lu_solve_with_factorization(
                    matrix,
                    &rhs_arr.as_array().to_owned(),
                    factorization,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            (
                PySparseLuFactorizationInner::F64 { matrix, factorization, .. },
                utils::RealReadonlyArray1::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::sparse_lu_solve_with_factorization(
                    matrix,
                    &rhs_arr.as_array().to_owned(),
                    factorization,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray1_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }

    fn solve_multiple<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array2(rhs, "rhs")?) {
            (
                PySparseLuFactorizationInner::F32 { matrix, factorization, .. },
                utils::RealReadonlyArray2::F32(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::sparse_lu_solve_multiple_with_factorization(
                    matrix,
                    &rhs_arr.as_array().to_owned(),
                    factorization,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
            (
                PySparseLuFactorizationInner::F64 { matrix, factorization, .. },
                utils::RealReadonlyArray2::F64(rhs_arr),
            ) => {
                let result = nabled_linalg::sparse::sparse_lu_solve_multiple_with_factorization(
                    matrix,
                    &rhs_arr.as_array().to_owned(),
                    factorization,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
            _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
        }
    }
}

/// Build a reusable Jacobi preconditioner for a CSR matrix.
#[pyfunction(name = "sparse_jacobi_preconditioner")]
pub fn jacobi_preconditioner(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PyJacobiPreconditioner> {
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
            let preconditioner =
                nabled_linalg::sparse::jacobi_preconditioner_view(&matrix).map_err(to_py_err)?;
            Ok(PyJacobiPreconditioner { inner: PyJacobiPreconditionerInner::F32(preconditioner) })
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
            let preconditioner =
                nabled_linalg::sparse::jacobi_preconditioner_view(&matrix).map_err(to_py_err)?;
            Ok(PyJacobiPreconditioner { inner: PyJacobiPreconditionerInner::F64(preconditioner) })
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
            let preconditioner =
                nabled_linalg::sparse::jacobi_preconditioner_view(&matrix).map_err(to_py_err)?;
            Ok(PyJacobiPreconditioner { inner: PyJacobiPreconditionerInner::F32(preconditioner) })
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
            let preconditioner =
                nabled_linalg::sparse::jacobi_preconditioner_view(&matrix).map_err(to_py_err)?;
            Ok(PyJacobiPreconditioner { inner: PyJacobiPreconditionerInner::F64(preconditioner) })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable ILU(0) factorization for a CSR matrix.
#[pyfunction(name = "sparse_ilu0_factor")]
pub fn ilu0_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PyIlu0Factorization> {
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
            let factorization =
                nabled_linalg::sparse::ilu0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIlu0Factorization {
                inner: PyIlu0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ilu0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIlu0Factorization {
                inner: PyIlu0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ilu0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIlu0Factorization {
                inner: PyIlu0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ilu0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIlu0Factorization {
                inner: PyIlu0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable ILUT factorization for a CSR matrix.
#[pyfunction(name = "sparse_ilut_factor")]
#[expect(clippy::too_many_lines)]
pub fn ilut_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    drop_tolerance: f64,
    max_fill: usize,
) -> PyResult<PyIlutFactorization> {
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
            let factorization = nabled_linalg::sparse::ilut_factor_view(
                &matrix,
                utils::f64_to_real::<f32>(drop_tolerance, "drop_tolerance")?,
                max_fill,
            )
            .map_err(to_py_err)?;
            Ok(PyIlutFactorization {
                inner: PyIlutFactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ilut_factor_view(&matrix, drop_tolerance, max_fill)
                    .map_err(to_py_err)?;
            Ok(PyIlutFactorization {
                inner: PyIlutFactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization = nabled_linalg::sparse::ilut_factor_view(
                &matrix,
                utils::f64_to_real::<f32>(drop_tolerance, "drop_tolerance")?,
                max_fill,
            )
            .map_err(to_py_err)?;
            Ok(PyIlutFactorization {
                inner: PyIlutFactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ilut_factor_view(&matrix, drop_tolerance, max_fill)
                    .map_err(to_py_err)?;
            Ok(PyIlutFactorization {
                inner: PyIlutFactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable ILU(k) factorization for a CSR matrix.
#[pyfunction(name = "sparse_iluk_factor")]
pub fn iluk_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    level_of_fill: usize,
) -> PyResult<PyIlukFactorization> {
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
            let factorization = nabled_linalg::sparse::iluk_factor_view(&matrix, level_of_fill)
                .map_err(to_py_err)?;
            Ok(PyIlukFactorization {
                inner: PyIlukFactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization = nabled_linalg::sparse::iluk_factor_view(&matrix, level_of_fill)
                .map_err(to_py_err)?;
            Ok(PyIlukFactorization {
                inner: PyIlukFactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization = nabled_linalg::sparse::iluk_factor_view(&matrix, level_of_fill)
                .map_err(to_py_err)?;
            Ok(PyIlukFactorization {
                inner: PyIlukFactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
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
            let factorization = nabled_linalg::sparse::iluk_factor_view(&matrix, level_of_fill)
                .map_err(to_py_err)?;
            Ok(PyIlukFactorization {
                inner: PyIlukFactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable IC(0) factorization for a CSR matrix.
#[pyfunction(name = "sparse_ic0_factor")]
pub fn ic0_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PyIc0Factorization> {
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
            let factorization =
                nabled_linalg::sparse::ic0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIc0Factorization {
                inner: PyIc0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ic0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIc0Factorization {
                inner: PyIc0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ic0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIc0Factorization {
                inner: PyIc0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ic0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIc0Factorization {
                inner: PyIc0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable ILDL(0) factorization for a CSR matrix.
#[pyfunction(name = "sparse_ildl0_factor")]
pub fn ildl0_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PyIldl0Factorization> {
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
            let factorization =
                nabled_linalg::sparse::ildl0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIldl0Factorization {
                inner: PyIldl0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ildl0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIldl0Factorization {
                inner: PyIldl0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ildl0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIldl0Factorization {
                inner: PyIldl0FactorizationInner::F32 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
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
            let factorization =
                nabled_linalg::sparse::ildl0_factor_view(&matrix).map_err(to_py_err)?;
            Ok(PyIldl0Factorization {
                inner: PyIldl0FactorizationInner::F64 {
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

/// Build a reusable sparse direct LU factorization for a CSR matrix.
#[pyfunction(name = "sparse_lu_factor")]
#[expect(clippy::too_many_lines)]
pub fn sparse_lu_factor(
    nrows: usize,
    ncols: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<PySparseLuFactorization> {
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
            let indptr_slice = indptr_arr.as_slice().map_err(py_value_error)?;
            let indices_slice = indices_arr.as_slice().map_err(py_value_error)?;
            let data_slice = data_arr.as_slice().map_err(py_value_error)?;
            let matrix_view = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_slice,
                indices_slice,
                data_slice,
            )
            .map_err(to_py_err)?;
            let matrix =
                owned_csr_from_slices(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let factorization =
                nabled_linalg::sparse::sparse_lu_factor_view(&matrix_view).map_err(to_py_err)?;
            Ok(PySparseLuFactorization {
                inner: PySparseLuFactorizationInner::F32 {
                    matrix,
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
        }
        (
            utils::IndexReadonlyArray1::I32(indptr_arr),
            utils::IndexReadonlyArray1::I32(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
        ) => {
            let indptr_slice = indptr_arr.as_slice().map_err(py_value_error)?;
            let indices_slice = indices_arr.as_slice().map_err(py_value_error)?;
            let data_slice = data_arr.as_slice().map_err(py_value_error)?;
            let matrix_view = nabled_linalg::sparse::CsrMatrixView::new(
                nrows,
                ncols,
                indptr_slice,
                indices_slice,
                data_slice,
            )
            .map_err(to_py_err)?;
            let matrix =
                owned_csr_from_slices(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let factorization =
                nabled_linalg::sparse::sparse_lu_factor_view(&matrix_view).map_err(to_py_err)?;
            Ok(PySparseLuFactorization {
                inner: PySparseLuFactorizationInner::F64 {
                    matrix,
                    factorization,
                    index_dtype: StoredIndexDtype::I32,
                },
            })
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F32(data_arr),
        ) => {
            let indptr_slice = indptr_arr.as_slice().map_err(py_value_error)?;
            let indices_slice = indices_arr.as_slice().map_err(py_value_error)?;
            let data_slice = data_arr.as_slice().map_err(py_value_error)?;
            let matrix_view =
                csr_view_from_slices_f32(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let matrix =
                owned_csr_from_slices(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let factorization =
                nabled_linalg::sparse::sparse_lu_factor_view(&matrix_view).map_err(to_py_err)?;
            Ok(PySparseLuFactorization {
                inner: PySparseLuFactorizationInner::F32 {
                    matrix,
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (
            utils::IndexReadonlyArray1::I64(indptr_arr),
            utils::IndexReadonlyArray1::I64(indices_arr),
            utils::RealReadonlyArray1::F64(data_arr),
        ) => {
            let indptr_slice = indptr_arr.as_slice().map_err(py_value_error)?;
            let indices_slice = indices_arr.as_slice().map_err(py_value_error)?;
            let data_slice = data_arr.as_slice().map_err(py_value_error)?;
            let matrix_view =
                csr_view_from_slices_f64(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let matrix =
                owned_csr_from_slices(nrows, ncols, indptr_slice, indices_slice, data_slice)?;
            let factorization =
                nabled_linalg::sparse::sparse_lu_factor_view(&matrix_view).map_err(to_py_err)?;
            Ok(PySparseLuFactorization {
                inner: PySparseLuFactorizationInner::F64 {
                    matrix,
                    factorization,
                    index_dtype: StoredIndexDtype::I64,
                },
            })
        }
        (utils::IndexReadonlyArray1::I32(_), utils::IndexReadonlyArray1::I64(_), _)
        | (utils::IndexReadonlyArray1::I64(_), utils::IndexReadonlyArray1::I32(_), _) => {
            Err(utils::matching_index_dtype_error(&["indptr", "indices"]))
        }
    }
}

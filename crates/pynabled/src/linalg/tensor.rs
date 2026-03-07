//! Tensor bindings for Python.

use numpy::{PyArray2, PyArray3, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Batched matrix-vector product: cube (B, m, n) @ vectors (B, n) -> (B, m).
#[pyfunction(name = "tensor_cube_matvec")]
pub fn cube_matvec<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    vectors: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let c = cube.readonly();
    let v = vectors.readonly();
    let result =
        nabled_linalg::tensor::cube_matvec(&c.as_array().to_owned(), &v.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Batched matrix-matrix product over last two axes of cubes.
#[pyfunction(name = "tensor_cube_matmat")]
pub fn cube_matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray3<f64>>,
    right: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    let l = left.readonly();
    let r = right.readonly();
    let result =
        nabled_linalg::tensor::cube_matmat(&l.as_array().to_owned(), &r.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

/// Sum over the last axis of a tensor.
#[pyfunction(name = "tensor_sum_last_axis")]
pub fn sum_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let t = tensor.readonly();
    let result =
        nabled_linalg::tensor::sum_last_axis(&t.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// L2 norm over the last axis.
#[pyfunction(name = "tensor_l2_norm_last_axis")]
pub fn l2_norm_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let t = tensor.readonly();
    let result =
        nabled_linalg::tensor::l2_norm_last_axis(&t.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// Normalize over the last axis.
#[pyfunction(name = "tensor_normalize_last_axis")]
pub fn normalize_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let t = tensor.readonly();
    let result =
        nabled_linalg::tensor::normalize_last_axis(&t.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// Batched dot product over the last axis.
#[pyfunction(name = "tensor_batched_dot_last_axis")]
pub fn batched_dot_last_axis<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::tensor::batched_dot_last_axis(
        &l.as_array().to_owned(),
        &r.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// Permute axes of a tensor.
#[pyfunction(name = "tensor_permute_axes")]
pub fn permute_axes<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    permutation: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let t = tensor.readonly();
    let result = nabled_linalg::tensor::permute_axes(&t.as_array().to_owned(), &permutation)
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// Contract axes between two tensors.
#[pyfunction(name = "tensor_contract_axes")]
pub fn contract_axes<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::tensor::contract_axes(
        &l.as_array().to_owned(),
        &r.as_array().to_owned(),
        &left_axes,
        &right_axes,
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// Batched matrix multiply over last two axes.
#[pyfunction(name = "tensor_batched_matmul_last_two")]
pub fn batched_matmul_last_two<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::tensor::batched_matmul_last_two(
        &l.as_array().to_owned(),
        &r.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, result).unbind())
}

/// HOSVD3 decomposition. Returns (core, u0, u1, u2).
#[pyfunction(name = "tensor_hosvd3")]
pub fn hosvd3<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    rank0: usize,
    rank1: usize,
    rank2: usize,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let c = cube.readonly();
    let result = nabled_linalg::tensor::hosvd3(&c.as_array().to_owned(), (rank0, rank1, rank2))
        .map_err(to_py_err)?;
    Ok((
        PyArray3::from_owned_array(py, result.core).unbind(),
        PyArray2::from_owned_array(py, result.u0).unbind(),
        PyArray2::from_owned_array(py, result.u1).unbind(),
        PyArray2::from_owned_array(py, result.u2).unbind(),
    ))
}

/// Reconstruct cube from HOSVD3 result.
#[pyfunction(name = "tensor_hosvd3_reconstruct")]
pub fn hosvd3_reconstruct<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyArray3<f64>>,
    u0: &Bound<'py, PyArray2<f64>>,
    u1: &Bound<'py, PyArray2<f64>>,
    u2: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    let r = nabled_linalg::tensor::Hosvd3Result {
        core: core.readonly().as_array().to_owned(),
        u0:   u0.readonly().as_array().to_owned(),
        u1:   u1.readonly().as_array().to_owned(),
        u2:   u2.readonly().as_array().to_owned(),
    };
    let result = nabled_linalg::tensor::hosvd3_reconstruct(&r).map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

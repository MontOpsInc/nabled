//! Tensor bindings for Python.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

type PyCpAls3Result = (Py<PyArray1<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>);
type PyCpAlsNdResult = (Py<PyArray1<f64>>, Vec<Py<PyArray2<f64>>>);
type PyCpMetrics = (f64, f64, f64, f64);
type PyCpConvergence = (usize, bool, f64);
type PyCpReport = (PyCpConvergence, PyCpMetrics);
type PyHosvdNdResult = (Py<PyArrayDyn<f64>>, Vec<Py<PyArray2<f64>>>);
type PyTensorTrainResult = Vec<Py<PyArray3<f64>>>;

fn standard_array2<T: Clone>(array: Array2<T>) -> Array2<T> {
    array.as_standard_layout().to_owned()
}

fn standard_array3<T: Clone>(array: Array3<T>) -> Array3<T> {
    array.as_standard_layout().to_owned()
}

fn standard_arrayd<T: Clone>(array: ndarray::ArrayD<T>) -> ndarray::ArrayD<T> {
    array.as_standard_layout().to_owned()
}

fn cp_als_config(
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_linalg::tensor::CpAlsConfig<f64> {
    let mut config = nabled_linalg::tensor::CpAlsConfig::<f64>::default();
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn hooi_config(
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_linalg::tensor::HooiConfig<f64> {
    let mut config = nabled_linalg::tensor::HooiConfig::<f64>::default();
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn tt_svd_config(
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_linalg::tensor::TtSvdConfig<f64> {
    let mut config = nabled_linalg::tensor::TtSvdConfig::<f64>::default();
    config.max_rank = max_rank;
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn tt_round_config(
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_linalg::tensor::TtRoundConfig<f64> {
    let mut config = nabled_linalg::tensor::TtRoundConfig::<f64>::default();
    config.max_rank = max_rank;
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn py_cp_metrics(metrics: nabled_linalg::tensor::CpErrorMetrics<f64>) -> PyCpMetrics {
    (metrics.signal_norm, metrics.residual_norm, metrics.relative_error, metrics.fit)
}

fn py_cp_report(report: nabled_linalg::tensor::CpAlsReport<f64>) -> PyCpReport {
    (
        (
            report.convergence.iterations_run,
            report.convergence.converged,
            report.convergence.final_max_factor_change,
        ),
        py_cp_metrics(report.metrics),
    )
}

fn py_cp_als3_result(
    py: Python<'_>,
    result: nabled_linalg::tensor::CpAls3Result<f64>,
) -> PyCpAls3Result {
    (
        PyArray1::from_owned_array(py, result.weights).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.factor_0)).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.factor_1)).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.factor_2)).unbind(),
    )
}

fn py_cp_als_nd_result(
    py: Python<'_>,
    result: nabled_linalg::tensor::CpAlsNdResult<f64>,
) -> PyCpAlsNdResult {
    (
        PyArray1::from_owned_array(py, result.weights).unbind(),
        result
            .factors
            .into_iter()
            .map(|factor| PyArray2::from_owned_array(py, standard_array2(factor)).unbind())
            .collect(),
    )
}

fn py_hosvd_nd_result(
    py: Python<'_>,
    result: nabled_linalg::tensor::HosvdNdResult<f64>,
) -> PyHosvdNdResult {
    (
        PyArrayDyn::from_owned_array(py, standard_arrayd(result.core)).unbind(),
        result
            .factors
            .into_iter()
            .map(|factor| PyArray2::from_owned_array(py, standard_array2(factor)).unbind())
            .collect(),
    )
}

fn py_tt_result(
    py: Python<'_>,
    result: nabled_linalg::tensor::TensorTrainResult<f64>,
) -> PyTensorTrainResult {
    result
        .cores
        .into_iter()
        .map(|core| PyArray3::from_owned_array(py, standard_array3(core)).unbind())
        .collect()
}

fn extract_array2_sequence(arrays: &Bound<'_, PyAny>) -> PyResult<Vec<Array2<f64>>> {
    let mut out = Vec::new();
    for item in arrays.try_iter()? {
        let item = item?;
        let array = item.cast::<PyArray2<f64>>()?;
        utils::require_contiguous(array)?;
        out.push(array.readonly().as_array().to_owned());
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("expected a non-empty sequence of 2D numpy arrays"));
    }
    Ok(out)
}

fn extract_array3_sequence(arrays: &Bound<'_, PyAny>) -> PyResult<Vec<Array3<f64>>> {
    let mut out = Vec::new();
    for item in arrays.try_iter()? {
        let item = item?;
        let array = item.cast::<PyArray3<f64>>()?;
        utils::require_contiguous(array)?;
        out.push(array.readonly().as_array().to_owned());
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("expected a non-empty sequence of 3D numpy arrays"));
    }
    Ok(out)
}

fn cp_als3_result_from_arrays(
    weights: &Bound<'_, PyArray1<f64>>,
    factor_0: &Bound<'_, PyArray2<f64>>,
    factor_1: &Bound<'_, PyArray2<f64>>,
    factor_2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<nabled_linalg::tensor::CpAls3Result<f64>> {
    utils::require_contiguous(weights)?;
    utils::require_contiguous(factor_0)?;
    utils::require_contiguous(factor_1)?;
    utils::require_contiguous(factor_2)?;
    Ok(nabled_linalg::tensor::CpAls3Result {
        weights:  weights.readonly().as_array().to_owned(),
        factor_0: factor_0.readonly().as_array().to_owned(),
        factor_1: factor_1.readonly().as_array().to_owned(),
        factor_2: factor_2.readonly().as_array().to_owned(),
    })
}

fn cp_als_nd_result_from_arrays(
    weights: &Bound<'_, PyArray1<f64>>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<nabled_linalg::tensor::CpAlsNdResult<f64>> {
    utils::require_contiguous(weights)?;
    let factors = extract_array2_sequence(factors)?;
    let shape = factors.iter().map(Array2::nrows).collect();
    Ok(nabled_linalg::tensor::CpAlsNdResult {
        weights: weights.readonly().as_array().to_owned(),
        factors,
        shape,
    })
}

fn hosvd_nd_result_from_arrays(
    core: &Bound<'_, PyArrayDyn<f64>>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<nabled_linalg::tensor::HosvdNdResult<f64>> {
    utils::require_contiguous(core)?;
    Ok(nabled_linalg::tensor::HosvdNdResult {
        core:    core.readonly().as_array().to_owned(),
        factors: extract_array2_sequence(factors)?,
    })
}

fn tt_result_from_cores(
    cores: &Bound<'_, PyAny>,
) -> PyResult<nabled_linalg::tensor::TensorTrainResult<f64>> {
    let cores = extract_array3_sequence(cores)?;
    let shape = cores.iter().map(|core| core.dim().1).collect();
    Ok(nabled_linalg::tensor::TensorTrainResult { cores, shape })
}

/// Batched matrix-vector product: cube `(B, m, n)` @ vectors `(B, n)` -> `(B, m)`.
#[pyfunction(name = "tensor_cube_matvec")]
pub fn cube_matvec<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    vectors: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(cube)?;
    utils::require_contiguous(vectors)?;
    let result = nabled_linalg::tensor::cube_matvec_view(
        &cube.readonly().as_array(),
        &vectors.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Complex batched matrix-vector product.
#[pyfunction(name = "tensor_cube_matvec_complex")]
pub fn cube_matvec_complex<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<Complex64>>,
    vectors: &Bound<'py, PyArray2<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(cube)?;
    utils::require_contiguous(vectors)?;
    let result = nabled_linalg::tensor::cube_matvec_complex_view(
        &cube.readonly().as_array(),
        &vectors.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Batched matrix-matrix product over the last two cube axes.
#[pyfunction(name = "tensor_cube_matmat")]
pub fn cube_matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray3<f64>>,
    right: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::cube_matmat_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

/// Complex batched matrix-matrix product over the last two cube axes.
#[pyfunction(name = "tensor_cube_matmat_complex")]
pub fn cube_matmat_complex<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray3<Complex64>>,
    right: &Bound<'py, PyArray3<Complex64>>,
) -> PyResult<Py<PyArray3<Complex64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::cube_matmat_complex_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

/// Sum over the last axis of a tensor.
#[pyfunction(name = "tensor_sum_last_axis")]
pub fn sum_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::sum_last_axis_view(&tensor.readonly().as_array())
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Sum over the last axis of a complex tensor.
#[pyfunction(name = "tensor_sum_last_axis_complex")]
pub fn sum_last_axis_complex<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::sum_last_axis_complex_view(&tensor.readonly().as_array())
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// L2 norm over the last axis.
#[pyfunction(name = "tensor_l2_norm_last_axis")]
pub fn l2_norm_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::l2_norm_last_axis_view(&tensor.readonly().as_array())
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// L2 norm over the last axis of a complex tensor.
#[pyfunction(name = "tensor_l2_norm_last_axis_complex")]
pub fn l2_norm_last_axis_complex<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let result =
        nabled_linalg::tensor::l2_norm_last_axis_complex_view(&tensor.readonly().as_array())
            .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Normalize over the last axis.
#[pyfunction(name = "tensor_normalize_last_axis")]
pub fn normalize_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::normalize_last_axis_view(&tensor.readonly().as_array())
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Normalize a complex tensor over the last axis.
#[pyfunction(name = "tensor_normalize_last_axis_complex")]
pub fn normalize_last_axis_complex<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(tensor)?;
    let result =
        nabled_linalg::tensor::normalize_last_axis_complex_view(&tensor.readonly().as_array())
            .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Batched dot product over the last axis.
#[pyfunction(name = "tensor_batched_dot_last_axis")]
pub fn batched_dot_last_axis<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::batched_dot_last_axis_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Complex batched dot product over the last axis.
#[pyfunction(name = "tensor_batched_dot_last_axis_complex")]
pub fn batched_dot_last_axis_complex<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<Complex64>>,
    right: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::batched_dot_last_axis_complex_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Permute tensor axes.
#[pyfunction(name = "tensor_permute_axes")]
pub fn permute_axes<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    permutation: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let result =
        nabled_linalg::tensor::permute_axes_view(&tensor.readonly().as_array(), &permutation)
            .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Permute complex tensor axes.
#[pyfunction(name = "tensor_permute_axes_complex")]
pub fn permute_axes_complex<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<Complex64>>,
    permutation: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::permute_axes_complex_view(
        &tensor.readonly().as_array(),
        &permutation,
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Contract explicit axes between two tensors.
#[pyfunction(name = "tensor_contract_axes")]
pub fn contract_axes<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::contract_axes_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        &left_axes,
        &right_axes,
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Contract explicit axes between two complex tensors.
#[pyfunction(name = "tensor_contract_axes_complex")]
pub fn contract_axes_complex<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<Complex64>>,
    right: &Bound<'py, PyArrayDyn<Complex64>>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::contract_axes_complex_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        &left_axes,
        &right_axes,
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Batched matrix multiply over the last two axes.
#[pyfunction(name = "tensor_batched_matmul_last_two")]
pub fn batched_matmul_last_two<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::batched_matmul_last_two_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Complex batched matrix multiply over the last two axes.
#[pyfunction(name = "tensor_batched_matmul_last_two_complex")]
pub fn batched_matmul_last_two_complex<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArrayDyn<Complex64>>,
    right: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::batched_matmul_last_two_complex_view(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Binary Einstein summation over real tensors.
#[pyfunction(name = "tensor_einsum")]
pub fn einsum<'py>(
    py: Python<'py>,
    equation: String,
    left: &Bound<'py, PyArrayDyn<f64>>,
    right: &Bound<'py, PyArrayDyn<f64>>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::einsum_view(
        &equation,
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Binary Einstein summation over complex tensors.
#[pyfunction(name = "tensor_einsum_complex")]
pub fn einsum_complex<'py>(
    py: Python<'py>,
    equation: String,
    left: &Bound<'py, PyArrayDyn<Complex64>>,
    right: &Bound<'py, PyArrayDyn<Complex64>>,
) -> PyResult<Py<PyArrayDyn<Complex64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let result = nabled_linalg::tensor::einsum_complex_view(
        &equation,
        &left.readonly().as_array(),
        &right.readonly().as_array(),
    )
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// HOSVD3 decomposition. Returns `(core, u0, u1, u2)`.
#[pyfunction(name = "tensor_hosvd3")]
pub fn hosvd3<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    rank0: usize,
    rank1: usize,
    rank2: usize,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    utils::require_contiguous(cube)?;
    let result = nabled_linalg::tensor::hosvd3(
        &cube.readonly().as_array().to_owned(),
        (rank0, rank1, rank2),
    )
    .map_err(to_py_err)?;
    Ok((
        PyArray3::from_owned_array(py, standard_array3(result.core)).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.u0)).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.u1)).unbind(),
        PyArray2::from_owned_array(py, standard_array2(result.u2)).unbind(),
    ))
}

/// Reconstruct a cube from an HOSVD3 decomposition.
#[pyfunction(name = "tensor_hosvd3_reconstruct")]
pub fn hosvd3_reconstruct<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyArray3<f64>>,
    u0: &Bound<'py, PyArray2<f64>>,
    u1: &Bound<'py, PyArray2<f64>>,
    u2: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    utils::require_contiguous(core)?;
    utils::require_contiguous(u0)?;
    utils::require_contiguous(u1)?;
    utils::require_contiguous(u2)?;
    let result = nabled_linalg::tensor::hosvd3_reconstruct(&nabled_linalg::tensor::Hosvd3Result {
        core: core.readonly().as_array().to_owned(),
        u0:   u0.readonly().as_array().to_owned(),
        u1:   u1.readonly().as_array().to_owned(),
        u2:   u2.readonly().as_array().to_owned(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

/// N-D HOSVD decomposition. Returns `(core, factors)`.
#[pyfunction(name = "tensor_hosvd_nd")]
pub fn hosvd_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    ranks: Vec<usize>,
) -> PyResult<PyHosvdNdResult> {
    utils::require_contiguous(tensor)?;
    let result = nabled_linalg::tensor::hosvd_nd_view(&tensor.readonly().as_array(), &ranks)
        .map_err(to_py_err)?;
    Ok(py_hosvd_nd_result(py, result))
}

/// N-D HOOI Tucker refinement. Returns `(core, factors)`.
#[pyfunction(name = "tensor_hooi_nd")]
pub fn hooi_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    ranks: Vec<usize>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyHosvdNdResult> {
    utils::require_contiguous(tensor)?;
    let config = hooi_config(max_iterations, tolerance);
    let result =
        nabled_linalg::tensor::hooi_nd_view(&tensor.readonly().as_array(), &ranks, &config)
            .map_err(to_py_err)?;
    Ok(py_hosvd_nd_result(py, result))
}

/// Reconstruct an N-D tensor from Tucker/HOSVD factors.
#[pyfunction(name = "tensor_hosvd_nd_reconstruct")]
pub fn hosvd_nd_reconstruct<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyArrayDyn<f64>>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let result =
        nabled_linalg::tensor::hosvd_nd_reconstruct(&hosvd_nd_result_from_arrays(core, factors)?)
            .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Project a tensor into a Tucker core using per-mode factors.
#[pyfunction(name = "tensor_tucker_project")]
pub fn tucker_project<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(tensor)?;
    let factors = extract_array2_sequence(factors)?;
    let result =
        nabled_linalg::tensor::tucker_project_view(&tensor.readonly().as_array(), &factors)
            .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Expand a Tucker core using per-mode factors.
#[pyfunction(name = "tensor_tucker_expand")]
pub fn tucker_expand<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyArrayDyn<f64>>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    utils::require_contiguous(core)?;
    let factors = extract_array2_sequence(factors)?;
    let result = nabled_linalg::tensor::tucker_expand_view(&core.readonly().as_array(), &factors)
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Rank-`R` CP-ALS decomposition over rank-3 tensors. Returns `(weights, factor_0, factor_1,
/// factor_2)`.
#[pyfunction(name = "tensor_cp_als3")]
pub fn cp_als3<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyCpAls3Result> {
    utils::require_contiguous(cube)?;
    let config = cp_als_config(max_iterations, tolerance);
    let result = nabled_linalg::tensor::cp_als3_view(&cube.readonly().as_array(), rank, &config)
        .map_err(to_py_err)?;
    Ok(py_cp_als3_result(py, result))
}

/// Rank-`R` CP-ALS decomposition with convergence and reconstruction report.
#[pyfunction(name = "tensor_cp_als3_with_report")]
pub fn cp_als3_with_report<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyArray3<f64>>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyCpAls3Result, PyCpReport)> {
    utils::require_contiguous(cube)?;
    let config = cp_als_config(max_iterations, tolerance);
    let (result, report) =
        nabled_linalg::tensor::cp_als3_view_with_report(&cube.readonly().as_array(), rank, &config)
            .map_err(to_py_err)?;
    Ok((py_cp_als3_result(py, result), py_cp_report(report)))
}

/// CP-ALS reconstruction diagnostics for rank-3 tensors.
#[pyfunction(name = "tensor_cp_als3_diagnostics")]
pub fn cp_als3_diagnostics(
    cube: &Bound<'_, PyArray3<f64>>,
    weights: &Bound<'_, PyArray1<f64>>,
    factor_0: &Bound<'_, PyArray2<f64>>,
    factor_1: &Bound<'_, PyArray2<f64>>,
    factor_2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<PyCpMetrics> {
    utils::require_contiguous(cube)?;
    let result = cp_als3_result_from_arrays(weights, factor_0, factor_1, factor_2)?;
    let metrics =
        nabled_linalg::tensor::cp_als3_diagnostics_view(&cube.readonly().as_array(), &result)
            .map_err(to_py_err)?;
    Ok(py_cp_metrics(metrics))
}

/// Reconstruct a rank-3 tensor from CP-ALS factors.
#[pyfunction(name = "tensor_cp_als3_reconstruct")]
pub fn cp_als3_reconstruct<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyArray1<f64>>,
    factor_0: &Bound<'py, PyArray2<f64>>,
    factor_1: &Bound<'py, PyArray2<f64>>,
    factor_2: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    let result = nabled_linalg::tensor::cp_als3_reconstruct(&cp_als3_result_from_arrays(
        weights, factor_0, factor_1, factor_2,
    )?)
    .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}

/// Rank-`R` CP-ALS decomposition over N-D tensors. Returns `(weights, factors)`.
#[pyfunction(name = "tensor_cp_als_nd")]
pub fn cp_als_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyCpAlsNdResult> {
    utils::require_contiguous(tensor)?;
    let config = cp_als_config(max_iterations, tolerance);
    let result =
        nabled_linalg::tensor::cp_als_nd_view(&tensor.readonly().as_array(), rank, &config)
            .map_err(to_py_err)?;
    Ok(py_cp_als_nd_result(py, result))
}

/// Rank-`R` N-D CP-ALS decomposition with convergence and reconstruction report.
#[pyfunction(name = "tensor_cp_als_nd_with_report")]
pub fn cp_als_nd_with_report<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyCpAlsNdResult, PyCpReport)> {
    utils::require_contiguous(tensor)?;
    let config = cp_als_config(max_iterations, tolerance);
    let (result, report) = nabled_linalg::tensor::cp_als_nd_view_with_report(
        &tensor.readonly().as_array(),
        rank,
        &config,
    )
    .map_err(to_py_err)?;
    Ok((py_cp_als_nd_result(py, result), py_cp_report(report)))
}

/// CP-ALS reconstruction diagnostics for N-D tensors.
#[pyfunction(name = "tensor_cp_als_nd_diagnostics")]
pub fn cp_als_nd_diagnostics(
    tensor: &Bound<'_, PyArrayDyn<f64>>,
    weights: &Bound<'_, PyArray1<f64>>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<PyCpMetrics> {
    utils::require_contiguous(tensor)?;
    let result = cp_als_nd_result_from_arrays(weights, factors)?;
    let metrics =
        nabled_linalg::tensor::cp_als_nd_diagnostics_view(&tensor.readonly().as_array(), &result)
            .map_err(to_py_err)?;
    Ok(py_cp_metrics(metrics))
}

/// Reconstruct an N-D tensor from CP-ALS factors.
#[pyfunction(name = "tensor_cp_als_nd_reconstruct")]
pub fn cp_als_nd_reconstruct<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyArray1<f64>>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let result = nabled_linalg::tensor::cp_als_nd_reconstruct(&cp_als_nd_result_from_arrays(
        weights, factors,
    )?)
    .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

/// Tensor-Train decomposition via TT-SVD. Returns a list of TT cores.
#[pyfunction(name = "tensor_tt_svd")]
pub fn tt_svd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyArrayDyn<f64>>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyTensorTrainResult> {
    utils::require_contiguous(tensor)?;
    let config = tt_svd_config(max_rank, tolerance);
    let result = nabled_linalg::tensor::tt_svd_view(&tensor.readonly().as_array(), &config)
        .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Left-orthogonalize TT cores while preserving the represented tensor.
#[pyfunction(name = "tensor_tt_orthogonalize_left")]
pub fn tt_orthogonalize_left<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    let result = nabled_linalg::tensor::tt_orthogonalize_left(&tt_result_from_cores(cores)?)
        .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Right-orthogonalize TT cores while preserving the represented tensor.
#[pyfunction(name = "tensor_tt_orthogonalize_right")]
pub fn tt_orthogonalize_right<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    let result = nabled_linalg::tensor::tt_orthogonalize_right(&tt_result_from_cores(cores)?)
        .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Round or compress TT cores with optional rank truncation.
#[pyfunction(name = "tensor_tt_round")]
pub fn tt_round<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyTensorTrainResult> {
    let config = tt_round_config(max_rank, tolerance);
    let result = nabled_linalg::tensor::tt_round(&tt_result_from_cores(cores)?, &config)
        .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Inner product between two Tensor-Train tensors.
#[pyfunction(name = "tensor_tt_inner")]
pub fn tt_inner(left_cores: &Bound<'_, PyAny>, right_cores: &Bound<'_, PyAny>) -> PyResult<f64> {
    nabled_linalg::tensor::tt_inner(
        &tt_result_from_cores(left_cores)?,
        &tt_result_from_cores(right_cores)?,
    )
    .map_err(to_py_err)
}

/// Frobenius norm of a Tensor-Train tensor.
#[pyfunction(name = "tensor_tt_norm")]
pub fn tt_norm(cores: &Bound<'_, PyAny>) -> PyResult<f64> {
    nabled_linalg::tensor::tt_norm(&tt_result_from_cores(cores)?).map_err(to_py_err)
}

/// Add two Tensor-Train tensors with identical shapes.
#[pyfunction(name = "tensor_tt_add")]
pub fn tt_add<'py>(
    py: Python<'py>,
    left_cores: &Bound<'py, PyAny>,
    right_cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    let result = nabled_linalg::tensor::tt_add(
        &tt_result_from_cores(left_cores)?,
        &tt_result_from_cores(right_cores)?,
    )
    .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Hadamard product between two Tensor-Train tensors with identical shapes.
#[pyfunction(name = "tensor_tt_hadamard")]
pub fn tt_hadamard<'py>(
    py: Python<'py>,
    left_cores: &Bound<'py, PyAny>,
    right_cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    let result = nabled_linalg::tensor::tt_hadamard(
        &tt_result_from_cores(left_cores)?,
        &tt_result_from_cores(right_cores)?,
    )
    .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Hadamard product followed by TT rounding.
#[pyfunction(name = "tensor_tt_hadamard_round")]
pub fn tt_hadamard_round<'py>(
    py: Python<'py>,
    left_cores: &Bound<'py, PyAny>,
    right_cores: &Bound<'py, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyTensorTrainResult> {
    let config = tt_round_config(max_rank, tolerance);
    let result = nabled_linalg::tensor::tt_hadamard_round(
        &tt_result_from_cores(left_cores)?,
        &tt_result_from_cores(right_cores)?,
        &config,
    )
    .map_err(to_py_err)?;
    Ok(py_tt_result(py, result))
}

/// Reconstruct a dense tensor from Tensor-Train cores.
#[pyfunction(name = "tensor_tt_svd_reconstruct")]
pub fn tt_svd_reconstruct<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let result = nabled_linalg::tensor::tt_svd_reconstruct(&tt_result_from_cores(cores)?)
        .map_err(to_py_err)?;
    Ok(PyArrayDyn::from_owned_array(py, standard_arrayd(result)).unbind())
}

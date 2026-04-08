//! Tensor bindings for Python.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, Array3, ArrayD};
use num_complex::Complex64;
use num_traits::{FromPrimitive, ToPrimitive};
use numpy::{
    Element, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadwriteArray2, PyReadwriteArray3,
    PyReadwriteArrayDyn,
};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

pub(crate) type PyHosvd3Result = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);
pub(crate) type PyCpAls3Result = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);
pub(crate) type PyCpAlsNdResult = (Py<PyAny>, Vec<Py<PyAny>>);
pub(crate) type PyCpMetrics = (f64, f64, f64, f64);
pub(crate) type PyCpConvergence = (usize, bool, f64);
pub(crate) type PyCpReport = (PyCpConvergence, PyCpMetrics);
pub(crate) type PyHosvdNdResult = (Py<PyAny>, Vec<Py<PyAny>>);
pub(crate) type PyTensorTrainResult = Vec<Py<PyAny>>;

fn output_array2<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArray2<'py, T>> {
    array
        .cast::<PyArray2<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label} and rank 2",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

fn output_array3<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArray3<'py, T>> {
    array
        .cast::<PyArray3<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label} and rank 3",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

fn output_arrayd<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArrayDyn<'py, T>> {
    array
        .cast::<PyArrayDyn<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label}",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

fn standard_array2<T: Clone>(array: Array2<T>) -> Array2<T> {
    array.as_standard_layout().to_owned()
}

fn standard_array3<T: Clone>(array: Array3<T>) -> Array3<T> {
    array.as_standard_layout().to_owned()
}

fn standard_arrayd<T: Clone>(array: ArrayD<T>) -> ArrayD<T> {
    array.as_standard_layout().to_owned()
}

pub(crate) fn real_scalar_to_f64<T: ToPrimitive>(value: T, name: &str) -> PyResult<f64> {
    value
        .to_f64()
        .ok_or_else(|| PyOverflowError::new_err(format!("{name} could not be represented as f64")))
}

pub(crate) fn cp_als_config<T>(
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_linalg::tensor::CpAlsConfig<T>>
where
    T: NabledReal + FromPrimitive,
    nabled_linalg::tensor::CpAlsConfig<T>: Default,
{
    let mut config = nabled_linalg::tensor::CpAlsConfig::<T>::default();
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_real::<T>(tolerance, "tolerance")?;
    }
    Ok(config)
}

pub(crate) fn hooi_config<T>(
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_linalg::tensor::HooiConfig<T>>
where
    T: NabledReal + FromPrimitive,
    nabled_linalg::tensor::HooiConfig<T>: Default,
{
    let mut config = nabled_linalg::tensor::HooiConfig::<T>::default();
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_real::<T>(tolerance, "tolerance")?;
    }
    Ok(config)
}

pub(crate) fn tt_svd_config<T>(
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_linalg::tensor::TtSvdConfig<T>>
where
    T: NabledReal + FromPrimitive,
    nabled_linalg::tensor::TtSvdConfig<T>: Default,
{
    let mut config = nabled_linalg::tensor::TtSvdConfig::<T> { max_rank, ..Default::default() };
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_real::<T>(tolerance, "tolerance")?;
    }
    Ok(config)
}

pub(crate) fn tt_round_config<T>(
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_linalg::tensor::TtRoundConfig<T>>
where
    T: NabledReal + FromPrimitive,
    nabled_linalg::tensor::TtRoundConfig<T>: Default,
{
    let mut config = nabled_linalg::tensor::TtRoundConfig::<T> { max_rank, ..Default::default() };
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_real::<T>(tolerance, "tolerance")?;
    }
    Ok(config)
}

pub(crate) fn py_cp_metrics<T: NabledReal + ToPrimitive>(
    metrics: nabled_linalg::tensor::CpErrorMetrics<T>,
) -> PyResult<PyCpMetrics> {
    Ok((
        real_scalar_to_f64(metrics.signal_norm, "signal_norm")?,
        real_scalar_to_f64(metrics.residual_norm, "residual_norm")?,
        real_scalar_to_f64(metrics.relative_error, "relative_error")?,
        real_scalar_to_f64(metrics.fit, "fit")?,
    ))
}

pub(crate) fn py_cp_report<T: NabledReal + ToPrimitive>(
    report: nabled_linalg::tensor::CpAlsReport<T>,
) -> PyResult<PyCpReport> {
    Ok((
        (
            report.convergence.iterations_run,
            report.convergence.converged,
            real_scalar_to_f64(
                report.convergence.final_max_factor_change,
                "final_max_factor_change",
            )?,
        ),
        py_cp_metrics(report.metrics)?,
    ))
}

pub(crate) fn py_hosvd3_result<T: Element + Clone + NabledReal>(
    py: Python<'_>,
    result: nabled_linalg::tensor::Hosvd3Result<T>,
) -> PyHosvd3Result {
    (
        utils::pyarray3_from_owned(py, standard_array3(result.core)),
        utils::pyarray2_from_owned(py, standard_array2(result.u0)),
        utils::pyarray2_from_owned(py, standard_array2(result.u1)),
        utils::pyarray2_from_owned(py, standard_array2(result.u2)),
    )
}

pub(crate) fn py_cp_als3_result<T: Element + Clone + NabledReal>(
    py: Python<'_>,
    result: nabled_linalg::tensor::CpAls3Result<T>,
) -> PyCpAls3Result {
    (
        utils::pyarray1_from_owned(py, result.weights),
        utils::pyarray2_from_owned(py, standard_array2(result.factor_0)),
        utils::pyarray2_from_owned(py, standard_array2(result.factor_1)),
        utils::pyarray2_from_owned(py, standard_array2(result.factor_2)),
    )
}

pub(crate) fn py_cp_als_nd_result<T: Element + Clone + NabledReal>(
    py: Python<'_>,
    result: nabled_linalg::tensor::CpAlsNdResult<T>,
) -> PyCpAlsNdResult {
    (
        utils::pyarray1_from_owned(py, result.weights),
        result
            .factors
            .into_iter()
            .map(|factor| utils::pyarray2_from_owned(py, standard_array2(factor)))
            .collect(),
    )
}

pub(crate) fn py_hosvd_nd_result<T: Element + Clone + NabledReal>(
    py: Python<'_>,
    result: nabled_linalg::tensor::HosvdNdResult<T>,
) -> PyHosvdNdResult {
    (
        utils::pyarrayd_from_owned(py, standard_arrayd(result.core)),
        result
            .factors
            .into_iter()
            .map(|factor| utils::pyarray2_from_owned(py, standard_array2(factor)))
            .collect(),
    )
}

pub(crate) fn py_tt_result<T: Element + Clone + NabledReal>(
    py: Python<'_>,
    result: nabled_linalg::tensor::TensorTrainResult<T>,
) -> PyTensorTrainResult {
    result
        .cores
        .into_iter()
        .map(|core| utils::pyarray3_from_owned(py, standard_array3(core)))
        .collect()
}

pub(crate) fn extract_array2_sequence<T: Element + Clone>(
    arrays: &Bound<'_, PyAny>,
) -> PyResult<Vec<Array2<T>>> {
    let mut out = Vec::new();
    for item in arrays.try_iter()? {
        let item = item?;
        let array = item.cast::<PyArray2<T>>().map_err(|_| {
            PyValueError::new_err(
                "expected a non-empty sequence of 2D NumPy arrays with matching float32/float64 \
                 dtype",
            )
        })?;
        utils::require_contiguous(array)?;
        out.push(array.readonly().as_array().to_owned());
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(
            "expected a non-empty sequence of 2D NumPy arrays with matching float32/float64 dtype",
        ));
    }
    Ok(out)
}

pub(crate) enum RealTensorTrainResult {
    F32(nabled_linalg::tensor::TensorTrainResult<f32>),
    F64(nabled_linalg::tensor::TensorTrainResult<f64>),
}

fn tensor_train_result_from_cores<T: NabledReal>(
    cores: Vec<Array3<T>>,
) -> nabled_linalg::tensor::TensorTrainResult<T> {
    let shape = cores.iter().map(|core| core.dim().1).collect();
    nabled_linalg::tensor::TensorTrainResult { cores, shape }
}

pub(crate) fn real_tt_result_from_cores(
    cores: &Bound<'_, PyAny>,
) -> PyResult<RealTensorTrainResult> {
    let mut iter = cores.try_iter()?;
    let Some(first) = iter.next() else {
        return Err(PyValueError::new_err(
            "expected a non-empty sequence of 3D NumPy arrays with matching float32/float64 dtype",
        ));
    };
    let first = first?;
    if let Ok(array) = first.cast::<PyArray3<f32>>() {
        utils::require_contiguous(array)?;
        let mut owned = vec![array.readonly().as_array().to_owned()];
        for item in iter {
            let item = item?;
            let array = item.cast::<PyArray3<f32>>().map_err(|_| {
                PyValueError::new_err(
                    "expected a non-empty sequence of 3D NumPy arrays with matching \
                     float32/float64 dtype",
                )
            })?;
            utils::require_contiguous(array)?;
            owned.push(array.readonly().as_array().to_owned());
        }
        return Ok(RealTensorTrainResult::F32(tensor_train_result_from_cores(owned)));
    }
    if let Ok(array) = first.cast::<PyArray3<f64>>() {
        utils::require_contiguous(array)?;
        let mut owned = vec![array.readonly().as_array().to_owned()];
        for item in iter {
            let item = item?;
            let array = item.cast::<PyArray3<f64>>().map_err(|_| {
                PyValueError::new_err(
                    "expected a non-empty sequence of 3D NumPy arrays with matching \
                     float32/float64 dtype",
                )
            })?;
            utils::require_contiguous(array)?;
            owned.push(array.readonly().as_array().to_owned());
        }
        return Ok(RealTensorTrainResult::F64(tensor_train_result_from_cores(owned)));
    }
    Err(PyValueError::new_err(
        "expected a non-empty sequence of 3D NumPy arrays with matching float32/float64 dtype",
    ))
}

fn cp_als3_result_from_arrays<T: Clone + NabledReal>(
    weights: Array1<T>,
    factor_0: Array2<T>,
    factor_1: Array2<T>,
    factor_2: Array2<T>,
) -> nabled_linalg::tensor::CpAls3Result<T> {
    nabled_linalg::tensor::CpAls3Result { weights, factor_0, factor_1, factor_2 }
}

fn cp_als_nd_result_from_arrays<T: Clone + NabledReal>(
    weights: Array1<T>,
    factors: Vec<Array2<T>>,
) -> nabled_linalg::tensor::CpAlsNdResult<T> {
    let shape = factors.iter().map(Array2::nrows).collect();
    nabled_linalg::tensor::CpAlsNdResult { weights, factors, shape }
}

fn hosvd_nd_result_from_arrays<T: Clone + NabledReal>(
    core: ArrayD<T>,
    factors: Vec<Array2<T>>,
) -> nabled_linalg::tensor::HosvdNdResult<T> {
    nabled_linalg::tensor::HosvdNdResult { core, factors }
}

/// Batched matrix-vector product: cube `(B, m, n)` @ vectors `(B, n)` -> `(B, m)`.
#[pyfunction(name = "tensor_cube_matvec")]
pub fn cube_matvec<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyAny>,
    vectors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array3(cube, "cube")?, utils::real_array2(vectors, "vectors")?) {
        (utils::RealReadonlyArray3::F32(cube_arr), utils::RealReadonlyArray2::F32(vectors_arr)) => {
            let result = nabled_linalg::tensor::cube_matvec_view(
                &cube_arr.as_array(),
                &vectors_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray3::F64(cube_arr), utils::RealReadonlyArray2::F64(vectors_arr)) => {
            let result = nabled_linalg::tensor::cube_matvec_view(
                &cube_arr.as_array(),
                &vectors_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["cube", "vectors"])),
    }
}

/// Batched matrix-vector product into a caller-provided output array.
#[pyfunction(name = "tensor_cube_matvec_into")]
pub fn cube_matvec_into(
    cube: &Bound<'_, PyAny>,
    vectors: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array3(cube, "cube")?, utils::real_array2(vectors, "vectors")?) {
        (utils::RealReadonlyArray3::F32(cube_arr), utils::RealReadonlyArray2::F32(vectors_arr)) => {
            let mut output_arr = output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::tensor::cube_matvec_view_into(
                &cube_arr.as_array(),
                &vectors_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray3::F64(cube_arr), utils::RealReadonlyArray2::F64(vectors_arr)) => {
            let mut output_arr = output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::tensor::cube_matvec_view_into(
                &cube_arr.as_array(),
                &vectors_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["cube", "vectors", "output"])),
    }
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

/// Complex batched matrix-vector product into a caller-provided output array.
#[pyfunction(name = "tensor_cube_matvec_complex_into")]
pub fn cube_matvec_complex_into(
    cube: &Bound<'_, PyArray3<Complex64>>,
    vectors: &Bound<'_, PyArray2<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(cube)?;
    utils::require_contiguous(vectors)?;
    let mut output_arr = output_array2::<Complex64>(output, "output", "complex128")?;
    nabled_linalg::tensor::cube_matvec_complex_view_into(
        &cube.readonly().as_array(),
        &vectors.readonly().as_array(),
        output_arr.as_array_mut(),
    )
    .map_err(to_py_err)
}

/// Batched matrix-matrix product over the last two cube axes.
#[pyfunction(name = "tensor_cube_matmat")]
pub fn cube_matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array3(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let result = nabled_linalg::tensor::cube_matmat_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let result = nabled_linalg::tensor::cube_matmat_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched matrix-matrix product into a caller-provided output array.
#[pyfunction(name = "tensor_cube_matmat_into")]
pub fn cube_matmat_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array3(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let mut output_arr = output_array3::<f32>(output, "output", "float32")?;
            nabled_linalg::tensor::cube_matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let mut output_arr = output_array3::<f64>(output, "output", "float64")?;
            nabled_linalg::tensor::cube_matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
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

/// Complex batched matrix-matrix product into a caller-provided output array.
#[pyfunction(name = "tensor_cube_matmat_complex_into")]
pub fn cube_matmat_complex_into(
    left: &Bound<'_, PyArray3<Complex64>>,
    right: &Bound<'_, PyArray3<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let mut output_arr = output_array3::<Complex64>(output, "output", "complex128")?;
    nabled_linalg::tensor::cube_matmat_complex_view_into(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        output_arr.as_array_mut(),
    )
    .map_err(to_py_err)
}

/// Sum over the last axis of a tensor.
#[pyfunction(name = "tensor_sum_last_axis")]
pub fn sum_last_axis<'py>(py: Python<'py>, tensor: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let result = nabled_linalg::tensor::sum_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let result = nabled_linalg::tensor::sum_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Sum over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_sum_last_axis_into")]
pub fn sum_last_axis_into(tensor: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::sum_last_axis_view_into(&tensor_arr.as_array(), &mut output_view)
                .map_err(to_py_err)
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::sum_last_axis_view_into(&tensor_arr.as_array(), &mut output_view)
                .map_err(to_py_err)
        }
    }
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

/// Sum over the last axis of a complex tensor into a caller-provided output array.
#[pyfunction(name = "tensor_sum_last_axis_complex_into")]
pub fn sum_last_axis_complex_into(
    tensor: &Bound<'_, PyArrayDyn<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(tensor)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::sum_last_axis_complex_view_into(
        &tensor.readonly().as_array(),
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// L2 norm over the last axis.
#[pyfunction(name = "tensor_l2_norm_last_axis")]
pub fn l2_norm_last_axis<'py>(py: Python<'py>, tensor: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let result = nabled_linalg::tensor::l2_norm_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let result = nabled_linalg::tensor::l2_norm_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// L2 norm over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_l2_norm_last_axis_into")]
pub fn l2_norm_last_axis_into(
    tensor: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::l2_norm_last_axis_view_into(
                &tensor_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::l2_norm_last_axis_view_into(
                &tensor_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
    }
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

/// L2 norm over the last axis of a complex tensor into a caller-provided output array.
#[pyfunction(name = "tensor_l2_norm_last_axis_complex_into")]
pub fn l2_norm_last_axis_complex_into(
    tensor: &Bound<'_, PyArrayDyn<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(tensor)?;
    let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::l2_norm_last_axis_complex_view_into(
        &tensor.readonly().as_array(),
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Normalize over the last axis.
#[pyfunction(name = "tensor_normalize_last_axis")]
pub fn normalize_last_axis<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let result = nabled_linalg::tensor::normalize_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let result = nabled_linalg::tensor::normalize_last_axis_view(&tensor_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Normalize over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_normalize_last_axis_into")]
pub fn normalize_last_axis_into(
    tensor: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::normalize_last_axis_view_into(
                &tensor_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::normalize_last_axis_view_into(
                &tensor_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
    }
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

/// Normalize a complex tensor over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_normalize_last_axis_complex_into")]
pub fn normalize_last_axis_complex_into(
    tensor: &Bound<'_, PyArrayDyn<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(tensor)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::normalize_last_axis_complex_view_into(
        &tensor.readonly().as_array(),
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Batched dot product over the last axis.
#[pyfunction(name = "tensor_batched_dot_last_axis")]
pub fn batched_dot_last_axis<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let result = nabled_linalg::tensor::batched_dot_last_axis_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let result = nabled_linalg::tensor::batched_dot_last_axis_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched dot product over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_batched_dot_last_axis_into")]
pub fn batched_dot_last_axis_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::batched_dot_last_axis_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::batched_dot_last_axis_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
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

/// Complex batched dot product over the last axis into a caller-provided output array.
#[pyfunction(name = "tensor_batched_dot_last_axis_complex_into")]
pub fn batched_dot_last_axis_complex_into(
    left: &Bound<'_, PyArrayDyn<Complex64>>,
    right: &Bound<'_, PyArrayDyn<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::batched_dot_last_axis_complex_view_into(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Permute tensor axes.
#[pyfunction(name = "tensor_permute_axes")]
pub fn permute_axes<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    permutation: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let result =
                nabled_linalg::tensor::permute_axes_view(&tensor_arr.as_array(), &permutation)
                    .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let result =
                nabled_linalg::tensor::permute_axes_view(&tensor_arr.as_array(), &permutation)
                    .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Permute tensor axes into a caller-provided output array.
#[pyfunction(name = "tensor_permute_axes_into")]
pub fn permute_axes_into(
    tensor: &Bound<'_, PyAny>,
    permutation: Vec<usize>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::permute_axes_view_into(
                &tensor_arr.as_array(),
                &permutation,
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::permute_axes_view_into(
                &tensor_arr.as_array(),
                &permutation,
                &mut output_view,
            )
            .map_err(to_py_err)
        }
    }
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

/// Permute complex tensor axes into a caller-provided output array.
#[pyfunction(name = "tensor_permute_axes_complex_into")]
pub fn permute_axes_complex_into(
    tensor: &Bound<'_, PyArrayDyn<Complex64>>,
    permutation: Vec<usize>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(tensor)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::permute_axes_complex_view_into(
        &tensor.readonly().as_array(),
        &permutation,
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Contract explicit axes between two tensors.
#[pyfunction(name = "tensor_contract_axes")]
pub fn contract_axes<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let result = nabled_linalg::tensor::contract_axes_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &left_axes,
                &right_axes,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let result = nabled_linalg::tensor::contract_axes_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &left_axes,
                &right_axes,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Contract explicit axes between two tensors into a caller-provided output array.
#[pyfunction(name = "tensor_contract_axes_into")]
pub fn contract_axes_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::contract_axes_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &left_axes,
                &right_axes,
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::contract_axes_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &left_axes,
                &right_axes,
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
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

/// Contract explicit axes between two complex tensors into a caller-provided output array.
#[pyfunction(name = "tensor_contract_axes_complex_into")]
pub fn contract_axes_complex_into(
    left: &Bound<'_, PyArrayDyn<Complex64>>,
    right: &Bound<'_, PyArrayDyn<Complex64>>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::contract_axes_complex_view_into(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        &left_axes,
        &right_axes,
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Batched matrix multiply over the last two axes.
#[pyfunction(name = "tensor_batched_matmul_last_two")]
pub fn batched_matmul_last_two<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let result = nabled_linalg::tensor::batched_matmul_last_two_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let result = nabled_linalg::tensor::batched_matmul_last_two_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched matrix multiply over the last two axes into a caller-provided output array.
#[pyfunction(name = "tensor_batched_matmul_last_two_into")]
pub fn batched_matmul_last_two_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f32>(output, "output", "float32")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::batched_matmul_last_two_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let mut output_arr = output_arrayd::<f64>(output, "output", "float64")?;
            let mut output_view = output_arr.as_array_mut();
            nabled_linalg::tensor::batched_matmul_last_two_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
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

/// Complex batched matrix multiply over the last two axes into a caller-provided output array.
#[pyfunction(name = "tensor_batched_matmul_last_two_complex_into")]
pub fn batched_matmul_last_two_complex_into(
    left: &Bound<'_, PyArrayDyn<Complex64>>,
    right: &Bound<'_, PyArrayDyn<Complex64>>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let mut output_arr = output_arrayd::<Complex64>(output, "output", "complex128")?;
    let mut output_view = output_arr.as_array_mut();
    nabled_linalg::tensor::batched_matmul_last_two_complex_view_into(
        &left.readonly().as_array(),
        &right.readonly().as_array(),
        &mut output_view,
    )
    .map_err(to_py_err)
}

/// Binary Einstein summation over real tensors.
#[pyfunction(name = "tensor_einsum")]
pub fn einsum<'py>(
    py: Python<'py>,
    equation: String,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_arrayd(left, "left")?, utils::real_arrayd(right, "right")?) {
        (
            utils::RealReadonlyArrayDyn::F32(left_arr),
            utils::RealReadonlyArrayDyn::F32(right_arr),
        ) => {
            let result = nabled_linalg::tensor::einsum_view(
                &equation,
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        (
            utils::RealReadonlyArrayDyn::F64(left_arr),
            utils::RealReadonlyArrayDyn::F64(right_arr),
        ) => {
            let result = nabled_linalg::tensor::einsum_view(
                &equation,
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
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
    cube: &Bound<'py, PyAny>,
    rank0: usize,
    rank1: usize,
    rank2: usize,
) -> PyResult<PyHosvd3Result> {
    match utils::real_array3(cube, "cube")? {
        utils::RealReadonlyArray3::F32(cube_arr) => {
            let result = nabled_linalg::tensor::hosvd3(
                &cube_arr.as_array().to_owned(),
                (rank0, rank1, rank2),
            )
            .map_err(to_py_err)?;
            Ok(py_hosvd3_result(py, result))
        }
        utils::RealReadonlyArray3::F64(cube_arr) => {
            let result = nabled_linalg::tensor::hosvd3(
                &cube_arr.as_array().to_owned(),
                (rank0, rank1, rank2),
            )
            .map_err(to_py_err)?;
            Ok(py_hosvd3_result(py, result))
        }
    }
}

/// Reconstruct a cube from an HOSVD3 decomposition.
#[pyfunction(name = "tensor_hosvd3_reconstruct")]
pub fn hosvd3_reconstruct<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyAny>,
    u0: &Bound<'py, PyAny>,
    u1: &Bound<'py, PyAny>,
    u2: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array3(core, "core")?,
        utils::real_array2(u0, "u0")?,
        utils::real_array2(u1, "u1")?,
        utils::real_array2(u2, "u2")?,
    ) {
        (
            utils::RealReadonlyArray3::F32(core_arr),
            utils::RealReadonlyArray2::F32(u0_arr),
            utils::RealReadonlyArray2::F32(u1_arr),
            utils::RealReadonlyArray2::F32(u2_arr),
        ) => {
            let result =
                nabled_linalg::tensor::hosvd3_reconstruct(&nabled_linalg::tensor::Hosvd3Result {
                    core: core_arr.as_array().to_owned(),
                    u0:   u0_arr.as_array().to_owned(),
                    u1:   u1_arr.as_array().to_owned(),
                    u2:   u2_arr.as_array().to_owned(),
                })
                .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray3::F64(core_arr),
            utils::RealReadonlyArray2::F64(u0_arr),
            utils::RealReadonlyArray2::F64(u1_arr),
            utils::RealReadonlyArray2::F64(u2_arr),
        ) => {
            let result =
                nabled_linalg::tensor::hosvd3_reconstruct(&nabled_linalg::tensor::Hosvd3Result {
                    core: core_arr.as_array().to_owned(),
                    u0:   u0_arr.as_array().to_owned(),
                    u1:   u1_arr.as_array().to_owned(),
                    u2:   u2_arr.as_array().to_owned(),
                })
                .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["core", "u0", "u1", "u2"])),
    }
}

/// N-D HOSVD decomposition. Returns `(core, factors)`.
#[pyfunction(name = "tensor_hosvd_nd")]
pub fn hosvd_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    ranks: Vec<usize>,
) -> PyResult<PyHosvdNdResult> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let result = nabled_linalg::tensor::hosvd_nd_view(&tensor_arr.as_array(), &ranks)
                .map_err(to_py_err)?;
            Ok(py_hosvd_nd_result(py, result))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let result = nabled_linalg::tensor::hosvd_nd_view(&tensor_arr.as_array(), &ranks)
                .map_err(to_py_err)?;
            Ok(py_hosvd_nd_result(py, result))
        }
    }
}

/// N-D HOOI Tucker refinement. Returns `(core, factors)`.
#[pyfunction(name = "tensor_hooi_nd")]
pub fn hooi_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    ranks: Vec<usize>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyHosvdNdResult> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let config = hooi_config::<f32>(max_iterations, tolerance)?;
            let result =
                nabled_linalg::tensor::hooi_nd_view(&tensor_arr.as_array(), &ranks, &config)
                    .map_err(to_py_err)?;
            Ok(py_hosvd_nd_result(py, result))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let config = hooi_config::<f64>(max_iterations, tolerance)?;
            let result =
                nabled_linalg::tensor::hooi_nd_view(&tensor_arr.as_array(), &ranks, &config)
                    .map_err(to_py_err)?;
            Ok(py_hosvd_nd_result(py, result))
        }
    }
}

/// Reconstruct an N-D tensor from Tucker/HOSVD factors.
#[pyfunction(name = "tensor_hosvd_nd_reconstruct")]
pub fn hosvd_nd_reconstruct<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(core, "core")? {
        utils::RealReadonlyArrayDyn::F32(core_arr) => {
            let result = nabled_linalg::tensor::hosvd_nd_reconstruct(&hosvd_nd_result_from_arrays(
                core_arr.as_array().to_owned(),
                extract_array2_sequence::<f32>(factors)?,
            ))
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(core_arr) => {
            let result = nabled_linalg::tensor::hosvd_nd_reconstruct(&hosvd_nd_result_from_arrays(
                core_arr.as_array().to_owned(),
                extract_array2_sequence::<f64>(factors)?,
            ))
            .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Project a tensor into a Tucker core using per-mode factors.
#[pyfunction(name = "tensor_tucker_project")]
pub fn tucker_project<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let factors = extract_array2_sequence::<f32>(factors)?;
            let result =
                nabled_linalg::tensor::tucker_project_view(&tensor_arr.as_array(), &factors)
                    .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let factors = extract_array2_sequence::<f64>(factors)?;
            let result =
                nabled_linalg::tensor::tucker_project_view(&tensor_arr.as_array(), &factors)
                    .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Expand a Tucker core using per-mode factors.
#[pyfunction(name = "tensor_tucker_expand")]
pub fn tucker_expand<'py>(
    py: Python<'py>,
    core: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_arrayd(core, "core")? {
        utils::RealReadonlyArrayDyn::F32(core_arr) => {
            let factors = extract_array2_sequence::<f32>(factors)?;
            let result = nabled_linalg::tensor::tucker_expand_view(&core_arr.as_array(), &factors)
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArrayDyn::F64(core_arr) => {
            let factors = extract_array2_sequence::<f64>(factors)?;
            let result = nabled_linalg::tensor::tucker_expand_view(&core_arr.as_array(), &factors)
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Rank-`R` CP-ALS decomposition over rank-3 tensors. Returns `(weights, factor_0, factor_1,
/// factor_2)`.
#[pyfunction(name = "tensor_cp_als3")]
pub fn cp_als3<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyAny>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyCpAls3Result> {
    match utils::real_array3(cube, "cube")? {
        utils::RealReadonlyArray3::F32(cube_arr) => {
            let config = cp_als_config::<f32>(max_iterations, tolerance)?;
            let result = nabled_linalg::tensor::cp_als3_view(&cube_arr.as_array(), rank, &config)
                .map_err(to_py_err)?;
            Ok(py_cp_als3_result(py, result))
        }
        utils::RealReadonlyArray3::F64(cube_arr) => {
            let config = cp_als_config::<f64>(max_iterations, tolerance)?;
            let result = nabled_linalg::tensor::cp_als3_view(&cube_arr.as_array(), rank, &config)
                .map_err(to_py_err)?;
            Ok(py_cp_als3_result(py, result))
        }
    }
}

/// Rank-`R` CP-ALS decomposition with convergence and reconstruction report.
#[pyfunction(name = "tensor_cp_als3_with_report")]
pub fn cp_als3_with_report<'py>(
    py: Python<'py>,
    cube: &Bound<'py, PyAny>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyCpAls3Result, PyCpReport)> {
    match utils::real_array3(cube, "cube")? {
        utils::RealReadonlyArray3::F32(cube_arr) => {
            let config = cp_als_config::<f32>(max_iterations, tolerance)?;
            let (result, report) = nabled_linalg::tensor::cp_als3_view_with_report(
                &cube_arr.as_array(),
                rank,
                &config,
            )
            .map_err(to_py_err)?;
            Ok((py_cp_als3_result(py, result), py_cp_report(report)?))
        }
        utils::RealReadonlyArray3::F64(cube_arr) => {
            let config = cp_als_config::<f64>(max_iterations, tolerance)?;
            let (result, report) = nabled_linalg::tensor::cp_als3_view_with_report(
                &cube_arr.as_array(),
                rank,
                &config,
            )
            .map_err(to_py_err)?;
            Ok((py_cp_als3_result(py, result), py_cp_report(report)?))
        }
    }
}

/// CP-ALS reconstruction diagnostics for rank-3 tensors.
#[pyfunction(name = "tensor_cp_als3_diagnostics")]
pub fn cp_als3_diagnostics(
    cube: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    factor_0: &Bound<'_, PyAny>,
    factor_1: &Bound<'_, PyAny>,
    factor_2: &Bound<'_, PyAny>,
) -> PyResult<PyCpMetrics> {
    match (
        utils::real_array3(cube, "cube")?,
        utils::real_array1(weights, "weights")?,
        utils::real_array2(factor_0, "factor_0")?,
        utils::real_array2(factor_1, "factor_1")?,
        utils::real_array2(factor_2, "factor_2")?,
    ) {
        (
            utils::RealReadonlyArray3::F32(cube_arr),
            utils::RealReadonlyArray1::F32(weights_arr),
            utils::RealReadonlyArray2::F32(factor_0_arr),
            utils::RealReadonlyArray2::F32(factor_1_arr),
            utils::RealReadonlyArray2::F32(factor_2_arr),
        ) => {
            let result = cp_als3_result_from_arrays(
                weights_arr.as_array().to_owned(),
                factor_0_arr.as_array().to_owned(),
                factor_1_arr.as_array().to_owned(),
                factor_2_arr.as_array().to_owned(),
            );
            let metrics =
                nabled_linalg::tensor::cp_als3_diagnostics_view(&cube_arr.as_array(), &result)
                    .map_err(to_py_err)?;
            py_cp_metrics(metrics)
        }
        (
            utils::RealReadonlyArray3::F64(cube_arr),
            utils::RealReadonlyArray1::F64(weights_arr),
            utils::RealReadonlyArray2::F64(factor_0_arr),
            utils::RealReadonlyArray2::F64(factor_1_arr),
            utils::RealReadonlyArray2::F64(factor_2_arr),
        ) => {
            let result = cp_als3_result_from_arrays(
                weights_arr.as_array().to_owned(),
                factor_0_arr.as_array().to_owned(),
                factor_1_arr.as_array().to_owned(),
                factor_2_arr.as_array().to_owned(),
            );
            let metrics =
                nabled_linalg::tensor::cp_als3_diagnostics_view(&cube_arr.as_array(), &result)
                    .map_err(to_py_err)?;
            py_cp_metrics(metrics)
        }
        _ => Err(utils::matching_real_dtype_error(&[
            "cube", "weights", "factor_0", "factor_1", "factor_2",
        ])),
    }
}

/// Reconstruct a rank-3 tensor from CP-ALS factors.
#[pyfunction(name = "tensor_cp_als3_reconstruct")]
pub fn cp_als3_reconstruct<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyAny>,
    factor_0: &Bound<'py, PyAny>,
    factor_1: &Bound<'py, PyAny>,
    factor_2: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array1(weights, "weights")?,
        utils::real_array2(factor_0, "factor_0")?,
        utils::real_array2(factor_1, "factor_1")?,
        utils::real_array2(factor_2, "factor_2")?,
    ) {
        (
            utils::RealReadonlyArray1::F32(weights_arr),
            utils::RealReadonlyArray2::F32(factor_0_arr),
            utils::RealReadonlyArray2::F32(factor_1_arr),
            utils::RealReadonlyArray2::F32(factor_2_arr),
        ) => {
            let result = nabled_linalg::tensor::cp_als3_reconstruct(&cp_als3_result_from_arrays(
                weights_arr.as_array().to_owned(),
                factor_0_arr.as_array().to_owned(),
                factor_1_arr.as_array().to_owned(),
                factor_2_arr.as_array().to_owned(),
            ))
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray1::F64(weights_arr),
            utils::RealReadonlyArray2::F64(factor_0_arr),
            utils::RealReadonlyArray2::F64(factor_1_arr),
            utils::RealReadonlyArray2::F64(factor_2_arr),
        ) => {
            let result = nabled_linalg::tensor::cp_als3_reconstruct(&cp_als3_result_from_arrays(
                weights_arr.as_array().to_owned(),
                factor_0_arr.as_array().to_owned(),
                factor_1_arr.as_array().to_owned(),
                factor_2_arr.as_array().to_owned(),
            ))
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => {
            Err(utils::matching_real_dtype_error(&["weights", "factor_0", "factor_1", "factor_2"]))
        }
    }
}

/// Rank-`R` CP-ALS decomposition over N-D tensors. Returns `(weights, factors)`.
#[pyfunction(name = "tensor_cp_als_nd")]
pub fn cp_als_nd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyCpAlsNdResult> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let config = cp_als_config::<f32>(max_iterations, tolerance)?;
            let result =
                nabled_linalg::tensor::cp_als_nd_view(&tensor_arr.as_array(), rank, &config)
                    .map_err(to_py_err)?;
            Ok(py_cp_als_nd_result(py, result))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let config = cp_als_config::<f64>(max_iterations, tolerance)?;
            let result =
                nabled_linalg::tensor::cp_als_nd_view(&tensor_arr.as_array(), rank, &config)
                    .map_err(to_py_err)?;
            Ok(py_cp_als_nd_result(py, result))
        }
    }
}

/// Rank-`R` N-D CP-ALS decomposition with convergence and reconstruction report.
#[pyfunction(name = "tensor_cp_als_nd_with_report")]
pub fn cp_als_nd_with_report<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyCpAlsNdResult, PyCpReport)> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let config = cp_als_config::<f32>(max_iterations, tolerance)?;
            let (result, report) = nabled_linalg::tensor::cp_als_nd_view_with_report(
                &tensor_arr.as_array(),
                rank,
                &config,
            )
            .map_err(to_py_err)?;
            Ok((py_cp_als_nd_result(py, result), py_cp_report(report)?))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let config = cp_als_config::<f64>(max_iterations, tolerance)?;
            let (result, report) = nabled_linalg::tensor::cp_als_nd_view_with_report(
                &tensor_arr.as_array(),
                rank,
                &config,
            )
            .map_err(to_py_err)?;
            Ok((py_cp_als_nd_result(py, result), py_cp_report(report)?))
        }
    }
}

/// CP-ALS reconstruction diagnostics for N-D tensors.
#[pyfunction(name = "tensor_cp_als_nd_diagnostics")]
pub fn cp_als_nd_diagnostics(
    tensor: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<PyCpMetrics> {
    match (utils::real_arrayd(tensor, "tensor")?, utils::real_array1(weights, "weights")?) {
        (
            utils::RealReadonlyArrayDyn::F32(tensor_arr),
            utils::RealReadonlyArray1::F32(weights_arr),
        ) => {
            let result = cp_als_nd_result_from_arrays(
                weights_arr.as_array().to_owned(),
                extract_array2_sequence::<f32>(factors)?,
            );
            let metrics =
                nabled_linalg::tensor::cp_als_nd_diagnostics_view(&tensor_arr.as_array(), &result)
                    .map_err(to_py_err)?;
            py_cp_metrics(metrics)
        }
        (
            utils::RealReadonlyArrayDyn::F64(tensor_arr),
            utils::RealReadonlyArray1::F64(weights_arr),
        ) => {
            let result = cp_als_nd_result_from_arrays(
                weights_arr.as_array().to_owned(),
                extract_array2_sequence::<f64>(factors)?,
            );
            let metrics =
                nabled_linalg::tensor::cp_als_nd_diagnostics_view(&tensor_arr.as_array(), &result)
                    .map_err(to_py_err)?;
            py_cp_metrics(metrics)
        }
        _ => Err(utils::matching_real_dtype_error(&["tensor", "weights", "factors"])),
    }
}

/// Reconstruct an N-D tensor from CP-ALS factors.
#[pyfunction(name = "tensor_cp_als_nd_reconstruct")]
pub fn cp_als_nd_reconstruct<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyAny>,
    factors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(weights, "weights")? {
        utils::RealReadonlyArray1::F32(weights_arr) => {
            let result =
                nabled_linalg::tensor::cp_als_nd_reconstruct(&cp_als_nd_result_from_arrays(
                    weights_arr.as_array().to_owned(),
                    extract_array2_sequence::<f32>(factors)?,
                ))
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        utils::RealReadonlyArray1::F64(weights_arr) => {
            let result =
                nabled_linalg::tensor::cp_als_nd_reconstruct(&cp_als_nd_result_from_arrays(
                    weights_arr.as_array().to_owned(),
                    extract_array2_sequence::<f64>(factors)?,
                ))
                .map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

/// Tensor-Train decomposition via TT-SVD. Returns a list of TT cores.
#[pyfunction(name = "tensor_tt_svd")]
pub fn tt_svd<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyTensorTrainResult> {
    match utils::real_arrayd(tensor, "tensor")? {
        utils::RealReadonlyArrayDyn::F32(tensor_arr) => {
            let config = tt_svd_config::<f32>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_svd_view(&tensor_arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        utils::RealReadonlyArrayDyn::F64(tensor_arr) => {
            let config = tt_svd_config::<f64>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_svd_view(&tensor_arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
    }
}

/// Left-orthogonalize TT cores while preserving the represented tensor.
#[pyfunction(name = "tensor_tt_orthogonalize_left")]
pub fn tt_orthogonalize_left<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    match real_tt_result_from_cores(cores)? {
        RealTensorTrainResult::F32(result) => {
            let result =
                nabled_linalg::tensor::tt_orthogonalize_left(&result).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        RealTensorTrainResult::F64(result) => {
            let result =
                nabled_linalg::tensor::tt_orthogonalize_left(&result).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
    }
}

/// Right-orthogonalize TT cores while preserving the represented tensor.
#[pyfunction(name = "tensor_tt_orthogonalize_right")]
pub fn tt_orthogonalize_right<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    match real_tt_result_from_cores(cores)? {
        RealTensorTrainResult::F32(result) => {
            let result =
                nabled_linalg::tensor::tt_orthogonalize_right(&result).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        RealTensorTrainResult::F64(result) => {
            let result =
                nabled_linalg::tensor::tt_orthogonalize_right(&result).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
    }
}

/// Round or compress TT cores with optional rank truncation.
#[pyfunction(name = "tensor_tt_round")]
pub fn tt_round<'py>(
    py: Python<'py>,
    cores: &Bound<'py, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyTensorTrainResult> {
    match real_tt_result_from_cores(cores)? {
        RealTensorTrainResult::F32(result) => {
            let config = tt_round_config::<f32>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_round(&result, &config).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        RealTensorTrainResult::F64(result) => {
            let config = tt_round_config::<f64>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_round(&result, &config).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
    }
}

/// Inner product between two Tensor-Train tensors.
#[pyfunction(name = "tensor_tt_inner")]
pub fn tt_inner(left_cores: &Bound<'_, PyAny>, right_cores: &Bound<'_, PyAny>) -> PyResult<f64> {
    match (real_tt_result_from_cores(left_cores)?, real_tt_result_from_cores(right_cores)?) {
        (RealTensorTrainResult::F32(left), RealTensorTrainResult::F32(right)) => {
            Ok(f64::from(nabled_linalg::tensor::tt_inner(&left, &right).map_err(to_py_err)?))
        }
        (RealTensorTrainResult::F64(left), RealTensorTrainResult::F64(right)) => {
            nabled_linalg::tensor::tt_inner(&left, &right).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left_cores", "right_cores"])),
    }
}

/// Frobenius norm of a Tensor-Train tensor.
#[pyfunction(name = "tensor_tt_norm")]
pub fn tt_norm(cores: &Bound<'_, PyAny>) -> PyResult<f64> {
    match real_tt_result_from_cores(cores)? {
        RealTensorTrainResult::F32(result) => {
            Ok(f64::from(nabled_linalg::tensor::tt_norm(&result).map_err(to_py_err)?))
        }
        RealTensorTrainResult::F64(result) => {
            nabled_linalg::tensor::tt_norm(&result).map_err(to_py_err)
        }
    }
}

/// Add two Tensor-Train tensors with identical shapes.
#[pyfunction(name = "tensor_tt_add")]
pub fn tt_add<'py>(
    py: Python<'py>,
    left_cores: &Bound<'py, PyAny>,
    right_cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    match (real_tt_result_from_cores(left_cores)?, real_tt_result_from_cores(right_cores)?) {
        (RealTensorTrainResult::F32(left), RealTensorTrainResult::F32(right)) => {
            let result = nabled_linalg::tensor::tt_add(&left, &right).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        (RealTensorTrainResult::F64(left), RealTensorTrainResult::F64(right)) => {
            let result = nabled_linalg::tensor::tt_add(&left, &right).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left_cores", "right_cores"])),
    }
}

/// Hadamard product between two Tensor-Train tensors with identical shapes.
#[pyfunction(name = "tensor_tt_hadamard")]
pub fn tt_hadamard<'py>(
    py: Python<'py>,
    left_cores: &Bound<'py, PyAny>,
    right_cores: &Bound<'py, PyAny>,
) -> PyResult<PyTensorTrainResult> {
    match (real_tt_result_from_cores(left_cores)?, real_tt_result_from_cores(right_cores)?) {
        (RealTensorTrainResult::F32(left), RealTensorTrainResult::F32(right)) => {
            let result = nabled_linalg::tensor::tt_hadamard(&left, &right).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        (RealTensorTrainResult::F64(left), RealTensorTrainResult::F64(right)) => {
            let result = nabled_linalg::tensor::tt_hadamard(&left, &right).map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left_cores", "right_cores"])),
    }
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
    match (real_tt_result_from_cores(left_cores)?, real_tt_result_from_cores(right_cores)?) {
        (RealTensorTrainResult::F32(left), RealTensorTrainResult::F32(right)) => {
            let config = tt_round_config::<f32>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_hadamard_round(&left, &right, &config)
                .map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        (RealTensorTrainResult::F64(left), RealTensorTrainResult::F64(right)) => {
            let config = tt_round_config::<f64>(max_rank, tolerance)?;
            let result = nabled_linalg::tensor::tt_hadamard_round(&left, &right, &config)
                .map_err(to_py_err)?;
            Ok(py_tt_result(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left_cores", "right_cores"])),
    }
}

/// Reconstruct a dense tensor from Tensor-Train cores.
#[pyfunction(name = "tensor_tt_svd_reconstruct")]
pub fn tt_svd_reconstruct<'py>(py: Python<'py>, cores: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match real_tt_result_from_cores(cores)? {
        RealTensorTrainResult::F32(result) => {
            let result = nabled_linalg::tensor::tt_svd_reconstruct(&result).map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
        RealTensorTrainResult::F64(result) => {
            let result = nabled_linalg::tensor::tt_svd_reconstruct(&result).map_err(to_py_err)?;
            Ok(utils::pyarrayd_from_owned(py, standard_arrayd(result)))
        }
    }
}

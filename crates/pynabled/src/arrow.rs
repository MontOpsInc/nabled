//! PyArrow bridge for nabled/ndarrow workflows.
//!
//! When built with the `arrow` feature, this module provides zero-copy conversion
//! from PyArrow arrays to nabled's Arrow-facing APIs.
#![expect(
    unused_qualifications,
    reason = "explicit ndarray type annotations keep the Arrow callback bridge readable"
)]

use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray, ListArray, PrimitiveArray, StructArray, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::PyArrowType;
use arrow_schema::{DataType, Field};
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::linalg::tensor as py_tensor;
use crate::ml::callbacks::{
    call_scalar_function_arrow_complex, call_scalar_function_arrow_f32,
    call_scalar_function_arrow_f64, call_vector_function_arrow_complex,
    call_vector_function_arrow_complex_with_iteration, call_vector_function_arrow_f32,
    call_vector_function_arrow_f32_with_iteration, call_vector_function_arrow_f64,
    call_vector_function_arrow_f64_with_iteration,
};
use crate::sparse::csr;
use crate::utils;

const DEFAULT_MAX_TERMS: usize = 64;
const DEFAULT_TOLERANCE: f64 = 1.0e-14;

enum RealPrimitiveArray {
    F32(PrimitiveArray<Float32Type>),
    F64(PrimitiveArray<Float64Type>),
}

enum RealFixedSizeListArray {
    F32(FixedSizeListArray),
    F64(FixedSizeListArray),
}

fn array_data_to_fixed_size_list(data: ArrayData) -> PyResult<FixedSizeListArray> {
    make_array(data)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .cloned()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected FixedSizeList array"))
}

fn array_data_to_struct(data: ArrayData) -> PyResult<StructArray> {
    make_array(data)
        .as_any()
        .downcast_ref::<StructArray>()
        .cloned()
        .ok_or_else(|| PyTypeError::new_err("expected Struct array"))
}

fn array_data_to_real_primitive(data: ArrayData) -> PyResult<RealPrimitiveArray> {
    let arr = make_array(data);
    if let Some(prim) = arr.as_any().downcast_ref::<PrimitiveArray<Float32Type>>() {
        return Ok(RealPrimitiveArray::F32(prim.clone()));
    }
    if let Some(prim) = arr.as_any().downcast_ref::<PrimitiveArray<Float64Type>>() {
        return Ok(RealPrimitiveArray::F64(prim.clone()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected float32 or float64 Arrow primitive array",
    ))
}

fn array_data_to_real_fixed_size_list(data: ArrayData) -> PyResult<RealFixedSizeListArray> {
    let fsl = array_data_to_fixed_size_list(data)?;
    let values = fsl.values();
    if values.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().is_some() {
        Ok(RealFixedSizeListArray::F32(fsl))
    } else if values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().is_some() {
        Ok(RealFixedSizeListArray::F64(fsl))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected FixedSizeList array with float32 or float64 values",
        ))
    }
}

fn primitive_array_into_pyarrow<T>(array: PrimitiveArray<T>) -> PyArrowType<ArrayData>
where
    T: ArrowPrimitiveType,
{
    PyArrowType(array.into_data())
}

fn fixed_size_list_into_pyarrow(array: FixedSizeListArray) -> PyArrowType<ArrayData> {
    PyArrowType(array.into_data())
}

fn extension_array_into_pyarrow(
    field: Field,
    array: FixedSizeListArray,
) -> (PyArrowType<Field>, PyArrowType<ArrayData>) {
    (PyArrowType(field), fixed_size_list_into_pyarrow(array))
}

fn extension_result_into_pyarrow(
    result: (Field, FixedSizeListArray),
) -> (PyArrowType<Field>, PyArrowType<ArrayData>) {
    extension_array_into_pyarrow(result.0, result.1)
}

fn struct_array_into_pyarrow(array: StructArray) -> PyArrowType<ArrayData> {
    PyArrowType(array.into_data())
}

fn struct_extension_array_into_pyarrow(
    field: Field,
    array: StructArray,
) -> (PyArrowType<Field>, PyArrowType<ArrayData>) {
    (PyArrowType(field), struct_array_into_pyarrow(array))
}

fn struct_extension_result_into_pyarrow(
    result: (Field, StructArray),
) -> (PyArrowType<Field>, PyArrowType<ArrayData>) {
    struct_extension_array_into_pyarrow(result.0, result.1)
}

fn fixed_size_list_with_item_nullability(
    array: &FixedSizeListArray,
    nullable: bool,
) -> FixedSizeListArray {
    FixedSizeListArray::new(
        Arc::new(Field::new("item", array.value_type().clone(), nullable)),
        array.value_length(),
        Arc::clone(array.values()),
        array.nulls().cloned(),
    )
}

fn fixed_size_list_with_non_null_item(array: &FixedSizeListArray) -> FixedSizeListArray {
    fixed_size_list_with_item_nullability(array, false)
}

fn fixed_size_list_with_nullable_item(array: &FixedSizeListArray) -> FixedSizeListArray {
    fixed_size_list_with_item_nullability(array, true)
}

fn field_with_array_storage(field: &Field, array: &FixedSizeListArray) -> Field {
    Field::new(field.name(), array.data_type().clone(), false)
        .with_metadata(field.metadata().clone())
}

fn sparse_values_list<'a>(matrix: &'a StructArray, column_name: &str) -> PyResult<&'a ListArray> {
    matrix
        .column_by_name(column_name)
        .ok_or_else(|| {
            PyTypeError::new_err(format!("expected sparse storage column '{column_name}'"))
        })?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "expected sparse storage column '{column_name}' as ListArray"
            ))
        })
}

fn sparse_value_type(matrix: &StructArray, column_name: &str) -> PyResult<DataType> {
    Ok(sparse_values_list(matrix, column_name)?.value_type())
}

fn owned_csr_from_extension_f32(
    field: &Field,
    array: &StructArray,
) -> PyResult<nabled_linalg::sparse::CsrMatrix<f32>> {
    let view =
        nabled::ndarrow::csr_view_from_extension::<Float32Type>(field, array).map_err(to_py_err)?;
    let row_ptrs = view
        .row_ptrs
        .iter()
        .copied()
        .map(|value| {
            usize::try_from(value).map_err(|_| PyTypeError::new_err("csr row_ptr exceeds usize"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let col_indices = view
        .col_indices
        .iter()
        .copied()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| PyTypeError::new_err("csr column index exceeds usize"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    nabled_linalg::sparse::CsrMatrix::new(
        view.nrows,
        view.ncols,
        row_ptrs,
        col_indices,
        view.values.to_vec(),
    )
    .map_err(to_py_err)
}

fn owned_csr_from_extension_f64(
    field: &Field,
    array: &StructArray,
) -> PyResult<nabled_linalg::sparse::CsrMatrix<f64>> {
    let view =
        nabled::ndarrow::csr_view_from_extension::<Float64Type>(field, array).map_err(to_py_err)?;
    let row_ptrs = view
        .row_ptrs
        .iter()
        .copied()
        .map(|value| {
            usize::try_from(value).map_err(|_| PyTypeError::new_err("csr row_ptr exceeds usize"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let col_indices = view
        .col_indices
        .iter()
        .copied()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| PyTypeError::new_err("csr column index exceeds usize"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    nabled_linalg::sparse::CsrMatrix::new(
        view.nrows,
        view.ncols,
        row_ptrs,
        col_indices,
        view.values.to_vec(),
    )
    .map_err(to_py_err)
}

fn sparse_tolerance_f32(tolerance: Option<f64>) -> PyResult<f32> {
    tolerance.map_or(Ok(1.0e-6_f32), |value| utils::f64_to_f32(value, "tolerance"))
}

fn sparse_tolerance_f64(tolerance: Option<f64>) -> f64 { tolerance.unwrap_or(1.0e-10) }

fn sparse_max_iterations(max_iterations: Option<usize>) -> usize { max_iterations.unwrap_or(5000) }

fn ilut_drop_tolerance_f32(drop_tolerance: Option<f64>) -> PyResult<f32> {
    drop_tolerance.map_or(Ok(1.0e-8_f32), |value| utils::f64_to_f32(value, "drop_tolerance"))
}

fn ilut_drop_tolerance_f64(drop_tolerance: Option<f64>) -> f64 { drop_tolerance.unwrap_or(1.0e-8) }

fn ilut_max_fill(max_fill: Option<usize>) -> usize { max_fill.unwrap_or(16) }

fn iluk_level_of_fill(level_of_fill: Option<usize>) -> usize { level_of_fill.unwrap_or(1) }

fn qr_config_f32(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> PyResult<nabled_linalg::qr::QRConfig<f32>> {
    let mut config = nabled_linalg::qr::QRConfig::<f32>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = utils::f64_to_f32(rank_tolerance, "rank_tolerance")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    Ok(config)
}

fn qr_config_f64(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> nabled_linalg::qr::QRConfig<f64> {
    let mut config = nabled_linalg::qr::QRConfig::<f64>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = rank_tolerance;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    config
}

fn qr_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> (Py<PyAny>, Py<PyAny>, usize) {
    (
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        result.rank,
    )
}

fn qr_pivoted_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    let permutation = result.p.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("internal QR pivoting result missing permutation")
    })?;
    Ok((
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        utils::pyarray2_from_owned(py, permutation),
        result.rank,
    ))
}

fn nonsymmetric_config_f32(
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> PyResult<nabled_linalg::eigen::NonsymmetricEigenConfig<f32>> {
    let mut config =
        nabled_linalg::eigen::NonsymmetricEigenConfig::<f32> { balance, ..Default::default() };
    if let Some(balance_max_iterations) = balance_max_iterations {
        config.balance_max_iterations = balance_max_iterations;
    }
    if let Some(balance_tolerance) = balance_tolerance {
        config.balance_tolerance = utils::f64_to_f32(balance_tolerance, "balance_tolerance")?;
    }
    Ok(config)
}

fn nonsymmetric_config_f64(
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> nabled_linalg::eigen::NonsymmetricEigenConfig<f64> {
    let mut config =
        nabled_linalg::eigen::NonsymmetricEigenConfig::<f64> { balance, ..Default::default() };
    if let Some(balance_max_iterations) = balance_max_iterations {
        config.balance_max_iterations = balance_max_iterations;
    }
    if let Some(balance_tolerance) = balance_tolerance {
        config.balance_tolerance = balance_tolerance;
    }
    config
}

fn iterative_config_f32(
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<nabled_ml::iterative::IterativeConfig<f32>> {
    let default = nabled_ml::iterative::IterativeConfig::<f32>::default_f32();
    Ok(nabled_ml::iterative::IterativeConfig {
        tolerance:      tolerance
            .map_or(Ok(default.tolerance), |value| utils::f64_to_f32(value, "tolerance"))?,
        max_iterations: max_iterations.unwrap_or(default.max_iterations),
    })
}

fn iterative_config_f64(
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> nabled_ml::iterative::IterativeConfig<f64> {
    let default = nabled_ml::iterative::IterativeConfig::<f64>::default_f64();
    nabled_ml::iterative::IterativeConfig {
        tolerance:      tolerance.unwrap_or(default.tolerance),
        max_iterations: max_iterations.unwrap_or(default.max_iterations),
    }
}

fn jacobian_config_f32(
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<nabled_ml::jacobian::JacobianConfig<f32>> {
    let mut config = nabled_ml::jacobian::JacobianConfig::<f32>::default();
    if let Some(step_size) = step_size {
        config.step_size = utils::f64_to_f32(step_size, "step_size")?;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_f32(tolerance, "tolerance")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.validate().map_err(to_py_err)?;
    Ok(config)
}

fn jacobian_config_f64(
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<nabled_ml::jacobian::JacobianConfig<f64>> {
    let mut config = nabled_ml::jacobian::JacobianConfig::<f64>::default();
    if let Some(step_size) = step_size {
        config.step_size = step_size;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.validate().map_err(to_py_err)?;
    Ok(config)
}

fn line_search_config_f32(
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<nabled_ml::optimization::LineSearchConfig<f32>> {
    let mut config = nabled_ml::optimization::LineSearchConfig::<f32>::default();
    if let Some(initial_step) = initial_step {
        config.initial_step = utils::f64_to_f32(initial_step, "initial_step")?;
    }
    if let Some(contraction) = contraction {
        config.contraction = utils::f64_to_f32(contraction, "contraction")?;
    }
    if let Some(sufficient_decrease) = sufficient_decrease {
        config.sufficient_decrease = utils::f64_to_f32(sufficient_decrease, "sufficient_decrease")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    Ok(config)
}

fn line_search_config_f64(
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> nabled_ml::optimization::LineSearchConfig<f64> {
    let mut config = nabled_ml::optimization::LineSearchConfig::<f64>::default();
    if let Some(initial_step) = initial_step {
        config.initial_step = initial_step;
    }
    if let Some(contraction) = contraction {
        config.contraction = contraction;
    }
    if let Some(sufficient_decrease) = sufficient_decrease {
        config.sufficient_decrease = sufficient_decrease;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config
}

fn sgd_config_f32(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::SGDConfig<f32>> {
    let mut config = nabled_ml::optimization::SGDConfig::<f32>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = utils::f64_to_f32(learning_rate, "learning_rate")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    Ok(config)
}

fn sgd_config_f64(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_ml::optimization::SGDConfig<f64> {
    let mut config = nabled_ml::optimization::SGDConfig::<f64>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn adam_config_f32(
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::AdamConfig<f32>> {
    let mut config = nabled_ml::optimization::AdamConfig::<f32>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = utils::f64_to_f32(learning_rate, "learning_rate")?;
    }
    if let Some(beta1) = beta1 {
        config.beta1 = utils::f64_to_f32(beta1, "beta1")?;
    }
    if let Some(beta2) = beta2 {
        config.beta2 = utils::f64_to_f32(beta2, "beta2")?;
    }
    config.epsilon =
        epsilon.map(|value| utils::f64_to_f32(value, "epsilon")).transpose()?.unwrap_or(1e-6);
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    Ok(config)
}

fn adam_config_f64(
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_ml::optimization::AdamConfig<f64> {
    let mut config = nabled_ml::optimization::AdamConfig::<f64>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(beta1) = beta1 {
        config.beta1 = beta1;
    }
    if let Some(beta2) = beta2 {
        config.beta2 = beta2;
    }
    if let Some(epsilon) = epsilon {
        config.epsilon = epsilon;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn momentum_config_f32(
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::MomentumConfig<f32>> {
    let mut config = nabled_ml::optimization::MomentumConfig::<f32>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = utils::f64_to_f32(learning_rate, "learning_rate")?;
    }
    if let Some(momentum) = momentum {
        config.momentum = utils::f64_to_f32(momentum, "momentum")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    Ok(config)
}

fn momentum_config_f64(
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_ml::optimization::MomentumConfig<f64> {
    let mut config = nabled_ml::optimization::MomentumConfig::<f64>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(momentum) = momentum {
        config.momentum = momentum;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn rmsprop_config_f32(
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::RMSPropConfig<f32>> {
    let mut config = nabled_ml::optimization::RMSPropConfig::<f32>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = utils::f64_to_f32(learning_rate, "learning_rate")?;
    }
    if let Some(rho) = rho {
        config.rho = utils::f64_to_f32(rho, "rho")?;
    }
    config.epsilon =
        epsilon.map(|value| utils::f64_to_f32(value, "epsilon")).transpose()?.unwrap_or(1e-6);
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    Ok(config)
}

fn rmsprop_config_f64(
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_ml::optimization::RMSPropConfig<f64> {
    let mut config = nabled_ml::optimization::RMSPropConfig::<f64>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(rho) = rho {
        config.rho = rho;
    }
    if let Some(epsilon) = epsilon {
        config.epsilon = epsilon;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn projected_gradient_config_f32(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::ProjectedGradientConfig<f32>> {
    let mut config = nabled_ml::optimization::ProjectedGradientConfig::<f32>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = utils::f64_to_f32(learning_rate, "learning_rate")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    Ok(config)
}

fn projected_gradient_config_f64(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> nabled_ml::optimization::ProjectedGradientConfig<f64> {
    let mut config = nabled_ml::optimization::ProjectedGradientConfig::<f64>::default();
    if let Some(learning_rate) = learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    config
}

fn bfgs_config_f32(
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<nabled_ml::optimization::BFGSConfig<f32>> {
    let mut config = nabled_ml::optimization::BFGSConfig::<f32>::default();
    if let Some(step_size) = step_size {
        config.step_size = utils::f64_to_f32(step_size, "step_size")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.tolerance =
        tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?.unwrap_or(1e-5);
    config.curvature_tolerance = curvature_tolerance
        .map(|value| utils::f64_to_f32(value, "curvature_tolerance"))
        .transpose()?
        .unwrap_or(1e-6);
    Ok(config)
}

fn bfgs_config_f64(
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> nabled_ml::optimization::BFGSConfig<f64> {
    let mut config = nabled_ml::optimization::BFGSConfig::<f64>::default();
    if let Some(step_size) = step_size {
        config.step_size = step_size;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = tolerance;
    }
    if let Some(curvature_tolerance) = curvature_tolerance {
        config.curvature_tolerance = curvature_tolerance;
    }
    config
}

fn map_callback_error<T, E: std::fmt::Display>(
    callback_error: &RefCell<Option<PyErr>>,
    result: Result<T, E>,
) -> PyResult<T> {
    if let Some(err) = callback_error.borrow_mut().take() {
        return Err(err);
    }
    result.map_err(to_py_err)
}

fn cp_als3_factor_arrays<'py, T: numpy::Element>(
    factors: &Bound<'py, PyAny>,
) -> PyResult<[PyReadonlyArray2<'py, T>; 3]> {
    py_tensor::extract_array2_sequence_views::<T>(factors)?.try_into().map_err(
        |factors: Vec<PyReadonlyArray2<'py, T>>| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "factors must contain exactly 3 contiguous 2D NumPy arrays, got {}",
                factors.len()
            ))
        },
    )
}

fn variable_shape_real_dtype(field: &Field) -> Option<&'static str> {
    let DataType::Struct(fields) = field.data_type() else {
        return None;
    };
    let (_, data_field) = fields.find("data")?;
    let DataType::List(item) = data_field.data_type() else {
        return None;
    };
    match item.data_type() {
        DataType::Float32 => Some("f32"),
        DataType::Float64 => Some("f64"),
        _ => None,
    }
}

/// Compute dot product of two real PyArrow arrays.
#[pyfunction(name = "arrow_dot")]
pub fn dot(left: PyArrowType<ArrayData>, right: PyArrowType<ArrayData>) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => {
            Ok(f64::from(nabled::arrow::vector::dot(&left_arr, &right_arr).map_err(to_py_err)?))
        }
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::dot(&left_arr, &right_arr).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute cosine similarity of two real PyArrow arrays.
#[pyfunction(name = "arrow_cosine_similarity")]
pub fn cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => Ok(f64::from(
            nabled::arrow::vector::cosine_similarity(&left_arr, &right_arr).map_err(to_py_err)?,
        )),
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::cosine_similarity(&left_arr, &right_arr).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute L2 norm of a real PyArrow array.
#[pyfunction(name = "arrow_l2_norm")]
pub fn l2_norm(array: PyArrowType<ArrayData>) -> PyResult<f64> {
    match array_data_to_real_primitive(array.0)? {
        RealPrimitiveArray::F32(arr) => {
            Ok(f64::from(nabled::arrow::vector::l2_norm(&arr).map_err(to_py_err)?))
        }
        RealPrimitiveArray::F64(arr) => nabled::arrow::vector::l2_norm(&arr).map_err(to_py_err),
    }
}

/// Compute cosine distance of two real PyArrow arrays.
#[pyfunction(name = "arrow_cosine_distance")]
pub fn cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => Ok(f64::from(
            nabled::arrow::vector::cosine_distance(&left_arr, &right_arr).map_err(to_py_err)?,
        )),
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::cosine_distance(&left_arr, &right_arr).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise L2 distances between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_l2_distance")]
pub fn pairwise_l2_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_l2_distance::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_l2_distance::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise cosine similarities between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_cosine_similarity")]
pub fn pairwise_cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_similarity::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_similarity::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise cosine distances between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_cosine_distance")]
pub fn pairwise_cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_distance::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_distance::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise dot products across Arrow row batches.
#[pyfunction(name = "arrow_batched_dot")]
pub fn batched_dot(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_dot::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_dot::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise L2 norms across Arrow row batches.
#[pyfunction(name = "arrow_batched_l2_norm")]
pub fn batched_l2_norm(array: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::vector::batched_l2_norm::<Float32Type>(&arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::vector::batched_l2_norm::<Float64Type>(&arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute row-wise cosine similarities across Arrow row batches.
#[pyfunction(name = "arrow_batched_cosine_similarity")]
pub fn batched_cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_similarity::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_similarity::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise cosine distances across Arrow row batches.
#[pyfunction(name = "arrow_batched_cosine_distance")]
pub fn batched_cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_distance::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_distance::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Normalize Arrow row batches.
#[pyfunction(name = "arrow_batched_normalize")]
pub fn batched_normalize(array: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::vector::batched_normalize::<Float32Type>(&arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::vector::batched_normalize::<Float64Type>(&arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute a dense matrix-vector product from PyArrow carriers.
#[pyfunction(name = "arrow_matvec")]
pub fn matvec(
    matrix: PyArrowType<ArrayData>,
    vector: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(vector.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(vector_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::matrix::matvec::<Float32Type>(&matrix_arr, &vector_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(vector_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::matrix::matvec::<Float64Type>(&matrix_arr, &vector_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "vector"])),
    }
}

/// Compute a dense matrix-matrix product from PyArrow carriers.
#[pyfunction(name = "arrow_matmat")]
pub fn matmat(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::matmat::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::matmat::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Apply one dense matrix to a batch of Arrow row vectors.
#[pyfunction(name = "arrow_batched_row_matvec")]
pub fn batched_row_matvec(
    batch_vectors: PyArrowType<ArrayData>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(batch_vectors.0)?,
        array_data_to_real_fixed_size_list(matrix.0)?,
    ) {
        (RealFixedSizeListArray::F32(batch_arr), RealFixedSizeListArray::F32(matrix_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::batched_row_matvec::<Float32Type>(&batch_arr, &matrix_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(batch_arr), RealFixedSizeListArray::F64(matrix_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::batched_row_matvec::<Float64Type>(&batch_arr, &matrix_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["batch_vectors", "matrix"])),
    }
}

/// Compute batched dense matrix-matrix products from Arrow fixed-shape tensors.
#[pyfunction(name = "arrow_batched_matmat")]
pub fn batched_matmat(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) = nabled::arrow::matrix::batched_matmat::<Float32Type>(
                &left_field,
                &left_arr,
                &right_field,
                &right_arr,
            )
            .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) = nabled::arrow::matrix::batched_matmat::<Float64Type>(
                &left_field,
                &left_arr,
                &right_field,
                &right_arr,
            )
            .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute batched dense matrix-matrix products with a broadcasted right operand.
#[pyfunction(name = "arrow_batched_matmat_broadcast_right")]
pub fn batched_matmat_broadcast_right(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_right::<Float32Type>(
                    &left_field,
                    &left_arr,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_right::<Float64Type>(
                    &left_field,
                    &left_arr,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute batched dense matrix-matrix products with a broadcasted left operand.
#[pyfunction(name = "arrow_batched_matmat_broadcast_left")]
pub fn batched_matmat_broadcast_left(
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_left::<Float32Type>(
                    &left_arr,
                    &right_field,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_left::<Float64Type>(
                    &left_arr,
                    &right_field,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute column means from a dense PyArrow matrix.
#[pyfunction(name = "arrow_column_means")]
pub fn column_means(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::stats::column_means_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::stats::column_means_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Center columns from a dense PyArrow matrix.
#[pyfunction(name = "arrow_center_columns")]
pub fn center_columns(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::center_columns_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::center_columns_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute covariance from a dense PyArrow matrix.
#[pyfunction(name = "arrow_covariance_matrix")]
pub fn covariance_matrix(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::covariance_matrix_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::covariance_matrix_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute correlation from a dense PyArrow matrix.
#[pyfunction(name = "arrow_correlation_matrix")]
pub fn correlation_matrix(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::correlation_matrix_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::correlation_matrix_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute SVD of a real PyArrow dense matrix.
/// Returns `(U, singular_values, Vt)` as NumPy arrays preserving `float32` or `float64`.
#[pyfunction(name = "arrow_svd_decompose")]
pub fn svd_decompose(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::svd::decompose_f32(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::svd::decompose_f64(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}

/// Compute truncated SVD of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_svd_decompose_truncated")]
pub fn svd_decompose_truncated(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    k: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::svd::decompose_truncated_f32(&fsl, k).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::svd::decompose_truncated_f64(&fsl, k).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}

/// Compute tolerance-driven SVD of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_svd_decompose_with_tolerance")]
pub fn svd_decompose_with_tolerance(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    tolerance: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let tolerance = utils::f64_to_f32(tolerance, "tolerance")?;
            let result = nabled::arrow::svd::decompose_with_tolerance_f32(&fsl, tolerance)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::svd::decompose_with_tolerance_f64(&fsl, tolerance)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}

/// Compute pseudo-inverse of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_svd_pseudo_inverse")]
pub fn svd_pseudo_inverse(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::svd::pseudo_inverse_f32(
                &fsl,
                nabled_linalg::svd::PseudoInverseConfig::default(),
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::svd::pseudo_inverse_f64(
                &fsl,
                nabled_linalg::svd::PseudoInverseConfig::default(),
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Compute right null-space basis of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_svd_null_space", signature = (matrix, tolerance=None))]
pub fn svd_null_space(
    matrix: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::svd::null_space_f32(
                &fsl,
                tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::svd::null_space_f64(&fsl, tolerance).map_err(to_py_err)?,
        )),
    }
}

/// Compute QR decomposition of a real PyArrow dense matrix. Returns `(Q, R, rank)`.
#[pyfunction(name = "arrow_qr_decompose", signature = (matrix, rank_tolerance=None, max_iterations=None))]
pub fn qr_decompose(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(qr_result_tuple(
            py,
            nabled::arrow::qr::decompose_f32(
                &fsl,
                &qr_config_f32(rank_tolerance, max_iterations, false)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(qr_result_tuple(
            py,
            nabled::arrow::qr::decompose_f64(
                &fsl,
                &qr_config_f64(rank_tolerance, max_iterations, false),
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Compute reduced QR decomposition of a real PyArrow dense matrix. Returns `(Q, R, rank)`.
#[pyfunction(
    name = "arrow_qr_decompose_reduced",
    signature = (matrix, rank_tolerance=None, max_iterations=None)
)]
pub fn qr_decompose_reduced(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(qr_result_tuple(
            py,
            nabled::arrow::qr::decompose_reduced_f32(
                &fsl,
                &qr_config_f32(rank_tolerance, max_iterations, false)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(qr_result_tuple(
            py,
            nabled::arrow::qr::decompose_reduced_f64(
                &fsl,
                &qr_config_f64(rank_tolerance, max_iterations, false),
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Compute pivoted QR decomposition of a real PyArrow dense matrix. Returns `(Q, R, P, rank)`.
#[pyfunction(
    name = "arrow_qr_decompose_pivoted",
    signature = (matrix, rank_tolerance=None, max_iterations=None)
)]
pub fn qr_decompose_pivoted(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => qr_pivoted_result_tuple(
            py,
            nabled::arrow::qr::decompose_with_pivoting_f32(
                &fsl,
                &qr_config_f32(rank_tolerance, max_iterations, true)?,
            )
            .map_err(to_py_err)?,
        ),
        RealFixedSizeListArray::F64(fsl) => qr_pivoted_result_tuple(
            py,
            nabled::arrow::qr::decompose_with_pivoting_f64(
                &fsl,
                &qr_config_f64(rank_tolerance, max_iterations, true),
            )
            .map_err(to_py_err)?,
        ),
    }
}

/// Solve least-squares from real PyArrow dense inputs.
#[pyfunction(
    name = "arrow_qr_solve_least_squares",
    signature = (matrix, rhs, rank_tolerance=None, max_iterations=None)
)]
pub fn qr_solve_least_squares(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::qr::solve_least_squares_f32(
                    &matrix_arr,
                    &rhs_arr,
                    &qr_config_f32(rank_tolerance, max_iterations, false)?,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::qr::solve_least_squares_f64(
                    &matrix_arr,
                    &rhs_arr,
                    &qr_config_f64(rank_tolerance, max_iterations, false),
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute LU decomposition of a real PyArrow dense matrix. Returns `(L, U, pivots,
/// permutation_sign)`.
#[pyfunction(name = "arrow_lu_decompose")]
pub fn lu_decompose(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i8)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let (result, pivots, permutation_sign) =
                nabled::arrow::lu::decompose_f32_with_metadata(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.l),
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(
                    py,
                    utils::usize_array1_to_i64(pivots, "pivots")
                        .expect("usize pivot indices should fit in Python int64 arrays"),
                ),
                permutation_sign,
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let (result, pivots, permutation_sign) =
                nabled::arrow::lu::decompose_f64_with_metadata(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.l),
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(
                    py,
                    utils::usize_array1_to_i64(pivots, "pivots")
                        .expect("usize pivot indices should fit in Python int64 arrays"),
                ),
                permutation_sign,
            ))
        }
    }
}

/// Solve a real dense system from PyArrow LU inputs.
#[pyfunction(name = "arrow_lu_solve")]
pub fn lu_solve(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::lu::solve_f32(&matrix_arr, &rhs_arr).map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::lu::solve_f64(&matrix_arr, &rhs_arr).map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute inverse of a real dense PyArrow matrix via LU.
#[pyfunction(name = "arrow_lu_inverse")]
pub fn lu_inverse(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::lu::inverse_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::lu::inverse_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute determinant of a real dense PyArrow matrix via LU.
#[pyfunction(name = "arrow_lu_determinant")]
pub fn lu_determinant(matrix: PyArrowType<ArrayData>) -> PyResult<f64> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            Ok(f64::from(nabled::arrow::lu::determinant_f32(&fsl).map_err(to_py_err)?))
        }
        RealFixedSizeListArray::F64(fsl) => {
            nabled::arrow::lu::determinant_f64(&fsl).map_err(to_py_err)
        }
    }
}

/// Compute signed log-determinant of a real dense PyArrow matrix via LU.
#[pyfunction(name = "arrow_lu_log_determinant")]
pub fn lu_log_determinant(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(i8, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::lu::log_determinant_f32(&fsl).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, f64::from(result.ln_abs_det))))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::lu::log_determinant_f64(&fsl).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det)))
        }
    }
}

/// Compute Cholesky decomposition of a real PyArrow dense matrix. Returns `L`.
#[pyfunction(name = "arrow_cholesky_decompose")]
pub fn cholesky_decompose(py: Python<'_>, matrix: PyArrowType<ArrayData>) -> PyResult<Py<PyAny>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::cholesky::decompose_f32(&fsl).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::cholesky::decompose_f64(&fsl).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
    }
}

/// Solve a real SPD system from PyArrow dense inputs using Cholesky.
#[pyfunction(name = "arrow_cholesky_solve")]
pub fn cholesky_solve(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::cholesky::solve_f32(&matrix_arr, &rhs_arr).map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::cholesky::solve_f64(&matrix_arr, &rhs_arr).map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute inverse of a real SPD PyArrow matrix using Cholesky.
#[pyfunction(name = "arrow_cholesky_inverse")]
pub fn cholesky_inverse(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::cholesky::inverse_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::cholesky::inverse_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute symmetric eigen decomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_eigen_symmetric")]
pub fn eigen_symmetric(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::eigen::symmetric_f32(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::eigen::symmetric_f64(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
    }
}

/// Compute generalized eigen decomposition of real PyArrow dense matrices.
#[pyfunction(name = "arrow_eigen_generalized")]
pub fn eigen_generalized(
    py: Python<'_>,
    matrix_a: PyArrowType<ArrayData>,
    matrix_b: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match (
        array_data_to_real_fixed_size_list(matrix_a.0)?,
        array_data_to_real_fixed_size_list(matrix_b.0)?,
    ) {
        (RealFixedSizeListArray::F32(a), RealFixedSizeListArray::F32(b)) => {
            let result = nabled::arrow::eigen::generalized_f32(&a, &b).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        (RealFixedSizeListArray::F64(a), RealFixedSizeListArray::F64(b)) => {
            let result = nabled::arrow::eigen::generalized_f64(&a, &b).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.eigenvectors),
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// Compute non-symmetric eigen decomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_eigen_nonsymmetric")]
pub fn eigen_nonsymmetric(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::eigen::nonsymmetric_f32(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.schur_vectors),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::eigen::nonsymmetric_f64(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.schur_vectors),
            ))
        }
    }
}

/// Compute matched left/right non-symmetric eigenvectors of a real PyArrow dense matrix.
#[pyfunction(
    name = "arrow_eigen_nonsymmetric_bi",
    signature = (matrix, balance=true, balance_max_iterations=None, balance_tolerance=None)
)]
pub fn eigen_nonsymmetric_bi(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    balance: bool,
    balance_max_iterations: Option<usize>,
    balance_tolerance: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::eigen::nonsymmetric_bi_f32(
                &fsl,
                &nonsymmetric_config_f32(balance, balance_max_iterations, balance_tolerance)?,
            )
            .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.right_eigenvectors),
                utils::pyarray2_from_owned(py, result.left_eigenvectors),
                utils::pyarray1_from_owned(py, result.balancing_diagonal),
                utils::pyarray2_from_owned(py, result.balanced_matrix),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::eigen::nonsymmetric_bi_f64(
                &fsl,
                &nonsymmetric_config_f64(balance, balance_max_iterations, balance_tolerance),
            )
            .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.eigenvalues),
                utils::pyarray2_from_owned(py, result.right_eigenvectors),
                utils::pyarray2_from_owned(py, result.left_eigenvectors),
                utils::pyarray1_from_owned(py, result.balancing_diagonal),
                utils::pyarray2_from_owned(py, result.balanced_matrix),
            ))
        }
    }
}

/// Compute Schur decomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_schur_compute")]
pub fn schur_compute(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::schur::compute_f32(&fsl).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::schur::compute_f64(&fsl).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
    }
}

/// Compute polar decomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_polar_compute")]
pub fn polar_compute(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::polar::compute_f32(&fsl).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::polar::compute_f64(&fsl).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
    }
}

/// Compute matrix exponential of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_exp", signature = (matrix, max_terms=None, tolerance=None))]
pub fn matrix_exp(
    matrix: PyArrowType<ArrayData>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    let max_terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::exp_f32(
                &fsl,
                max_terms,
                utils::f64_to_f32(tolerance, "tolerance")?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::exp_f64(&fsl, max_terms, tolerance)
                .map_err(to_py_err)?,
        )),
    }
}

/// Compute matrix exponential via eigendecomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_exp_eigen")]
pub fn matrix_exp_eigen(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::exp_eigen_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::exp_eigen_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute matrix logarithm via Taylor expansion of a real PyArrow dense matrix.
#[pyfunction(
    name = "arrow_matrix_log_taylor",
    signature = (matrix, max_terms=None, tolerance=None)
)]
pub fn matrix_log_taylor(
    matrix: PyArrowType<ArrayData>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    let max_terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_taylor_f32(
                &fsl,
                max_terms,
                utils::f64_to_f32(tolerance, "tolerance")?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_taylor_f64(&fsl, max_terms, tolerance)
                .map_err(to_py_err)?,
        )),
    }
}

/// Compute matrix logarithm via eigendecomposition of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_log_eigen")]
pub fn matrix_log_eigen(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_eigen_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_eigen_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute matrix logarithm via SVD of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_log_svd")]
pub fn matrix_log_svd(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_svd_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::log_svd_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute a real matrix power from a PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_power")]
pub fn matrix_power(
    matrix: PyArrowType<ArrayData>,
    power: f64,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::power_f32(&fsl, utils::f64_to_f32(power, "power")?)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::power_f64(&fsl, power).map_err(to_py_err)?,
        )),
    }
}

/// Compute the matrix sign function of a real PyArrow dense matrix.
#[pyfunction(name = "arrow_matrix_sign")]
pub fn matrix_sign(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::sign_f32(&fsl).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(fsl) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::matrix_functions::sign_f64(&fsl).map_err(to_py_err)?,
        )),
    }
}

/// Compute PCA from a real PyArrow dense matrix.
#[pyfunction(name = "arrow_compute_pca", signature = (matrix, n_components=None))]
pub fn compute_pca(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    n_components: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::pca::compute_f32(&fsl, n_components).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.components),
                utils::pyarray1_from_owned(py, result.explained_variance),
                utils::pyarray1_from_owned(py, result.explained_variance_ratio),
                utils::pyarray1_from_owned(py, result.mean),
                utils::pyarray2_from_owned(py, result.scores),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::pca::compute_f64(&fsl, n_components).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.components),
                utils::pyarray1_from_owned(py, result.explained_variance),
                utils::pyarray1_from_owned(py, result.explained_variance_ratio),
                utils::pyarray1_from_owned(py, result.mean),
                utils::pyarray2_from_owned(py, result.scores),
            ))
        }
    }
}

/// Project Arrow dense data into PCA score space using a typed PCA result.
#[pyfunction(name = "arrow_pca_transform")]
pub fn pca_transform(
    matrix: PyArrowType<ArrayData>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(matrix.0)?,
        utils::real_array2(components, "components")?,
        utils::real_array1(mean, "mean")?,
    ) {
        (
            RealFixedSizeListArray::F32(matrix_arr),
            utils::RealReadonlyArray2::F32(components_arr),
            utils::RealReadonlyArray1::F32(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::transform_f32_from_components_view(
                &matrix_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        (
            RealFixedSizeListArray::F64(matrix_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::transform_f64_from_components_view(
                &matrix_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["matrix", "components", "mean"])),
    }
}

/// Reconstruct Arrow dense data from PCA scores using a typed PCA result.
#[pyfunction(name = "arrow_pca_inverse_transform")]
pub fn pca_inverse_transform(
    scores: PyArrowType<ArrayData>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(scores.0)?,
        utils::real_array2(components, "components")?,
        utils::real_array1(mean, "mean")?,
    ) {
        (
            RealFixedSizeListArray::F32(scores_arr),
            utils::RealReadonlyArray2::F32(components_arr),
            utils::RealReadonlyArray1::F32(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::inverse_transform_f32_from_components_view(
                &scores_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        (
            RealFixedSizeListArray::F64(scores_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::inverse_transform_f64_from_components_view(
                &scores_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["scores", "components", "mean"])),
    }
}

/// Solve linear regression directly from real PyArrow dense inputs.
#[pyfunction(name = "arrow_linear_regression", signature = (x, y, add_intercept=true))]
pub fn linear_regression(
    py: Python<'_>,
    x: PyArrowType<ArrayData>,
    y: PyArrowType<ArrayData>,
    add_intercept: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    match (array_data_to_real_fixed_size_list(x.0)?, array_data_to_real_primitive(y.0)?) {
        (RealFixedSizeListArray::F32(x_arr), RealPrimitiveArray::F32(y_arr)) => {
            let result =
                nabled::arrow::regression::linear_regression_f32(&x_arr, &y_arr, add_intercept)
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.coefficients),
                utils::pyarray1_from_owned(py, result.fitted_values),
                utils::pyarray1_from_owned(py, result.residuals),
                f64::from(result.r_squared),
            ))
        }
        (RealFixedSizeListArray::F64(x_arr), RealPrimitiveArray::F64(y_arr)) => {
            let result =
                nabled::arrow::regression::linear_regression_f64(&x_arr, &y_arr, add_intercept)
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.coefficients),
                utils::pyarray1_from_owned(py, result.fitted_values),
                utils::pyarray1_from_owned(py, result.residuals),
                result.r_squared,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "y"])),
    }
}

/// Compute a sparse matrix-vector product directly from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_matvec")]
pub fn sparse_matvec_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::matvec_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::matvec_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute a sparse-dense matrix product directly from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_matmat_dense")]
pub fn sparse_matmat_dense_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    dense: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_fixed_size_list(dense.0)?)
    {
        (DataType::Float32, RealFixedSizeListArray::F32(dense_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::sparse::matmat_dense_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &dense_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealFixedSizeListArray::F64(dense_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::sparse::matmat_dense_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &dense_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "dense"])),
    }
}

/// Solve a sparse linear system directly from a canonical Arrow CSR extension via sparse LU.
#[pyfunction(name = "arrow_sparse_lu_solve")]
pub fn sparse_lu_solve_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::sparse_lu_solve_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::sparse_lu_solve_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a sparse linear system via Jacobi iteration from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_jacobi_solve", signature = (field, matrix, rhs, tolerance=None, max_iterations=None))]
pub fn sparse_jacobi_solve_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::jacobi_solve_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f32(tolerance)?,
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::jacobi_solve_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f64(tolerance),
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a sparse linear system via Gauss-Seidel iteration from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_gauss_seidel_solve", signature = (field, matrix, rhs, tolerance=None, max_iterations=None))]
pub fn sparse_gauss_seidel_solve_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::gauss_seidel_solve_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f32(tolerance)?,
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::gauss_seidel_solve_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f64(tolerance),
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a sparse linear system via conjugate gradient from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_conjugate_gradient_solve", signature = (field, matrix, rhs, tolerance=None, max_iterations=None))]
pub fn sparse_conjugate_gradient_solve_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::conjugate_gradient_solve_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f32(tolerance)?,
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::conjugate_gradient_solve_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f64(tolerance),
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a sparse linear system via PCG from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_pcg_solve", signature = (field, matrix, rhs, tolerance=None, max_iterations=None))]
pub fn sparse_pcg_solve_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (sparse_value_type(&matrix_arr, "values")?, array_data_to_real_primitive(rhs.0)?) {
        (DataType::Float32, RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::pcg_solve_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f32(tolerance)?,
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::pcg_solve_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &rhs_arr,
                    sparse_tolerance_f64(tolerance),
                    sparse_max_iterations(max_iterations),
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute batched sparse matrix-vector products from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_batched_matvec")]
pub fn sparse_batched_matvec_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    batch_vectors: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (
        sparse_value_type(&matrix_arr, "values")?,
        array_data_to_real_fixed_size_list(batch_vectors.0)?,
    ) {
        (DataType::Float32, RealFixedSizeListArray::F32(batch_vectors_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::sparse::batched_matvec_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                    &batch_vectors_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (DataType::Float64, RealFixedSizeListArray::F64(batch_vectors_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::sparse::batched_matvec_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                    &batch_vectors_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "batch_vectors"])),
    }
}

/// Transpose a canonical Arrow CSR extension and return canonical Python CSR components.
#[pyfunction(name = "arrow_sparse_transpose")]
pub fn sparse_transpose_arrow(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyCsrParts> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => {
            let result = nabled::arrow::sparse::transpose_csr_extension::<Float32Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csr_parts_i32_f32(py, result)
        }
        DataType::Float64 => {
            let result = nabled::arrow::sparse::transpose_csr_extension::<Float64Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csr_parts_i32_f64(py, result)
        }
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Convert a canonical Arrow CSR extension to canonical Python CSC components.
#[pyfunction(name = "arrow_sparse_csr_to_csc")]
pub fn sparse_csr_to_csc_arrow(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyCsrParts> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => {
            let result = nabled::arrow::sparse::csr_to_csc_csr_extension::<Float32Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csc_parts_f32(py, result, csr::StoredIndexDtype::I32)
        }
        DataType::Float64 => {
            let result = nabled::arrow::sparse::csr_to_csc_csr_extension::<Float64Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csc_parts_f64(py, result, csr::StoredIndexDtype::I32)
        }
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Compute a sparse-sparse matrix product directly from canonical Arrow CSR extensions.
#[pyfunction(name = "arrow_sparse_matmat_sparse")]
pub fn sparse_matmat_sparse_arrow(
    py: Python<'_>,
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<csr::PyCsrParts> {
    let left_arr = array_data_to_struct(left.0)?;
    let right_arr = array_data_to_struct(right.0)?;
    match (sparse_value_type(&left_arr, "values")?, sparse_value_type(&right_arr, "values")?) {
        (DataType::Float32, DataType::Float32) => {
            let result = nabled::arrow::sparse::matmat_sparse_csr_extension::<Float32Type>(
                &left_field.0,
                &left_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csr_parts_i32_f32(py, result)
        }
        (DataType::Float64, DataType::Float64) => {
            let result = nabled::arrow::sparse::matmat_sparse_csr_extension::<Float64Type>(
                &left_field.0,
                &left_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?;
            csr::py_csr_parts_i32_f64(py, result)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Build a reusable Jacobi preconditioner from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_jacobi_preconditioner")]
pub fn sparse_jacobi_preconditioner_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyJacobiPreconditioner> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyJacobiPreconditioner::from_f32(
            nabled::arrow::sparse::jacobi_preconditioner_csr_extension::<Float32Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyJacobiPreconditioner::from_f64(
            nabled::arrow::sparse::jacobi_preconditioner_csr_extension::<Float64Type>(
                &field.0,
                &matrix_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable Jacobi preconditioner to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_jacobi_preconditioner")]
pub fn sparse_apply_jacobi_preconditioner_arrow(
    preconditioner: &csr::PyJacobiPreconditioner,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&preconditioner.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyJacobiPreconditionerInner::F32(preconditioner),
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_jacobi_preconditioner::<Float32Type>(
                preconditioner,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        (
            csr::PyJacobiPreconditionerInner::F64(preconditioner),
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_jacobi_preconditioner::<Float64Type>(
                preconditioner,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["preconditioner", "rhs"])),
    }
}

/// Build a reusable ILU(0) factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_ilu0_factor")]
pub fn sparse_ilu0_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyIlu0Factorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyIlu0Factorization::from_f32(
            nabled::arrow::sparse::ilu0_factor_csr_extension::<Float32Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyIlu0Factorization::from_f64(
            nabled::arrow::sparse::ilu0_factor_csr_extension::<Float64Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable ILU(0) factorization to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_ilu0_preconditioner")]
pub fn sparse_apply_ilu0_preconditioner_arrow(
    factorization: &csr::PyIlu0Factorization,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyIlu0FactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_ilu0_preconditioner::<Float32Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        (
            csr::PyIlu0FactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_ilu0_preconditioner::<Float64Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Build a reusable ILUT factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_ilut_factor", signature = (field, matrix, drop_tolerance=None, max_fill=None))]
pub fn sparse_ilut_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    drop_tolerance: Option<f64>,
    max_fill: Option<usize>,
) -> PyResult<csr::PyIlutFactorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyIlutFactorization::from_f32(
            nabled::arrow::sparse::ilut_factor_csr_extension::<Float32Type>(
                &field.0,
                &matrix_arr,
                ilut_drop_tolerance_f32(drop_tolerance)?,
                ilut_max_fill(max_fill),
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyIlutFactorization::from_f64(
            nabled::arrow::sparse::ilut_factor_csr_extension::<Float64Type>(
                &field.0,
                &matrix_arr,
                ilut_drop_tolerance_f64(drop_tolerance),
                ilut_max_fill(max_fill),
            )
            .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable ILUT factorization to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_ilut_preconditioner")]
pub fn sparse_apply_ilut_preconditioner_arrow(
    factorization: &csr::PyIlutFactorization,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyIlutFactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_ilut_preconditioner::<Float32Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        (
            csr::PyIlutFactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_ilut_preconditioner::<Float64Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Build a reusable ILU(k) factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_iluk_factor", signature = (field, matrix, level_of_fill=None))]
pub fn sparse_iluk_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    level_of_fill: Option<usize>,
) -> PyResult<csr::PyIlukFactorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyIlukFactorization::from_f32(
            nabled::arrow::sparse::iluk_factor_csr_extension::<Float32Type>(
                &field.0,
                &matrix_arr,
                iluk_level_of_fill(level_of_fill),
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyIlukFactorization::from_f64(
            nabled::arrow::sparse::iluk_factor_csr_extension::<Float64Type>(
                &field.0,
                &matrix_arr,
                iluk_level_of_fill(level_of_fill),
            )
            .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable ILU(k) factorization to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_iluk_preconditioner")]
pub fn sparse_apply_iluk_preconditioner_arrow(
    factorization: &csr::PyIlukFactorization,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyIlukFactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_iluk_preconditioner::<Float32Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        (
            csr::PyIlukFactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_iluk_preconditioner::<Float64Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Build a reusable IC(0) factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_ic0_factor")]
pub fn sparse_ic0_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyIc0Factorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyIc0Factorization::from_f32(
            nabled::arrow::sparse::ic0_factor_csr_extension::<Float32Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyIc0Factorization::from_f64(
            nabled::arrow::sparse::ic0_factor_csr_extension::<Float64Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable IC(0) factorization to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_ic0_preconditioner")]
pub fn sparse_apply_ic0_preconditioner_arrow(
    factorization: &csr::PyIc0Factorization,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyIc0FactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_ic0_preconditioner::<Float32Type>(factorization, &rhs_arr)
                .map_err(to_py_err)?,
        )),
        (
            csr::PyIc0FactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_ic0_preconditioner::<Float64Type>(factorization, &rhs_arr)
                .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Build a reusable ILDL(0) factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_ildl0_factor")]
pub fn sparse_ildl0_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PyIldl0Factorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => Ok(csr::PyIldl0Factorization::from_f32(
            nabled::arrow::sparse::ildl0_factor_csr_extension::<Float32Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(csr::PyIldl0Factorization::from_f64(
            nabled::arrow::sparse::ildl0_factor_csr_extension::<Float64Type>(&field.0, &matrix_arr)
                .map_err(to_py_err)?,
        )),
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Apply a reusable ILDL(0) factorization to an Arrow dense vector.
#[pyfunction(name = "arrow_sparse_apply_ildl0_preconditioner")]
pub fn sparse_apply_ildl0_preconditioner_arrow(
    factorization: &csr::PyIldl0Factorization,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PyIldl0FactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float32Type>(
            nabled::arrow::sparse::apply_ildl0_preconditioner::<Float32Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        (
            csr::PyIldl0FactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => Ok(primitive_array_into_pyarrow::<Float64Type>(
            nabled::arrow::sparse::apply_ildl0_preconditioner::<Float64Type>(
                factorization,
                &rhs_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Build a reusable sparse LU factorization from a canonical Arrow CSR extension.
#[pyfunction(name = "arrow_sparse_lu_factor")]
pub fn sparse_lu_factor_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<csr::PySparseLuFactorization> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match sparse_value_type(&matrix_arr, "values")? {
        DataType::Float32 => {
            let owned_matrix = owned_csr_from_extension_f32(&field.0, &matrix_arr)?;
            let factorization =
                nabled::arrow::sparse::sparse_lu_factor_csr_extension::<Float32Type>(
                    &field.0,
                    &matrix_arr,
                )
                .map_err(to_py_err)?;
            Ok(csr::PySparseLuFactorization::from_f32(owned_matrix, factorization))
        }
        DataType::Float64 => {
            let owned_matrix = owned_csr_from_extension_f64(&field.0, &matrix_arr)?;
            let factorization =
                nabled::arrow::sparse::sparse_lu_factor_csr_extension::<Float64Type>(
                    &field.0,
                    &matrix_arr,
                )
                .map_err(to_py_err)?;
            Ok(csr::PySparseLuFactorization::from_f64(owned_matrix, factorization))
        }
        _ => {
            Err(PyTypeError::new_err("expected sparse Arrow values with float32 or float64 dtype"))
        }
    }
}

/// Solve a sparse linear system using a reusable sparse LU factorization.
#[pyfunction(name = "arrow_sparse_lu_solve_with_factorization")]
pub fn sparse_lu_solve_with_factorization_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    factorization: &csr::PySparseLuFactorization,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (&factorization.inner, array_data_to_real_primitive(rhs.0)?) {
        (
            csr::PySparseLuFactorizationInner::F32 { factorization, .. },
            RealPrimitiveArray::F32(rhs_arr),
        ) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::sparse::sparse_lu_solve_with_factorization_csr_extension::<
                    Float32Type,
                >(&field.0, &matrix_arr, &rhs_arr, factorization)
                .map_err(to_py_err)?,
            ))
        }
        (
            csr::PySparseLuFactorizationInner::F64 { factorization, .. },
            RealPrimitiveArray::F64(rhs_arr),
        ) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::sparse::sparse_lu_solve_with_factorization_csr_extension::<
                    Float64Type,
                >(&field.0, &matrix_arr, &rhs_arr, factorization)
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Solve multiple sparse right-hand sides using a reusable sparse LU factorization.
#[pyfunction(name = "arrow_sparse_lu_solve_multiple_with_factorization")]
pub fn sparse_lu_solve_multiple_with_factorization_arrow(
    field: PyArrowType<Field>,
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    factorization: &csr::PySparseLuFactorization,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_struct(matrix.0)?;
    match (&factorization.inner, array_data_to_real_fixed_size_list(rhs.0)?) {
        (
            csr::PySparseLuFactorizationInner::F32 { factorization, .. },
            RealFixedSizeListArray::F32(rhs_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::sparse::sparse_lu_solve_multiple_with_factorization_csr_extension::<
                Float32Type,
            >(&field.0, &matrix_arr, &rhs_arr, factorization)
            .map_err(to_py_err)?,
        )),
        (
            csr::PySparseLuFactorizationInner::F64 { factorization, .. },
            RealFixedSizeListArray::F64(rhs_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::sparse::sparse_lu_solve_multiple_with_factorization_csr_extension::<
                Float64Type,
            >(&field.0, &matrix_arr, &rhs_arr, factorization)
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["factorization", "rhs"])),
    }
}

/// Compute row-wise sparse matrix-vector products from a canonical Arrow CSR batch extension.
#[pyfunction(name = "arrow_sparse_batch_matvec")]
pub fn sparse_batch_matvec_arrow(
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
    vectors_field: PyArrowType<Field>,
    vectors: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrices_arr = array_data_to_struct(matrices.0)?;
    let vectors_arr = array_data_to_struct(vectors.0)?;
    match sparse_value_type(&matrices_arr, "values")? {
        DataType::Float32 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matvec_csr_batch_extension::<Float32Type>(
                &field.0,
                &matrices_arr,
                &vectors_field.0,
                &vectors_arr,
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matvec_csr_batch_extension::<Float64Type>(
                &field.0,
                &matrices_arr,
                &vectors_field.0,
                &vectors_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected sparse Arrow batch values with float32 or float64 dtype",
        )),
    }
}

/// Compute row-wise sparse-dense matrix products from a canonical Arrow CSR batch extension.
#[pyfunction(name = "arrow_sparse_batch_matmat_dense")]
pub fn sparse_batch_matmat_dense_arrow(
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrices_arr = array_data_to_struct(matrices.0)?;
    let right_arr = array_data_to_struct(right.0)?;
    match sparse_value_type(&matrices_arr, "values")? {
        DataType::Float32 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matmat_dense_csr_batch_extension::<Float32Type>(
                &field.0,
                &matrices_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matmat_dense_csr_batch_extension::<Float64Type>(
                &field.0,
                &matrices_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected sparse Arrow batch values with float32 or float64 dtype",
        )),
    }
}

/// Transpose each sparse matrix in a canonical Arrow CSR batch extension.
#[pyfunction(name = "arrow_sparse_batch_transpose")]
pub fn sparse_batch_transpose_arrow(
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrices_arr = array_data_to_struct(matrices.0)?;
    match sparse_value_type(&matrices_arr, "values")? {
        DataType::Float32 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::transpose_csr_batch_extension::<Float32Type>(
                &field.0,
                &matrices_arr,
            )
            .map_err(to_py_err)?,
        )),
        DataType::Float64 => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::transpose_csr_batch_extension::<Float64Type>(
                &field.0,
                &matrices_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected sparse Arrow batch values with float32 or float64 dtype",
        )),
    }
}

/// Compute row-wise sparse-sparse products from canonical Arrow CSR batch extensions.
#[pyfunction(name = "arrow_sparse_batch_matmat_sparse")]
pub fn sparse_batch_matmat_sparse_arrow(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let left_arr = array_data_to_struct(left.0)?;
    let right_arr = array_data_to_struct(right.0)?;
    match (sparse_value_type(&left_arr, "values")?, sparse_value_type(&right_arr, "values")?) {
        (DataType::Float32, DataType::Float32) => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matmat_sparse_csr_batch_extension::<Float32Type>(
                &left_field.0,
                &left_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?,
        )),
        (DataType::Float64, DataType::Float64) => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::sparse::matmat_sparse_csr_batch_extension::<Float64Type>(
                &left_field.0,
                &left_arr,
                &right_field.0,
                &right_arr,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute the Hermitian dot product of two complex PyArrow vectors.
#[pyfunction(name = "arrow_dot_hermitian")]
pub fn dot_hermitian(
    py: Python<'_>,
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<Py<PyAny>> {
    let left_arr = array_data_to_fixed_size_list(left.0)?;
    let right_arr = array_data_to_fixed_size_list(right.0)?;
    Ok(utils::py_complex(
        py,
        nabled::arrow::vector::dot_hermitian(&left_field.0, &left_arr, &right_field.0, &right_arr)
            .map_err(to_py_err)?,
    ))
}

/// Compute the L2 norm of a complex PyArrow vector.
#[pyfunction(name = "arrow_l2_norm_complex")]
pub fn l2_norm_complex(field: PyArrowType<Field>, vector: PyArrowType<ArrayData>) -> PyResult<f64> {
    let vector_arr = array_data_to_fixed_size_list(vector.0)?;
    nabled::arrow::vector::l2_norm_complex(&field.0, &vector_arr).map_err(to_py_err)
}

/// Compute the cosine similarity of two complex PyArrow vectors.
#[pyfunction(name = "arrow_cosine_similarity_complex")]
pub fn cosine_similarity_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let left_arr = array_data_to_fixed_size_list(left.0)?;
    let right_arr = array_data_to_fixed_size_list(right.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::vector::cosine_similarity_complex(
            &left_field.0,
            &left_arr,
            &right_field.0,
            &right_arr,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute row-wise Hermitian dot products for complex Arrow row batches.
#[pyfunction(name = "arrow_batched_dot_hermitian")]
pub fn batched_dot_hermitian(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let left_arr = array_data_to_fixed_size_list(left.0)?;
    let right_arr = array_data_to_fixed_size_list(right.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::vector::batched_dot_hermitian(&left_arr, &right_arr).map_err(to_py_err)?,
    ))
}

/// Compute row-wise complex-vector norms for Arrow row batches.
#[pyfunction(name = "arrow_batched_l2_norm_complex")]
pub fn batched_l2_norm_complex(rows: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let rows_arr = array_data_to_fixed_size_list(rows.0)?;
    Ok(primitive_array_into_pyarrow::<Float64Type>(
        nabled::arrow::vector::batched_l2_norm_complex(&rows_arr).map_err(to_py_err)?,
    ))
}

/// Compute row-wise complex cosine similarities for Arrow row batches.
#[pyfunction(name = "arrow_batched_cosine_similarity_complex")]
pub fn batched_cosine_similarity_complex(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let left_arr = array_data_to_fixed_size_list(left.0)?;
    let right_arr = array_data_to_fixed_size_list(right.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::vector::batched_cosine_similarity_complex(&left_arr, &right_arr)
            .map_err(to_py_err)?,
    ))
}

/// Normalize each row of a complex Arrow dense matrix.
#[pyfunction(name = "arrow_batched_normalize_complex")]
pub fn batched_normalize_complex(rows: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let rows_arr = array_data_to_fixed_size_list(rows.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::vector::batched_normalize_complex(&rows_arr).map_err(to_py_err)?,
    ))
}

/// Compute a complex matrix-vector product directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_matvec_complex")]
pub fn matvec_complex(
    matrix: PyArrowType<ArrayData>,
    vector_field: PyArrowType<Field>,
    vector: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let vector_arr = array_data_to_fixed_size_list(vector.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::matrix::matvec_complex(&matrix_arr, &vector_field.0, &vector_arr)
            .map_err(to_py_err)?,
    ))
}

/// Compute a complex matrix-matrix product directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_matmat_complex")]
pub fn matmat_complex(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let left_arr = array_data_to_fixed_size_list(left.0)?;
    let right_arr = array_data_to_fixed_size_list(right.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix::matmat_complex(&left_arr, &right_arr).map_err(to_py_err)?,
    ))
}

/// Compute complex column means directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_column_means_complex")]
pub fn column_means_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::stats::column_means_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Center complex columns directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_center_columns_complex")]
pub fn center_columns_complex(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::stats::center_columns_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute a complex covariance matrix directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_covariance_matrix_complex")]
pub fn covariance_matrix_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::stats::covariance_matrix_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute a complex correlation matrix directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_correlation_matrix_complex")]
pub fn correlation_matrix_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::stats::correlation_matrix_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute modified Gram-Schmidt orthogonalization directly from real PyArrow dense inputs.
#[pyfunction(name = "arrow_gram_schmidt")]
pub fn gram_schmidt(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::orthogonalization::gram_schmidt_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::orthogonalization::gram_schmidt_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute modified Gram-Schmidt orthogonalization directly from complex PyArrow dense inputs.
#[pyfunction(name = "arrow_gram_schmidt_complex")]
pub fn gram_schmidt_complex(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::orthogonalization::gram_schmidt_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute classical Gram-Schmidt orthogonalization directly from real PyArrow dense inputs.
#[pyfunction(name = "arrow_gram_schmidt_classic")]
pub fn gram_schmidt_classic(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::orthogonalization::gram_schmidt_classic_f32(&matrix_arr)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::orthogonalization::gram_schmidt_classic_f64(&matrix_arr)
                .map_err(to_py_err)?,
        )),
    }
}

/// Solve a lower-triangular real system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_lower")]
pub fn solve_lower(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::triangular::solve_lower_f32(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::triangular::solve_lower_f64(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a lower-triangular complex system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_lower_complex")]
pub fn solve_lower_complex(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::triangular::solve_lower_complex(&matrix_arr, &rhs_field.0, &rhs_arr)
            .map_err(to_py_err)?,
    ))
}

/// Solve an upper-triangular real system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_upper")]
pub fn solve_upper(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::triangular::solve_upper_f32(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::triangular::solve_upper_f64(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve an upper-triangular complex system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_upper_complex")]
pub fn solve_upper_complex(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::triangular::solve_upper_complex(&matrix_arr, &rhs_field.0, &rhs_arr)
            .map_err(to_py_err)?,
    ))
}

/// Solve a lower-triangular matrix-RHS system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_lower_matrix")]
pub fn solve_lower_matrix(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(matrix.0)?,
        array_data_to_real_fixed_size_list(rhs.0)?,
    ) {
        (RealFixedSizeListArray::F32(matrix_arr), RealFixedSizeListArray::F32(rhs_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::triangular::solve_lower_matrix_f32(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealFixedSizeListArray::F64(rhs_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::triangular::solve_lower_matrix_f64(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve an upper-triangular matrix-RHS system directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_solve_upper_matrix")]
pub fn solve_upper_matrix(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(matrix.0)?,
        array_data_to_real_fixed_size_list(rhs.0)?,
    ) {
        (RealFixedSizeListArray::F32(matrix_arr), RealFixedSizeListArray::F32(rhs_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::triangular::solve_upper_matrix_f32(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealFixedSizeListArray::F64(rhs_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::triangular::solve_upper_matrix_f64(&matrix_arr, &rhs_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Compute batched QR decomposition directly from Arrow fixed-shape tensor input.
#[pyfunction(name = "arrow_batched_qr", signature = (field, matrices, rank_tolerance=None, max_iterations=None))]
pub fn batched_qr(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, usize)>> {
    let matrices_arr =
        fixed_size_list_with_non_null_item(&array_data_to_fixed_size_list(matrices.0)?);
    let field = field_with_array_storage(&field.0, &matrices_arr);
    match matrices_arr.value_type() {
        arrow_schema::DataType::Float32 => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            Ok(nabled::arrow::batched::qr_f32(&field, &matrices_arr, &config)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| qr_result_tuple(py, result))
                .collect())
        }
        arrow_schema::DataType::Float64 => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            Ok(nabled::arrow::batched::qr_f64(&field, &matrices_arr, &config)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| qr_result_tuple(py, result))
                .collect())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected fixed-shape tensor with float32 or float64 values",
        )),
    }
}

/// Compute batched SVD directly from Arrow fixed-shape tensor input.
#[pyfunction(name = "arrow_batched_svd")]
pub fn batched_svd(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>> {
    let matrices_arr =
        fixed_size_list_with_non_null_item(&array_data_to_fixed_size_list(matrices.0)?);
    let field = field_with_array_storage(&field.0, &matrices_arr);
    match matrices_arr.value_type() {
        arrow_schema::DataType::Float32 => {
            Ok(nabled::arrow::batched::svd_f32(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| {
                    (
                        utils::pyarray2_from_owned(py, result.u),
                        utils::pyarray1_from_owned(py, result.singular_values),
                        utils::pyarray2_from_owned(py, result.vt),
                    )
                })
                .collect())
        }
        arrow_schema::DataType::Float64 => {
            Ok(nabled::arrow::batched::svd_f64(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| {
                    (
                        utils::pyarray2_from_owned(py, result.u),
                        utils::pyarray1_from_owned(py, result.singular_values),
                        utils::pyarray2_from_owned(py, result.vt),
                    )
                })
                .collect())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected fixed-shape tensor with float32 or float64 values",
        )),
    }
}

/// Compute batched LU decomposition directly from Arrow fixed-shape tensor input.
#[pyfunction(name = "arrow_batched_lu")]
pub fn batched_lu(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i8)>> {
    let matrices_arr =
        fixed_size_list_with_non_null_item(&array_data_to_fixed_size_list(matrices.0)?);
    let field = field_with_array_storage(&field.0, &matrices_arr);
    match matrices_arr.value_type() {
        arrow_schema::DataType::Float32 => {
            Ok(nabled::arrow::batched::lu_f32_with_metadata(&field, &matrices_arr)
                .map_err(to_py_err)?
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
        arrow_schema::DataType::Float64 => {
            Ok(nabled::arrow::batched::lu_f64_with_metadata(&field, &matrices_arr)
                .map_err(to_py_err)?
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
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected fixed-shape tensor with float32 or float64 values",
        )),
    }
}

/// Compute batched Cholesky decomposition directly from Arrow fixed-shape tensor input.
#[pyfunction(name = "arrow_batched_cholesky")]
pub fn batched_cholesky(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
) -> PyResult<Vec<Py<PyAny>>> {
    let matrices_arr =
        fixed_size_list_with_non_null_item(&array_data_to_fixed_size_list(matrices.0)?);
    let field = field_with_array_storage(&field.0, &matrices_arr);
    match matrices_arr.value_type() {
        arrow_schema::DataType::Float32 => {
            Ok(nabled::arrow::batched::cholesky_f32(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| utils::pyarray2_from_owned(py, result.l))
                .collect())
        }
        arrow_schema::DataType::Float64 => {
            Ok(nabled::arrow::batched::cholesky_f64(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| utils::pyarray2_from_owned(py, result.l))
                .collect())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected fixed-shape tensor with float32 or float64 values",
        )),
    }
}

/// Compute batched symmetric eigen decomposition directly from Arrow fixed-shape tensor input.
#[pyfunction(name = "arrow_batched_symmetric_eigen")]
pub fn batched_symmetric_eigen(
    py: Python<'_>,
    field: PyArrowType<Field>,
    matrices: PyArrowType<ArrayData>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
    let matrices_arr =
        fixed_size_list_with_non_null_item(&array_data_to_fixed_size_list(matrices.0)?);
    let field = field_with_array_storage(&field.0, &matrices_arr);
    match matrices_arr.value_type() {
        arrow_schema::DataType::Float32 => {
            Ok(nabled::arrow::batched::symmetric_eigen_f32(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| {
                    (
                        utils::pyarray1_from_owned(py, result.eigenvalues),
                        utils::pyarray2_from_owned(py, result.eigenvectors),
                    )
                })
                .collect())
        }
        arrow_schema::DataType::Float64 => {
            Ok(nabled::arrow::batched::symmetric_eigen_f64(&field, &matrices_arr)
                .map_err(to_py_err)?
                .into_iter()
                .map(|result| {
                    (
                        utils::pyarray1_from_owned(py, result.eigenvalues),
                        utils::pyarray2_from_owned(py, result.eigenvectors),
                    )
                })
                .collect())
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "expected fixed-shape tensor with float32 or float64 values",
        )),
    }
}

/// Compute complex SVD directly from PyArrow dense input.
#[pyfunction(name = "arrow_svd_decompose_complex")]
pub fn svd_decompose_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::svd::decompose_complex(&matrix_arr).map_err(to_py_err)?;
    Ok((
        utils::pyarray2_from_owned(py, result.u),
        utils::pyarray1_from_owned(py, result.singular_values),
        utils::pyarray2_from_owned(py, result.vt),
    ))
}

/// Compute complex QR directly from PyArrow dense input.
#[pyfunction(name = "arrow_qr_decompose_complex")]
pub fn qr_decompose_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let config = nabled_linalg::qr::QRConfig::<f64>::default();
    let result = nabled::arrow::qr::decompose_complex(&matrix_arr, &config).map_err(to_py_err)?;
    Ok((
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        result.rank,
    ))
}

/// Solve a complex linear system directly from PyArrow dense input.
#[pyfunction(name = "arrow_lu_solve_complex")]
pub fn lu_solve_complex(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::lu::solve_complex(&matrix_arr, &rhs_field.0, &rhs_arr).map_err(to_py_err)?,
    ))
}

/// Compute the complex inverse of a PyArrow dense matrix.
#[pyfunction(name = "arrow_lu_inverse_complex")]
pub fn lu_inverse_complex(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::lu::inverse_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute the complex determinant of a PyArrow dense matrix.
#[pyfunction(name = "arrow_lu_determinant_complex")]
pub fn lu_determinant_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<Py<PyAny>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(utils::py_complex(
        py,
        nabled::arrow::lu::determinant_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute complex Cholesky decomposition directly from PyArrow dense input.
#[pyfunction(name = "arrow_cholesky_decompose_complex")]
pub fn cholesky_decompose_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<Py<PyAny>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::cholesky::decompose_complex(&matrix_arr).map_err(to_py_err)?;
    Ok(utils::pyarray2_from_owned(py, result.l))
}

/// Solve a complex Hermitian system directly from PyArrow dense input.
#[pyfunction(name = "arrow_cholesky_solve_complex")]
pub fn cholesky_solve_complex(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    Ok(extension_result_into_pyarrow(
        nabled::arrow::cholesky::solve_complex(&matrix_arr, &rhs_field.0, &rhs_arr)
            .map_err(to_py_err)?,
    ))
}

/// Compute the complex inverse of a Hermitian positive-definite PyArrow matrix.
#[pyfunction(name = "arrow_cholesky_inverse_complex")]
pub fn cholesky_inverse_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::cholesky::inverse_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute complex non-symmetric eigen decomposition directly from PyArrow dense input.
#[pyfunction(name = "arrow_eigen_nonsymmetric_complex")]
pub fn eigen_nonsymmetric_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::eigen::nonsymmetric_complex(&matrix_arr).map_err(to_py_err)?;
    Ok((
        utils::pyarray1_from_owned(py, result.eigenvalues),
        utils::pyarray2_from_owned(py, result.schur_vectors),
    ))
}

/// Compute complex Schur decomposition directly from PyArrow dense input.
#[pyfunction(name = "arrow_schur_compute_complex")]
pub fn schur_compute_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::schur::compute_complex(&matrix_arr).map_err(to_py_err)?;
    Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
}

/// Compute complex polar decomposition directly from PyArrow dense input.
#[pyfunction(name = "arrow_polar_compute_complex")]
pub fn polar_compute_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::polar::compute_complex(&matrix_arr).map_err(to_py_err)?;
    Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
}

/// Compute the complex matrix exponential directly from PyArrow dense input.
#[pyfunction(name = "arrow_matrix_exp_complex", signature = (matrix, max_terms=None, tolerance=None))]
pub fn matrix_exp_complex(
    matrix: PyArrowType<ArrayData>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::exp_complex(
            &matrix_arr,
            max_terms.unwrap_or(DEFAULT_MAX_TERMS),
            tolerance.unwrap_or(DEFAULT_TOLERANCE),
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute the complex matrix exponential via eigen decomposition.
#[pyfunction(name = "arrow_matrix_exp_eigen_complex")]
pub fn matrix_exp_eigen_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::exp_eigen_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute the complex matrix logarithm via eigen decomposition.
#[pyfunction(name = "arrow_matrix_log_eigen_complex")]
pub fn matrix_log_eigen_complex(
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::log_eigen_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute the complex matrix logarithm via SVD.
#[pyfunction(name = "arrow_matrix_log_svd_complex")]
pub fn matrix_log_svd_complex(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::log_svd_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute the complex matrix power.
#[pyfunction(name = "arrow_matrix_power_complex")]
pub fn matrix_power_complex(
    matrix: PyArrowType<ArrayData>,
    power: f64,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::power_complex(&matrix_arr, power).map_err(to_py_err)?,
    ))
}

/// Compute the complex matrix sign.
#[pyfunction(name = "arrow_matrix_sign_complex")]
pub fn matrix_sign_complex(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::matrix_functions::sign_complex(&matrix_arr).map_err(to_py_err)?,
    ))
}

/// Compute complex PCA directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_compute_pca_complex", signature = (matrix, n_components=None))]
pub fn compute_pca_complex(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
    n_components: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let result =
        nabled::arrow::pca::compute_complex(&matrix_arr, n_components).map_err(to_py_err)?;
    Ok((
        utils::pyarray2_from_owned(py, result.components),
        utils::pyarray1_from_owned(py, result.explained_variance),
        utils::pyarray1_from_owned(py, result.explained_variance_ratio),
        utils::pyarray1_from_owned(py, result.mean),
        utils::pyarray2_from_owned(py, result.scores),
    ))
}

/// Project complex Arrow dense data into PCA score space.
#[pyfunction(name = "arrow_pca_transform_complex")]
pub fn pca_transform_complex(
    matrix: PyArrowType<ArrayData>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
) -> PyResult<PyArrowType<ArrayData>> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    match (utils::numeric_array2(components, "components")?, utils::numeric_array1(mean, "mean")?) {
        (
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::transform_complex_from_components_view(
                &matrix_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "components and mean must both have dtype complex128",
        )),
    }
}

/// Reconstruct complex Arrow dense data from PCA scores.
#[pyfunction(name = "arrow_pca_inverse_transform_complex")]
pub fn pca_inverse_transform_complex(
    scores: PyArrowType<ArrayData>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
) -> PyResult<PyArrowType<ArrayData>> {
    let scores_arr = array_data_to_fixed_size_list(scores.0)?;
    match (utils::numeric_array2(components, "components")?, utils::numeric_array1(mean, "mean")?) {
        (
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::pca::inverse_transform_complex_from_components_view(
                &scores_arr,
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "components and mean must both have dtype complex128",
        )),
    }
}

/// Solve complex linear regression directly from PyArrow dense inputs.
#[pyfunction(name = "arrow_linear_regression_complex", signature = (x, y_field, y, add_intercept=true))]
pub fn linear_regression_complex(
    py: Python<'_>,
    x: PyArrowType<ArrayData>,
    y_field: PyArrowType<Field>,
    y: PyArrowType<ArrayData>,
    add_intercept: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    let x_arr = array_data_to_fixed_size_list(x.0)?;
    let y_arr = array_data_to_fixed_size_list(y.0)?;
    let result = nabled::arrow::regression::linear_regression_complex(
        &x_arr,
        &y_field.0,
        &y_arr,
        add_intercept,
    )
    .map_err(to_py_err)?;
    Ok((
        utils::pyarray1_from_owned(py, result.coefficients),
        utils::pyarray1_from_owned(py, result.fitted_values),
        utils::pyarray1_from_owned(py, result.residuals),
        result.r_squared,
    ))
}

/// Solve an SPD system directly from PyArrow dense inputs via conjugate gradient.
#[pyfunction(name = "arrow_conjugate_gradient", signature = (matrix, rhs, tolerance=None, max_iterations=None))]
pub fn conjugate_gradient(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            let config = iterative_config_f32(tolerance, max_iterations)?;
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::iterative::conjugate_gradient_f32(&matrix_arr, &rhs_arr, &config)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            let config = iterative_config_f64(tolerance, max_iterations);
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::iterative::conjugate_gradient_f64(&matrix_arr, &rhs_arr, &config)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a complex SPD system directly from canonical complex PyArrow carriers.
#[pyfunction(name = "arrow_conjugate_gradient_complex", signature = (matrix, rhs_field, rhs, tolerance=None, max_iterations=None))]
pub fn conjugate_gradient_complex_arrow(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    let config = iterative_config_f64(tolerance, max_iterations);
    Ok(extension_result_into_pyarrow(
        nabled::arrow::iterative::conjugate_gradient_complex(
            &matrix_arr,
            &rhs_field.0,
            &rhs_arr,
            &config,
        )
        .map_err(to_py_err)?,
    ))
}

/// Solve a general linear system directly from PyArrow dense inputs via GMRES.
#[pyfunction(name = "arrow_gmres", signature = (matrix, rhs, tolerance=None, max_iterations=None))]
pub fn gmres_arrow(
    matrix: PyArrowType<ArrayData>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(rhs.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(rhs_arr)) => {
            let config = iterative_config_f32(tolerance, max_iterations)?;
            Ok(primitive_array_into_pyarrow::<Float32Type>(
                nabled::arrow::iterative::gmres_f32(&matrix_arr, &rhs_arr, &config)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(rhs_arr)) => {
            let config = iterative_config_f64(tolerance, max_iterations);
            Ok(primitive_array_into_pyarrow::<Float64Type>(
                nabled::arrow::iterative::gmres_f64(&matrix_arr, &rhs_arr, &config)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve a general complex linear system directly from canonical complex PyArrow carriers.
#[pyfunction(name = "arrow_gmres_complex", signature = (matrix, rhs_field, rhs, tolerance=None, max_iterations=None))]
pub fn gmres_complex_arrow(
    matrix: PyArrowType<ArrayData>,
    rhs_field: PyArrowType<Field>,
    rhs: PyArrowType<ArrayData>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let matrix_arr = array_data_to_fixed_size_list(matrix.0)?;
    let rhs_arr = array_data_to_fixed_size_list(rhs.0)?;
    let config = iterative_config_f64(tolerance, max_iterations);
    Ok(extension_result_into_pyarrow(
        nabled::arrow::iterative::gmres_complex(&matrix_arr, &rhs_field.0, &rhs_arr, &config)
            .map_err(to_py_err)?,
    ))
}

/// Compute a numerical Jacobian via forward differences from a PyArrow dense vector.
#[pyfunction(name = "arrow_numerical_jacobian", signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian_arrow(
    function: &Bound<'_, PyAny>,
    x: PyArrowType<ArrayData>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(x.0)? {
        RealPrimitiveArray::F32(x_arr) => {
            let config = jacobian_config_f32(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f32>| -> Result<ndarray::Array1<f32>, _> {
                match call_vector_function_arrow_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_jacobian(&wrapped, &x_arr, &config),
            )?))
        }
        RealPrimitiveArray::F64(x_arr) => {
            let config = jacobian_config_f64(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, _> {
                match call_vector_function_arrow_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_jacobian(&wrapped, &x_arr, &config),
            )?))
        }
    }
}

/// Compute a numerical Jacobian via central differences from a PyArrow dense vector.
#[pyfunction(name = "arrow_numerical_jacobian_central", signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian_central_arrow(
    function: &Bound<'_, PyAny>,
    x: PyArrowType<ArrayData>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(x.0)? {
        RealPrimitiveArray::F32(x_arr) => {
            let config = jacobian_config_f32(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f32>| -> Result<ndarray::Array1<f32>, _> {
                match call_vector_function_arrow_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_jacobian_central(&wrapped, &x_arr, &config),
            )?))
        }
        RealPrimitiveArray::F64(x_arr) => {
            let config = jacobian_config_f64(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, _> {
                match call_vector_function_arrow_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_jacobian_central(&wrapped, &x_arr, &config),
            )?))
        }
    }
}

/// Compute a numerical gradient from a scalar-valued Python callback over a PyArrow vector.
#[pyfunction(name = "arrow_numerical_gradient", signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_gradient_arrow(
    function: &Bound<'_, PyAny>,
    x: PyArrowType<ArrayData>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(x.0)? {
        RealPrimitiveArray::F32(x_arr) => {
            let config = jacobian_config_f32(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f32>| -> Result<f32, _> {
                match call_scalar_function_arrow_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_gradient(&wrapped, &x_arr, &config),
            )?))
        }
        RealPrimitiveArray::F64(x_arr) => {
            let config = jacobian_config_f64(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, _> {
                match call_scalar_function_arrow_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_gradient(&wrapped, &x_arr, &config),
            )?))
        }
    }
}

/// Compute a numerical Hessian from a scalar-valued Python callback over a PyArrow vector.
#[pyfunction(name = "arrow_numerical_hessian", signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_hessian_arrow(
    function: &Bound<'_, PyAny>,
    x: PyArrowType<ArrayData>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(x.0)? {
        RealPrimitiveArray::F32(x_arr) => {
            let config = jacobian_config_f32(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f32>| -> Result<f32, _> {
                match call_scalar_function_arrow_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_hessian(&wrapped, &x_arr, &config),
            )?))
        }
        RealPrimitiveArray::F64(x_arr) => {
            let config = jacobian_config_f64(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, _> {
                match call_scalar_function_arrow_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(nabled_ml::jacobian::JacobianError::FunctionError(
                            "python callback raised".to_string(),
                        ))
                    }
                }
            };
            Ok(fixed_size_list_into_pyarrow(map_callback_error(
                &callback_error,
                nabled::arrow::jacobian::numerical_hessian(&wrapped, &x_arr, &config),
            )?))
        }
    }
}

/// Perform Armijo backtracking line search from Arrow dense vectors.
#[pyfunction(name = "arrow_backtracking_line_search", signature = (point, direction, objective, gradient, initial_step=None, contraction=None, sufficient_decrease=None, max_iterations=None))]
pub fn backtracking_line_search_arrow(
    point: PyArrowType<ArrayData>,
    direction: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<f64> {
    match (array_data_to_real_primitive(point.0)?, array_data_to_real_primitive(direction.0)?) {
        (RealPrimitiveArray::F32(point_arr), RealPrimitiveArray::F32(direction_arr)) => {
            let config = line_search_config_f32(
                initial_step,
                contraction,
                sufficient_decrease,
                max_iterations,
            )?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            map_callback_error(
                &callback_error,
                nabled::arrow::optimization::backtracking_line_search(
                    &point_arr,
                    &direction_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )
            .map(f64::from)
        }
        (RealPrimitiveArray::F64(point_arr), RealPrimitiveArray::F64(direction_arr)) => {
            let config = line_search_config_f64(
                initial_step,
                contraction,
                sufficient_decrease,
                max_iterations,
            );
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            map_callback_error(
                &callback_error,
                nabled::arrow::optimization::backtracking_line_search(
                    &point_arr,
                    &direction_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )
        }
        _ => Err(utils::matching_real_dtype_error(&["point", "direction"])),
    }
}

/// Perform Armijo backtracking line search from complex Arrow dense vectors.
#[pyfunction(name = "arrow_backtracking_line_search_complex", signature = (point_field, point, direction_field, direction, objective, gradient, initial_step=None, contraction=None, sufficient_decrease=None, max_iterations=None))]
pub fn backtracking_line_search_complex_arrow(
    point_field: PyArrowType<Field>,
    point: PyArrowType<ArrayData>,
    direction_field: PyArrowType<Field>,
    direction: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<f64> {
    let point_arr = array_data_to_fixed_size_list(point.0)?;
    let direction_arr = array_data_to_fixed_size_list(direction.0)?;
    let config =
        line_search_config_f64(initial_step, contraction, sufficient_decrease, max_iterations);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    map_callback_error(
        &callback_error,
        nabled::arrow::optimization::backtracking_line_search_complex(
            &point_field.0,
            &point_arr,
            &direction_field.0,
            &direction_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )
}

/// Minimize an objective with gradient descent from an Arrow dense vector.
#[pyfunction(name = "arrow_gradient_descent", signature = (initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn gradient_descent_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config = sgd_config_f32(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::gradient_descent(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config = sgd_config_f64(learning_rate, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::gradient_descent(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
    }
}

/// Minimize an objective with gradient descent from a complex Arrow dense vector.
#[pyfunction(name = "arrow_gradient_descent_complex", signature = (initial_field, initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn gradient_descent_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = sgd_config_f64(learning_rate, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::gradient_descent_complex(
            &initial_field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Minimize an objective with Adam from an Arrow dense vector.
#[pyfunction(name = "arrow_adam", signature = (initial, objective, gradient, learning_rate=None, beta1=None, beta2=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn adam_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config =
                adam_config_f32(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::adam(&initial_arr, objective_fn, gradient_fn, &config),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config =
                adam_config_f64(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::adam(&initial_arr, objective_fn, gradient_fn, &config),
            )?))
        }
    }
}

/// Minimize an objective with Adam from a complex Arrow dense vector.
#[pyfunction(name = "arrow_adam_complex", signature = (initial_field, initial, objective, gradient, learning_rate=None, beta1=None, beta2=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn adam_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = adam_config_f64(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::adam_complex(
            &initial_field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Minimize an objective with momentum descent from an Arrow dense vector.
#[pyfunction(name = "arrow_momentum_descent", signature = (initial, objective, gradient, learning_rate=None, momentum=None, max_iterations=None, tolerance=None))]
pub fn momentum_descent_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config = momentum_config_f32(learning_rate, momentum, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::momentum_descent(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config = momentum_config_f64(learning_rate, momentum, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::momentum_descent(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
    }
}

/// Minimize an objective with momentum descent from a complex Arrow dense vector.
#[pyfunction(name = "arrow_momentum_descent_complex", signature = (initial_field, initial, objective, gradient, learning_rate=None, momentum=None, max_iterations=None, tolerance=None))]
pub fn momentum_descent_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = momentum_config_f64(learning_rate, momentum, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::momentum_descent_complex(
            &initial_field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Minimize an objective with RMSProp from an Arrow dense vector.
#[pyfunction(name = "arrow_rmsprop", signature = (initial, objective, gradient, learning_rate=None, rho=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn rmsprop_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config =
                rmsprop_config_f32(learning_rate, rho, epsilon, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::rmsprop(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config = rmsprop_config_f64(learning_rate, rho, epsilon, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::rmsprop(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
    }
}

/// Minimize an objective with RMSProp from a complex Arrow dense vector.
#[pyfunction(name = "arrow_rmsprop_complex", signature = (initial_field, initial, objective, gradient, learning_rate=None, rho=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn rmsprop_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = rmsprop_config_f64(learning_rate, rho, epsilon, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::rmsprop_complex(
            &initial_field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Minimize an objective with projected gradient descent and box constraints from Arrow vectors.
#[pyfunction(name = "arrow_projected_gradient_descent_box", signature = (initial, objective, gradient, lower_bounds, upper_bounds, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn projected_gradient_descent_box_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    lower_bounds: PyArrowType<ArrayData>,
    upper_bounds: PyArrowType<ArrayData>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_primitive(initial.0)?,
        array_data_to_real_primitive(lower_bounds.0)?,
        array_data_to_real_primitive(upper_bounds.0)?,
    ) {
        (
            RealPrimitiveArray::F32(initial_arr),
            RealPrimitiveArray::F32(lower_arr),
            RealPrimitiveArray::F32(upper_arr),
        ) => {
            let config = projected_gradient_config_f32(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::projected_gradient_descent_box(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &lower_arr,
                    &upper_arr,
                    &config,
                ),
            )?))
        }
        (
            RealPrimitiveArray::F64(initial_arr),
            RealPrimitiveArray::F64(lower_arr),
            RealPrimitiveArray::F64(upper_arr),
        ) => {
            let config = projected_gradient_config_f64(learning_rate, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::projected_gradient_descent_box(
                    &initial_arr,
                    objective_fn,
                    gradient_fn,
                    &lower_arr,
                    &upper_arr,
                    &config,
                ),
            )?))
        }
        _ => Err(utils::matching_real_dtype_error(&["initial", "lower_bounds", "upper_bounds"])),
    }
}

/// Minimize an objective with projected gradient descent and box constraints from complex Arrow
/// vectors.
#[pyfunction(name = "arrow_projected_gradient_descent_box_complex", signature = (field, initial, objective, gradient, lower_bounds, upper_bounds, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn projected_gradient_descent_box_complex_arrow(
    field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    lower_bounds: PyArrowType<ArrayData>,
    upper_bounds: PyArrowType<ArrayData>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let lower_arr = array_data_to_fixed_size_list(lower_bounds.0)?;
    let upper_arr = array_data_to_fixed_size_list(upper_bounds.0)?;
    let config = projected_gradient_config_f64(learning_rate, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::projected_gradient_descent_box_complex(
            &field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &lower_arr,
            &upper_arr,
            &config,
        ),
    )?))
}

/// Run stochastic gradient descent from an Arrow dense vector.
#[pyfunction(name = "arrow_stochastic_gradient_descent", signature = (initial, stochastic_gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn stochastic_gradient_descent_arrow(
    initial: PyArrowType<ArrayData>,
    stochastic_gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config = sgd_config_f32(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let gradient_fn =
                |input: &ndarray::Array1<f32>, iteration: usize| -> ndarray::Array1<f32> {
                    match call_vector_function_arrow_f32_with_iteration(
                        stochastic_gradient,
                        input,
                        iteration,
                    ) {
                        Ok(value) => value,
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            ndarray::Array1::from_elem(input.len(), f32::NAN)
                        }
                    }
                };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::stochastic_gradient_descent(
                    &initial_arr,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config = sgd_config_f64(learning_rate, max_iterations, tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let gradient_fn =
                |input: &ndarray::Array1<f64>, iteration: usize| -> ndarray::Array1<f64> {
                    match call_vector_function_arrow_f64_with_iteration(
                        stochastic_gradient,
                        input,
                        iteration,
                    ) {
                        Ok(value) => value,
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            ndarray::Array1::from_elem(input.len(), f64::NAN)
                        }
                    }
                };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::stochastic_gradient_descent(
                    &initial_arr,
                    gradient_fn,
                    &config,
                ),
            )?))
        }
    }
}

/// Run stochastic gradient descent from a complex Arrow dense vector.
#[pyfunction(name = "arrow_stochastic_gradient_descent_complex", signature = (initial_field, initial, stochastic_gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn stochastic_gradient_descent_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    stochastic_gradient: &Bound<'_, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = sgd_config_f64(learning_rate, max_iterations, tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let gradient_fn =
        |input: &ndarray::Array1<Complex64>, iteration: usize| -> ndarray::Array1<Complex64> {
            match call_vector_function_arrow_complex_with_iteration(
                stochastic_gradient,
                input,
                iteration,
            ) {
                Ok(value) => value,
                Err(err) => {
                    *callback_error.borrow_mut() = Some(err);
                    ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
                }
            }
        };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::stochastic_gradient_descent_complex(
            &initial_field.0,
            &initial_arr,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Minimize an objective with BFGS from an Arrow dense vector.
#[pyfunction(name = "arrow_bfgs", signature = (initial, objective, gradient, step_size=None, max_iterations=None, tolerance=None, curvature_tolerance=None))]
pub fn bfgs_arrow(
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_primitive(initial.0)? {
        RealPrimitiveArray::F32(initial_arr) => {
            let config =
                bfgs_config_f32(step_size, max_iterations, tolerance, curvature_tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_arrow_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_arrow_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float32Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::bfgs(&initial_arr, objective_fn, gradient_fn, &config),
            )?))
        }
        RealPrimitiveArray::F64(initial_arr) => {
            let config = bfgs_config_f64(step_size, max_iterations, tolerance, curvature_tolerance);
            let callback_error = RefCell::<Option<PyErr>>::default();
            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_arrow_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_arrow_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };
            Ok(primitive_array_into_pyarrow::<Float64Type>(map_callback_error(
                &callback_error,
                nabled::arrow::optimization::bfgs(&initial_arr, objective_fn, gradient_fn, &config),
            )?))
        }
    }
}

/// Minimize an objective with BFGS from a complex Arrow dense vector.
#[pyfunction(name = "arrow_bfgs_complex", signature = (initial_field, initial, objective, gradient, step_size=None, max_iterations=None, tolerance=None, curvature_tolerance=None))]
pub fn bfgs_complex_arrow(
    initial_field: PyArrowType<Field>,
    initial: PyArrowType<ArrayData>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let initial_arr = array_data_to_fixed_size_list(initial.0)?;
    let config = bfgs_config_f64(step_size, max_iterations, tolerance, curvature_tolerance);
    let callback_error = RefCell::<Option<PyErr>>::default();
    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_arrow_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_arrow_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };
    Ok(extension_result_into_pyarrow(map_callback_error(
        &callback_error,
        nabled::arrow::optimization::bfgs_complex(
            &initial_field.0,
            &initial_arr,
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?))
}

/// Reduce the last axis of a real fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_sum_last_axis_fixed")]
pub fn tensor_sum_last_axis_fixed(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::sum_last_axis::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::sum_last_axis::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
    }
}

/// Reduce the last axis of a real variable-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_sum_last_axis_variable")]
pub fn tensor_sum_last_axis_variable(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let array = array_data_to_struct(array.0)?;
    match variable_shape_real_dtype(&field.0) {
        Some("f32") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::sum_last_axis_variable::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        Some("f64") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::sum_last_axis_variable::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected arrow.variable_shape_tensor with float32 or float64 values",
        )),
    }
}

/// Compute last-axis norms for a real fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_l2_norm_last_axis_fixed")]
pub fn tensor_l2_norm_last_axis_fixed(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::l2_norm_last_axis::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::l2_norm_last_axis::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
    }
}

/// Compute last-axis norms for a real variable-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_l2_norm_last_axis_variable")]
pub fn tensor_l2_norm_last_axis_variable(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let array = array_data_to_struct(array.0)?;
    match variable_shape_real_dtype(&field.0) {
        Some("f32") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::l2_norm_last_axis_variable::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        Some("f64") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::l2_norm_last_axis_variable::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected arrow.variable_shape_tensor with float32 or float64 values",
        )),
    }
}

/// Normalize a real fixed-shape Arrow tensor batch over its last axis.
#[pyfunction(name = "arrow_tensor_normalize_last_axis_fixed")]
pub fn tensor_normalize_last_axis_fixed(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::normalize_last_axis::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::normalize_last_axis::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
    }
}

/// Normalize a real variable-shape Arrow tensor batch over its last axis.
#[pyfunction(name = "arrow_tensor_normalize_last_axis_variable")]
pub fn tensor_normalize_last_axis_variable(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let array = array_data_to_struct(array.0)?;
    match variable_shape_real_dtype(&field.0) {
        Some("f32") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::normalize_last_axis_variable::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        Some("f64") => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::normalize_last_axis_variable::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        _ => Err(PyTypeError::new_err(
            "expected arrow.variable_shape_tensor with float32 or float64 values",
        )),
    }
}

/// Compute last-axis batched dot products for real fixed-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_batched_dot_last_axis_fixed")]
pub fn tensor_batched_dot_last_axis_fixed(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left), RealFixedSizeListArray::F32(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::batched_dot_last_axis::<Float32Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left), RealFixedSizeListArray::F64(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::batched_dot_last_axis::<Float64Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute last-axis batched dot products for real variable-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_batched_dot_last_axis_variable")]
pub fn tensor_batched_dot_last_axis_variable(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    let left = array_data_to_struct(left.0)?;
    let right = array_data_to_struct(right.0)?;
    match (variable_shape_real_dtype(&left_field.0), variable_shape_real_dtype(&right_field.0)) {
        (Some("f32"), Some("f32")) => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::batched_dot_last_axis_variable::<Float32Type>(
                &left_field.0,
                &left,
                &right_field.0,
                &right,
            )
            .map_err(to_py_err)?,
        )),
        (Some("f64"), Some("f64")) => Ok(struct_extension_result_into_pyarrow(
            nabled::arrow::tensor::batched_dot_last_axis_variable::<Float64Type>(
                &left_field.0,
                &left,
                &right_field.0,
                &right,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Reduce the last axis of a complex fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_sum_last_axis_fixed_complex")]
pub fn tensor_sum_last_axis_fixed_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::sum_last_axis_complex(
            &field.0,
            &array_data_to_fixed_size_list(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Reduce the last axis of a complex variable-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_sum_last_axis_variable_complex")]
pub fn tensor_sum_last_axis_variable_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(struct_extension_result_into_pyarrow(
        nabled::arrow::tensor::sum_last_axis_variable_complex(
            &field.0,
            &array_data_to_struct(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute last-axis norms for a complex fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_l2_norm_last_axis_fixed_complex")]
pub fn tensor_l2_norm_last_axis_fixed_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::l2_norm_last_axis_complex(
            &field.0,
            &array_data_to_fixed_size_list(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute last-axis norms for a complex variable-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_l2_norm_last_axis_variable_complex")]
pub fn tensor_l2_norm_last_axis_variable_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(struct_extension_result_into_pyarrow(
        nabled::arrow::tensor::l2_norm_last_axis_variable_complex(
            &field.0,
            &array_data_to_struct(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Normalize a complex fixed-shape Arrow tensor batch over its last axis.
#[pyfunction(name = "arrow_tensor_normalize_last_axis_fixed_complex")]
pub fn tensor_normalize_last_axis_fixed_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::normalize_last_axis_complex(
            &field.0,
            &array_data_to_fixed_size_list(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Normalize a complex variable-shape Arrow tensor batch over its last axis.
#[pyfunction(name = "arrow_tensor_normalize_last_axis_variable_complex")]
pub fn tensor_normalize_last_axis_variable_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(struct_extension_result_into_pyarrow(
        nabled::arrow::tensor::normalize_last_axis_variable_complex(
            &field.0,
            &array_data_to_struct(array.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute last-axis batched dot products for complex fixed-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_batched_dot_last_axis_fixed_complex")]
pub fn tensor_batched_dot_last_axis_fixed_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::batched_dot_last_axis_complex(
            &left_field.0,
            &array_data_to_fixed_size_list(left.0)?,
            &right_field.0,
            &array_data_to_fixed_size_list(right.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute last-axis batched dot products for complex variable-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_batched_dot_last_axis_variable_complex")]
pub fn tensor_batched_dot_last_axis_variable_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(struct_extension_result_into_pyarrow(
        nabled::arrow::tensor::batched_dot_last_axis_variable_complex(
            &left_field.0,
            &array_data_to_struct(left.0)?,
            &right_field.0,
            &array_data_to_struct(right.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Permute a real fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_permute_axes")]
pub fn tensor_permute_axes(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    permutation: Vec<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::permute_axes::<Float32Type>(&field.0, &array, &permutation)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(extension_result_into_pyarrow(
            nabled::arrow::tensor::permute_axes::<Float64Type>(&field.0, &array, &permutation)
                .map_err(to_py_err)?,
        )),
    }
}

/// Permute a complex fixed-shape Arrow tensor batch.
#[pyfunction(name = "arrow_tensor_permute_axes_complex")]
pub fn tensor_permute_axes_complex(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    permutation: Vec<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::permute_axes_complex(
            &field.0,
            &array_data_to_fixed_size_list(array.0)?,
            &permutation,
        )
        .map_err(to_py_err)?,
    ))
}

/// Contract two real fixed-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_contract_axes")]
pub fn tensor_contract_axes(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left), RealFixedSizeListArray::F32(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::contract_axes::<Float32Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                    &left_axes,
                    &right_axes,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left), RealFixedSizeListArray::F64(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::contract_axes::<Float64Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                    &left_axes,
                    &right_axes,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Contract two complex fixed-shape Arrow tensor batches.
#[pyfunction(name = "arrow_tensor_contract_axes_complex")]
pub fn tensor_contract_axes_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
    left_axes: Vec<usize>,
    right_axes: Vec<usize>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::contract_axes_complex(
            &left_field.0,
            &array_data_to_fixed_size_list(left.0)?,
            &right_field.0,
            &array_data_to_fixed_size_list(right.0)?,
            &left_axes,
            &right_axes,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute batched matrix multiplication across the last two axes of real fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_batched_matmul_last_two")]
pub fn tensor_batched_matmul_last_two(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left), RealFixedSizeListArray::F32(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::batched_matmul_last_two::<Float32Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left), RealFixedSizeListArray::F64(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::batched_matmul_last_two::<Float64Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute batched matrix multiplication across the last two axes of complex fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_batched_matmul_last_two_complex")]
pub fn tensor_batched_matmul_last_two_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::batched_matmul_last_two_complex(
            &left_field.0,
            &array_data_to_fixed_size_list(left.0)?,
            &right_field.0,
            &array_data_to_fixed_size_list(right.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute cube matvec over real fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_cube_matvec")]
pub fn tensor_cube_matvec(
    cube_field: PyArrowType<Field>,
    cube: PyArrowType<ArrayData>,
    vectors: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(cube.0)?,
        array_data_to_real_fixed_size_list(vectors.0)?,
    ) {
        (RealFixedSizeListArray::F32(cube), RealFixedSizeListArray::F32(vectors)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::tensor::cube_matvec::<Float32Type>(&cube_field.0, &cube, &vectors)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(cube), RealFixedSizeListArray::F64(vectors)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::tensor::cube_matvec::<Float64Type>(&cube_field.0, &cube, &vectors)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["cube", "vectors"])),
    }
}

/// Compute cube matvec over complex fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_cube_matvec_complex")]
pub fn tensor_cube_matvec_complex(
    cube_field: PyArrowType<Field>,
    cube: PyArrowType<ArrayData>,
    vectors: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    Ok(fixed_size_list_into_pyarrow(
        nabled::arrow::tensor::cube_matvec_complex(
            &cube_field.0,
            &array_data_to_fixed_size_list(cube.0)?,
            &array_data_to_fixed_size_list(vectors.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute cube matmat over real fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_cube_matmat")]
pub fn tensor_cube_matmat(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left), RealFixedSizeListArray::F32(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cube_matmat::<Float32Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left), RealFixedSizeListArray::F64(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cube_matmat::<Float64Type>(
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute cube matmat over complex fixed-shape tensors.
#[pyfunction(name = "arrow_tensor_cube_matmat_complex")]
pub fn tensor_cube_matmat_complex(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::cube_matmat_complex(
            &left_field.0,
            &array_data_to_fixed_size_list(left.0)?,
            &right_field.0,
            &array_data_to_fixed_size_list(right.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Flatten a real fixed-shape rank-3 Arrow tensor batch into a dense matrix.
#[pyfunction(name = "arrow_tensor_flatten_cubes")]
pub fn tensor_flatten_cubes(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::tensor::flatten_cubes::<Float32Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::tensor::flatten_cubes::<Float64Type>(&field.0, &array)
                .map_err(to_py_err)?,
        )),
    }
}

/// Evaluate two-operand Einstein summation over real fixed-shape Arrow tensors.
#[pyfunction(name = "arrow_tensor_einsum")]
pub fn tensor_einsum(
    expression: &str,
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left), RealFixedSizeListArray::F32(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::einsum::<Float32Type>(
                    expression,
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left), RealFixedSizeListArray::F64(right)) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::einsum::<Float64Type>(
                    expression,
                    &left_field.0,
                    &left,
                    &right_field.0,
                    &right,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Evaluate two-operand Einstein summation over complex fixed-shape Arrow tensors.
#[pyfunction(name = "arrow_tensor_einsum_complex")]
pub fn tensor_einsum_complex(
    expression: &str,
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    Ok(extension_result_into_pyarrow(
        nabled::arrow::tensor::einsum_complex(
            expression,
            &left_field.0,
            &array_data_to_fixed_size_list(left.0)?,
            &right_field.0,
            &array_data_to_fixed_size_list(right.0)?,
        )
        .map_err(to_py_err)?,
    ))
}

/// Compute CP-ALS for a rank-3 real Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als3", signature = (field, array, rank, max_iterations=None, tolerance=None))]
pub fn tensor_cp_als3(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyCpAls3Result> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(py_tensor::py_cp_als3_result(
            py,
            nabled::arrow::tensor::cp_als3::<Float32Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f32>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(py_tensor::py_cp_als3_result(
            py,
            nabled::arrow::tensor::cp_als3::<Float64Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f64>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Compute CP-ALS with diagnostics for a rank-3 real Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als3_with_report", signature = (field, array, rank, max_iterations=None, tolerance=None))]
pub fn tensor_cp_als3_with_report(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(py_tensor::PyCpAls3Result, py_tensor::PyCpReport)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => {
            let (result, report) = nabled::arrow::tensor::cp_als3_with_report::<Float32Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f32>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?;
            Ok((py_tensor::py_cp_als3_result(py, result), py_tensor::py_cp_report(report)?))
        }
        RealFixedSizeListArray::F64(array) => {
            let (result, report) = nabled::arrow::tensor::cp_als3_with_report::<Float64Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f64>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?;
            Ok((py_tensor::py_cp_als3_result(py, result), py_tensor::py_cp_report(report)?))
        }
    }
}

/// Compute reconstruction diagnostics for a rank-3 real Arrow CP result.
#[pyfunction(name = "arrow_tensor_cp_als3_diagnostics")]
pub fn tensor_cp_als3_diagnostics(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    weights: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyCpMetrics> {
    match (array_data_to_real_fixed_size_list(array.0)?, utils::real_array1(weights, "weights")?) {
        (RealFixedSizeListArray::F32(array), utils::RealReadonlyArray1::F32(weights_arr)) => {
            let [factor_0, factor_1, factor_2] = cp_als3_factor_arrays::<f32>(factors)?;
            py_tensor::py_cp_metrics(
                nabled::arrow::tensor::cp_als3_diagnostics_from_factors_view::<Float32Type>(
                    &field.0,
                    &array,
                    &weights_arr.as_array(),
                    &factor_0.as_array(),
                    &factor_1.as_array(),
                    &factor_2.as_array(),
                )
                .map_err(to_py_err)?,
            )
        }
        (RealFixedSizeListArray::F64(array), utils::RealReadonlyArray1::F64(weights_arr)) => {
            let [factor_0, factor_1, factor_2] = cp_als3_factor_arrays::<f64>(factors)?;
            py_tensor::py_cp_metrics(
                nabled::arrow::tensor::cp_als3_diagnostics_from_factors_view::<Float64Type>(
                    &field.0,
                    &array,
                    &weights_arr.as_array(),
                    &factor_0.as_array(),
                    &factor_1.as_array(),
                    &factor_2.as_array(),
                )
                .map_err(to_py_err)?,
            )
        }
        _ => Err(utils::matching_real_dtype_error(&["tensor", "weights", "factors"])),
    }
}

/// Reconstruct a rank-3 real Arrow CP result into an Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als3_reconstruct")]
pub fn tensor_cp_als3_reconstruct(
    field_name: &str,
    weights: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match utils::real_array1(weights, "weights")? {
        utils::RealReadonlyArray1::F32(weights_arr) => {
            let [factor_0, factor_1, factor_2] = cp_als3_factor_arrays::<f32>(factors)?;
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cp_als3_reconstruct_from_factors_view::<Float32Type>(
                    field_name,
                    &weights_arr.as_array(),
                    &factor_0.as_array(),
                    &factor_1.as_array(),
                    &factor_2.as_array(),
                )
                .map_err(to_py_err)?,
            ))
        }
        utils::RealReadonlyArray1::F64(weights_arr) => {
            let [factor_0, factor_1, factor_2] = cp_als3_factor_arrays::<f64>(factors)?;
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cp_als3_reconstruct_from_factors_view::<Float64Type>(
                    field_name,
                    &weights_arr.as_array(),
                    &factor_0.as_array(),
                    &factor_1.as_array(),
                    &factor_2.as_array(),
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Compute CP-ALS for an N-D real Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als_nd", signature = (field, array, rank, max_iterations=None, tolerance=None))]
pub fn tensor_cp_als_nd(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyCpAlsNdResult> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(py_tensor::py_cp_als_nd_result(
            py,
            nabled::arrow::tensor::cp_als_nd::<Float32Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f32>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(py_tensor::py_cp_als_nd_result(
            py,
            nabled::arrow::tensor::cp_als_nd::<Float64Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f64>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Compute CP-ALS with diagnostics for an N-D real Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als_nd_with_report", signature = (field, array, rank, max_iterations=None, tolerance=None))]
pub fn tensor_cp_als_nd_with_report(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    rank: usize,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<(py_tensor::PyCpAlsNdResult, py_tensor::PyCpReport)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => {
            let (result, report) = nabled::arrow::tensor::cp_als_nd_with_report::<Float32Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f32>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?;
            Ok((py_tensor::py_cp_als_nd_result(py, result), py_tensor::py_cp_report(report)?))
        }
        RealFixedSizeListArray::F64(array) => {
            let (result, report) = nabled::arrow::tensor::cp_als_nd_with_report::<Float64Type>(
                &field.0,
                &array,
                rank,
                &py_tensor::cp_als_config::<f64>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?;
            Ok((py_tensor::py_cp_als_nd_result(py, result), py_tensor::py_cp_report(report)?))
        }
    }
}

/// Compute reconstruction diagnostics for an N-D real Arrow CP result.
#[pyfunction(name = "arrow_tensor_cp_als_nd_diagnostics")]
pub fn tensor_cp_als_nd_diagnostics(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    weights: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyCpMetrics> {
    match (array_data_to_real_fixed_size_list(array.0)?, utils::real_array1(weights, "weights")?) {
        (RealFixedSizeListArray::F32(array), utils::RealReadonlyArray1::F32(weights_arr)) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f32>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            py_tensor::py_cp_metrics(
                nabled::arrow::tensor::cp_als_nd_diagnostics_from_factors_view::<Float32Type>(
                    &field.0,
                    &array,
                    &weights_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            )
        }
        (RealFixedSizeListArray::F64(array), utils::RealReadonlyArray1::F64(weights_arr)) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f64>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            py_tensor::py_cp_metrics(
                nabled::arrow::tensor::cp_als_nd_diagnostics_from_factors_view::<Float64Type>(
                    &field.0,
                    &array,
                    &weights_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            )
        }
        _ => Err(utils::matching_real_dtype_error(&["tensor", "weights", "factors"])),
    }
}

/// Reconstruct an N-D real Arrow CP result into an Arrow tensor.
#[pyfunction(name = "arrow_tensor_cp_als_nd_reconstruct")]
pub fn tensor_cp_als_nd_reconstruct(
    field_name: &str,
    weights: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match utils::real_array1(weights, "weights")? {
        utils::RealReadonlyArray1::F32(weights_arr) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f32>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cp_als_nd_reconstruct_from_factors_view::<Float32Type>(
                    field_name,
                    &weights_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
        utils::RealReadonlyArray1::F64(weights_arr) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f64>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::cp_als_nd_reconstruct_from_factors_view::<Float64Type>(
                    field_name,
                    &weights_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Compute HOSVD for an N-D real Arrow tensor.
#[pyfunction(name = "arrow_tensor_hosvd_nd")]
pub fn tensor_hosvd_nd(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    ranks: Vec<usize>,
) -> PyResult<py_tensor::PyHosvdNdResult> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(py_tensor::py_hosvd_nd_result(
            py,
            nabled::arrow::tensor::hosvd_nd::<Float32Type>(&field.0, &array, &ranks)
                .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(py_tensor::py_hosvd_nd_result(
            py,
            nabled::arrow::tensor::hosvd_nd::<Float64Type>(&field.0, &array, &ranks)
                .map_err(to_py_err)?,
        )),
    }
}

/// Compute HOOI/Tucker refinement for an N-D real Arrow tensor.
#[pyfunction(name = "arrow_tensor_hooi_nd", signature = (field, array, ranks, max_iterations=None, tolerance=None))]
pub fn tensor_hooi_nd(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    ranks: Vec<usize>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyHosvdNdResult> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(py_tensor::py_hosvd_nd_result(
            py,
            nabled::arrow::tensor::hooi_nd::<Float32Type>(
                &field.0,
                &array,
                &ranks,
                &py_tensor::hooi_config::<f32>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(py_tensor::py_hosvd_nd_result(
            py,
            nabled::arrow::tensor::hooi_nd::<Float64Type>(
                &field.0,
                &array,
                &ranks,
                &py_tensor::hooi_config::<f64>(max_iterations, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Reconstruct an N-D real Arrow HOSVD/Tucker result.
#[pyfunction(name = "arrow_tensor_hosvd_nd_reconstruct")]
pub fn tensor_hosvd_nd_reconstruct(
    field_name: &str,
    core: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match utils::real_arrayd(core, "core")? {
        utils::RealReadonlyArrayDyn::F32(core_arr) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f32>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::hosvd_nd_reconstruct_from_factors_view::<Float32Type>(
                    field_name,
                    &core_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
        utils::RealReadonlyArrayDyn::F64(core_arr) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f64>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::hosvd_nd_reconstruct_from_factors_view::<Float64Type>(
                    field_name,
                    &core_arr.as_array(),
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Project a real Arrow tensor into Tucker core coordinates.
#[pyfunction(name = "arrow_tensor_tucker_project")]
pub fn tensor_tucker_project(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f32>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tucker_project_from_factors_view::<Float32Type>(
                    &field.0,
                    &array,
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
        RealFixedSizeListArray::F64(array) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f64>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tucker_project_from_factors_view::<Float64Type>(
                    &field.0,
                    &array,
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Expand a real Arrow Tucker core into the original tensor space.
#[pyfunction(name = "arrow_tensor_tucker_expand")]
pub fn tensor_tucker_expand(
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    factors: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f32>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tucker_expand_from_factors_view::<Float32Type>(
                    &field.0,
                    &array,
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
        RealFixedSizeListArray::F64(array) => {
            let factor_arrays = py_tensor::extract_array2_sequence_views::<f64>(factors)?;
            let factor_views =
                factor_arrays.iter().map(PyReadonlyArray2::as_array).collect::<Vec<_>>();
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tucker_expand_from_factors_view::<Float64Type>(
                    &field.0,
                    &array,
                    &factor_views,
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Compute Tensor-Train decomposition for a real Arrow tensor.
#[pyfunction(name = "arrow_tensor_tt_svd", signature = (field, array, max_rank=None, tolerance=None))]
pub fn tensor_tt_svd(
    py: Python<'_>,
    field: PyArrowType<Field>,
    array: PyArrowType<ArrayData>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(array) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_svd::<Float32Type>(
                &field.0,
                &array,
                &py_tensor::tt_svd_config::<f32>(max_rank, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(array) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_svd::<Float64Type>(
                &field.0,
                &array,
                &py_tensor::tt_svd_config::<f64>(max_rank, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
    }
}

/// Left-orthogonalize a real Tensor-Train result.
#[pyfunction(name = "arrow_tensor_tt_orthogonalize_left")]
pub fn tensor_tt_orthogonalize_left(
    py: Python<'_>,
    cores: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match py_tensor::real_tt_core_arrays(cores)? {
        py_tensor::RealTensorTrainCoreArrays::F32(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_orthogonalize_left_from_cores_view(&core_views)
                    .map_err(to_py_err)?,
            ))
        }
        py_tensor::RealTensorTrainCoreArrays::F64(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_orthogonalize_left_from_cores_view(&core_views)
                    .map_err(to_py_err)?,
            ))
        }
    }
}

/// Right-orthogonalize a real Tensor-Train result.
#[pyfunction(name = "arrow_tensor_tt_orthogonalize_right")]
pub fn tensor_tt_orthogonalize_right(
    py: Python<'_>,
    cores: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match py_tensor::real_tt_core_arrays(cores)? {
        py_tensor::RealTensorTrainCoreArrays::F32(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_orthogonalize_right_from_cores_view(&core_views)
                    .map_err(to_py_err)?,
            ))
        }
        py_tensor::RealTensorTrainCoreArrays::F64(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_orthogonalize_right_from_cores_view(&core_views)
                    .map_err(to_py_err)?,
            ))
        }
    }
}

/// Round/compress a real Tensor-Train result.
#[pyfunction(name = "arrow_tensor_tt_round", signature = (cores, max_rank=None, tolerance=None))]
pub fn tensor_tt_round(
    py: Python<'_>,
    cores: &Bound<'_, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match py_tensor::real_tt_core_arrays(cores)? {
        py_tensor::RealTensorTrainCoreArrays::F32(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_round_from_cores_view(
                    &core_views,
                    &py_tensor::tt_round_config::<f32>(max_rank, tolerance)?,
                )
                .map_err(to_py_err)?,
            ))
        }
        py_tensor::RealTensorTrainCoreArrays::F64(core_arrays) => {
            let core_views = core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>();
            Ok(py_tensor::py_tt_result(
                py,
                nabled::arrow::tensor::tt_round_from_cores_view(
                    &core_views,
                    &py_tensor::tt_round_config::<f64>(max_rank, tolerance)?,
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

/// Compute the inner product of two real Tensor-Train results.
#[pyfunction(name = "arrow_tensor_tt_inner")]
pub fn tensor_tt_inner(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<f64> {
    match (py_tensor::real_tt_core_arrays(left)?, py_tensor::real_tt_core_arrays(right)?) {
        (
            py_tensor::RealTensorTrainCoreArrays::F32(left),
            py_tensor::RealTensorTrainCoreArrays::F32(right),
        ) => py_tensor::real_scalar_to_f64(
            nabled::arrow::tensor::tt_inner_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
            "tt_inner",
        ),
        (
            py_tensor::RealTensorTrainCoreArrays::F64(left),
            py_tensor::RealTensorTrainCoreArrays::F64(right),
        ) => py_tensor::real_scalar_to_f64(
            nabled::arrow::tensor::tt_inner_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
            "tt_inner",
        ),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute the Frobenius norm of a real Tensor-Train result.
#[pyfunction(name = "arrow_tensor_tt_norm")]
pub fn tensor_tt_norm(cores: &Bound<'_, PyAny>) -> PyResult<f64> {
    match py_tensor::real_tt_core_arrays(cores)? {
        py_tensor::RealTensorTrainCoreArrays::F32(core_arrays) => py_tensor::real_scalar_to_f64(
            nabled::arrow::tensor::tt_norm_from_cores_view(
                &core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
            "tt_norm",
        ),
        py_tensor::RealTensorTrainCoreArrays::F64(core_arrays) => py_tensor::real_scalar_to_f64(
            nabled::arrow::tensor::tt_norm_from_cores_view(
                &core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
            "tt_norm",
        ),
    }
}

/// Add two real Tensor-Train results.
#[pyfunction(name = "arrow_tensor_tt_add")]
pub fn tensor_tt_add(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match (py_tensor::real_tt_core_arrays(left)?, py_tensor::real_tt_core_arrays(right)?) {
        (
            py_tensor::RealTensorTrainCoreArrays::F32(left),
            py_tensor::RealTensorTrainCoreArrays::F32(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_add_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
        )),
        (
            py_tensor::RealTensorTrainCoreArrays::F64(left),
            py_tensor::RealTensorTrainCoreArrays::F64(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_add_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute the Hadamard product of two real Tensor-Train results.
#[pyfunction(name = "arrow_tensor_tt_hadamard")]
pub fn tensor_tt_hadamard(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match (py_tensor::real_tt_core_arrays(left)?, py_tensor::real_tt_core_arrays(right)?) {
        (
            py_tensor::RealTensorTrainCoreArrays::F32(left),
            py_tensor::RealTensorTrainCoreArrays::F32(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_hadamard_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
        )),
        (
            py_tensor::RealTensorTrainCoreArrays::F64(left),
            py_tensor::RealTensorTrainCoreArrays::F64(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_hadamard_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute the Hadamard product of two real Tensor-Train results and round it.
#[pyfunction(name = "arrow_tensor_tt_hadamard_round", signature = (left, right, max_rank=None, tolerance=None))]
pub fn tensor_tt_hadamard_round(
    py: Python<'_>,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    max_rank: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<py_tensor::PyTensorTrainResult> {
    match (py_tensor::real_tt_core_arrays(left)?, py_tensor::real_tt_core_arrays(right)?) {
        (
            py_tensor::RealTensorTrainCoreArrays::F32(left),
            py_tensor::RealTensorTrainCoreArrays::F32(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_hadamard_round_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &py_tensor::tt_round_config::<f32>(max_rank, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        (
            py_tensor::RealTensorTrainCoreArrays::F64(left),
            py_tensor::RealTensorTrainCoreArrays::F64(right),
        ) => Ok(py_tensor::py_tt_result(
            py,
            nabled::arrow::tensor::tt_hadamard_round_from_cores_view(
                &left.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &right.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                &py_tensor::tt_round_config::<f64>(max_rank, tolerance)?,
            )
            .map_err(to_py_err)?,
        )),
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Reconstruct a real Tensor-Train result into an Arrow tensor.
#[pyfunction(name = "arrow_tensor_tt_svd_reconstruct")]
pub fn tensor_tt_svd_reconstruct(
    field_name: &str,
    cores: &Bound<'_, PyAny>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match py_tensor::real_tt_core_arrays(cores)? {
        py_tensor::RealTensorTrainCoreArrays::F32(core_arrays) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tt_svd_reconstruct_from_cores_view::<Float32Type>(
                    field_name,
                    &core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                )
                .map_err(to_py_err)?,
            ))
        }
        py_tensor::RealTensorTrainCoreArrays::F64(core_arrays) => {
            Ok(extension_result_into_pyarrow(
                nabled::arrow::tensor::tt_svd_reconstruct_from_cores_view::<Float64Type>(
                    field_name,
                    &core_arrays.iter().map(|core| core.as_array()).collect::<Vec<_>>(),
                )
                .map_err(to_py_err)?,
            ))
        }
    }
}

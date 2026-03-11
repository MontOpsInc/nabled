//! Arrow-facing adapters over `nabled`'s ndarray-native APIs.
//!
//! This module is intentionally facade-only and feature-gated behind `arrow`.
//! It uses `ndarrow` for zero-copy Arrow -> ndarray bridging, then delegates to
//! existing ndarray-native `nabled` domains.
//!
//! Core crates remain Arrow-free:
//!
//! 1. `nabled-core`
//! 2. `nabled-linalg`
//! 3. `nabled-ml`
//!
//! Current scope favors operations with view-native ndarray entrypoints so the
//! bridge path stays explicit and copy-light.

use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, FixedSizeListArray, ListArray, PrimitiveArray, StructArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::extension::{ExtensionType, FixedShapeTensor};
use arrow_schema::{DataType, Field};
use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD};
use ndarrow::{AsNdarray, NdarrowElement};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod batched;
pub mod cholesky;
pub mod eigen;
pub mod iterative;
pub mod jacobian;
pub mod lu;
pub mod matrix;
pub mod matrix_functions;
pub mod optimization;
pub mod orthogonalization;
pub mod pca;
pub mod polar;
pub mod qr;
pub mod regression;
pub mod schur;
pub mod sparse;
pub mod stats;
pub mod svd;
pub mod tensor;
pub mod triangular;
pub mod vector;

/// Error type for Arrow-facing `nabled` adapters.
#[derive(Debug, Error)]
pub enum ArrowInteropError {
    /// Error while bridging Arrow data to/from ndarray.
    #[error(transparent)]
    Bridge(#[from] ndarrow::NdarrowError),
    /// Error from vector primitives.
    #[error(transparent)]
    Vector(#[from] crate::linalg::vector::VectorError),
    /// Error from dense matrix primitives.
    #[error(transparent)]
    Matrix(#[from] crate::linalg::matrix::MatrixError),
    /// Error from LU operations.
    #[error(transparent)]
    LU(#[from] crate::linalg::lu::LUError),
    /// Error from Cholesky operations.
    #[error(transparent)]
    Cholesky(#[from] crate::linalg::cholesky::CholeskyError),
    /// Error from QR operations.
    #[error(transparent)]
    QR(#[from] crate::linalg::qr::QRError),
    /// Error from SVD operations.
    #[error(transparent)]
    SVD(#[from] crate::linalg::svd::SVDError),
    /// Error from eigen operations.
    #[error(transparent)]
    Eigen(#[from] crate::linalg::eigen::EigenError),
    /// Error from Schur operations.
    #[error(transparent)]
    Schur(#[from] crate::linalg::schur::SchurError),
    /// Error from polar decomposition operations.
    #[error(transparent)]
    Polar(#[from] crate::linalg::polar::PolarError),
    /// Error from matrix functions.
    #[error(transparent)]
    MatrixFunctions(#[from] crate::linalg::matrix_functions::MatrixFunctionError),
    /// Error from orthogonalization routines.
    #[error(transparent)]
    Orthogonalization(#[from] crate::linalg::orthogonalization::OrthogonalizationError),
    /// Error from sparse operations.
    #[error(transparent)]
    Sparse(#[from] crate::linalg::sparse::SparseError),
    /// Error from tensor operations.
    #[error(transparent)]
    Tensor(#[from] crate::linalg::tensor::TensorError),
    /// Error from triangular solves.
    #[error(transparent)]
    Triangular(#[from] crate::linalg::triangular::TriangularError),
    /// Error from iterative solves.
    #[error(transparent)]
    Iterative(#[from] crate::ml::iterative::IterativeError),
    /// Error from Jacobian workflows.
    #[error(transparent)]
    Jacobian(#[from] crate::ml::jacobian::JacobianError),
    /// Error from optimization workflows.
    #[error(transparent)]
    Optimization(#[from] crate::ml::optimization::OptimizationError),
    /// Error from PCA workflows.
    #[error(transparent)]
    PCA(#[from] crate::ml::pca::PCAError),
    /// Error from regression workflows.
    #[error(transparent)]
    Regression(#[from] crate::ml::regression::RegressionError),
    /// Error from statistics workflows.
    #[error(transparent)]
    Stats(#[from] crate::ml::stats::StatsError),
    /// Error when an Arrow-backed tensor cannot be interpreted with the expected rank/shape.
    #[error("invalid Arrow tensor shape: {0}")]
    InvalidShape(String),
}

pub(crate) fn primitive_array_view<T>(
    array: &PrimitiveArray<T>,
) -> Result<ArrayView1<'_, T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    Ok(array.as_ndarray()?)
}

pub(crate) fn fixed_size_list_view<T>(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    Ok(ndarrow::fixed_size_list_as_array2::<T>(array)?)
}

pub(crate) fn fixed_shape_tensor_viewd<'a, T>(
    field: &'a Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayViewD<'a, T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    Ok(ndarrow::fixed_shape_tensor_as_array_viewd::<T>(field, array)?)
}

#[derive(Debug, Deserialize, Serialize)]
struct FixedShapeTensorWireMetadata {
    shape:        Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dim_names:    Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    permutations: Option<Vec<usize>>,
}

fn owned_array1_into_vec<T: Copy>(array: Array1<T>) -> Vec<T> {
    let len = array.len();
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let (mut raw, offset) = standard.into_raw_vec_and_offset();
    let off = offset.unwrap_or(0);
    if off == 0 {
        raw.truncate(len);
        raw
    } else {
        raw[off..off + len].to_vec()
    }
}

fn owned_array2_into_vec<T: Copy>(array: Array2<T>) -> (usize, Vec<T>) {
    let cols = array.ncols();
    let len = array.len();
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let (mut raw, offset) = standard.into_raw_vec_and_offset();
    let off = offset.unwrap_or(0);
    if off == 0 {
        raw.truncate(len);
        (cols, raw)
    } else {
        (cols, raw[off..off + len].to_vec())
    }
}

pub(crate) fn primitive_array_from_owned<T>(array: Array1<T::Native>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    PrimitiveArray::new(ScalarBuffer::from(owned_array1_into_vec(array)), None)
}

pub(crate) fn fixed_size_list_from_owned<T>(
    array: Array2<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let (cols, values) = owned_array2_into_vec(array);
    let value_length = i32::try_from(cols).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
        message: format!("matrix column count {cols} exceeds Arrow i32 value_length limits"),
    })?;
    let values = PrimitiveArray::<T>::new(ScalarBuffer::from(values), None);
    Ok(FixedSizeListArray::new(
        Arc::new(Field::new("item", T::DATA_TYPE, false)),
        value_length,
        Arc::new(values),
        None,
    ))
}

pub(crate) fn complex64_vector_view<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayView1<'a, Complex64>, ArrowInteropError> {
    Ok(ndarrow::complex64_as_array_view1(field, array)?)
}

pub(crate) fn complex64_vector_from_owned(
    field_name: &str,
    array: Array1<Complex64>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    Ok(ndarrow::array1_complex64_to_extension(field_name, array)?)
}

pub(crate) fn complex64_matrix_view(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, Complex64>, ArrowInteropError> {
    if array.null_count() > 0 {
        return Err(ndarrow::NdarrowError::NullsPresent { null_count: array.null_count() }.into());
    }

    let cols = usize::try_from(array.value_length()).map_err(|_| {
        ndarrow::NdarrowError::ShapeMismatch {
            message: format!(
                "negative Arrow fixed-size list value length: {}",
                array.value_length()
            ),
        }
    })?;
    let inner_field = match array.data_type() {
        DataType::FixedSizeList(item, _) => item.as_ref(),
        data_type => {
            return Err(ndarrow::NdarrowError::TypeMismatch {
                message: format!(
                    "expected nested complex matrix storage as FixedSizeList, found {data_type}"
                ),
            }
            .into());
        }
    };
    let inner_array =
        array.values().as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
            ndarrow::NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex matrix inner storage as FixedSizeListArray, found {}",
                    array.values().data_type()
                ),
            }
        })?;
    let flat = ndarrow::complex64_as_array_view1(inner_field, inner_array)?;
    flat.into_shape_with_order((array.len(), cols))
        .map_err(|error| ArrowInteropError::InvalidShape(error.to_string()))
}

pub(crate) fn complex64_matrix_from_owned(
    array: Array2<Complex64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let (cols, values) = owned_array2_into_vec(array);
    let (item_field, item_array) =
        ndarrow::array1_complex64_to_extension("item", Array1::from_vec(values))?;
    let value_length = i32::try_from(cols).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
        message: format!("matrix column count {cols} exceeds Arrow i32 value_length limits"),
    })?;
    Ok(FixedSizeListArray::new(Arc::new(item_field), value_length, Arc::new(item_array), None))
}

fn fixed_shape_tensor_shape(field: &Field) -> Result<Vec<usize>, ArrowInteropError> {
    let extension =
        field.try_extension_type::<FixedShapeTensor>().map_err(ndarrow::NdarrowError::from)?;
    extension.supports_data_type(field.data_type()).map_err(ndarrow::NdarrowError::from)?;
    let raw_metadata =
        field.extension_type_metadata().ok_or_else(|| ndarrow::NdarrowError::InvalidMetadata {
            message: "arrow.fixed_shape_tensor metadata missing".to_owned(),
        })?;
    let metadata: FixedShapeTensorWireMetadata =
        serde_json::from_str(raw_metadata).map_err(|e| ndarrow::NdarrowError::InvalidMetadata {
            message: format!("arrow.fixed_shape_tensor metadata parse failed: {e}"),
        })?;
    Ok(metadata.shape)
}

pub(crate) fn complex64_fixed_shape_tensor_viewd<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayViewD<'a, Complex64>, ArrowInteropError> {
    if array.null_count() > 0 {
        return Err(ndarrow::NdarrowError::NullsPresent { null_count: array.null_count() }.into());
    }
    if !field.data_type().equals_datatype(array.data_type()) {
        return Err(ndarrow::NdarrowError::TypeMismatch {
            message: format!(
                "field data type ({}) does not match array data type ({})",
                field.data_type(),
                array.data_type()
            ),
        }
        .into());
    }

    let tensor_shape = fixed_shape_tensor_shape(field)?;
    let element_count = tensor_shape.iter().product::<usize>();
    if usize::try_from(array.value_length()).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
        message: format!("negative Arrow fixed-size list value length: {}", array.value_length()),
    })? != element_count
    {
        return Err(ArrowInteropError::InvalidShape(format!(
            "fixed-shape tensor element count mismatch: metadata shape {:?} implies {}, array \
             stores {}",
            tensor_shape,
            element_count,
            array.value_length()
        )));
    }

    let inner_field = match array.data_type() {
        DataType::FixedSizeList(item, _) => item.as_ref(),
        data_type => {
            return Err(ndarrow::NdarrowError::TypeMismatch {
                message: format!(
                    "expected nested complex tensor storage as FixedSizeList, found {data_type}"
                ),
            }
            .into());
        }
    };
    let inner_array =
        array.values().as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
            ndarrow::NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex tensor inner storage as FixedSizeListArray, found {}",
                    array.values().data_type()
                ),
            }
        })?;
    let flat = ndarrow::complex64_as_array_view1(inner_field, inner_array)?;

    let mut full_shape = Vec::with_capacity(tensor_shape.len() + 1);
    full_shape.push(array.len());
    full_shape.extend_from_slice(&tensor_shape);

    flat.into_shape_with_order(ndarray::IxDyn(&full_shape))
        .map_err(|error| ArrowInteropError::InvalidShape(error.to_string()))
}

pub(crate) fn complex64_fixed_shape_tensor_from_owned(
    field_name: &str,
    array: ArrayD<Complex64>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let shape = array.shape().to_vec();
    if shape.is_empty() {
        return Err(ndarrow::NdarrowError::ShapeMismatch {
            message: "ArrayD must have at least one dimension (batch axis)".to_owned(),
        }
        .into());
    }

    let batch = shape[0];
    let tensor_shape = shape[1..].to_vec();
    let element_count = if tensor_shape.is_empty() {
        1
    } else {
        tensor_shape.iter().try_fold(1_usize, |acc, dim| acc.checked_mul(*dim)).ok_or_else(
            || ndarrow::NdarrowError::ShapeMismatch {
                message: format!("tensor shape product overflows usize: {tensor_shape:?}"),
            },
        )?
    };
    let expected_len =
        batch.checked_mul(element_count).ok_or_else(|| ndarrow::NdarrowError::ShapeMismatch {
            message: format!("batch × tensor_shape product overflows usize: {shape:?}"),
        })?;
    if expected_len != array.len() {
        return Err(ndarrow::NdarrowError::ShapeMismatch {
            message: format!(
                "array length ({}) does not match batch × tensor element count ({expected_len})",
                array.len()
            ),
        }
        .into());
    }

    let (cols, values) = owned_array2_into_vec(
        array
            .into_shape_with_order((batch, element_count))
            .map_err(|error| ArrowInteropError::InvalidShape(error.to_string()))?,
    );
    let (item_field, item_array) =
        ndarrow::array1_complex64_to_extension("item", Array1::from_vec(values))?;
    let value_length = i32::try_from(cols).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
        message: format!("tensor element count {cols} exceeds Arrow i32 value_length limits"),
    })?;
    let fsl =
        FixedSizeListArray::new(Arc::new(item_field), value_length, Arc::new(item_array), None);

    let extension =
        FixedShapeTensor::try_new(fsl.value_type().clone(), tensor_shape.clone(), None, None)
            .map_err(ndarrow::NdarrowError::from)?;
    extension.supports_data_type(fsl.data_type()).map_err(ndarrow::NdarrowError::from)?;
    let metadata_json = serde_json::to_string(&FixedShapeTensorWireMetadata {
        shape:        tensor_shape,
        dim_names:    None,
        permutations: None,
    })
    .map_err(|e| ndarrow::NdarrowError::InvalidMetadata {
        message: format!("arrow.fixed_shape_tensor metadata serialization failed: {e}"),
    })?;

    let mut metadata = std::collections::HashMap::new();
    drop(metadata.insert(
        arrow_schema::extension::EXTENSION_TYPE_NAME_KEY.to_owned(),
        FixedShapeTensor::NAME.to_owned(),
    ));
    drop(
        metadata
            .insert(arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY.to_owned(), metadata_json),
    );
    let field = Field::new(field_name, fsl.data_type().clone(), false).with_metadata(metadata);
    Ok((field, fsl))
}

pub(crate) fn fixed_shape_tensor_from_owned<T>(
    field_name: &str,
    array: ArrayD<T::Native>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    Ok(ndarrow::arrayd_to_fixed_shape_tensor(field_name, array)?)
}

pub(crate) fn csr_matrix_view_from_columns<'a, T>(
    indices: &'a ListArray,
    values: &'a ListArray,
    ncols: usize,
) -> Result<crate::linalg::sparse::CsrMatrixView<'a, i32, T::Native, u32>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let view = ndarrow::csr_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::CsrMatrixView::new(
        view.nrows,
        view.ncols,
        view.row_ptrs,
        view.col_indices,
        view.values,
    )?)
}

pub(crate) fn csr_matrix_view_from_extension<'a, T>(
    field: &Field,
    array: &'a StructArray,
) -> Result<crate::linalg::sparse::CsrMatrixView<'a, i32, T::Native, u32>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let view = ndarrow::csr_view_from_extension::<T>(field, array)?;
    Ok(crate::linalg::sparse::CsrMatrixView::new(
        view.nrows,
        view.ncols,
        view.row_ptrs,
        view.col_indices,
        view.values,
    )?)
}

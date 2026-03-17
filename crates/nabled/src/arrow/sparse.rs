//! Arrow adapters for sparse CSR primitives.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, FixedSizeListArray, ListArray, PrimitiveArray, StructArray};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, csr_matrix_view_from_columns, csr_matrix_view_from_extension,
    fixed_size_list_from_owned, fixed_size_list_view, primitive_array_from_owned,
    primitive_array_view,
};

type CsrBatchRowParts<T> = ([usize; 2], Vec<i32>, Vec<u32>, Vec<T>);

macro_rules! sparse_iterative_solver_wrappers {
    ($columns_name:ident, $extension_name:ident, $call:path) => {
        /// Solve a sparse linear system from Arrow CSR columns and an Arrow dense RHS vector.
        ///
        /// # Errors
        /// Returns an error when CSR structure is invalid, inputs contain nulls, or the solver
        /// fails to converge.
        pub fn $columns_name<T>(
            indices: &ListArray,
            values: &ListArray,
            ncols: usize,
            rhs: &PrimitiveArray<T>,
            tolerance: T::Native,
            max_iterations: usize,
        ) -> Result<PrimitiveArray<T>, ArrowInteropError>
        where
            T: ArrowPrimitiveType,
            T::Native: NabledReal + NdarrowElement,
        {
            let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
            let rhs_view = primitive_array_view(rhs)?;
            let output = $call(&matrix_view, &rhs_view, tolerance, max_iterations)?;
            Ok(primitive_array_from_owned::<T>(output))
        }

        /// Solve a sparse linear system from an Arrow `ndarrow.csr_matrix` extension and an Arrow
        /// dense RHS vector.
        ///
        /// # Errors
        /// Returns an error when extension metadata is invalid, inputs contain nulls, or the solver
        /// fails to converge.
        pub fn $extension_name<T>(
            field: &Field,
            matrix: &StructArray,
            rhs: &PrimitiveArray<T>,
            tolerance: T::Native,
            max_iterations: usize,
        ) -> Result<PrimitiveArray<T>, ArrowInteropError>
        where
            T: ArrowPrimitiveType,
            T::Native: NabledReal + NdarrowElement,
        {
            let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
            let rhs_view = primitive_array_view(rhs)?;
            let output = $call(&matrix_view, &rhs_view, tolerance, max_iterations)?;
            Ok(primitive_array_from_owned::<T>(output))
        }
    };
}

fn csr_matrix_to_batch_parts<T: NabledReal>(
    matrix: crate::linalg::sparse::CsrMatrix<T>,
) -> Result<CsrBatchRowParts<T>, ArrowInteropError> {
    let row_ptrs = matrix
        .indptr
        .into_iter()
        .map(|index| {
            i32::try_from(index).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
                message: format!("CSR row pointer {index} exceeds i32 limits for batch output"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let col_indices = matrix
        .indices
        .into_iter()
        .map(|index| {
            u32::try_from(index).map_err(|_| ndarrow::NdarrowError::ShapeMismatch {
                message: format!("CSR column index {index} exceeds u32 limits for batch output"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(([matrix.nrows, matrix.ncols], row_ptrs, col_indices, matrix.data))
}

/// Compute sparse-dense `y = A x` directly from Arrow CSR columns and an Arrow dense vector.
///
/// `indices` and `values` must describe the same CSR matrix via Arrow `ListArray` columns using
/// the `ndarrow` CSR contract.
///
/// # Errors
/// Returns an error when CSR structure is invalid, inputs contain nulls, or dimensions mismatch.
pub fn matvec_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    vector: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let vector_view = primitive_array_view(vector)?;
    let output = crate::linalg::sparse::matvec_view(&matrix_view, &vector_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Compute sparse-dense `y = A x` directly from an Arrow `ndarrow.csr_matrix` extension array.
///
/// # Errors
/// Returns an error when extension metadata is invalid, inputs contain nulls, or dimensions
/// mismatch.
pub fn matvec_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    vector: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let vector_view = primitive_array_view(vector)?;
    let output = crate::linalg::sparse::matvec_view(&matrix_view, &vector_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Compute sparse-dense `C = A B` directly from Arrow CSR columns and an Arrow dense matrix.
///
/// # Errors
/// Returns an error when CSR structure is invalid, inputs contain nulls, or dimensions mismatch.
pub fn matmat_dense_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::sparse::matmat_dense_view(&matrix_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute sparse-dense `C = A B` directly from an Arrow `ndarrow.csr_matrix` extension array.
///
/// # Errors
/// Returns an error when extension metadata is invalid, inputs contain nulls, or dimensions
/// mismatch.
pub fn matmat_dense_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::sparse::matmat_dense_view(&matrix_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute sparse direct LU factorization directly from Arrow CSR columns.
///
/// # Errors
/// Returns an error when CSR structure is invalid or factorization fails.
pub fn sparse_lu_factor_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
) -> Result<crate::linalg::sparse::SparseLUFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::sparse_lu_factor_view(&matrix_view)?)
}

/// Compute sparse direct LU factorization directly from an Arrow CSR extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid or factorization fails.
pub fn sparse_lu_factor_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
) -> Result<crate::linalg::sparse::SparseLUFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::sparse_lu_factor_view(&matrix_view)?)
}

/// Solve a sparse linear system directly from Arrow CSR columns using sparse LU factorization.
///
/// # Errors
/// Returns an error when CSR structure is invalid, inputs contain nulls, or factorization fails.
pub fn sparse_lu_solve_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    rhs: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Solve a sparse linear system directly from an Arrow CSR extension using sparse LU factorization.
///
/// # Errors
/// Returns an error when extension metadata is invalid, inputs contain nulls, or factorization
/// fails.
pub fn sparse_lu_solve_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    rhs: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

sparse_iterative_solver_wrappers!(
    jacobi_solve_csr_columns,
    jacobi_solve_csr_extension,
    crate::linalg::sparse::jacobi_solve_view
);
sparse_iterative_solver_wrappers!(
    gauss_seidel_solve_csr_columns,
    gauss_seidel_solve_csr_extension,
    crate::linalg::sparse::gauss_seidel_solve_view
);
sparse_iterative_solver_wrappers!(
    conjugate_gradient_solve_csr_columns,
    conjugate_gradient_solve_csr_extension,
    crate::linalg::sparse::conjugate_gradient_solve_view
);
sparse_iterative_solver_wrappers!(
    pcg_solve_csr_columns,
    pcg_solve_csr_extension,
    crate::linalg::sparse::pcg_solve_view
);

/// Transpose a CSR matrix described by Arrow CSR columns.
///
/// # Errors
/// Returns an error when CSR structure is invalid.
pub fn transpose_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
) -> Result<crate::linalg::sparse::CsrMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::transpose_view(&matrix_view)?)
}

/// Transpose a CSR matrix described by an Arrow CSR extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid.
pub fn transpose_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
) -> Result<crate::linalg::sparse::CsrMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::transpose_view(&matrix_view)?)
}

/// Convert a CSR matrix described by Arrow CSR columns into CSC form.
///
/// # Errors
/// Returns an error when CSR structure is invalid.
pub fn csr_to_csc_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
) -> Result<crate::linalg::sparse::CscMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::csr_to_csc_view(&matrix_view)?)
}

/// Convert a CSR matrix described by an Arrow CSR extension into CSC form.
///
/// # Errors
/// Returns an error when extension metadata is invalid.
pub fn csr_to_csc_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
) -> Result<crate::linalg::sparse::CscMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::csr_to_csc_view(&matrix_view)?)
}

/// Compute sparse-sparse matrix multiplication from Arrow CSR column inputs.
///
/// # Errors
/// Returns an error when either CSR structure is invalid or dimensions mismatch.
pub fn matmat_sparse_csr_columns<T>(
    left_indices: &ListArray,
    left_values: &ListArray,
    left_ncols: usize,
    right_indices: &ListArray,
    right_values: &ListArray,
    right_ncols: usize,
) -> Result<crate::linalg::sparse::CsrMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = csr_matrix_view_from_columns::<T>(left_indices, left_values, left_ncols)?;
    let right_view = csr_matrix_view_from_columns::<T>(right_indices, right_values, right_ncols)?;
    Ok(crate::linalg::sparse::matmat_sparse_view(&left_view, &right_view)?)
}

/// Compute sparse-sparse matrix multiplication from Arrow CSR extension inputs.
///
/// # Errors
/// Returns an error when either extension is invalid or dimensions mismatch.
pub fn matmat_sparse_csr_extension<T>(
    left_field: &Field,
    left: &StructArray,
    right_field: &Field,
    right: &StructArray,
) -> Result<crate::linalg::sparse::CsrMatrix<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = csr_matrix_view_from_extension::<T>(left_field, left)?;
    let right_view = csr_matrix_view_from_extension::<T>(right_field, right)?;
    Ok(crate::linalg::sparse::matmat_sparse_view(&left_view, &right_view)?)
}

/// Compute row-wise sparse-dense `y_i = A_i x_i` from an Arrow `ndarrow.csr_matrix_batch`
/// extension and a variable-shape tensor batch of dense vectors.
///
/// # Errors
/// Returns an error when either extension is invalid, contains nulls, batch lengths mismatch, or
/// per-row dimensions are incompatible.
pub fn matvec_csr_batch_extension<T>(
    field: &Field,
    matrices: &StructArray,
    vectors_field: &Field,
    vectors: &StructArray,
) -> Result<(Field, StructArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    if matrices.len() != vectors.len() {
        return Err(ArrowInteropError::InvalidShape(format!(
            "sparse matrix batch row count mismatch: {} vs {}",
            matrices.len(),
            vectors.len()
        )));
    }

    let mut outputs = Vec::with_capacity(matrices.len());
    let mut vector_iter = ndarrow::variable_shape_tensor_iter::<T>(vectors_field, vectors)?;
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(field, matrices)? {
        let (_, matrix_view) = matrix_row?;
        let (_, vector_view) = vector_iter.next().ok_or_else(|| {
            ArrowInteropError::InvalidShape("dense vector batch iterator ended early".to_owned())
        })??;
        let vector_view = vector_view
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|error| ArrowInteropError::InvalidShape(error.to_string()))?;
        let matrix_view = crate::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )?;
        outputs.push(crate::linalg::sparse::matvec_view(&matrix_view, &vector_view)?.into_dyn());
    }

    Ok(ndarrow::arrays_to_variable_shape_tensor(field.name(), outputs, Some(vec![None]))?)
}

/// Compute row-wise sparse-dense `C_i = A_i B_i` from an Arrow `ndarrow.csr_matrix_batch`
/// extension and a variable-shape tensor batch of dense matrices.
///
/// # Errors
/// Returns an error when either extension is invalid, contains nulls, batch lengths mismatch, or
/// per-row dimensions are incompatible.
pub fn matmat_dense_csr_batch_extension<T>(
    field: &Field,
    matrices: &StructArray,
    right_field: &Field,
    right: &StructArray,
) -> Result<(Field, StructArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    if matrices.len() != right.len() {
        return Err(ArrowInteropError::InvalidShape(format!(
            "sparse matrix batch row count mismatch: {} vs {}",
            matrices.len(),
            right.len()
        )));
    }

    let mut outputs = Vec::with_capacity(matrices.len());
    let mut right_iter = ndarrow::variable_shape_tensor_iter::<T>(right_field, right)?;
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(field, matrices)? {
        let (_, matrix_view) = matrix_row?;
        let (_, right_view) = right_iter.next().ok_or_else(|| {
            ArrowInteropError::InvalidShape("dense matrix batch iterator ended early".to_owned())
        })??;
        let right_view = right_view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|error| ArrowInteropError::InvalidShape(error.to_string()))?;
        let matrix_view = crate::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )?;
        outputs
            .push(crate::linalg::sparse::matmat_dense_view(&matrix_view, &right_view)?.into_dyn());
    }

    Ok(ndarrow::arrays_to_variable_shape_tensor(field.name(), outputs, Some(vec![None, None]))?)
}

/// Transpose each sparse matrix in an Arrow `ndarrow.csr_matrix_batch` extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid or any row violates CSR invariants.
pub fn transpose_csr_batch_extension<T>(
    field: &Field,
    matrices: &StructArray,
) -> Result<(Field, StructArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let mut shapes = Vec::with_capacity(matrices.len());
    let mut row_ptrs = Vec::with_capacity(matrices.len());
    let mut col_indices = Vec::with_capacity(matrices.len());
    let mut values = Vec::with_capacity(matrices.len());

    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(field, matrices)? {
        let (_, matrix_view) = matrix_row?;
        let matrix_view = crate::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )?;
        let transposed = crate::linalg::sparse::transpose_view(&matrix_view)?;
        let (shape, row_ptr, cols, row_values) = csr_matrix_to_batch_parts(transposed)?;
        shapes.push(shape);
        row_ptrs.push(row_ptr);
        col_indices.push(cols);
        values.push(row_values);
    }

    Ok(ndarrow::csr_batch_to_extension_array(
        field.name(),
        shapes,
        row_ptrs,
        col_indices,
        values,
    )?)
}

/// Compute row-wise sparse-sparse matrix products from Arrow `ndarrow.csr_matrix_batch`
/// extensions.
///
/// # Errors
/// Returns an error when either extension is invalid, contains nulls, batch lengths mismatch, or
/// per-row dimensions are incompatible.
pub fn matmat_sparse_csr_batch_extension<T>(
    left_field: &Field,
    left: &StructArray,
    right_field: &Field,
    right: &StructArray,
) -> Result<(Field, StructArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    if left.len() != right.len() {
        return Err(ArrowInteropError::InvalidShape(format!(
            "sparse matrix batch row count mismatch: {} vs {}",
            left.len(),
            right.len()
        )));
    }

    let mut shapes = Vec::with_capacity(left.len());
    let mut row_ptrs = Vec::with_capacity(left.len());
    let mut col_indices = Vec::with_capacity(left.len());
    let mut values = Vec::with_capacity(left.len());
    let mut right_iter = ndarrow::csr_matrix_batch_iter::<T>(right_field, right)?;

    for left_row in ndarrow::csr_matrix_batch_iter::<T>(left_field, left)? {
        let (_, left_view) = left_row?;
        let (_, right_view) = right_iter.next().ok_or_else(|| {
            ArrowInteropError::InvalidShape("right sparse batch iterator ended early".to_owned())
        })??;
        let left_view = crate::linalg::sparse::CsrMatrixView::new(
            left_view.nrows,
            left_view.ncols,
            left_view.row_ptrs,
            left_view.col_indices,
            left_view.values,
        )?;
        let right_view = crate::linalg::sparse::CsrMatrixView::new(
            right_view.nrows,
            right_view.ncols,
            right_view.row_ptrs,
            right_view.col_indices,
            right_view.values,
        )?;
        let product = crate::linalg::sparse::matmat_sparse_view(&left_view, &right_view)?;
        let (shape, row_ptr, cols, row_values) = csr_matrix_to_batch_parts(product)?;
        shapes.push(shape);
        row_ptrs.push(row_ptr);
        col_indices.push(cols);
        values.push(row_values);
    }

    Ok(ndarrow::csr_batch_to_extension_array(
        left_field.name(),
        shapes,
        row_ptrs,
        col_indices,
        values,
    )?)
}

/// Compute batched sparse matrix-vector products from Arrow CSR columns and an Arrow dense matrix.
///
/// # Errors
/// Returns an error when CSR structure is invalid, inputs contain nulls, or dimensions mismatch.
pub fn batched_matvec_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    batch_vectors: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let batch_vectors_view = fixed_size_list_view::<T>(batch_vectors)?;
    let output = crate::linalg::sparse::batched_matvec_view(&matrix_view, &batch_vectors_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute batched sparse matrix-vector products from an Arrow CSR extension and Arrow dense
/// matrix.
///
/// # Errors
/// Returns an error when extension metadata is invalid, inputs contain nulls, or dimensions
/// mismatch.
pub fn batched_matvec_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    batch_vectors: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let batch_vectors_view = fixed_size_list_view::<T>(batch_vectors)?;
    let output = crate::linalg::sparse::batched_matvec_view(&matrix_view, &batch_vectors_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Build a Jacobi preconditioner from Arrow CSR columns.
///
/// # Errors
/// Returns an error when CSR structure is invalid or the diagonal is singular.
pub fn jacobi_preconditioner_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
) -> Result<crate::linalg::sparse::JacobiPreconditioner<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::jacobi_preconditioner_view(&matrix_view)?)
}

/// Build a Jacobi preconditioner from an Arrow CSR extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid or the diagonal is singular.
pub fn jacobi_preconditioner_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
) -> Result<crate::linalg::sparse::JacobiPreconditioner<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::jacobi_preconditioner_view(&matrix_view)?)
}

/// Apply a Jacobi preconditioner to an Arrow dense vector.
///
/// # Errors
/// Returns an error when dimensions mismatch.
pub fn apply_jacobi_preconditioner<T>(
    preconditioner: &crate::linalg::sparse::JacobiPreconditioner<T::Native>,
    rhs: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::sparse::apply_jacobi_preconditioner(preconditioner, &rhs_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

macro_rules! sparse_factorization_wrappers {
    ($columns_name:ident, $extension_name:ident, $result_ty:ident, $call:path) => {
        /// Compute a sparse factorization from Arrow CSR column storage.
        ///
        /// # Errors
        /// Returns an error when CSR structure is invalid or factorization fails.
        pub fn $columns_name<T>(
            indices: &ListArray,
            values: &ListArray,
            ncols: usize,
        ) -> Result<crate::linalg::sparse::$result_ty<T::Native>, ArrowInteropError>
        where
            T: ArrowPrimitiveType,
            T::Native: NabledReal + NdarrowElement,
        {
            let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
            Ok($call(&matrix_view)?)
        }

        /// Compute a sparse factorization from an Arrow CSR extension.
        ///
        /// # Errors
        /// Returns an error when extension metadata is invalid or factorization fails.
        pub fn $extension_name<T>(
            field: &Field,
            matrix: &StructArray,
        ) -> Result<crate::linalg::sparse::$result_ty<T::Native>, ArrowInteropError>
        where
            T: ArrowPrimitiveType,
            T::Native: NabledReal + NdarrowElement,
        {
            let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
            Ok($call(&matrix_view)?)
        }
    };
}

sparse_factorization_wrappers!(
    ilu0_factor_csr_columns,
    ilu0_factor_csr_extension,
    ILU0Factorization,
    crate::linalg::sparse::ilu0_factor_view
);
sparse_factorization_wrappers!(
    ic0_factor_csr_columns,
    ic0_factor_csr_extension,
    IC0Factorization,
    crate::linalg::sparse::ic0_factor_view
);
sparse_factorization_wrappers!(
    ildl0_factor_csr_columns,
    ildl0_factor_csr_extension,
    ILDL0Factorization,
    crate::linalg::sparse::ildl0_factor_view
);

/// Compute ILUT factorization from Arrow CSR columns.
///
/// # Errors
/// Returns an error when CSR structure is invalid or factorization breaks down.
pub fn ilut_factor_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    drop_tolerance: T::Native,
    max_fill: usize,
) -> Result<crate::linalg::sparse::ILUTFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::ilut_factor_view(&matrix_view, drop_tolerance, max_fill)?)
}

/// Compute ILUT factorization from an Arrow CSR extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid or factorization breaks down.
pub fn ilut_factor_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    drop_tolerance: T::Native,
    max_fill: usize,
) -> Result<crate::linalg::sparse::ILUTFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::ilut_factor_view(&matrix_view, drop_tolerance, max_fill)?)
}

/// Compute ILU(k) factorization from Arrow CSR columns.
///
/// # Errors
/// Returns an error when CSR structure is invalid or factorization breaks down.
pub fn iluk_factor_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    level_of_fill: usize,
) -> Result<crate::linalg::sparse::ILUKFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    Ok(crate::linalg::sparse::iluk_factor_view(&matrix_view, level_of_fill)?)
}

/// Compute ILU(k) factorization from an Arrow CSR extension.
///
/// # Errors
/// Returns an error when extension metadata is invalid or factorization breaks down.
pub fn iluk_factor_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    level_of_fill: usize,
) -> Result<crate::linalg::sparse::ILUKFactorization<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    Ok(crate::linalg::sparse::iluk_factor_view(&matrix_view, level_of_fill)?)
}

macro_rules! sparse_apply_preconditioner_wrappers {
    ($name:ident, $factor_ty:ident, $call:path) => {
        /// Apply a sparse preconditioner to an Arrow dense vector.
        ///
        /// # Errors
        /// Returns an error when dimensions mismatch or the Arrow vector is invalid.
        pub fn $name<T>(
            factorization: &crate::linalg::sparse::$factor_ty<T::Native>,
            rhs: &PrimitiveArray<T>,
        ) -> Result<PrimitiveArray<T>, ArrowInteropError>
        where
            T: ArrowPrimitiveType,
            T::Native: NabledReal + NdarrowElement,
        {
            let rhs_view = primitive_array_view(rhs)?;
            let output = $call(factorization, &rhs_view)?;
            Ok(primitive_array_from_owned::<T>(output))
        }
    };
}

sparse_apply_preconditioner_wrappers!(
    apply_ilu0_preconditioner,
    ILU0Factorization,
    crate::linalg::sparse::apply_ilu0_preconditioner
);
sparse_apply_preconditioner_wrappers!(
    apply_ilut_preconditioner,
    ILUTFactorization,
    crate::linalg::sparse::apply_ilut_preconditioner
);
sparse_apply_preconditioner_wrappers!(
    apply_iluk_preconditioner,
    ILUKFactorization,
    crate::linalg::sparse::apply_iluk_preconditioner
);
sparse_apply_preconditioner_wrappers!(
    apply_ic0_preconditioner,
    IC0Factorization,
    crate::linalg::sparse::apply_ic0_preconditioner
);
sparse_apply_preconditioner_wrappers!(
    apply_ildl0_preconditioner,
    ILDL0Factorization,
    crate::linalg::sparse::apply_ildl0_preconditioner
);

/// Solve a sparse linear system from Arrow CSR columns with a precomputed sparse LU factorization.
///
/// # Errors
/// Returns an error when CSR structure is invalid, dimensions mismatch, or factors are singular.
pub fn sparse_lu_solve_with_factorization_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    rhs: &PrimitiveArray<T>,
    factorization: &crate::linalg::sparse::SparseLUFactorization<T::Native>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_with_factorization_view(
        &matrix_view,
        &rhs_view,
        factorization,
    )?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Solve a sparse linear system from an Arrow CSR extension with a precomputed sparse LU
/// factorization.
///
/// # Errors
/// Returns an error when extension metadata is invalid, dimensions mismatch, or factors are
/// singular.
pub fn sparse_lu_solve_with_factorization_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    rhs: &PrimitiveArray<T>,
    factorization: &crate::linalg::sparse::SparseLUFactorization<T::Native>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_with_factorization_view(
        &matrix_view,
        &rhs_view,
        factorization,
    )?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Solve multiple sparse right-hand sides from Arrow CSR columns with a precomputed sparse LU
/// factorization.
///
/// # Errors
/// Returns an error when CSR structure is invalid, dimensions mismatch, or factors are singular.
pub fn sparse_lu_solve_multiple_with_factorization_csr_columns<T>(
    indices: &ListArray,
    values: &ListArray,
    ncols: usize,
    rhs: &FixedSizeListArray,
    factorization: &crate::linalg::sparse::SparseLUFactorization<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_columns::<T>(indices, values, ncols)?;
    let rhs_view = fixed_size_list_view::<T>(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_multiple_with_factorization_view(
        &matrix_view,
        &rhs_view,
        factorization,
    )?;
    fixed_size_list_from_owned::<T>(output)
}

/// Solve multiple sparse right-hand sides from an Arrow CSR extension with a precomputed sparse LU
/// factorization.
///
/// # Errors
/// Returns an error when extension metadata is invalid, dimensions mismatch, or factors are
/// singular.
pub fn sparse_lu_solve_multiple_with_factorization_csr_extension<T>(
    field: &Field,
    matrix: &StructArray,
    rhs: &FixedSizeListArray,
    factorization: &crate::linalg::sparse::SparseLUFactorization<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = csr_matrix_view_from_extension::<T>(field, matrix)?;
    let rhs_view = fixed_size_list_view::<T>(rhs)?;
    let output = crate::linalg::sparse::sparse_lu_solve_multiple_with_factorization_view(
        &matrix_view,
        &rhs_view,
        factorization,
    )?;
    fixed_size_list_from_owned::<T>(output)
}

//! Arrow adapters for sparse CSR primitives.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, ListArray, PrimitiveArray, StructArray};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, csr_matrix_view_from_columns, csr_matrix_view_from_extension,
    fixed_size_list_from_owned, fixed_size_list_view, primitive_array_from_owned,
    primitive_array_view,
};

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

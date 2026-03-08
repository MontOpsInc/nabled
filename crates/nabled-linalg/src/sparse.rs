//! Sparse matrix primitives and iterative solves over CSR/CSC/COO structures.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2};
use thiserror::Error;

#[cfg(feature = "magma-system")]
use crate::provider::magma_sparse;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

fn default_tolerance<T: NabledReal>() -> T {
    T::from_f64(DEFAULT_TOLERANCE).unwrap_or(T::epsilon())
}

fn dot<T: NabledReal>(left: &Array1<T>, right: &Array1<T>) -> Result<T, SparseError> {
    if left.len() != right.len() {
        return Err(SparseError::DimensionMismatch);
    }
    let mut sum = T::zero();
    for i in 0..left.len() {
        sum += left[i] * right[i];
    }
    Ok(sum)
}

fn solve_dense_system<T: NabledReal>(
    matrix: Array2<T>,
    rhs: Array1<T>,
) -> Result<Array1<T>, SparseError> {
    let (nrows, ncols) = matrix.dim();
    if nrows == 0 || ncols == 0 || rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }
    if nrows != ncols || rhs.len() != nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let mut a = matrix;
    let mut b = rhs;
    let tolerance = default_tolerance::<T>();

    for pivot in 0..nrows {
        let mut best_row = pivot;
        let mut best_value = a[[pivot, pivot]].abs();
        for row in (pivot + 1)..nrows {
            let candidate = a[[row, pivot]].abs();
            if candidate > best_value {
                best_row = row;
                best_value = candidate;
            }
        }

        if best_value <= tolerance {
            return Err(SparseError::SingularMatrix);
        }

        if best_row != pivot {
            for col in pivot..ncols {
                let tmp = a[[pivot, col]];
                a[[pivot, col]] = a[[best_row, col]];
                a[[best_row, col]] = tmp;
            }
            let tmp = b[pivot];
            b[pivot] = b[best_row];
            b[best_row] = tmp;
        }

        let diagonal = a[[pivot, pivot]];
        let pivot_rhs = b[pivot];
        for row in (pivot + 1)..nrows {
            let factor = a[[row, pivot]] / diagonal;
            if factor.abs() <= tolerance {
                continue;
            }
            a[[row, pivot]] = T::zero();
            for col in (pivot + 1)..ncols {
                let pivot_value = a[[pivot, col]];
                a[[row, col]] -= factor * pivot_value;
            }
            b[row] -= factor * pivot_rhs;
        }
    }

    let mut x = Array1::<T>::zeros(nrows);
    for row_rev in 0..nrows {
        let row = nrows - 1 - row_rev;
        let mut sum = b[row];
        for col in (row + 1)..ncols {
            sum -= a[[row, col]] * x[col];
        }
        let diagonal = a[[row, row]];
        if diagonal.abs() <= tolerance {
            return Err(SparseError::SingularMatrix);
        }
        x[row] = sum / diagonal;
    }

    Ok(x)
}

/// Error type for sparse matrix operations.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SparseError {
    /// Matrix or vector input is empty.
    #[error("input cannot be empty")]
    EmptyInput,
    /// CSR structure is invalid.
    #[error("invalid CSR structure")]
    InvalidStructure,
    /// Dimensions are incompatible.
    #[error("input dimensions are incompatible")]
    DimensionMismatch,
    /// Matrix is singular.
    #[error("matrix is singular")]
    SingularMatrix,
    /// Iterative solve exceeded iteration budget.
    #[error("maximum iterations exceeded")]
    MaxIterationsExceeded,
}

#[cfg(feature = "magma-system")]
fn map_magma_sparse_error(error: &'static str) -> SparseError {
    match error {
        "empty" => SparseError::EmptyInput,
        "invalid_dimensions" | "bad_dimensions" => SparseError::DimensionMismatch,
        _ => SparseError::InvalidStructure,
    }
}

/// Compressed sparse row (CSR) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T: NabledReal = f64> {
    /// Number of rows.
    pub nrows:   usize,
    /// Number of columns.
    pub ncols:   usize,
    /// Row pointer offsets (`len = nrows + 1`).
    pub indptr:  Vec<usize>,
    /// Column index for each non-zero value.
    pub indices: Vec<usize>,
    /// Non-zero values.
    pub data:    Vec<T>,
}

/// Index contract for borrowed CSR views.
pub trait CsrIndex: Copy {
    /// Convert index to `usize` with bounds/validity checks.
    ///
    /// # Errors
    /// Returns an error when the index value cannot be represented as `usize`.
    fn to_usize(self) -> Result<usize, SparseError>;
}

impl CsrIndex for usize {
    fn to_usize(self) -> Result<usize, SparseError> { Ok(self) }
}

impl CsrIndex for u32 {
    fn to_usize(self) -> Result<usize, SparseError> {
        usize::try_from(self).map_err(|_| SparseError::InvalidStructure)
    }
}

impl CsrIndex for i32 {
    fn to_usize(self) -> Result<usize, SparseError> {
        if self < 0 {
            return Err(SparseError::InvalidStructure);
        }
        usize::try_from(self).map_err(|_| SparseError::InvalidStructure)
    }
}

/// Borrowed compressed sparse row (CSR) matrix view.
#[derive(Debug, Clone, Copy)]
pub struct CsrMatrixView<'a, R: CsrIndex = usize, T = f64, C: CsrIndex = R> {
    /// Number of rows.
    pub nrows:       usize,
    /// Number of columns.
    pub ncols:       usize,
    /// Row pointer offsets (`len = nrows + 1`).
    pub row_ptrs:    &'a [R],
    /// Column index for each non-zero value.
    pub col_indices: &'a [C],
    /// Non-zero values.
    pub values:      &'a [T],
}

impl<'a, R: CsrIndex, T, C: CsrIndex> CsrMatrixView<'a, R, T, C> {
    /// Construct a borrowed CSR matrix view after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSR arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptrs: &'a [R],
        col_indices: &'a [C],
        values: &'a [T],
    ) -> Result<Self, SparseError> {
        let view = Self { nrows, ncols, row_ptrs, col_indices, values };
        view.validate()?;
        Ok(view)
    }

    /// Validate this view's CSR structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSR arrays are inconsistent.
    pub fn validate(&self) -> Result<(), SparseError> {
        if self.nrows == 0 || self.ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if self.row_ptrs.len() != self.nrows + 1 {
            return Err(SparseError::InvalidStructure);
        }
        if self.col_indices.len() != self.values.len() {
            return Err(SparseError::InvalidStructure);
        }

        if self.row_ptrs[0].to_usize()? != 0 {
            return Err(SparseError::InvalidStructure);
        }
        if self.row_ptrs[self.nrows].to_usize()? != self.col_indices.len() {
            return Err(SparseError::InvalidStructure);
        }

        for row in 0..self.nrows {
            let start = self.row_ptrs[row].to_usize()?;
            let end = self.row_ptrs[row + 1].to_usize()?;
            if start > end {
                return Err(SparseError::InvalidStructure);
            }
        }

        for &index in self.col_indices {
            if index.to_usize()? >= self.ncols {
                return Err(SparseError::InvalidStructure);
            }
        }

        Ok(())
    }

    /// Get row [start, end) bounds in the value/index arrays.
    ///
    /// # Errors
    /// Returns an error if row pointers contain invalid index values.
    pub fn row_bounds(&self, row: usize) -> Result<(usize, usize), SparseError> {
        let start = self.row_ptrs[row].to_usize()?;
        let end = self.row_ptrs[row + 1].to_usize()?;
        Ok((start, end))
    }
}

impl<'a, T: NabledReal> From<&'a CsrMatrix<T>> for CsrMatrixView<'a, usize, T, usize> {
    fn from(matrix: &'a CsrMatrix<T>) -> Self {
        Self {
            nrows:       matrix.nrows,
            ncols:       matrix.ncols,
            row_ptrs:    &matrix.indptr,
            col_indices: &matrix.indices,
            values:      &matrix.data,
        }
    }
}

impl<T: NabledReal> CsrMatrix<T> {
    /// Construct a CSR matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSR arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, SparseError> {
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if indptr.len() != nrows + 1 {
            return Err(SparseError::InvalidStructure);
        }
        if indptr.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err(SparseError::InvalidStructure);
        }
        if indices.len() != data.len() {
            return Err(SparseError::InvalidStructure);
        }
        if indptr[nrows] != indices.len() {
            return Err(SparseError::InvalidStructure);
        }
        for row in 0..nrows {
            if indptr[row] > indptr[row + 1] {
                return Err(SparseError::InvalidStructure);
            }
        }
        if indices.iter().any(|&index| index >= ncols) {
            return Err(SparseError::InvalidStructure);
        }

        Ok(Self { nrows, ncols, indptr, indices, data })
    }

    /// Borrow this owned CSR matrix as a zero-copy view.
    #[must_use]
    pub fn as_view(&self) -> CsrMatrixView<'_, usize, T> { self.into() }
}

/// Coordinate list (COO) sparse matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix<T: NabledReal = f64> {
    /// Number of rows.
    pub nrows:       usize,
    /// Number of columns.
    pub ncols:       usize,
    /// Row index for each non-zero entry.
    pub row_indices: Vec<usize>,
    /// Column index for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Non-zero values.
    pub data:        Vec<T>,
}

/// Compressed sparse column (CSC) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix<T: NabledReal = f64> {
    /// Number of rows.
    pub nrows:   usize,
    /// Number of columns.
    pub ncols:   usize,
    /// Column pointer offsets (`len = ncols + 1`).
    pub indptr:  Vec<usize>,
    /// Row index for each non-zero value.
    pub indices: Vec<usize>,
    /// Non-zero values.
    pub data:    Vec<T>,
}

impl<T: NabledReal> CscMatrix<T> {
    /// Construct a CSC matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSC arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, SparseError> {
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if indptr.len() != ncols + 1 {
            return Err(SparseError::InvalidStructure);
        }
        if indptr.first().copied().unwrap_or(usize::MAX) != 0 {
            return Err(SparseError::InvalidStructure);
        }
        if indices.len() != data.len() {
            return Err(SparseError::InvalidStructure);
        }
        if indptr[ncols] != indices.len() {
            return Err(SparseError::InvalidStructure);
        }
        for col in 0..ncols {
            if indptr[col] > indptr[col + 1] {
                return Err(SparseError::InvalidStructure);
            }
        }
        if indices.iter().any(|&index| index >= nrows) {
            return Err(SparseError::InvalidStructure);
        }

        Ok(Self { nrows, ncols, indptr, indices, data })
    }

    /// Convert CSC to CSR.
    ///
    /// # Errors
    /// Returns an error if conversion encounters invalid structure.
    pub fn to_csr(&self) -> Result<CsrMatrix<T>, SparseError> {
        let nnz = self.data.len();
        let mut counts = vec![0_usize; self.nrows];
        for &row in &self.indices {
            counts[row] += 1;
        }

        let mut indptr = vec![0_usize; self.nrows + 1];
        for row in 0..self.nrows {
            indptr[row + 1] = indptr[row] + counts[row];
        }

        let mut next = indptr[..self.nrows].to_vec();
        let mut indices = vec![0_usize; nnz];
        let mut data = vec![T::zero(); nnz];

        for col in 0..self.ncols {
            for entry in self.indptr[col]..self.indptr[col + 1] {
                let row = self.indices[entry];
                let destination = next[row];
                indices[destination] = col;
                data[destination] = self.data[entry];
                next[row] += 1;
            }
        }

        CsrMatrix::new(self.nrows, self.ncols, indptr, indices, data)
    }
}

/// Diagonal (Jacobi) preconditioner for iterative sparse solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobiPreconditioner<T: NabledReal = f64> {
    /// Inverse diagonal values.
    pub inverse_diagonal: Array1<T>,
}

/// Incomplete LU(0) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILU0Factorization<T: NabledReal = f64> {
    /// Unit-lower factor.
    pub l: CsrMatrix<T>,
    /// Upper factor.
    pub u: CsrMatrix<T>,
}

/// Incomplete LU with threshold/drop tolerance (ILUT) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILUTFactorization<T: NabledReal = f64> {
    /// Unit-lower factor.
    pub l: CsrMatrix<T>,
    /// Upper factor.
    pub u: CsrMatrix<T>,
}

/// Incomplete LU with level-of-fill (ILU(k)) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILUKFactorization<T: NabledReal = f64> {
    /// Unit-lower factor.
    pub l:             CsrMatrix<T>,
    /// Upper factor.
    pub u:             CsrMatrix<T>,
    /// Requested level-of-fill used during construction.
    pub level_of_fill: usize,
}

/// Sparse direct LU factorization with partial row pivoting.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseLUFactorization<T: NabledReal = f64> {
    /// Unit-lower factor.
    pub l:           CsrMatrix<T>,
    /// Upper factor.
    pub u:           CsrMatrix<T>,
    /// Row permutation such that `P * A = L * U`.
    pub permutation: Vec<usize>,
}

/// Configuration for ILUT-based sparse factorization and solves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ILUTConfig<T: NabledReal = f64> {
    /// Drop entries with absolute magnitude less than or equal to this value.
    pub drop_tolerance: T,
    /// Maximum number of retained off-diagonal entries per row in each factor.
    pub max_fill:       usize,
}

impl<T: NabledReal> ILUTConfig<T> {
    /// Conservative profile prioritizing sparsity.
    #[must_use]
    pub fn conservative() -> Self {
        Self { drop_tolerance: T::from_f64(1e-6).unwrap_or(T::epsilon()), max_fill: 8 }
    }

    /// Balanced profile for general sparse workloads.
    #[must_use]
    pub fn balanced() -> Self {
        Self { drop_tolerance: T::from_f64(1e-8).unwrap_or(T::epsilon()), max_fill: 16 }
    }

    /// Aggressive profile prioritizing preconditioner quality.
    #[must_use]
    pub fn aggressive() -> Self {
        Self { drop_tolerance: T::from_f64(1e-10).unwrap_or(T::epsilon()), max_fill: 32 }
    }

    /// Size-aware default profile.
    #[must_use]
    pub fn for_dimension(dimension: usize) -> Self {
        let fill = if dimension <= 32 {
            8
        } else if dimension <= 256 {
            16
        } else {
            32
        };
        Self {
            drop_tolerance: T::from_f64(1e-8).unwrap_or(T::epsilon()),
            max_fill:       fill.min(dimension.max(1)),
        }
    }
}

impl<T: NabledReal> Default for ILUTConfig<T> {
    fn default() -> Self { Self::balanced() }
}

/// Configuration profile for ILU(k)-based sparse factorization and solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ILUKConfig {
    /// Maximum allowed level-of-fill.
    pub level_of_fill: usize,
}

impl ILUKConfig {
    /// Conservative profile with no extra fill beyond the original sparsity pattern.
    #[must_use]
    pub const fn conservative() -> Self { Self { level_of_fill: 0 } }

    /// Balanced profile allowing limited controlled fill.
    #[must_use]
    pub const fn balanced() -> Self { Self { level_of_fill: 1 } }

    /// Aggressive profile allowing deeper fill for stronger preconditioning.
    #[must_use]
    pub const fn aggressive() -> Self { Self { level_of_fill: 2 } }
}

impl Default for ILUKConfig {
    fn default() -> Self { Self::balanced() }
}

/// Incomplete Cholesky(0) sparse factorization for SPD systems.
#[derive(Debug, Clone, PartialEq)]
pub struct IC0Factorization<T: NabledReal = f64> {
    /// Lower-triangular factor with diagonal terms.
    pub l:           CsrMatrix<T>,
    /// Cached transpose of `l` for backward substitution.
    pub l_transpose: CsrMatrix<T>,
}

/// Incomplete LDL(0) sparse factorization for symmetric sparse systems.
#[derive(Debug, Clone, PartialEq)]
pub struct ILDL0Factorization<T: NabledReal = f64> {
    /// Unit-lower factor.
    pub l:           CsrMatrix<T>,
    /// Diagonal factor.
    pub d:           Array1<T>,
    /// Cached transpose of `l` for backward substitution.
    pub l_transpose: CsrMatrix<T>,
}

impl<T: NabledReal> CooMatrix<T> {
    /// Construct a COO matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or COO arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<T>,
    ) -> Result<Self, SparseError> {
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::EmptyInput);
        }
        if row_indices.len() != col_indices.len() || row_indices.len() != data.len() {
            return Err(SparseError::InvalidStructure);
        }
        if row_indices.iter().any(|&row| row >= nrows)
            || col_indices.iter().any(|&col| col >= ncols)
        {
            return Err(SparseError::InvalidStructure);
        }

        Ok(Self { nrows, ncols, row_indices, col_indices, data })
    }

    /// Convert COO to CSR. Duplicate coordinates are summed.
    ///
    /// # Errors
    /// Returns an error if COO structure is invalid.
    pub fn to_csr(&self) -> Result<CsrMatrix<T>, SparseError> {
        let mut entries = self
            .row_indices
            .iter()
            .copied()
            .zip(self.col_indices.iter().copied())
            .zip(self.data.iter().copied())
            .map(|((row, col), value)| (row, col, value))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(row, col, _)| (row, col));

        let mut collapsed = Vec::<(usize, usize, T)>::new();
        for (row, col, value) in entries {
            if let Some((last_row, last_col, last_value)) = collapsed.last_mut()
                && *last_row == row
                && *last_col == col
            {
                *last_value += value;
            } else {
                collapsed.push((row, col, value));
            }
        }

        let mut indptr = vec![0_usize; self.nrows + 1];
        for &(row, _, _) in &collapsed {
            indptr[row + 1] += 1;
        }
        for row in 0..self.nrows {
            indptr[row + 1] += indptr[row];
        }

        let indices = collapsed.iter().map(|&(_, col, _)| col).collect::<Vec<_>>();
        let data = collapsed.iter().map(|&(_, _, value)| value).collect::<Vec<_>>();
        CsrMatrix::new(self.nrows, self.ncols, indptr, indices, data)
    }
}

/// Compute sparse matrix-vector product `y = A x`.
///
/// # Errors
/// Returns an error if vector length mismatches matrix columns.
pub fn matvec<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    matvec_view(&matrix.as_view(), vector)
}

/// Compute sparse matrix-vector product `y = A x` from a borrowed CSR view.
///
/// # Errors
/// Returns an error if vector length mismatches matrix columns.
pub fn matvec_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    vector: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    let mut output = Array1::<T>::zeros(matrix.nrows);
    matvec_view_into(matrix, vector, &mut output)?;
    Ok(output)
}

/// Compute sparse matrix-vector product `y = A x` via MAGMA sparse (`f64`).
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matvec_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    matrix.validate()?;
    if vector.len() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    match magma_sparse::spmv_f64(
        matrix.nrows,
        matrix.ncols,
        matrix.row_ptrs,
        matrix.col_indices,
        matrix.values,
        vector,
    ) {
        Ok(result) => Ok(result),
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_sparse_error(error));
            }
            matvec_view(matrix, vector)
        }
    }
}

/// Compute sparse matrix-vector product `y = A x` via MAGMA sparse (`f32`).
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matvec_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    vector: &Array1<f32>,
) -> Result<Array1<f32>, SparseError> {
    matrix.validate()?;
    if vector.len() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    match magma_sparse::spmv_f32(
        matrix.nrows,
        matrix.ncols,
        matrix.row_ptrs,
        matrix.col_indices,
        matrix.values,
        vector,
    ) {
        Ok(result) => Ok(result),
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_sparse_error(error));
            }
            matvec_view(matrix, vector)
        }
    }
}

/// Compute sparse matrix-vector product `y = A x` via MAGMA sparse (`f64`) into `output`.
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matvec_magma_f64_view_into(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    vector: &Array1<f64>,
    output: &mut Array1<f64>,
) -> Result<(), SparseError> {
    if output.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    let result = matvec_magma_f64_view(matrix, vector)?;
    output.assign(&result);
    Ok(())
}

/// Compute sparse matrix-vector product `y = A x` via MAGMA sparse (`f32`) into `output`.
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matvec_magma_f32_view_into(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    vector: &Array1<f32>,
    output: &mut Array1<f32>,
) -> Result<(), SparseError> {
    if output.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    let result = matvec_magma_f32_view(matrix, vector)?;
    output.assign(&result);
    Ok(())
}

/// Compute sparse matrix-vector product `y = A x` into `output`.
///
/// # Errors
/// Returns an error if input/output dimensions are incompatible.
pub fn matvec_into<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), SparseError> {
    matvec_view_into(&matrix.as_view(), vector, output)
}

/// Compute sparse matrix-vector product `y = A x` into `output` from a borrowed CSR view.
///
/// # Errors
/// Returns an error if input/output dimensions are incompatible.
pub fn matvec_view_into<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    vector: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), SparseError> {
    matrix.validate()?;
    if vector.len() != matrix.ncols || output.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        let mut sum = T::zero();
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            sum += matrix.values[entry] * vector[col];
        }
        output[row] = sum;
    }

    Ok(())
}

/// Compute transpose of CSR matrix.
///
/// # Errors
/// Returns an error if transpose construction fails due to invalid structure.
pub fn transpose<T: NabledReal>(matrix: &CsrMatrix<T>) -> Result<CsrMatrix<T>, SparseError> {
    transpose_view(&matrix.as_view())
}

/// Compute transpose of a borrowed CSR matrix view.
///
/// # Errors
/// Returns an error if transpose construction fails due to invalid structure.
pub fn transpose_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<CsrMatrix<T>, SparseError> {
    matrix.validate()?;

    let nnz = matrix.values.len();
    let mut counts = vec![0_usize; matrix.ncols];
    for &col in matrix.col_indices {
        counts[col.to_usize()?] += 1;
    }

    let mut indptr_t = vec![0_usize; matrix.ncols + 1];
    for row in 0..matrix.ncols {
        indptr_t[row + 1] = indptr_t[row] + counts[row];
    }

    let mut next = indptr_t[..matrix.ncols].to_vec();
    let mut indices_t = vec![0_usize; nnz];
    let mut data_t = vec![T::zero(); nnz];

    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            let destination = next[col];
            indices_t[destination] = row;
            data_t[destination] = matrix.values[entry];
            next[col] += 1;
        }
    }

    CsrMatrix::new(matrix.ncols, matrix.nrows, indptr_t, indices_t, data_t)
}

/// Convert CSR matrix to CSC.
///
/// # Errors
/// Returns an error if conversion encounters invalid structure.
pub fn csr_to_csc<T: NabledReal>(matrix: &CsrMatrix<T>) -> Result<CscMatrix<T>, SparseError> {
    csr_to_csc_view(&matrix.as_view())
}

/// Convert a borrowed CSR matrix view to CSC.
///
/// # Errors
/// Returns an error if conversion encounters invalid structure.
pub fn csr_to_csc_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<CscMatrix<T>, SparseError> {
    matrix.validate()?;

    let nnz = matrix.values.len();
    let mut counts = vec![0_usize; matrix.ncols];
    for &col in matrix.col_indices {
        counts[col.to_usize()?] += 1;
    }

    let mut indptr = vec![0_usize; matrix.ncols + 1];
    for col in 0..matrix.ncols {
        indptr[col + 1] = indptr[col] + counts[col];
    }

    let mut next = indptr[..matrix.ncols].to_vec();
    let mut indices = vec![0_usize; nnz];
    let mut data = vec![T::zero(); nnz];

    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            let destination = next[col];
            indices[destination] = row;
            data[destination] = matrix.values[entry];
            next[col] += 1;
        }
    }

    CscMatrix::new(matrix.nrows, matrix.ncols, indptr, indices, data)
}

/// Compute sparse matrix-vector product `y = A x` for CSC format.
///
/// # Errors
/// Returns an error if vector length mismatches matrix columns.
pub fn matvec_csc<T: NabledReal>(
    matrix: &CscMatrix<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if vector.len() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    let mut output = Array1::<T>::zeros(matrix.nrows);
    for col in 0..matrix.ncols {
        let x = vector[col];
        for entry in matrix.indptr[col]..matrix.indptr[col + 1] {
            let row = matrix.indices[entry];
            output[row] += matrix.data[entry] * x;
        }
    }
    Ok(output)
}

/// Build Jacobi preconditioner from sparse matrix diagonal.
///
/// # Errors
/// Returns an error if matrix is not square or diagonal contains zeros.
pub fn jacobi_preconditioner<T: NabledReal>(
    matrix: &CsrMatrix<T>,
) -> Result<JacobiPreconditioner<T>, SparseError> {
    jacobi_preconditioner_view(&matrix.as_view())
}

/// Build Jacobi preconditioner from a borrowed sparse matrix view.
///
/// # Errors
/// Returns an error if matrix is not square or diagonal contains zeros.
pub fn jacobi_preconditioner_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<JacobiPreconditioner<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    let mut inverse_diagonal = Array1::<T>::zeros(matrix.nrows);
    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        let mut diagonal = T::zero();
        for entry in start..end {
            if matrix.col_indices[entry].to_usize()? == row {
                diagonal = matrix.values[entry];
                break;
            }
        }
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        inverse_diagonal[row] = T::one() / diagonal;
    }
    Ok(JacobiPreconditioner { inverse_diagonal })
}

/// Apply Jacobi preconditioner to a vector.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn apply_jacobi_preconditioner<T: NabledReal>(
    preconditioner: &JacobiPreconditioner<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if preconditioner.inverse_diagonal.len() != vector.len() {
        return Err(SparseError::DimensionMismatch);
    }
    let mut output = Array1::<T>::zeros(vector.len());
    for i in 0..vector.len() {
        output[i] = preconditioner.inverse_diagonal[i] * vector[i];
    }
    Ok(output)
}

fn row_positions_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<Vec<HashMap<usize, usize>>, SparseError> {
    let mut positions = Vec::<HashMap<usize, usize>>::with_capacity(matrix.nrows);
    for row in 0..matrix.nrows {
        let mut map = HashMap::<usize, usize>::new();
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let _ = map.insert(matrix.col_indices[entry].to_usize()?, entry);
        }
        positions.push(map);
    }
    Ok(positions)
}

fn is_symmetric_from_positions_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    positions: &[HashMap<usize, usize>],
    tolerance: T,
) -> Result<bool, SparseError> {
    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            let value = matrix.values[entry];
            let Some(&mirror_entry) = positions[col].get(&row) else {
                return Ok(false);
            };
            if (value - matrix.values[mirror_entry]).abs() > tolerance {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Compute ILU(0) factorization for a square sparse matrix.
///
/// The non-zero pattern of factors follows the input pattern (`level-of-fill = 0`).
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn ilu0_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
) -> Result<ILU0Factorization<T>, SparseError> {
    ilu0_factor_view(&matrix.as_view())
}

/// Compute ILU(0) factorization for a square sparse matrix from a borrowed view.
///
/// The non-zero pattern of factors follows the input pattern (`level-of-fill = 0`).
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn ilu0_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<ILU0Factorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let positions = row_positions_view(matrix)?;
    let mut factors = matrix.values.to_vec();

    for row in 0..n {
        let mut lower_entries = Vec::<(usize, usize)>::new();
        let (row_start, row_end) = matrix.row_bounds(row)?;
        for entry in row_start..row_end {
            let col = matrix.col_indices[entry].to_usize()?;
            if col < row {
                lower_entries.push((col, entry));
            }
        }
        lower_entries.sort_unstable_by_key(|&(col, _)| col);

        for (col_j, row_col_entry) in lower_entries {
            let Some(&diag_index) = positions[col_j].get(&col_j) else {
                return Err(SparseError::SingularMatrix);
            };
            let diagonal = factors[diag_index];
            if diagonal.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }

            let multiplier = factors[row_col_entry] / diagonal;
            factors[row_col_entry] = multiplier;

            let (upper_start, upper_end) = matrix.row_bounds(col_j)?;
            for upper_entry in upper_start..upper_end {
                let col_k = matrix.col_indices[upper_entry].to_usize()?;
                if col_k <= col_j {
                    continue;
                }
                if let Some(&update_index) = positions[row].get(&col_k) {
                    let upper_value = factors[upper_entry];
                    factors[update_index] -= multiplier * upper_value;
                }
            }
        }

        let Some(&row_diagonal_index) = positions[row].get(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        if factors[row_diagonal_index].abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);

    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<T>::new();
    u_indptr.push(0);

    for row in 0..n {
        let mut l_row = Vec::<(usize, T)>::new();
        l_row.push((row, T::one()));
        let mut u_row = Vec::<(usize, T)>::new();

        let (row_start, row_end) = matrix.row_bounds(row)?;
        for (&col, &value) in
            matrix.col_indices[row_start..row_end].iter().zip(factors[row_start..row_end].iter())
        {
            let col = col.to_usize()?;
            if col < row {
                l_row.push((col, value));
            } else {
                u_row.push((col, value));
            }
        }

        l_row.sort_unstable_by_key(|&(col, _)| col);
        u_row.sort_unstable_by_key(|&(col, _)| col);

        for (col, value) in l_row {
            l_indices.push(col);
            l_data.push(value);
        }
        for (col, value) in u_row {
            u_indices.push(col);
            u_data.push(value);
        }

        l_indptr.push(l_indices.len());
        u_indptr.push(u_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let u = CsrMatrix::new(n, n, u_indptr, u_indices, u_data)?;
    Ok(ILU0Factorization { l, u })
}

fn retain_strongest_entries<T: NabledReal>(entries: &mut Vec<(usize, T)>, max_entries: usize) {
    if entries.len() <= max_entries {
        entries.sort_unstable_by_key(|&(col, _)| col);
        return;
    }
    entries.sort_unstable_by(|left, right| {
        right.1.abs().partial_cmp(&left.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(max_entries);
    entries.sort_unstable_by_key(|&(col, _)| col);
}

/// Compute incomplete LU with threshold/drop tolerance (ILUT) factorization.
///
/// `drop_tolerance` controls magnitude pruning and `max_fill` limits the
/// retained sub/super-diagonal entries per row.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ilut_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<ILUTFactorization<T>, SparseError> {
    ilut_factor_view(&matrix.as_view(), drop_tolerance, max_fill)
}

/// Compute incomplete LU with threshold/drop tolerance (ILUT) factorization from a borrowed view.
///
/// `drop_tolerance` controls magnitude pruning and `max_fill` limits the
/// retained sub/super-diagonal entries per row.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ilut_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<ILUTFactorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let tolerance = drop_tolerance.max(T::zero());
    let mut lower_rows = Vec::<Vec<(usize, T)>>::with_capacity(n);
    let mut upper_rows = Vec::<Vec<(usize, T)>>::with_capacity(n);
    let mut upper_diagonal = vec![T::zero(); n];

    for row in 0..n {
        let mut working = BTreeMap::<usize, T>::new();
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            let value = matrix.values[entry];
            if col == row || value.abs() > tolerance {
                let _ = working.insert(col, value);
            }
        }

        let mut lower_candidates = Vec::<(usize, T)>::new();
        let lower_columns = working.keys().copied().filter(|&col| col < row).collect::<Vec<_>>();
        for col_j in lower_columns {
            let Some(value) = working.remove(&col_j) else {
                continue;
            };
            if value.abs() <= tolerance {
                continue;
            }

            let diagonal = upper_diagonal[col_j];
            if diagonal.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let multiplier = value / diagonal;
            if multiplier.abs() <= tolerance {
                continue;
            }
            lower_candidates.push((col_j, multiplier));

            for &(col_k, upper_value) in &upper_rows[col_j] {
                if col_k <= col_j {
                    continue;
                }
                let updated =
                    working.get(&col_k).copied().unwrap_or(T::zero()) - multiplier * upper_value;
                if updated.abs() <= tolerance {
                    let _ = working.remove(&col_k);
                } else {
                    let _ = working.insert(col_k, updated);
                }
            }
        }

        let Some(diagonal) = working.remove(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        upper_diagonal[row] = diagonal;

        let mut upper_candidates = working
            .into_iter()
            .filter(|&(col, value)| col > row && value.abs() > tolerance)
            .collect::<Vec<_>>();

        retain_strongest_entries(&mut lower_candidates, max_fill);
        retain_strongest_entries(&mut upper_candidates, max_fill);

        let mut lower_row = lower_candidates;
        lower_row.push((row, T::one()));
        lower_row.sort_unstable_by_key(|&(col, _)| col);

        let mut upper_row = Vec::<(usize, T)>::with_capacity(upper_candidates.len() + 1);
        upper_row.push((row, diagonal));
        upper_row.extend(upper_candidates);
        upper_row.sort_unstable_by_key(|&(col, _)| col);

        lower_rows.push(lower_row);
        upper_rows.push(upper_row);
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);

    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<T>::new();
    u_indptr.push(0);

    for row in 0..n {
        for &(col, value) in &lower_rows[row] {
            l_indices.push(col);
            l_data.push(value);
        }
        for &(col, value) in &upper_rows[row] {
            u_indices.push(col);
            u_data.push(value);
        }
        l_indptr.push(l_indices.len());
        u_indptr.push(u_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let u = CsrMatrix::new(n, n, u_indptr, u_indices, u_data)?;
    Ok(ILUTFactorization { l, u })
}

/// Compute ILUT factorization using a configuration profile.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn ilut_factor_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    config: ILUTConfig<T>,
) -> Result<ILUTFactorization<T>, SparseError> {
    ilut_factor_with_config_view(&matrix.as_view(), config)
}

/// Compute ILUT factorization from a borrowed sparse view using a configuration profile.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn ilut_factor_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    config: ILUTConfig<T>,
) -> Result<ILUTFactorization<T>, SparseError> {
    ilut_factor_view(matrix, config.drop_tolerance, config.max_fill)
}

fn iluk_initial_row_state_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    row: usize,
) -> Result<IlukRowState<T>, SparseError> {
    let mut values = BTreeMap::<usize, T>::new();
    let mut levels = HashMap::<usize, usize>::new();
    let (start, end) = matrix.row_bounds(row)?;
    for entry in start..end {
        let col = matrix.col_indices[entry].to_usize()?;
        let value = matrix.values[entry];
        let _ = values.insert(col, value);
        let _ = levels.insert(col, 0);
    }
    Ok((values, levels))
}

type SparseRowEntries<T> = Vec<(usize, T)>;
type SparseUpperRowWithLevel<T> = Vec<(usize, T, usize)>;
type IlukRowState<T> = (BTreeMap<usize, T>, HashMap<usize, usize>);
type IlukRowFactors<T> = (SparseRowEntries<T>, SparseRowEntries<T>, SparseUpperRowWithLevel<T>, T);

#[allow(clippy::many_single_char_names)]
fn iluk_eliminate_row_entries<T: NabledReal>(
    row: usize,
    level_of_fill: usize,
    values: &mut BTreeMap<usize, T>,
    levels: &mut HashMap<usize, usize>,
    upper_rows_with_levels: &[Vec<(usize, T, usize)>],
    upper_diagonal: &[T],
) -> Result<(), SparseError> {
    let mut pending_lower =
        values.keys().copied().filter(|&col| col < row).collect::<BTreeSet<_>>();
    while let Some(col_j) = pending_lower.pop_first() {
        let Some(level_ij) = levels.get(&col_j).copied() else {
            continue;
        };
        if level_ij > level_of_fill {
            continue;
        }

        let Some(value_ij) = values.get(&col_j).copied() else {
            continue;
        };
        let diagonal = upper_diagonal[col_j];
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let multiplier = value_ij / diagonal;
        if multiplier.abs() <= default_tolerance::<T>() {
            let _ = values.remove(&col_j);
            let _ = levels.remove(&col_j);
            continue;
        }
        let _ = values.insert(col_j, multiplier);

        for &(col_k, upper_value, level_jk) in &upper_rows_with_levels[col_j] {
            if col_k <= col_j {
                continue;
            }
            let candidate_level = level_ij.saturating_add(level_jk).saturating_add(1);
            if candidate_level > level_of_fill {
                continue;
            }

            let existing_level = levels.get(&col_k).copied();
            if existing_level.is_none_or(|current_level| candidate_level < current_level) {
                let _ = levels.insert(col_k, candidate_level);
            }

            let updated =
                values.get(&col_k).copied().unwrap_or(T::zero()) - multiplier * upper_value;
            if updated.abs() <= default_tolerance::<T>() {
                let _ = values.remove(&col_k);
                let _ = levels.remove(&col_k);
            } else {
                let _ = values.insert(col_k, updated);
                if col_k < row && existing_level.is_none() {
                    let _ = pending_lower.insert(col_k);
                }
            }
        }
    }
    Ok(())
}

fn iluk_finalize_row_entries<T: NabledReal>(
    row: usize,
    level_of_fill: usize,
    values: &BTreeMap<usize, T>,
    levels: &HashMap<usize, usize>,
) -> Result<IlukRowFactors<T>, SparseError> {
    let Some(diagonal) = values.get(&row).copied() else {
        return Err(SparseError::SingularMatrix);
    };
    if diagonal.abs() <= default_tolerance::<T>() {
        return Err(SparseError::SingularMatrix);
    }

    let mut lower_row = vec![(row, T::one())];
    let mut upper_row = vec![(row, diagonal)];
    let mut upper_row_levels = vec![(row, diagonal, 0)];
    for (&col, &value) in values {
        if col < row {
            if levels.get(&col).copied().unwrap_or(level_of_fill + 1) <= level_of_fill {
                lower_row.push((col, value));
            }
            continue;
        }
        if col == row {
            continue;
        }
        let entry_level = levels.get(&col).copied().unwrap_or(level_of_fill + 1);
        if entry_level <= level_of_fill {
            upper_row.push((col, value));
            upper_row_levels.push((col, value, entry_level));
        }
    }

    lower_row.sort_unstable_by_key(|&(col, _)| col);
    upper_row.sort_unstable_by_key(|&(col, _)| col);
    upper_row_levels.sort_unstable_by_key(|&(col, _, _)| col);
    Ok((lower_row, upper_row, upper_row_levels, diagonal))
}

/// Compute incomplete LU with level-of-fill (ILU(k)) factorization.
///
/// `level_of_fill` controls the maximum accepted fill-in level in each factor.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn iluk_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    level_of_fill: usize,
) -> Result<ILUKFactorization<T>, SparseError> {
    iluk_factor_view(&matrix.as_view(), level_of_fill)
}

/// Compute incomplete LU with level-of-fill (ILU(k)) factorization from a borrowed view.
///
/// `level_of_fill` controls the maximum accepted fill-in level in each factor.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn iluk_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    level_of_fill: usize,
) -> Result<ILUKFactorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let mut lower_rows = Vec::<Vec<(usize, T)>>::with_capacity(n);
    let mut upper_rows = Vec::<Vec<(usize, T)>>::with_capacity(n);
    let mut upper_rows_with_levels = vec![Vec::<(usize, T, usize)>::new(); n];
    let mut upper_diagonal = vec![T::zero(); n];

    for row in 0..n {
        let (mut values, mut levels) = iluk_initial_row_state_view(matrix, row)?;
        iluk_eliminate_row_entries(
            row,
            level_of_fill,
            &mut values,
            &mut levels,
            &upper_rows_with_levels,
            &upper_diagonal,
        )?;
        let (lower_row, upper_row, upper_row_levels, diagonal) =
            iluk_finalize_row_entries(row, level_of_fill, &values, &levels)?;

        upper_diagonal[row] = diagonal;
        lower_rows.push(lower_row);
        upper_rows.push(upper_row);
        upper_rows_with_levels[row] = upper_row_levels;
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);
    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<T>::new();
    u_indptr.push(0);

    for row in 0..n {
        for &(col, value) in &lower_rows[row] {
            l_indices.push(col);
            l_data.push(value);
        }
        for &(col, value) in &upper_rows[row] {
            u_indices.push(col);
            u_data.push(value);
        }
        l_indptr.push(l_indices.len());
        u_indptr.push(u_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let u = CsrMatrix::new(n, n, u_indptr, u_indices, u_data)?;
    Ok(ILUKFactorization { l, u, level_of_fill })
}

/// Compute ILU(k) factorization using a configuration profile.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn iluk_factor_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    config: ILUKConfig,
) -> Result<ILUKFactorization<T>, SparseError> {
    iluk_factor_with_config_view(&matrix.as_view(), config)
}

/// Compute ILU(k) factorization from a borrowed sparse view using a configuration profile.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn iluk_factor_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    config: ILUKConfig,
) -> Result<ILUKFactorization<T>, SparseError> {
    iluk_factor_view(matrix, config.level_of_fill)
}

fn csr_rows_as_maps_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<Vec<BTreeMap<usize, T>>, SparseError> {
    let mut rows = Vec::<BTreeMap<usize, T>>::with_capacity(matrix.nrows);
    for row in 0..matrix.nrows {
        let mut map = BTreeMap::<usize, T>::new();
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let _ = map.insert(matrix.col_indices[entry].to_usize()?, matrix.values[entry]);
        }
        rows.push(map);
    }
    Ok(rows)
}

fn split_lu_rows_to_csr<T: NabledReal>(
    rows: &[BTreeMap<usize, T>],
) -> Result<(CsrMatrix<T>, CsrMatrix<T>), SparseError> {
    let n = rows.len();
    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);

    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<T>::new();
    u_indptr.push(0);

    for (row, row_map) in rows.iter().enumerate() {
        l_indices.push(row);
        l_data.push(T::one());

        for (&col, &value) in row_map {
            if col < row {
                l_indices.push(col);
                l_data.push(value);
            } else {
                u_indices.push(col);
                u_data.push(value);
            }
        }

        l_indptr.push(l_indices.len());
        u_indptr.push(u_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let u = CsrMatrix::new(n, n, u_indptr, u_indices, u_data)?;
    Ok((l, u))
}

/// Compute sparse direct LU factorization with partial row pivoting.
///
/// The resulting factorization satisfies `P * A = L * U`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn sparse_lu_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
) -> Result<SparseLUFactorization<T>, SparseError> {
    sparse_lu_factor_view(&matrix.as_view())
}

/// Compute sparse direct LU factorization with partial row pivoting from a borrowed view.
///
/// The resulting factorization satisfies `P * A = L * U`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn sparse_lu_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<SparseLUFactorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if matrix.nrows == 0 {
        return Err(SparseError::EmptyInput);
    }

    let n = matrix.nrows;
    let mut rows = csr_rows_as_maps_view(matrix)?;
    let mut permutation = (0..n).collect::<Vec<_>>();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_value = rows[k].get(&k).copied().unwrap_or(T::zero()).abs();
        for (candidate_row, row_map) in rows.iter().enumerate().skip(k + 1) {
            let candidate = row_map.get(&k).copied().unwrap_or(T::zero()).abs();
            if candidate > pivot_value {
                pivot_value = candidate;
                pivot_row = candidate_row;
            }
        }
        if pivot_value <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if pivot_row != k {
            rows.swap(k, pivot_row);
            permutation.swap(k, pivot_row);
        }

        let diagonal = rows[k].get(&k).copied().unwrap_or(T::zero());
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for row in (k + 1)..n {
            let Some(entry_ik) = rows[row].get(&k).copied() else {
                continue;
            };
            let multiplier = entry_ik / diagonal;
            if multiplier.abs() <= default_tolerance::<T>() {
                let _ = rows[row].remove(&k);
                continue;
            }
            let _ = rows[row].insert(k, multiplier);

            let pivot_updates = rows[k]
                .iter()
                .filter_map(|(&col, &value)| (col > k).then_some((col, value)))
                .collect::<Vec<_>>();
            for (col, pivot_value_col) in pivot_updates {
                let updated = rows[row].get(&col).copied().unwrap_or(T::zero())
                    - multiplier * pivot_value_col;
                if updated.abs() <= default_tolerance::<T>() {
                    let _ = rows[row].remove(&col);
                } else {
                    let _ = rows[row].insert(col, updated);
                }
            }
        }
    }

    let (l, u) = split_lu_rows_to_csr(&rows)?;
    Ok(SparseLUFactorization { l, u, permutation })
}

fn apply_sparse_row_permutation<T: NabledReal>(
    permutation: &[usize],
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if permutation.len() != rhs.len() {
        return Err(SparseError::DimensionMismatch);
    }
    let mut permuted = Array1::<T>::zeros(rhs.len());
    for (row, source_index) in permutation.iter().copied().enumerate() {
        permuted[row] = rhs[source_index];
    }
    Ok(permuted)
}

/// Solve sparse linear system `A x = b` using direct sparse LU factorization.
///
/// # Errors
/// Returns an error for invalid dimensions or singular systems.
pub fn sparse_lu_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    sparse_lu_solve_view(&matrix.as_view(), rhs)
}

/// Solve sparse linear system `A x = b` using direct sparse LU factorization from a borrowed view.
///
/// # Errors
/// Returns an error for invalid dimensions or singular systems.
pub fn sparse_lu_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    let factorization = sparse_lu_factor_view(matrix)?;
    sparse_lu_solve_with_factorization_view(matrix, rhs, &factorization)
}

/// Solve sparse linear system `A x = b` with a precomputed sparse LU factorization.
///
/// # Errors
/// Returns an error for invalid dimensions or singular factors.
pub fn sparse_lu_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    factorization: &SparseLUFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    sparse_lu_solve_with_factorization_view(&matrix.as_view(), rhs, factorization)
}

/// Solve sparse linear system `A x = b` with a precomputed sparse LU factorization from a borrowed
/// view.
///
/// # Errors
/// Returns an error for invalid dimensions or singular factors.
pub fn sparse_lu_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    factorization: &SparseLUFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols
        || factorization.l.nrows != matrix.nrows
        || factorization.u.nrows != matrix.nrows
        || factorization.permutation.len() != matrix.nrows
    {
        return Err(SparseError::DimensionMismatch);
    }
    let permuted_rhs = apply_sparse_row_permutation(&factorization.permutation, rhs)?;
    apply_lu_preconditioner(&factorization.l, &factorization.u, &permuted_rhs)
}

/// Solve sparse linear systems `A X = B` using a precomputed sparse LU factorization.
///
/// # Errors
/// Returns an error for invalid dimensions or singular factors.
pub fn sparse_lu_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    factorization: &SparseLUFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    sparse_lu_solve_multiple_with_factorization_view(&matrix.as_view(), rhs, factorization)
}

/// Solve sparse linear systems `A X = B` using a precomputed sparse LU factorization from a
/// borrowed view.
///
/// # Errors
/// Returns an error for invalid dimensions or singular factors.
pub fn sparse_lu_solve_multiple_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    factorization: &SparseLUFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        sparse_lu_solve_with_factorization_view(matrix, rhs_column, factorization)
    })
}

fn apply_lu_preconditioner<T: NabledReal>(
    lower: &CsrMatrix<T>,
    upper: &CsrMatrix<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if lower.nrows != lower.ncols || upper.nrows != upper.ncols || lower.nrows != upper.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != lower.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let n = rhs.len();
    let mut intermediate = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = rhs[row];
        for entry in lower.indptr[row]..lower.indptr[row + 1] {
            let col = lower.indices[entry];
            if col < row {
                sum -= lower.data[entry] * intermediate[col];
            }
        }
        intermediate[row] = sum;
    }

    let mut output = Array1::<T>::zeros(n);
    for row_reverse in 0..n {
        let row = n - 1 - row_reverse;
        let mut sum = intermediate[row];
        let mut diagonal = None;
        for entry in upper.indptr[row]..upper.indptr[row + 1] {
            let col = upper.indices[entry];
            let value = upper.data[entry];
            if col == row {
                diagonal = Some(value);
            } else if col > row {
                sum -= value * output[col];
            }
        }

        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        output[row] = sum / diagonal;
    }

    Ok(output)
}

/// Apply an ILU(0) preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`ilu0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ilu0_preconditioner<T: NabledReal>(
    factorization: &ILU0Factorization<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    apply_lu_preconditioner(&factorization.l, &factorization.u, rhs)
}

/// Apply an ILUT preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`ilut_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ilut_preconditioner<T: NabledReal>(
    factorization: &ILUTFactorization<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    apply_lu_preconditioner(&factorization.l, &factorization.u, rhs)
}

/// Apply an ILU(k) preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`iluk_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_iluk_preconditioner<T: NabledReal>(
    factorization: &ILUKFactorization<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    apply_lu_preconditioner(&factorization.l, &factorization.u, rhs)
}

/// Compute incomplete Cholesky(0) factorization for an SPD sparse matrix.
///
/// The non-zero pattern of `L` follows the lower-triangular pattern of `A`
/// (`level-of-fill = 0`), including the diagonal.
///
/// # Errors
/// Returns an error if dimensions are incompatible or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ic0_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
) -> Result<IC0Factorization<T>, SparseError> {
    ic0_factor_view(&matrix.as_view())
}

/// Compute incomplete Cholesky(0) factorization for an SPD sparse matrix from a borrowed view.
///
/// The non-zero pattern of `L` follows the lower-triangular pattern of `A`
/// (`level-of-fill = 0`), including the diagonal.
///
/// # Errors
/// Returns an error if dimensions are incompatible or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ic0_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<IC0Factorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let positions = row_positions_view(matrix)?;
    let mut factors = matrix.values.to_vec();

    for row in 0..n {
        let mut lower_entries = Vec::<(usize, usize)>::new();
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            if col < row {
                lower_entries.push((col, entry));
            }
        }
        lower_entries.sort_unstable_by_key(|&(col, _)| col);

        for idx in 0..lower_entries.len() {
            let (col_j, row_col_entry) = lower_entries[idx];
            let mut sum = factors[row_col_entry];

            for &(col_k, row_k_entry) in lower_entries.iter().take(idx) {
                if let Some(&jk_index) = positions[col_j].get(&col_k) {
                    sum -= factors[row_k_entry] * factors[jk_index];
                }
            }

            let Some(&diag_index) = positions[col_j].get(&col_j) else {
                return Err(SparseError::SingularMatrix);
            };
            let diagonal = factors[diag_index];
            if diagonal.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }

            factors[row_col_entry] = sum / diagonal;
        }

        let Some(&row_diagonal_index) = positions[row].get(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        let mut diagonal = factors[row_diagonal_index];
        for &(_, row_col_entry) in &lower_entries {
            let value = factors[row_col_entry];
            diagonal -= value * value;
        }
        if diagonal <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        factors[row_diagonal_index] = diagonal.sqrt();
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);

    for row in 0..n {
        let mut row_entries = Vec::<(usize, T)>::new();
        let (row_start, row_end) = matrix.row_bounds(row)?;
        for (&col, &value) in
            matrix.col_indices[row_start..row_end].iter().zip(factors[row_start..row_end].iter())
        {
            let col = col.to_usize()?;
            if col <= row {
                row_entries.push((col, value));
            }
        }
        row_entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, value) in row_entries {
            l_indices.push(col);
            l_data.push(value);
        }
        l_indptr.push(l_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let l_transpose = transpose_view(&l.as_view())?;
    Ok(IC0Factorization { l, l_transpose })
}

/// Apply an IC(0) preconditioner to a dense vector.
///
/// Solves `L L^T x = rhs` where `L` comes from [`ic0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ic0_preconditioner<T: NabledReal>(
    factorization: &IC0Factorization<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if factorization.l.nrows != factorization.l.ncols
        || factorization.l_transpose.nrows != factorization.l_transpose.ncols
        || factorization.l.nrows != factorization.l_transpose.nrows
    {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != factorization.l.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let n = rhs.len();
    let mut intermediate = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = rhs[row];
        let mut diagonal = None;
        for entry in factorization.l.indptr[row]..factorization.l.indptr[row + 1] {
            let col = factorization.l.indices[entry];
            let value = factorization.l.data[entry];
            if col < row {
                sum -= value * intermediate[col];
            } else if col == row {
                diagonal = Some(value);
            }
        }
        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        intermediate[row] = sum / diagonal;
    }

    let mut output = Array1::<T>::zeros(n);
    for row_reverse in 0..n {
        let row = n - 1 - row_reverse;
        let mut sum = intermediate[row];
        let mut diagonal = None;
        for entry in
            factorization.l_transpose.indptr[row]..factorization.l_transpose.indptr[row + 1]
        {
            let col = factorization.l_transpose.indices[entry];
            let value = factorization.l_transpose.data[entry];
            if col > row {
                sum -= value * output[col];
            } else if col == row {
                diagonal = Some(value);
            }
        }
        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        output[row] = sum / diagonal;
    }

    Ok(output)
}

/// Compute incomplete LDL(0) factorization for a square symmetric sparse matrix.
///
/// The non-zero pattern of `L` follows the strictly-lower pattern of `A`
/// (`level-of-fill = 0`) with unit diagonal in `L` and diagonal terms in `D`.
///
/// # Errors
/// Returns an error if dimensions are incompatible, input is non-symmetric,
/// or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ildl0_factor<T: NabledReal>(
    matrix: &CsrMatrix<T>,
) -> Result<ILDL0Factorization<T>, SparseError> {
    ildl0_factor_view(&matrix.as_view())
}

/// Compute incomplete LDL(0) factorization for a square symmetric sparse matrix from a borrowed
/// view.
///
/// The non-zero pattern of `L` follows the strictly-lower pattern of `A`
/// (`level-of-fill = 0`) with unit diagonal in `L` and diagonal terms in `D`.
///
/// # Errors
/// Returns an error if dimensions are incompatible, input is non-symmetric,
/// or factorization breaks down.
#[allow(clippy::many_single_char_names)]
pub fn ildl0_factor_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
) -> Result<ILDL0Factorization<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let positions = row_positions_view(matrix)?;
    if !is_symmetric_from_positions_view(matrix, &positions, default_tolerance::<T>())? {
        return Err(SparseError::DimensionMismatch);
    }

    let mut factors = matrix.values.to_vec();
    let mut diagonal = Array1::<T>::zeros(n);

    for row in 0..n {
        let mut lower_entries = Vec::<(usize, usize)>::new();
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            if col < row {
                lower_entries.push((col, entry));
            }
        }
        lower_entries.sort_unstable_by_key(|&(col, _)| col);

        for idx in 0..lower_entries.len() {
            let (col_j, row_col_entry) = lower_entries[idx];
            let mut sum = factors[row_col_entry];

            for &(col_k, row_k_entry) in lower_entries.iter().take(idx) {
                if let Some(&jk_index) = positions[col_j].get(&col_k) {
                    sum -= factors[row_k_entry] * diagonal[col_k] * factors[jk_index];
                }
            }

            let d_j = diagonal[col_j];
            if d_j.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            factors[row_col_entry] = sum / d_j;
        }

        let Some(&row_diagonal_index) = positions[row].get(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        let mut d_i = factors[row_diagonal_index];
        for &(_, row_col_entry) in &lower_entries {
            let col_k = matrix.col_indices[row_col_entry].to_usize()?;
            let l_ik = factors[row_col_entry];
            d_i -= l_ik * l_ik * diagonal[col_k];
        }
        if d_i.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        diagonal[row] = d_i;
        factors[row_diagonal_index] = d_i;
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<T>::new();
    l_indptr.push(0);

    for row in 0..n {
        let mut row_entries = Vec::<(usize, T)>::new();
        row_entries.push((row, T::one()));
        let (row_start, row_end) = matrix.row_bounds(row)?;
        for (&col, &value) in
            matrix.col_indices[row_start..row_end].iter().zip(factors[row_start..row_end].iter())
        {
            let col = col.to_usize()?;
            if col < row {
                row_entries.push((col, value));
            }
        }
        row_entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, value) in row_entries {
            l_indices.push(col);
            l_data.push(value);
        }
        l_indptr.push(l_indices.len());
    }

    let l = CsrMatrix::new(n, n, l_indptr, l_indices, l_data)?;
    let l_transpose = transpose_view(&l.as_view())?;
    Ok(ILDL0Factorization { l, d: diagonal, l_transpose })
}

/// Apply an ILDL(0) preconditioner to a dense vector.
///
/// Solves `L D L^T x = rhs` where factors come from [`ildl0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ildl0_preconditioner<T: NabledReal>(
    factorization: &ILDL0Factorization<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SparseError> {
    if factorization.l.nrows != factorization.l.ncols
        || factorization.l_transpose.nrows != factorization.l_transpose.ncols
        || factorization.l.nrows != factorization.l_transpose.nrows
    {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != factorization.l.nrows || factorization.d.len() != rhs.len() {
        return Err(SparseError::DimensionMismatch);
    }

    let n = rhs.len();
    let mut intermediate = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = rhs[row];
        let mut diagonal = None;
        for entry in factorization.l.indptr[row]..factorization.l.indptr[row + 1] {
            let col = factorization.l.indices[entry];
            let value = factorization.l.data[entry];
            if col < row {
                sum -= value * intermediate[col];
            } else if col == row {
                diagonal = Some(value);
            }
        }
        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        intermediate[row] = sum / diagonal;
    }

    let mut scaled = Array1::<T>::zeros(n);
    for row in 0..n {
        let diagonal = factorization.d[row];
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        scaled[row] = intermediate[row] / diagonal;
    }

    let mut output = Array1::<T>::zeros(n);
    for row_reverse in 0..n {
        let row = n - 1 - row_reverse;
        let mut sum = scaled[row];
        let mut diagonal = None;
        for entry in
            factorization.l_transpose.indptr[row]..factorization.l_transpose.indptr[row + 1]
        {
            let col = factorization.l_transpose.indices[entry];
            let value = factorization.l_transpose.data[entry];
            if col > row {
                sum -= value * output[col];
            } else if col == row {
                diagonal = Some(value);
            }
        }
        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        output[row] = sum / diagonal;
    }

    Ok(output)
}

/// Compute sparse-dense matrix multiplication `Y = A B`.
///
/// `A` is sparse CSR `(m, n)` and `B` is dense `(n, k)`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_dense<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    dense: &Array2<T>,
) -> Result<Array2<T>, SparseError> {
    matmat_dense_view(&matrix.as_view(), dense)
}

/// Compute sparse-dense matrix multiplication `Y = A B` from a borrowed CSR view.
///
/// `A` is sparse CSR `(m, n)` and `B` is dense `(n, k)`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_dense_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    dense: &Array2<T>,
) -> Result<Array2<T>, SparseError> {
    matrix.validate()?;
    let mut output = Array2::<T>::zeros((matrix.nrows, dense.ncols()));
    matmat_dense_view_into(matrix, dense, &mut output)?;
    Ok(output)
}

/// Compute sparse-dense matrix multiplication `Y = A B` via MAGMA sparse (`f64`).
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matmat_dense_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    dense: &Array2<f64>,
) -> Result<Array2<f64>, SparseError> {
    matrix.validate()?;
    if dense.nrows() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if !MagmaProviderPolicy::verify_force_mode()
        && (matrix.nrows < 16 || matrix.ncols < 16 || dense.ncols() < 16)
    {
        return matmat_dense_view(matrix, dense);
    }
    match magma_sparse::spmm_f64(
        matrix.nrows,
        matrix.ncols,
        matrix.row_ptrs,
        matrix.col_indices,
        matrix.values,
        dense,
    ) {
        Ok(result) => Ok(result),
        Err(error) => {
            if error == "provider_failure" && !MagmaProviderPolicy::fail_fast_mode() {
                return matmat_dense_view(matrix, dense);
            }
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_sparse_error(error));
            }
            matmat_dense_view(matrix, dense)
        }
    }
}

/// Compute sparse-dense matrix multiplication `Y = A B` via MAGMA sparse (`f32`).
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matmat_dense_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    dense: &Array2<f32>,
) -> Result<Array2<f32>, SparseError> {
    matrix.validate()?;
    if dense.nrows() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if !MagmaProviderPolicy::verify_force_mode()
        && (matrix.nrows < 16 || matrix.ncols < 16 || dense.ncols() < 16)
    {
        return matmat_dense_view(matrix, dense);
    }
    match magma_sparse::spmm_f32(
        matrix.nrows,
        matrix.ncols,
        matrix.row_ptrs,
        matrix.col_indices,
        matrix.values,
        dense,
    ) {
        Ok(result) => Ok(result),
        Err(error) => {
            if error == "provider_failure" && !MagmaProviderPolicy::fail_fast_mode() {
                return matmat_dense_view(matrix, dense);
            }
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_sparse_error(error));
            }
            matmat_dense_view(matrix, dense)
        }
    }
}

/// Compute sparse-dense matrix multiplication `Y = A B` via MAGMA sparse (`f64`) into `output`.
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matmat_dense_magma_f64_view_into(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    dense: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), SparseError> {
    if output.dim() != (matrix.nrows, dense.ncols()) {
        return Err(SparseError::DimensionMismatch);
    }
    let result = matmat_dense_magma_f64_view(matrix, dense)?;
    output.assign(&result);
    Ok(())
}

/// Compute sparse-dense matrix multiplication `Y = A B` via MAGMA sparse (`f32`) into `output`.
///
/// This path requires feature `magma-system` and an `i32`-indexed CSR view.
///
/// # Errors
/// Returns an error if dimensions/structure are invalid or provider execution fails.
#[cfg(feature = "magma-system")]
pub fn matmat_dense_magma_f32_view_into(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    dense: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<(), SparseError> {
    if output.dim() != (matrix.nrows, dense.ncols()) {
        return Err(SparseError::DimensionMismatch);
    }
    let result = matmat_dense_magma_f32_view(matrix, dense)?;
    output.assign(&result);
    Ok(())
}

#[cfg(feature = "magma-system")]
fn validate_magma_iterative_inputs<T>(
    matrix: &CsrMatrixView<'_, i32, T, i32>,
    rhs: &Array1<T>,
) -> Result<(), SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols || rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }
    Ok(())
}

#[cfg(feature = "magma-system")]
fn expect_vector_len<T>(vector: &Array1<T>, expected: usize) -> Result<(), SparseError> {
    if vector.len() == expected { Ok(()) } else { Err(SparseError::DimensionMismatch) }
}

#[cfg(feature = "magma-system")]
fn conjugate_gradient_with_operator<T: NabledReal>(
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    mut matvec: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
) -> Result<Array1<T>, SparseError> {
    let n = rhs.len();
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut solution = Array1::<T>::zeros(n);
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut residual_norm_sq = dot(&residual, &residual)?;

    if residual_norm_sq.sqrt() <= tolerance {
        return Ok(solution);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec(&direction)?;
        expect_vector_len(&matrix_direction, n)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let alpha = residual_norm_sq / denominator;
        for i in 0..n {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        let next_residual_norm_sq = dot(&residual, &residual)?;
        if next_residual_norm_sq.sqrt() <= tolerance {
            return Ok(solution);
        }

        let beta = next_residual_norm_sq / residual_norm_sq;
        for i in 0..n {
            direction[i] = residual[i] + beta * direction[i];
        }
        residual_norm_sq = next_residual_norm_sq;
    }

    Err(SparseError::MaxIterationsExceeded)
}

#[cfg(feature = "magma-system")]
fn pcg_with_operator<T: NabledReal>(
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    mut matvec: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
    mut precondition: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
) -> Result<Array1<T>, SparseError> {
    let n = rhs.len();
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut solution = Array1::<T>::zeros(n);
    let mut residual = rhs.clone();
    let mut preconditioned_residual = precondition(&residual)?;
    expect_vector_len(&preconditioned_residual, n)?;
    let mut direction = preconditioned_residual.clone();
    let mut rho = dot(&residual, &preconditioned_residual)?;

    if rho.sqrt() <= tolerance {
        return Ok(solution);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec(&direction)?;
        expect_vector_len(&matrix_direction, n)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let alpha = rho / denominator;
        for i in 0..n {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        preconditioned_residual = precondition(&residual)?;
        expect_vector_len(&preconditioned_residual, n)?;
        let rho_next = dot(&residual, &preconditioned_residual)?;
        let beta = rho_next / rho;
        for i in 0..n {
            direction[i] = preconditioned_residual[i] + beta * direction[i];
        }
        rho = rho_next;
    }

    Err(SparseError::MaxIterationsExceeded)
}

#[cfg(feature = "magma-system")]
#[allow(clippy::many_single_char_names)]
fn gmres_with_operator<T: NabledReal>(
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    mut matvec: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
    mut precondition: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
) -> Result<Array1<T>, SparseError> {
    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let preconditioned_rhs = precondition(rhs)?;
    expect_vector_len(&preconditioned_rhs, n)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }

    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<T>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec(&vj)?;
        expect_vector_len(&av, n)?;
        let mut w = precondition(&av)?;
        expect_vector_len(&w, n)?;

        for i in 0..=j {
            let mut hij = T::zero();
            for row in 0..n {
                hij += basis[[row, i]] * w[row];
            }
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = dot(&w, &w)?.sqrt();
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let mut h = Array2::<T>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }

    let ht = h.t();
    let normal_matrix = ht.dot(&h);
    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);
    let y = solve_dense_system(normal_matrix, normal_rhs)?;

    let mut solution = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = T::zero();
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec(&solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

#[cfg(feature = "magma-system")]
fn bicgstab_with_operator<T: NabledReal>(
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    mut matvec: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
    mut precondition: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
) -> Result<Array1<T>, SparseError> {
    let tolerance = tolerance.max(default_tolerance::<T>());
    let n = rhs.len();
    let mut solution = Array1::<T>::zeros(n);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(n);
    let mut search_direction = Array1::<T>::zeros(n);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..n {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = precondition(&search_direction)?;
        expect_vector_len(&preconditioned_search, n)?;
        krylov_vector = matvec(&preconditioned_search)?;
        expect_vector_len(&krylov_vector, n)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..n {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        if dot(&auxiliary_residual, &auxiliary_residual)?.sqrt() <= tolerance {
            for i in 0..n {
                solution[i] += alpha * preconditioned_search[i];
            }
            return Ok(solution);
        }

        let preconditioned_auxiliary = precondition(&auxiliary_residual)?;
        expect_vector_len(&preconditioned_auxiliary, n)?;
        let transformed_auxiliary = matvec(&preconditioned_auxiliary)?;
        expect_vector_len(&transformed_auxiliary, n)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..n {
            solution[i] += alpha * preconditioned_search[i] + omega * preconditioned_auxiliary[i];
        }
        for i in 0..n {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }
        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with MAGMA-backed conjugate gradient (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn conjugate_gradient_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    conjugate_gradient_with_operator(rhs, tolerance, max_iterations, |vector| {
        matvec_magma_f64_view(matrix, vector)
    })
}

/// Solve sparse linear system `A x = b` with MAGMA-backed conjugate gradient (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn conjugate_gradient_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    conjugate_gradient_with_operator(rhs, tolerance, max_iterations, |vector| {
        matvec_magma_f32_view(matrix, vector)
    })
}

/// Solve sparse linear system `A x = b` with MAGMA-backed Jacobi-preconditioned CG (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn pcg_jacobi_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let preconditioner = jacobi_preconditioner_view(matrix)?;
    pcg_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f64_view(matrix, vector),
        |vector| apply_jacobi_preconditioner(&preconditioner, vector),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed Jacobi-preconditioned CG (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn pcg_jacobi_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let preconditioner = jacobi_preconditioner_view(matrix)?;
    pcg_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f32_view(matrix, vector),
        |vector| apply_jacobi_preconditioner(&preconditioner, vector),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed left-preconditioned GMRES (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn gmres_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    gmres_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f64_view(matrix, vector),
        |vector| Ok(vector.clone()),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed left-preconditioned GMRES (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn gmres_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    gmres_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f32_view(matrix, vector),
        |vector| Ok(vector.clone()),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed ILU(0)-preconditioned GMRES (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, factorization breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn gmres_ilu0_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let factorization = ilu0_factor_view(matrix)?;
    gmres_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f64_view(matrix, vector),
        |vector| apply_ilu0_preconditioner(&factorization, vector),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed ILU(0)-preconditioned GMRES (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, factorization breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn gmres_ilu0_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let factorization = ilu0_factor_view(matrix)?;
    gmres_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f32_view(matrix, vector),
        |vector| apply_ilu0_preconditioner(&factorization, vector),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed left-preconditioned `BiCGSTAB` (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn bicgstab_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    bicgstab_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f64_view(matrix, vector),
        |vector| Ok(vector.clone()),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed left-preconditioned `BiCGSTAB` (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, singular breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn bicgstab_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    bicgstab_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f32_view(matrix, vector),
        |vector| Ok(vector.clone()),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed ILU(0)-preconditioned `BiCGSTAB` (`f64`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, factorization breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn bicgstab_ilu0_magma_f64_view(
    matrix: &CsrMatrixView<'_, i32, f64, i32>,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let factorization = ilu0_factor_view(matrix)?;
    bicgstab_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f64_view(matrix, vector),
        |vector| apply_ilu0_preconditioner(&factorization, vector),
    )
}

/// Solve sparse linear system `A x = b` with MAGMA-backed ILU(0)-preconditioned `BiCGSTAB` (`f32`).
///
/// # Errors
/// Returns an error for invalid dimensions/structure, factorization breakdown, or non-convergence.
#[cfg(feature = "magma-system")]
pub fn bicgstab_ilu0_magma_f32_view(
    matrix: &CsrMatrixView<'_, i32, f32, i32>,
    rhs: &Array1<f32>,
    tolerance: f32,
    max_iterations: usize,
) -> Result<Array1<f32>, SparseError> {
    validate_magma_iterative_inputs(matrix, rhs)?;
    let factorization = ilu0_factor_view(matrix)?;
    bicgstab_with_operator(
        rhs,
        tolerance,
        max_iterations,
        |vector| matvec_magma_f32_view(matrix, vector),
        |vector| apply_ilu0_preconditioner(&factorization, vector),
    )
}

/// Compute sparse-dense matrix multiplication `Y = A B` into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_dense_into<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    dense: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SparseError> {
    matmat_dense_view_into(&matrix.as_view(), dense, output)
}

/// Compute sparse-dense matrix multiplication `Y = A B` into `output` from a borrowed CSR view.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_dense_view_into<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    dense: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SparseError> {
    matrix.validate()?;
    if dense.nrows() != matrix.ncols || output.dim() != (matrix.nrows, dense.ncols()) {
        return Err(SparseError::DimensionMismatch);
    }

    output.fill(T::zero());
    for row in 0..matrix.nrows {
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            let col = matrix.col_indices[entry].to_usize()?;
            let value = matrix.values[entry];
            for dense_col in 0..dense.ncols() {
                output[[row, dense_col]] += value * dense[[col, dense_col]];
            }
        }
    }
    Ok(())
}

/// Compute sparse-sparse matrix multiplication `C = A B` in CSR format.
///
/// # Errors
/// Returns an error if dimensions are incompatible or sparse structure is invalid.
pub fn matmat_sparse<T: NabledReal>(
    left: &CsrMatrix<T>,
    right: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, SparseError> {
    matmat_sparse_view(&left.as_view(), &right.as_view())
}

/// Compute sparse-sparse matrix multiplication `C = A B` in CSR format from borrowed CSR views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or sparse structure is invalid.
pub fn matmat_sparse_view<T: NabledReal, LR: CsrIndex, LC: CsrIndex, RR: CsrIndex, RC: CsrIndex>(
    left: &CsrMatrixView<'_, LR, T, LC>,
    right: &CsrMatrixView<'_, RR, T, RC>,
) -> Result<CsrMatrix<T>, SparseError> {
    left.validate()?;
    right.validate()?;
    if left.ncols != right.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let mut row_indptr = Vec::<usize>::with_capacity(left.nrows + 1);
    let mut col_indices = Vec::<usize>::new();
    let mut values = Vec::<T>::new();
    row_indptr.push(0);

    for row in 0..left.nrows {
        let mut accumulator = HashMap::<usize, T>::new();
        let (left_start, left_end) = left.row_bounds(row)?;
        for left_entry in left_start..left_end {
            let left_col = left.col_indices[left_entry].to_usize()?;
            let left_value = left.values[left_entry];
            let (right_start, right_end) = right.row_bounds(left_col)?;
            for right_entry in right_start..right_end {
                let right_col = right.col_indices[right_entry].to_usize()?;
                let right_value = right.values[right_entry];
                let next = accumulator.get(&right_col).copied().unwrap_or(T::zero())
                    + left_value * right_value;
                if next.abs() <= default_tolerance::<T>() {
                    let _ = accumulator.remove(&right_col);
                } else {
                    let _ = accumulator.insert(right_col, next);
                }
            }
        }

        let mut entries = accumulator.into_iter().collect::<Vec<_>>();
        entries.sort_unstable_by_key(|&(col, _)| col);
        for (col, value) in entries {
            col_indices.push(col);
            values.push(value);
        }
        row_indptr.push(col_indices.len());
    }

    CsrMatrix::new(left.nrows, right.ncols, row_indptr, col_indices, values)
}

/// Compute batched sparse matrix-vector products.
///
/// Inputs are row vectors with shape `(batch, ncols)` and output is `(batch, nrows)`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    batch_vectors: &Array2<T>,
) -> Result<Array2<T>, SparseError> {
    let mut output = Array2::<T>::zeros((batch_vectors.nrows(), matrix.nrows));
    batched_matvec_view_into(&matrix.as_view(), batch_vectors, &mut output)?;
    Ok(output)
}

/// Compute batched sparse matrix-vector products from a borrowed CSR view.
///
/// Inputs are row vectors with shape `(batch, ncols)` and output is `(batch, nrows)`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    batch_vectors: &Array2<T>,
) -> Result<Array2<T>, SparseError> {
    let mut output = Array2::<T>::zeros((batch_vectors.nrows(), matrix.nrows));
    batched_matvec_view_into(matrix, batch_vectors, &mut output)?;
    Ok(output)
}

/// Compute batched sparse matrix-vector products into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec_into<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    batch_vectors: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SparseError> {
    batched_matvec_view_into(&matrix.as_view(), batch_vectors, output)
}

/// Compute batched sparse matrix-vector products into `output` from a borrowed CSR view.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec_view_into<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    batch_vectors: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SparseError> {
    matrix.validate()?;
    if batch_vectors.ncols() != matrix.ncols
        || output.dim() != (batch_vectors.nrows(), matrix.nrows)
    {
        return Err(SparseError::DimensionMismatch);
    }

    output.fill(T::zero());
    for batch in 0..batch_vectors.nrows() {
        for row in 0..matrix.nrows {
            let mut sum = T::zero();
            let (start, end) = matrix.row_bounds(row)?;
            for entry in start..end {
                let col = matrix.col_indices[entry].to_usize()?;
                sum += matrix.values[entry] * batch_vectors[[batch, col]];
            }
            output[[batch, row]] = sum;
        }
    }
    Ok(())
}

/// Solve sparse linear system `A x = b` with Jacobi iteration.
///
/// # Errors
/// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
pub fn jacobi_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    jacobi_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with Jacobi iteration from a borrowed CSR view.
///
/// # Errors
/// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
pub fn jacobi_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = matrix.nrows;
    let tolerance = tolerance.max(default_tolerance::<T>());

    let mut diagonal = Array1::<T>::zeros(n);
    for row in 0..n {
        let (start, end) = matrix.row_bounds(row)?;
        for entry in start..end {
            if matrix.col_indices[entry].to_usize()? == row {
                diagonal[row] = matrix.values[entry];
                break;
            }
        }
        if diagonal[row].abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
    }

    let mut x = Array1::<T>::zeros(n);
    let mut x_next = Array1::<T>::zeros(n);

    for _ in 0..max_iterations.max(1) {
        for row in 0..n {
            let (start, end) = matrix.row_bounds(row)?;
            let mut off_diagonal = T::zero();
            for entry in start..end {
                let col = matrix.col_indices[entry].to_usize()?;
                if col != row {
                    off_diagonal += matrix.values[entry] * x[col];
                }
            }
            x_next[row] = (rhs[row] - off_diagonal) / diagonal[row];
        }

        let mut delta_inf = T::zero();
        for i in 0..n {
            delta_inf = delta_inf.max((x_next[i] - x[i]).abs());
            x[i] = x_next[i];
        }

        if delta_inf <= tolerance {
            return Ok(x);
        }
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with Gauss-Seidel iteration.
///
/// # Errors
/// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
pub fn gauss_seidel_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    gauss_seidel_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with Gauss-Seidel iteration from a borrowed CSR view.
///
/// # Errors
/// Returns an error for invalid dimensions, singular diagonals, or non-convergence.
pub fn gauss_seidel_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = matrix.nrows;
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut x = Array1::<T>::zeros(n);

    for _ in 0..max_iterations.max(1) {
        let previous = x.clone();
        for row in 0..n {
            let (start, end) = matrix.row_bounds(row)?;

            let mut diagonal = T::zero();
            let mut sum = T::zero();
            for entry in start..end {
                let col = matrix.col_indices[entry].to_usize()?;
                let value = matrix.values[entry];
                if col == row {
                    diagonal = value;
                } else {
                    sum += value * x[col];
                }
            }

            if diagonal.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }

            x[row] = (rhs[row] - sum) / diagonal;
        }

        let mut delta_inf = T::zero();
        for i in 0..n {
            delta_inf = delta_inf.max((x[i] - previous[i]).abs());
        }
        if delta_inf <= tolerance {
            return Ok(x);
        }
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with conjugate gradient iteration.
///
/// This routine assumes an SPD matrix `A`.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn conjugate_gradient_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    conjugate_gradient_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with conjugate gradient iteration from a borrowed CSR view.
///
/// This routine assumes an SPD matrix `A`.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn conjugate_gradient_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut x = Array1::<T>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut residual_norm_sq = dot(&residual, &residual)?;

    if residual_norm_sq.sqrt() <= tolerance {
        return Ok(x);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec_view(matrix, &direction)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let alpha = residual_norm_sq / denominator;
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        let next_residual_norm_sq = dot(&residual, &residual)?;
        if next_residual_norm_sq.sqrt() <= tolerance {
            return Ok(x);
        }

        let beta = next_residual_norm_sq / residual_norm_sq;
        for i in 0..direction.len() {
            direction[i] = residual[i] + beta * direction[i];
        }
        residual_norm_sq = next_residual_norm_sq;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with preconditioned conjugate gradient.
///
/// This routine assumes an SPD matrix `A` and uses a Jacobi preconditioner.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn pcg_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    pcg_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with preconditioned conjugate gradient from a borrowed CSR
/// view.
///
/// This routine assumes an SPD matrix `A` and uses a Jacobi preconditioner.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn pcg_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let preconditioner = jacobi_preconditioner_view(matrix)?;
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut solution = Array1::<T>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut preconditioned_residual = apply_jacobi_preconditioner(&preconditioner, &residual)?;
    let mut direction = preconditioned_residual.clone();
    let mut rho = dot(&residual, &preconditioned_residual)?;

    if rho.sqrt() <= tolerance {
        return Ok(solution);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec_view(matrix, &direction)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let alpha = rho / denominator;
        for i in 0..solution.len() {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        let residual_norm = dot(&residual, &residual)?.sqrt();
        if residual_norm <= tolerance {
            return Ok(solution);
        }

        preconditioned_residual = apply_jacobi_preconditioner(&preconditioner, &residual)?;
        let rho_next = dot(&residual, &preconditioned_residual)?;
        let beta = rho_next / rho;
        for i in 0..direction.len() {
            direction[i] = preconditioned_residual[i] + beta * direction[i];
        }
        rho = rho_next;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with IC(0)-preconditioned conjugate gradient.
///
/// This routine assumes an SPD matrix `A`.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn pcg_ic0_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    pcg_ic0_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with IC(0)-preconditioned conjugate gradient from a
/// borrowed CSR view.
///
/// This routine assumes an SPD matrix `A`.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn pcg_ic0_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let factorization = ic0_factor_view(matrix)?;
    let tolerance = tolerance.max(default_tolerance::<T>());
    let mut solution = Array1::<T>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut preconditioned_residual = apply_ic0_preconditioner(&factorization, &residual)?;
    let mut direction = preconditioned_residual.clone();
    let mut rho = dot(&residual, &preconditioned_residual)?;

    if rho.sqrt() <= tolerance {
        return Ok(solution);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec_view(matrix, &direction)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        let alpha = rho / denominator;
        for i in 0..solution.len() {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * matrix_direction[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        preconditioned_residual = apply_ic0_preconditioner(&factorization, &residual)?;
        let rho_next = dot(&residual, &preconditioned_residual)?;
        let beta = rho_next / rho;
        for i in 0..direction.len() {
            direction[i] = preconditioned_residual[i] + beta * direction[i];
        }
        rho = rho_next;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned GMRES.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilu0_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    gmres_ilu0_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned GMRES from a borrowed CSR view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilu0_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilu0_factor_view(matrix)?;
    gmres_ilu0_solve_with_factorization_view(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilu0_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    gmres_ilu0_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilu0_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(default_tolerance::<T>());

    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ilu0_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<T>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec_view(matrix, &vj)?;
        let mut w = apply_ilu0_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = T::zero();
            for row in 0..n {
                hij += basis[[row, i]] * w[row];
            }
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = dot(&w, &w)?.sqrt();
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let mut h = Array2::<T>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y = solve_dense_system(normal_matrix, normal_rhs)?;

    let mut solution = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = T::zero();
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec_view(matrix, &solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilut_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<Array1<T>, SparseError> {
    gmres_ilut_solve_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        drop_tolerance,
        max_fill,
    )
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES from a borrowed CSR view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilut_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilut_factor_view(matrix, drop_tolerance, max_fill)?;
    gmres_ilut_solve_with_factorization_view(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilut_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    gmres_ilut_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilut_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(default_tolerance::<T>());

    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ilut_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<T>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec_view(matrix, &vj)?;
        let mut w = apply_ilut_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = T::zero();
            for row in 0..n {
                hij += basis[[row, i]] * w[row];
            }
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = dot(&w, &w)?.sqrt();
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let mut h = Array2::<T>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y = solve_dense_system(normal_matrix, normal_rhs)?;

    let mut solution = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = T::zero();
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec_view(matrix, &solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES.
///
/// Uses an [`ILUTConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilut_solve_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUTConfig<T>,
) -> Result<Array1<T>, SparseError> {
    gmres_ilut_solve_with_config_view(&matrix.as_view(), rhs, tolerance, max_iterations, config)
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES from a borrowed CSR view.
///
/// Uses an [`ILUTConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilut_solve_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUTConfig<T>,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilut_factor_with_config_view(matrix, config)?;
    gmres_ilut_solve_with_factorization_view(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<T>, SparseError> {
    gmres_iluk_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations, level_of_fill)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES from a borrowed CSR view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = iluk_factor_view(matrix, level_of_fill)?;
    gmres_iluk_solve_with_factorization_view(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    gmres_iluk_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(default_tolerance::<T>());

    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let preconditioned_rhs = apply_iluk_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<T>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec_view(matrix, &vj)?;
        let mut w = apply_iluk_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = T::zero();
            for row in 0..n {
                hij += basis[[row, i]] * w[row];
            }
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = dot(&w, &w)?.sqrt();
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let mut h = Array2::<T>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y = solve_dense_system(normal_matrix, normal_rhs)?;

    let mut solution = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = T::zero();
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec_view(matrix, &solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES.
///
/// Uses an [`ILUKConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_iluk_solve_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<T>, SparseError> {
    gmres_iluk_solve_with_config_view(&matrix.as_view(), rhs, tolerance, max_iterations, config)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses an [`ILUKConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_iluk_solve_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<T>, SparseError> {
    let factorization = iluk_factor_with_config_view(matrix, config)?;
    gmres_iluk_solve_with_factorization_view(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    gmres_ildl0_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES from a borrowed CSR view.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ildl0_factor_view(matrix)?;
    gmres_ildl0_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    gmres_ildl0_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(default_tolerance::<T>());

    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ildl0_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<T>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec_view(matrix, &vj)?;
        let mut w = apply_ildl0_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = T::zero();
            for row in 0..n {
                hij += basis[[row, i]] * w[row];
            }
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = dot(&w, &w)?.sqrt();
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let mut h = Array2::<T>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y = solve_dense_system(normal_matrix, normal_rhs)?;

    let mut solution = Array1::<T>::zeros(n);
    for row in 0..n {
        let mut sum = T::zero();
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec_view(matrix, &solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

fn solve_multiple_rhs_with_solver_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    mut solve_column: impl FnMut(&Array1<T>) -> Result<Array1<T>, SparseError>,
) -> Result<Array2<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.nrows() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let mut output = Array2::<T>::zeros((matrix.ncols, rhs.ncols()));
    for col in 0..rhs.ncols() {
        let rhs_column = rhs.column(col).to_owned();
        let solution = solve_column(&rhs_column)?;
        if solution.len() != output.nrows() {
            return Err(SparseError::DimensionMismatch);
        }
        output.column_mut(col).assign(&solution);
    }
    Ok(output)
}

/// Solve sparse linear system `A x = b` with `BiCGSTAB` iteration.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn bicgstab_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    bicgstab_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with `BiCGSTAB` iteration from a borrowed CSR view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn bicgstab_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let dimension = rhs.len();
    let mut solution = Array1::<T>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(dimension);
    let mut search_direction = Array1::<T>::zeros(dimension);

    let residual_norm = dot(&residual, &residual)?.sqrt();
    if residual_norm <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        krylov_vector = matvec_view(matrix, &search_direction)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..dimension {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        let auxiliary_norm = dot(&auxiliary_residual, &auxiliary_residual)?.sqrt();
        if auxiliary_norm <= tolerance {
            for i in 0..dimension {
                solution[i] += alpha * search_direction[i];
            }
            return Ok(solution);
        }

        let transformed_auxiliary = matvec_view(matrix, &auxiliary_residual)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..dimension {
            solution[i] += alpha * search_direction[i] + omega * auxiliary_residual[i];
        }

        for i in 0..dimension {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        let residual_norm = dot(&residual, &residual)?.sqrt();
        if residual_norm <= tolerance {
            return Ok(solution);
        }

        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned `BiCGSTAB`.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ilu0_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilu0_factor_view(matrix)?;
    bicgstab_ilu0_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ilu0_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let dimension = rhs.len();
    let mut solution = Array1::<T>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(dimension);
    let mut search_direction = Array1::<T>::zeros(dimension);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = apply_ilu0_preconditioner(factorization, &search_direction)?;
        krylov_vector = matvec_view(matrix, &preconditioned_search)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..dimension {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        if dot(&auxiliary_residual, &auxiliary_residual)?.sqrt() <= tolerance {
            for i in 0..dimension {
                solution[i] += alpha * preconditioned_search[i];
            }
            return Ok(solution);
        }

        let preconditioned_auxiliary =
            apply_ilu0_preconditioner(factorization, &auxiliary_residual)?;
        let transformed_auxiliary = matvec_view(matrix, &preconditioned_auxiliary)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..dimension {
            solution[i] += alpha * preconditioned_search[i] + omega * preconditioned_auxiliary[i];
        }

        for i in 0..dimension {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear systems `A X = B` with ILU(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    bicgstab_ilu0_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILU(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        bicgstab_ilu0_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB`.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ilut_solve_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        drop_tolerance,
        max_fill,
    )
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    drop_tolerance: T,
    max_fill: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilut_factor_view(matrix, drop_tolerance, max_fill)?;
    bicgstab_ilut_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ilut_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let dimension = rhs.len();
    let mut solution = Array1::<T>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(dimension);
    let mut search_direction = Array1::<T>::zeros(dimension);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = apply_ilut_preconditioner(factorization, &search_direction)?;
        krylov_vector = matvec_view(matrix, &preconditioned_search)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..dimension {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        if dot(&auxiliary_residual, &auxiliary_residual)?.sqrt() <= tolerance {
            for i in 0..dimension {
                solution[i] += alpha * preconditioned_search[i];
            }
            return Ok(solution);
        }

        let preconditioned_auxiliary =
            apply_ilut_preconditioner(factorization, &auxiliary_residual)?;
        let transformed_auxiliary = matvec_view(matrix, &preconditioned_auxiliary)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..dimension {
            solution[i] += alpha * preconditioned_search[i] + omega * preconditioned_auxiliary[i];
        }

        for i in 0..dimension {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear systems `A X = B` with ILUT-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    bicgstab_ilut_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILUT-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        bicgstab_ilut_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear systems `A X = B` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    bicgstab_iluk_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILU(k)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        bicgstab_iluk_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB`.
///
/// Uses an [`ILUTConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUTConfig<T>,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ilut_solve_with_config_view(&matrix.as_view(), rhs, tolerance, max_iterations, config)
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses an [`ILUTConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUTConfig<T>,
) -> Result<Array1<T>, SparseError> {
    let factorization = ilut_factor_with_config_view(matrix, config)?;
    bicgstab_ilut_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<T>, SparseError> {
    bicgstab_iluk_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations, level_of_fill)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = iluk_factor_view(matrix, level_of_fill)?;
    bicgstab_iluk_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    bicgstab_iluk_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let dimension = rhs.len();
    let mut solution = Array1::<T>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(dimension);
    let mut search_direction = Array1::<T>::zeros(dimension);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = apply_iluk_preconditioner(factorization, &search_direction)?;
        krylov_vector = matvec_view(matrix, &preconditioned_search)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..dimension {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        if dot(&auxiliary_residual, &auxiliary_residual)?.sqrt() <= tolerance {
            for i in 0..dimension {
                solution[i] += alpha * preconditioned_search[i];
            }
            return Ok(solution);
        }

        let preconditioned_auxiliary =
            apply_iluk_preconditioner(factorization, &auxiliary_residual)?;
        let transformed_auxiliary = matvec_view(matrix, &preconditioned_auxiliary)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..dimension {
            solution[i] += alpha * preconditioned_search[i] + omega * preconditioned_auxiliary[i];
        }

        for i in 0..dimension {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// Uses an [`ILUKConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_config<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<T>, SparseError> {
    bicgstab_iluk_solve_with_config_view(&matrix.as_view(), rhs, tolerance, max_iterations, config)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses an [`ILUKConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_config_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<T>, SparseError> {
    let factorization = iluk_factor_with_config_view(matrix, config)?;
    bicgstab_iluk_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ildl0_solve_view(&matrix.as_view(), rhs, tolerance, max_iterations)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
) -> Result<Array1<T>, SparseError> {
    let factorization = ildl0_factor_view(matrix)?;
    bicgstab_ildl0_solve_with_factorization_view(
        matrix,
        rhs,
        tolerance,
        max_iterations,
        &factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    bicgstab_ildl0_solve_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_with_factorization_view<T: NabledReal, R: CsrIndex, C: CsrIndex>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array1<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array1<T>, SparseError> {
    matrix.validate()?;
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(default_tolerance::<T>());
    let dimension = rhs.len();
    let mut solution = Array1::<T>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = T::one();
    let mut alpha = T::one();
    let mut omega = T::one();
    let mut krylov_vector = Array1::<T>::zeros(dimension);
    let mut search_direction = Array1::<T>::zeros(dimension);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= default_tolerance::<T>() {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = apply_ildl0_preconditioner(factorization, &search_direction)?;
        krylov_vector = matvec_view(matrix, &preconditioned_search)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        alpha = rho / denominator;

        let mut auxiliary_residual = residual.clone();
        for i in 0..dimension {
            auxiliary_residual[i] -= alpha * krylov_vector[i];
        }

        if dot(&auxiliary_residual, &auxiliary_residual)?.sqrt() <= tolerance {
            for i in 0..dimension {
                solution[i] += alpha * preconditioned_search[i];
            }
            return Ok(solution);
        }

        let preconditioned_auxiliary =
            apply_ildl0_preconditioner(factorization, &auxiliary_residual)?;
        let transformed_auxiliary = matvec_view(matrix, &preconditioned_auxiliary)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= default_tolerance::<T>() {
            return Err(SparseError::SingularMatrix);
        }

        for i in 0..dimension {
            solution[i] += alpha * preconditioned_search[i] + omega * preconditioned_auxiliary[i];
        }

        for i in 0..dimension {
            residual[i] = auxiliary_residual[i] - omega * transformed_auxiliary[i];
        }

        if dot(&residual, &residual)?.sqrt() <= tolerance {
            return Ok(solution);
        }

        rho_prev = rho;
    }

    Err(SparseError::MaxIterationsExceeded)
}

/// Solve sparse linear systems `A X = B` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    bicgstab_ildl0_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILDL(0)-preconditioned `BiCGSTAB` from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        bicgstab_ildl0_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear systems `A X = B` with ILU(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilu0_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    gmres_ilu0_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILU(0)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilu0_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILU0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        gmres_ilu0_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear systems `A X = B` with ILUT-preconditioned GMRES.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilut_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    gmres_ilut_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILUT-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ilut_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUTFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        gmres_ilut_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear systems `A X = B` with ILU(k)-preconditioned GMRES.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_iluk_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    gmres_iluk_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILU(k)-preconditioned GMRES from a borrowed CSR view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_iluk_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILUKFactorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        gmres_iluk_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

/// Solve sparse linear systems `A X = B` with ILDL(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ildl0_solve_multiple_with_factorization<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    gmres_ildl0_solve_multiple_with_factorization_view(
        &matrix.as_view(),
        rhs,
        tolerance,
        max_iterations,
        factorization,
    )
}

/// Solve sparse linear systems `A X = B` with ILDL(0)-preconditioned GMRES from a borrowed CSR
/// view.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn gmres_ildl0_solve_multiple_with_factorization_view<
    T: NabledReal,
    R: CsrIndex,
    C: CsrIndex,
>(
    matrix: &CsrMatrixView<'_, R, T, C>,
    rhs: &Array2<T>,
    tolerance: T,
    max_iterations: usize,
    factorization: &ILDL0Factorization<T>,
) -> Result<Array2<T>, SparseError> {
    solve_multiple_rhs_with_solver_view(matrix, rhs, |rhs_column| {
        gmres_ildl0_solve_with_factorization_view(
            matrix,
            rhs_column,
            tolerance,
            max_iterations,
            factorization,
        )
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, arr1};

    use super::*;

    fn toy_matrix() -> CsrMatrix {
        // [4 1 0]
        // [1 3 1]
        // [0 1 2]
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap()
    }

    fn toy_matrix_f32() -> CsrMatrix<f32> {
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f32, 1.0_f32, 1.0_f32, 3.0_f32, 1.0_f32, 1.0_f32, 2.0_f32,
        ])
        .unwrap()
    }

    fn symmetric_indefinite_matrix() -> CsrMatrix {
        // [4 1 0]
        // [1 0 1]
        // [0 1 3]
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 3.0_f64,
        ])
        .unwrap()
    }

    fn csr_to_dense<T: NabledReal>(matrix: &CsrMatrix<T>) -> Array2<T> {
        let mut dense = Array2::<T>::zeros((matrix.nrows, matrix.ncols));
        for row in 0..matrix.nrows {
            for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
                dense[[row, matrix.indices[entry]]] = matrix.data[entry];
            }
        }
        dense
    }

    #[test]
    fn matvec_matches_expected() {
        let matrix = toy_matrix();
        let vector = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let y = matvec(&matrix, &vector).unwrap();
        assert!((y[0] - 6.0_f64).abs() < 1e-12_f64);
        assert!((y[1] - 10.0_f64).abs() < 1e-12_f64);
        assert!((y[2] - 8.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn real_f32_core_and_view_paths_work() {
        let matrix = toy_matrix_f32();
        let rhs = arr1(&[1.0_f32, 2.0_f32, 3.0_f32]);

        let owned_product = matvec(&matrix, &rhs).unwrap();
        let mut into_product = Array1::<f32>::zeros(rhs.len());
        matvec_into(&matrix, &rhs, &mut into_product).unwrap();
        for i in 0..rhs.len() {
            assert!((owned_product[i] - into_product[i]).abs() < 1e-6_f32);
        }

        let row_ptrs = vec![0_i32, 2, 5, 7];
        let col_indices = vec![0_u32, 1, 0, 1, 2, 1, 2];
        let view = CsrMatrixView::new(3, 3, &row_ptrs, &col_indices, &matrix.data).unwrap();
        let view_product = matvec_view(&view, &rhs).unwrap();
        for i in 0..rhs.len() {
            assert!((owned_product[i] - view_product[i]).abs() < 1e-6_f32);
        }

        let dense_rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f32, 0.0_f32, 0.5_f32, 2.0_f32, 2.0_f32, -1.0_f32,
        ])
        .unwrap();
        let dense_owned = matmat_dense(&matrix, &dense_rhs).unwrap();
        let dense_view = matmat_dense_view(&view, &dense_rhs).unwrap();
        assert_eq!(dense_owned, dense_view);

        let transpose_owned = transpose(&matrix).unwrap();
        let transpose_viewed = transpose_view(&view).unwrap();
        assert_eq!(transpose_owned, transpose_viewed);

        let jacobi_solution = jacobi_solve_view(&view, &rhs, 1e-5_f32, 5_000).unwrap();
        let reconstructed = matvec(&matrix, &jacobi_solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-3_f32);
        }

        let level0_factorization = ilu0_factor_view(&view).unwrap();
        let level0_result = apply_ilu0_preconditioner(&level0_factorization, &rhs).unwrap();
        assert_eq!(level0_result.len(), rhs.len());

        let threshold_factorization =
            ilut_factor_with_config_view(&view, ILUTConfig::<f32>::balanced()).unwrap();
        let threshold_result = apply_ilut_preconditioner(&threshold_factorization, &rhs).unwrap();
        assert_eq!(threshold_result.len(), rhs.len());

        let fill_factorization =
            iluk_factor_with_config_view(&view, ILUKConfig::balanced()).unwrap();
        let fill_result = apply_iluk_preconditioner(&fill_factorization, &rhs).unwrap();
        assert_eq!(fill_result.len(), rhs.len());

        let lu = sparse_lu_factor_view(&view).unwrap();
        let lu_solution = sparse_lu_solve_with_factorization_view(&view, &rhs, &lu).unwrap();
        let lu_reconstructed = matvec(&matrix, &lu_solution).unwrap();
        for i in 0..rhs.len() {
            assert!((lu_reconstructed[i] - rhs[i]).abs() < 1e-4_f32);
        }
    }

    #[test]
    fn csr_view_i32_u32_matches_owned_matvec_and_products() {
        let matrix = toy_matrix();
        let vector = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let dense_right = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, 0.5_f64, 2.0_f64, 2.0_f64, -1.0_f64,
        ])
        .unwrap();

        let row_ptrs = vec![0_i32, 2, 5, 7];
        let col_indices = vec![0_u32, 1, 0, 1, 2, 1, 2];
        let view = CsrMatrixView::new(3, 3, &row_ptrs, &col_indices, &matrix.data).unwrap();

        let owned_y = matvec(&matrix, &vector).unwrap();
        let view_y = matvec_view(&view, &vector).unwrap();
        assert_eq!(view_y, owned_y);

        let owned_dense = matmat_dense(&matrix, &dense_right).unwrap();
        let view_dense = matmat_dense_view(&view, &dense_right).unwrap();
        assert_eq!(view_dense, owned_dense);

        let owned_transpose = transpose(&matrix).unwrap();
        let view_transpose = transpose_view(&view).unwrap();
        assert_eq!(view_transpose, owned_transpose);
    }

    #[test]
    fn csr_view_i32_u32_iterative_and_factorization_paths_match_owned() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let row_ptrs = vec![0_i32, 2, 5, 7];
        let col_indices = vec![0_u32, 1, 0, 1, 2, 1, 2];
        let view = CsrMatrixView::new(3, 3, &row_ptrs, &col_indices, &matrix.data).unwrap();

        let owned_jacobi = jacobi_solve(&matrix, &rhs, 1.0e-10_f64, 5_000).unwrap();
        let view_jacobi = jacobi_solve_view(&view, &rhs, 1.0e-10_f64, 5_000).unwrap();
        for i in 0..rhs.len() {
            assert!((owned_jacobi[i] - view_jacobi[i]).abs() < 1.0e-8_f64);
        }

        let owned_lu = sparse_lu_factor(&matrix).unwrap();
        let view_lu = sparse_lu_factor_view(&view).unwrap();
        assert_eq!(owned_lu.permutation, view_lu.permutation);
        assert_eq!(owned_lu.l, view_lu.l);
        assert_eq!(owned_lu.u, view_lu.u);
    }

    #[test]
    fn jacobi_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let solution = jacobi_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn rejects_invalid_structure() {
        let result = CsrMatrix::new(2, 2, vec![0, 1], vec![0], vec![1.0_f64]);
        assert!(matches!(result, Err(SparseError::InvalidStructure)));
    }

    #[test]
    fn coo_to_csr_roundtrip_matvec() {
        let coo = CooMatrix::new(3, 3, vec![0, 0, 1, 1, 1, 2, 2], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let csr = coo.to_csr().unwrap();
        let vector = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let y = matvec(&csr, &vector).unwrap();
        assert!((y[0] - 6.0_f64).abs() < 1e-12_f64);
        assert!((y[1] - 10.0_f64).abs() < 1e-12_f64);
        assert!((y[2] - 8.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn gauss_seidel_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let solution = gauss_seidel_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn transpose_roundtrip_matches_original() {
        let matrix = toy_matrix();
        let transposed = transpose(&matrix).unwrap();
        let twice = transpose(&transposed).unwrap();
        assert_eq!(matrix.nrows, twice.nrows);
        assert_eq!(matrix.ncols, twice.ncols);
        assert_eq!(matrix.indptr, twice.indptr);
        assert_eq!(matrix.indices, twice.indices);
        assert_eq!(matrix.data, twice.data);
    }

    #[test]
    fn csc_conversion_and_matvec_match_csr() {
        let matrix = toy_matrix();
        let csc = csr_to_csc(&matrix).unwrap();
        let roundtrip = csc.to_csr().unwrap();
        assert_eq!(matrix.nrows, roundtrip.nrows);
        assert_eq!(matrix.ncols, roundtrip.ncols);
        assert_eq!(matrix.indptr, roundtrip.indptr);
        assert_eq!(matrix.indices, roundtrip.indices);
        assert_eq!(matrix.data, roundtrip.data);

        let vector = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let reference_product = matvec(&matrix, &vector).unwrap();
        let converted_product = matvec_csc(&csc, &vector).unwrap();
        for i in 0..vector.len() {
            assert!((reference_product[i] - converted_product[i]).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn sparse_dense_matmat_matches_expected() {
        let matrix = toy_matrix();
        let dense = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
        ])
        .unwrap();
        let y = matmat_dense(&matrix, &dense).unwrap();
        assert_eq!(y.dim(), (3, 2));
        assert!((y[[0, 0]] - 4.0_f64).abs() < 1e-12_f64);
        assert!((y[[0, 1]] - 1.0_f64).abs() < 1e-12_f64);
        assert!((y[[1, 0]] - 2.0_f64).abs() < 1e-12_f64);
        assert!((y[[1, 1]] - 4.0_f64).abs() < 1e-12_f64);
        assert!((y[[2, 0]] - 2.0_f64).abs() < 1e-12_f64);
        assert!((y[[2, 1]] - 3.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn batched_matvec_matches_single_path() {
        let matrix = toy_matrix();
        let batch = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 3.0_f64, 2.0_f64, 1.0_f64,
        ])
        .unwrap();
        let batched = batched_matvec(&matrix, &batch).unwrap();
        assert_eq!(batched.dim(), (2, 3));
        let first = matvec(&matrix, &arr1(&[1.0_f64, 2.0_f64, 3.0_f64])).unwrap();
        let second = matvec(&matrix, &arr1(&[3.0_f64, 2.0_f64, 1.0_f64])).unwrap();
        for i in 0..3 {
            assert!((batched[[0, i]] - first[i]).abs() < 1e-12_f64);
            assert!((batched[[1, i]] - second[i]).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn conjugate_gradient_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let solution = conjugate_gradient_solve(&matrix, &rhs, 1e-10_f64, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn jacobi_preconditioner_builds_and_applies() {
        let matrix = toy_matrix();
        let preconditioner = jacobi_preconditioner(&matrix).unwrap();
        assert_eq!(preconditioner.inverse_diagonal.len(), 3);
        assert!((preconditioner.inverse_diagonal[0] - 0.25_f64).abs() < 1e-12_f64);
        assert!((preconditioner.inverse_diagonal[1] - (1.0_f64 / 3.0_f64)).abs() < 1e-12_f64);
        assert!((preconditioner.inverse_diagonal[2] - 0.5_f64).abs() < 1e-12_f64);

        let rhs = arr1(&[4.0_f64, 6.0_f64, 2.0_f64]);
        let transformed = apply_jacobi_preconditioner(&preconditioner, &rhs).unwrap();
        assert!((transformed[0] - 1.0_f64).abs() < 1e-12_f64);
        assert!((transformed[1] - 2.0_f64).abs() < 1e-12_f64);
        assert!((transformed[2] - 1.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn pcg_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let solution = pcg_solve(&matrix, &rhs, 1e-10_f64, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn sparse_sparse_matmat_matches_dense_result() {
        let matrix = toy_matrix();
        let product = matmat_sparse(&matrix, &matrix).unwrap();
        let vector = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);

        let lhs = matvec(&matrix, &vector).unwrap();
        let expected = matvec(&matrix, &lhs).unwrap();
        let observed = matvec(&product, &vector).unwrap();
        for i in 0..vector.len() {
            assert!((expected[i] - observed[i]).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_system() {
        // [4 1 0]
        // [2 3 1]
        // [0 1 2]
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = bicgstab_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn ilu0_factorization_reconstructs_toy_matrix() {
        let matrix = toy_matrix();
        let factorization = ilu0_factor(&matrix).unwrap();

        let dense_l = csr_to_dense(&factorization.l);
        let dense_u = csr_to_dense(&factorization.u);
        let reconstructed = dense_l.dot(&dense_u);
        let expected = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            for col in 0..matrix.ncols {
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn apply_ilu0_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ilu0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = apply_ilu0_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn ilut_factorization_reconstructs_toy_matrix() {
        let matrix = toy_matrix();
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();

        let dense_l = csr_to_dense(&factorization.l);
        let dense_u = csr_to_dense(&factorization.u);
        let reconstructed = dense_l.dot(&dense_u);
        let expected = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            for col in 0..matrix.ncols {
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn apply_ilut_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = apply_ilut_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn iluk_factorization_reconstructs_toy_matrix_with_fill() {
        let matrix = toy_matrix();
        let factorization = iluk_factor(&matrix, 1).unwrap();

        let dense_l = csr_to_dense(&factorization.l);
        let dense_u = csr_to_dense(&factorization.u);
        let reconstructed = dense_l.dot(&dense_u);
        let expected = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            for col in 0..matrix.ncols {
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn apply_iluk_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = apply_iluk_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn sparse_lu_factorization_reconstructs_permuted_matrix() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            0.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let factorization = sparse_lu_factor(&matrix).unwrap();
        let dense_l = csr_to_dense(&factorization.l);
        let dense_u = csr_to_dense(&factorization.u);
        let lu = dense_l.dot(&dense_u);
        let dense_a = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            let original_row = factorization.permutation[row];
            for col in 0..matrix.ncols {
                assert!((lu[[row, col]] - dense_a[[original_row, col]]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn sparse_lu_solve_reconstructs_rhs() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            0.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = sparse_lu_solve(&matrix, &rhs).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-7_f64);
        }
    }

    #[test]
    fn sparse_lu_reuse_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            0.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = sparse_lu_factor(&matrix).unwrap();
        let direct = sparse_lu_solve(&matrix, &rhs).unwrap();
        let reused = sparse_lu_solve_with_factorization(&matrix, &rhs, &factorization).unwrap();
        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn sparse_lu_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            0.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, -2.0_f64, 1.5_f64, 3.0_f64, -1.0_f64,
        ])
        .unwrap();
        let factorization = sparse_lu_factor(&matrix).unwrap();
        let multi =
            sparse_lu_solve_multiple_with_factorization(&matrix, &rhs, &factorization).unwrap();
        for col in 0..rhs.ncols() {
            let single = sparse_lu_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn sparse_lu_rejects_singular_matrix() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 4, 6], vec![0, 1, 0, 1, 1, 2], vec![
            1.0_f64, 2.0_f64, 2.0_f64, 4.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let result = sparse_lu_factor(&matrix);
        assert!(matches!(result, Err(SparseError::SingularMatrix)));
    }

    #[test]
    fn ic0_factorization_reconstructs_toy_matrix() {
        let matrix = toy_matrix();
        let factorization = ic0_factor(&matrix).unwrap();

        let dense_l = csr_to_dense(&factorization.l);
        let reconstructed = dense_l.dot(&dense_l.t());
        let expected = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            for col in 0..matrix.ncols {
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn apply_ic0_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ic0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = apply_ic0_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn ildl0_factorization_reconstructs_symmetric_indefinite_matrix() {
        let matrix = symmetric_indefinite_matrix();
        let factorization = ildl0_factor(&matrix).unwrap();

        let dense_l = csr_to_dense(&factorization.l);
        let mut dense_ld = dense_l.clone();
        for row in 0..dense_ld.nrows() {
            for col in 0..dense_ld.ncols() {
                dense_ld[[row, col]] *= factorization.d[col];
            }
        }
        let reconstructed = dense_ld.dot(&dense_l.t());
        let expected = csr_to_dense(&matrix);

        for row in 0..matrix.nrows {
            for col in 0..matrix.ncols {
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn apply_ildl0_preconditioner_rejects_bad_dimensions() {
        let matrix = symmetric_indefinite_matrix();
        let factorization = ildl0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = apply_ildl0_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn ildl0_factorization_rejects_nonsymmetric_input() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let result = ildl0_factor(&matrix);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn pcg_ic0_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64, 3.0_f64]);
        let solution = pcg_ic0_solve(&matrix, &rhs, 1e-10_f64, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_ilu0_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = bicgstab_ilu0_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_ilu0_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ilu0_factor(&matrix).unwrap();

        let direct = bicgstab_ilu0_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reused =
            bicgstab_ilu0_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn bicgstab_ilut_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = bicgstab_ilut_solve(&matrix, &rhs, 1e-10_f64, 5000, 0.0_f64, 8).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_ilut_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution =
            bicgstab_ilut_solve_with_config(&matrix, &rhs, 1e-10_f64, 5000, ILUTConfig::balanced())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_ilut_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();

        let direct = bicgstab_ilut_solve(&matrix, &rhs, 1e-10_f64, 5000, 0.0_f64, 8).unwrap();
        let reused =
            bicgstab_ilut_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn bicgstab_iluk_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = bicgstab_iluk_solve(&matrix, &rhs, 1e-10_f64, 5000, 1).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_iluk_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution =
            bicgstab_iluk_solve_with_config(&matrix, &rhs, 1e-10_f64, 5000, ILUKConfig::balanced())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_iluk_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = iluk_factor(&matrix, 1).unwrap();

        let direct = bicgstab_iluk_solve(&matrix, &rhs, 1e-10_f64, 5000, 1).unwrap();
        let reused =
            bicgstab_iluk_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn bicgstab_ildl0_solves_symmetric_indefinite_system() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = bicgstab_ildl0_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn bicgstab_ildl0_with_factorization_matches_direct() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ildl0_factor(&matrix).unwrap();

        let direct = bicgstab_ildl0_solve(&matrix, &rhs, 1e-10_f64, 5000).unwrap();
        let reused =
            bicgstab_ildl0_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn gmres_ilu0_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = gmres_ilu0_solve(&matrix, &rhs, 1e-10_f64, 10).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn gmres_ilu0_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ilu0_factor(&matrix).unwrap();

        let direct = gmres_ilu0_solve(&matrix, &rhs, 1e-10_f64, 10).unwrap();
        let reused =
            gmres_ilu0_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 10, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn gmres_ilut_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = gmres_ilut_solve(&matrix, &rhs, 1e-10_f64, 10, 0.0_f64, 8).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn gmres_ilut_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();

        let direct = gmres_ilut_solve(&matrix, &rhs, 1e-10_f64, 10, 0.0_f64, 8).unwrap();
        let reused =
            gmres_ilut_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 10, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn gmres_iluk_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = gmres_iluk_solve(&matrix, &rhs, 1e-10_f64, 10, 1).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn gmres_iluk_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = iluk_factor(&matrix, 1).unwrap();

        let direct = gmres_iluk_solve(&matrix, &rhs, 1e-10_f64, 10, 1).unwrap();
        let reused =
            gmres_iluk_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 10, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn gmres_iluk_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution =
            gmres_iluk_solve_with_config(&matrix, &rhs, 1e-10_f64, 10, ILUKConfig::aggressive())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn gmres_ildl0_solves_symmetric_indefinite_system() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution = gmres_ildl0_solve(&matrix, &rhs, 1e-10_f64, 32).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn gmres_ildl0_with_factorization_matches_direct() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let factorization = ildl0_factor(&matrix).unwrap();

        let direct = gmres_ildl0_solve(&matrix, &rhs, 1e-10_f64, 32).unwrap();
        let reused =
            gmres_ildl0_solve_with_factorization(&matrix, &rhs, 1e-10_f64, 32, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn gmres_ilut_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0_f64, 3.0_f64]);
        let solution =
            gmres_ilut_solve_with_config(&matrix, &rhs, 1e-10_f64, 10, ILUTConfig::aggressive())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn ilut_config_profiles_are_ordered() {
        let conservative = ILUTConfig::<f64>::conservative();
        let balanced = ILUTConfig::<f64>::balanced();
        let aggressive = ILUTConfig::<f64>::aggressive();
        assert!(conservative.drop_tolerance >= balanced.drop_tolerance);
        assert!(balanced.drop_tolerance >= aggressive.drop_tolerance);
        assert!(conservative.max_fill <= balanced.max_fill);
        assert!(balanced.max_fill <= aggressive.max_fill);

        let small = ILUTConfig::<f64>::for_dimension(16);
        let medium = ILUTConfig::<f64>::for_dimension(128);
        let large = ILUTConfig::<f64>::for_dimension(2048);
        assert!(small.max_fill <= medium.max_fill);
        assert!(medium.max_fill <= large.max_fill);
    }

    #[test]
    fn iluk_config_profiles_are_ordered() {
        let conservative = ILUKConfig::conservative();
        let balanced = ILUKConfig::balanced();
        let aggressive = ILUKConfig::aggressive();
        assert!(conservative.level_of_fill <= balanced.level_of_fill);
        assert!(balanced.level_of_fill <= aggressive.level_of_fill);
    }

    #[test]
    fn gmres_ilut_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = gmres_ilut_solve(&matrix, &rhs, 1e-8_f64, 10, 0.0_f64, 8);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_iluk_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = gmres_iluk_solve(&matrix, &rhs, 1e-8_f64, 10, 1);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn bicgstab_iluk_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = bicgstab_iluk_solve(&matrix, &rhs, 1e-8_f64, 10, 1);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_ilu0_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = gmres_ilu0_solve(&matrix, &rhs, 1e-8_f64, 10);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_ildl0_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0_f64]);
        let result = gmres_ildl0_solve(&matrix, &rhs, 1e-8_f64, 10);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn bicgstab_ilu0_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, -2.0_f64, 0.5_f64, 3.0_f64, -1.0_f64,
        ])
        .unwrap();
        let factorization = ilu0_factor(&matrix).unwrap();
        let multi = bicgstab_ilu0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ilu0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn bicgstab_ilut_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, -1.0_f64, -2.0_f64, 2.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();
        let multi = bicgstab_ilut_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ilut_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn bicgstab_iluk_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, -1.0_f64, -2.0_f64, 2.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let multi = bicgstab_iluk_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_iluk_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn bicgstab_ildl0_multi_rhs_matches_single_column_reuse() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, -2.0_f64, 1.5_f64, 3.0_f64, -1.0_f64,
        ])
        .unwrap();
        let factorization = ildl0_factor(&matrix).unwrap();
        let multi = bicgstab_ildl0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ildl0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn gmres_ilu0_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, -2.0_f64, 0.5_f64, 3.0_f64, -1.0_f64,
        ])
        .unwrap();
        let factorization = ilu0_factor(&matrix).unwrap();
        let multi = gmres_ilu0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            32,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ilu0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn gmres_ilut_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, -1.0_f64, -2.0_f64, 2.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();
        let factorization = ilut_factor(&matrix, 0.0_f64, 8).unwrap();
        let multi = gmres_ilut_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            32,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ilut_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn gmres_iluk_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, -1.0_f64, -2.0_f64, 2.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let multi = gmres_iluk_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            32,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_iluk_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn gmres_ildl0_multi_rhs_matches_single_column_reuse() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 0.0_f64, -2.0_f64, 1.5_f64, 3.0_f64, -1.0_f64,
        ])
        .unwrap();
        let factorization = ildl0_factor(&matrix).unwrap();
        let multi = gmres_ildl0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10_f64,
            32,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ildl0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10_f64,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9_f64);
            }
        }
    }

    #[test]
    fn multi_rhs_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = Array2::zeros((2, 2));
        let factorization = ilu0_factor(&matrix).unwrap();
        let result = gmres_ilu0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-8_f64,
            10,
            &factorization,
        );
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_matvec_view_matches_internal_f64() {
        let row_ptrs = vec![0_i32, 2, 4];
        let col_indices = vec![0_i32, 1, 0, 1];
        let values = vec![4.0_f64, 1.0, 2.0, 3.0];
        let view = CsrMatrixView::new(2, 2, &row_ptrs, &col_indices, &values).unwrap();
        let vector = Array1::from_vec(vec![1.0_f64, -2.0_f64]);

        let expected = matvec_view(&view, &vector).unwrap();
        let magma = matvec_magma_f64_view(&view, &vector).unwrap();
        for index in 0..expected.len() {
            assert!((expected[index] - magma[index]).abs() < 1e-9_f64);
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_matvec_view_matches_internal_f32() {
        let row_ptrs = vec![0_i32, 2, 4];
        let col_indices = vec![0_i32, 1, 0, 1];
        let values = vec![4.0_f32, 1.0, 2.0, 3.0];
        let view = CsrMatrixView::new(2, 2, &row_ptrs, &col_indices, &values).unwrap();
        let vector = Array1::from_vec(vec![1.0_f32, -2.0_f32]);

        let expected = matvec_view(&view, &vector).unwrap();
        let magma = matvec_magma_f32_view(&view, &vector).unwrap();
        for index in 0..expected.len() {
            assert!((expected[index] - magma[index]).abs() < 1e-5_f32);
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_matmat_dense_view_matches_internal_f64() {
        let row_ptrs = vec![0_i32, 2, 4];
        let col_indices = vec![0_i32, 1, 0, 1];
        let values = vec![4.0_f64, 1.0, 2.0, 3.0];
        let view = CsrMatrixView::new(2, 2, &row_ptrs, &col_indices, &values).unwrap();
        let dense = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();

        let expected = matmat_dense_view(&view, &dense).unwrap();
        let magma = matmat_dense_magma_f64_view(&view, &dense).unwrap();
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert!((expected[[row, col]] - magma[[row, col]]).abs() < 1e-9_f64);
            }
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_matmat_dense_view_matches_internal_f32() {
        let row_ptrs = vec![0_i32, 2, 4];
        let col_indices = vec![0_i32, 1, 0, 1];
        let values = vec![4.0_f32, 1.0, 2.0, 3.0];
        let view = CsrMatrixView::new(2, 2, &row_ptrs, &col_indices, &values).unwrap();
        let dense = Array2::from_shape_vec((2, 3), vec![
            1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32,
        ])
        .unwrap();

        let expected = matmat_dense_view(&view, &dense).unwrap();
        let magma = matmat_dense_magma_f32_view(&view, &dense).unwrap();
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert!((expected[[row, col]] - magma[[row, col]]).abs() < 1e-4_f32);
            }
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_iterative_sparse_solvers_match_internal_f64() {
        let spd_row_ptrs = vec![0_i32, 2, 5, 7];
        let spd_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
        let spd_values = vec![4.0_f64, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let spd_view =
            CsrMatrixView::new(3, 3, &spd_row_ptrs, &spd_col_indices, &spd_values).unwrap();
        let rhs_spd = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);

        let cg_expected =
            conjugate_gradient_solve_view(&spd_view, &rhs_spd, 1e-10_f64, 256).unwrap();
        let cg_magma =
            conjugate_gradient_magma_f64_view(&spd_view, &rhs_spd, 1e-10_f64, 256).unwrap();
        for i in 0..cg_expected.len() {
            assert!((cg_expected[i] - cg_magma[i]).abs() < 1e-8_f64);
        }

        let pcg_expected = pcg_solve_view(&spd_view, &rhs_spd, 1e-10_f64, 256).unwrap();
        let pcg_magma = pcg_jacobi_magma_f64_view(&spd_view, &rhs_spd, 1e-10_f64, 256).unwrap();
        for i in 0..pcg_expected.len() {
            assert!((pcg_expected[i] - pcg_magma[i]).abs() < 1e-8_f64);
        }

        let nonsym_row_ptrs = vec![0_i32, 2, 5, 7];
        let nonsym_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
        let nonsym_values = vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0];
        let nonsym_view =
            CsrMatrixView::new(3, 3, &nonsym_row_ptrs, &nonsym_col_indices, &nonsym_values)
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);

        let gmres_expected = gmres_ilu0_solve_view(&nonsym_view, &rhs, 1e-10_f64, 16).unwrap();
        let gmres_magma = gmres_ilu0_magma_f64_view(&nonsym_view, &rhs, 1e-10_f64, 16).unwrap();
        for i in 0..gmres_expected.len() {
            assert!((gmres_expected[i] - gmres_magma[i]).abs() < 1e-7_f64);
        }

        let bicg_expected = bicgstab_ilu0_solve_view(&nonsym_view, &rhs, 1e-10_f64, 256).unwrap();
        let bicg_magma = bicgstab_ilu0_magma_f64_view(&nonsym_view, &rhs, 1e-10_f64, 256).unwrap();
        for i in 0..bicg_expected.len() {
            assert!((bicg_expected[i] - bicg_magma[i]).abs() < 1e-7_f64);
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn magma_iterative_sparse_solvers_match_internal_f32() {
        let spd_row_ptrs = vec![0_i32, 2, 5, 7];
        let spd_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
        let spd_values = vec![4.0_f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let spd_view =
            CsrMatrixView::new(3, 3, &spd_row_ptrs, &spd_col_indices, &spd_values).unwrap();
        let rhs_spd = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);

        let cg_expected =
            conjugate_gradient_solve_view(&spd_view, &rhs_spd, 1e-6_f32, 256).unwrap();
        let cg_magma =
            conjugate_gradient_magma_f32_view(&spd_view, &rhs_spd, 1e-6_f32, 256).unwrap();
        for i in 0..cg_expected.len() {
            assert!((cg_expected[i] - cg_magma[i]).abs() < 5e-4_f32);
        }

        let pcg_expected = pcg_solve_view(&spd_view, &rhs_spd, 1e-6_f32, 256).unwrap();
        let pcg_magma = pcg_jacobi_magma_f32_view(&spd_view, &rhs_spd, 1e-6_f32, 256).unwrap();
        for i in 0..pcg_expected.len() {
            assert!((pcg_expected[i] - pcg_magma[i]).abs() < 5e-4_f32);
        }

        let nonsym_row_ptrs = vec![0_i32, 2, 5, 7];
        let nonsym_col_indices = vec![0_i32, 1, 0, 1, 2, 1, 2];
        let nonsym_values = vec![4.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0];
        let nonsym_view =
            CsrMatrixView::new(3, 3, &nonsym_row_ptrs, &nonsym_col_indices, &nonsym_values)
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f32, 2.0, 3.0]);

        let gmres_expected = gmres_ilu0_solve_view(&nonsym_view, &rhs, 1e-6_f32, 16).unwrap();
        let gmres_magma = gmres_ilu0_magma_f32_view(&nonsym_view, &rhs, 1e-6_f32, 16).unwrap();
        for i in 0..gmres_expected.len() {
            assert!((gmres_expected[i] - gmres_magma[i]).abs() < 2e-3_f32);
        }

        let bicg_expected = bicgstab_ilu0_solve_view(&nonsym_view, &rhs, 1e-6_f32, 256).unwrap();
        let bicg_magma = bicgstab_ilu0_magma_f32_view(&nonsym_view, &rhs, 1e-6_f32, 256).unwrap();
        for i in 0..bicg_expected.len() {
            assert!((bicg_expected[i] - bicg_magma[i]).abs() < 2e-3_f32);
        }
    }
}

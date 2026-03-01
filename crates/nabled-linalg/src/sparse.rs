//! Sparse matrix primitives and iterative solves over CSR/CSC/COO structures.

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use thiserror::Error;

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, SparseError> {
    if left.len() != right.len() {
        return Err(SparseError::DimensionMismatch);
    }
    let mut sum = 0.0_f64;
    for i in 0..left.len() {
        sum += left[i] * right[i];
    }
    Ok(sum)
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

/// Compressed sparse row (CSR) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows:   usize,
    /// Number of columns.
    pub ncols:   usize,
    /// Row pointer offsets (`len = nrows + 1`).
    pub indptr:  Vec<usize>,
    /// Column index for each non-zero value.
    pub indices: Vec<usize>,
    /// Non-zero values.
    pub data:    Vec<f64>,
}

impl CsrMatrix {
    /// Construct a CSR matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSR arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
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
}

/// Coordinate list (COO) sparse matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    /// Number of rows.
    pub nrows:       usize,
    /// Number of columns.
    pub ncols:       usize,
    /// Row index for each non-zero entry.
    pub row_indices: Vec<usize>,
    /// Column index for each non-zero entry.
    pub col_indices: Vec<usize>,
    /// Non-zero values.
    pub data:        Vec<f64>,
}

/// Compressed sparse column (CSC) matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix {
    /// Number of rows.
    pub nrows:   usize,
    /// Number of columns.
    pub ncols:   usize,
    /// Column pointer offsets (`len = ncols + 1`).
    pub indptr:  Vec<usize>,
    /// Row index for each non-zero value.
    pub indices: Vec<usize>,
    /// Non-zero values.
    pub data:    Vec<f64>,
}

impl CscMatrix {
    /// Construct a CSC matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or CSC arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
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
    pub fn to_csr(&self) -> Result<CsrMatrix, SparseError> {
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
        let mut data = vec![0.0_f64; nnz];

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
pub struct JacobiPreconditioner {
    /// Inverse diagonal values.
    pub inverse_diagonal: Array1<f64>,
}

/// Incomplete LU(0) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILU0Factorization {
    /// Unit-lower factor.
    pub l: CsrMatrix,
    /// Upper factor.
    pub u: CsrMatrix,
}

impl CooMatrix {
    /// Construct a COO matrix after validating structure.
    ///
    /// # Errors
    /// Returns an error if dimensions are empty or COO arrays are inconsistent.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        data: Vec<f64>,
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
    pub fn to_csr(&self) -> Result<CsrMatrix, SparseError> {
        let mut entries = self
            .row_indices
            .iter()
            .copied()
            .zip(self.col_indices.iter().copied())
            .zip(self.data.iter().copied())
            .map(|((row, col), value)| (row, col, value))
            .collect::<Vec<_>>();
        entries.sort_by_key(|&(row, col, _)| (row, col));

        let mut collapsed = Vec::<(usize, usize, f64)>::new();
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
pub fn matvec(matrix: &CsrMatrix, vector: &Array1<f64>) -> Result<Array1<f64>, SparseError> {
    let mut output = Array1::<f64>::zeros(matrix.nrows);
    matvec_into(matrix, vector, &mut output)?;
    Ok(output)
}

/// Compute sparse matrix-vector product `y = A x` into `output`.
///
/// # Errors
/// Returns an error if input/output dimensions are incompatible.
pub fn matvec_into(
    matrix: &CsrMatrix,
    vector: &Array1<f64>,
    output: &mut Array1<f64>,
) -> Result<(), SparseError> {
    if vector.len() != matrix.ncols || output.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    for row in 0..matrix.nrows {
        let start = matrix.indptr[row];
        let end = matrix.indptr[row + 1];
        let mut sum = 0.0_f64;
        for entry in start..end {
            sum += matrix.data[entry] * vector[matrix.indices[entry]];
        }
        output[row] = sum;
    }

    Ok(())
}

/// Compute transpose of CSR matrix.
///
/// # Errors
/// Returns an error if transpose construction fails due to invalid structure.
pub fn transpose(matrix: &CsrMatrix) -> Result<CsrMatrix, SparseError> {
    let nnz = matrix.data.len();
    let mut counts = vec![0_usize; matrix.ncols];
    for &col in &matrix.indices {
        counts[col] += 1;
    }

    let mut indptr_t = vec![0_usize; matrix.ncols + 1];
    for row in 0..matrix.ncols {
        indptr_t[row + 1] = indptr_t[row] + counts[row];
    }

    let mut next = indptr_t[..matrix.ncols].to_vec();
    let mut indices_t = vec![0_usize; nnz];
    let mut data_t = vec![0.0_f64; nnz];

    for row in 0..matrix.nrows {
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let destination = next[col];
            indices_t[destination] = row;
            data_t[destination] = matrix.data[entry];
            next[col] += 1;
        }
    }

    CsrMatrix::new(matrix.ncols, matrix.nrows, indptr_t, indices_t, data_t)
}

/// Convert CSR matrix to CSC.
///
/// # Errors
/// Returns an error if conversion encounters invalid structure.
pub fn csr_to_csc(matrix: &CsrMatrix) -> Result<CscMatrix, SparseError> {
    let nnz = matrix.data.len();
    let mut counts = vec![0_usize; matrix.ncols];
    for &col in &matrix.indices {
        counts[col] += 1;
    }

    let mut indptr = vec![0_usize; matrix.ncols + 1];
    for col in 0..matrix.ncols {
        indptr[col + 1] = indptr[col] + counts[col];
    }

    let mut next = indptr[..matrix.ncols].to_vec();
    let mut indices = vec![0_usize; nnz];
    let mut data = vec![0.0_f64; nnz];

    for row in 0..matrix.nrows {
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let destination = next[col];
            indices[destination] = row;
            data[destination] = matrix.data[entry];
            next[col] += 1;
        }
    }

    CscMatrix::new(matrix.nrows, matrix.ncols, indptr, indices, data)
}

/// Compute sparse matrix-vector product `y = A x` for CSC format.
///
/// # Errors
/// Returns an error if vector length mismatches matrix columns.
pub fn matvec_csc(matrix: &CscMatrix, vector: &Array1<f64>) -> Result<Array1<f64>, SparseError> {
    if vector.len() != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    let mut output = Array1::<f64>::zeros(matrix.nrows);
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
pub fn jacobi_preconditioner(matrix: &CsrMatrix) -> Result<JacobiPreconditioner, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    let mut inverse_diagonal = Array1::<f64>::zeros(matrix.nrows);
    for row in 0..matrix.nrows {
        let mut diagonal = 0.0_f64;
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            if matrix.indices[entry] == row {
                diagonal = matrix.data[entry];
                break;
            }
        }
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        inverse_diagonal[row] = 1.0 / diagonal;
    }
    Ok(JacobiPreconditioner { inverse_diagonal })
}

/// Apply Jacobi preconditioner to a vector.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn apply_jacobi_preconditioner(
    preconditioner: &JacobiPreconditioner,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    if preconditioner.inverse_diagonal.len() != vector.len() {
        return Err(SparseError::DimensionMismatch);
    }
    let mut output = Array1::<f64>::zeros(vector.len());
    for i in 0..vector.len() {
        output[i] = preconditioner.inverse_diagonal[i] * vector[i];
    }
    Ok(output)
}

fn row_positions(matrix: &CsrMatrix) -> Vec<HashMap<usize, usize>> {
    let mut positions = Vec::<HashMap<usize, usize>>::with_capacity(matrix.nrows);
    for row in 0..matrix.nrows {
        let mut map = HashMap::<usize, usize>::new();
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let _ = map.insert(matrix.indices[entry], entry);
        }
        positions.push(map);
    }
    positions
}

/// Compute ILU(0) factorization for a square sparse matrix.
///
/// The non-zero pattern of factors follows the input pattern (`level-of-fill = 0`).
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factorization breaks down.
pub fn ilu0_factor(matrix: &CsrMatrix) -> Result<ILU0Factorization, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let positions = row_positions(matrix);
    let mut factors = matrix.data.clone();

    for row in 0..n {
        let mut lower_entries = Vec::<(usize, usize)>::new();
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
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
            if diagonal.abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }

            let multiplier = factors[row_col_entry] / diagonal;
            factors[row_col_entry] = multiplier;

            for upper_entry in matrix.indptr[col_j]..matrix.indptr[col_j + 1] {
                let col_k = matrix.indices[upper_entry];
                if col_k <= col_j {
                    continue;
                }
                if let Some(&update_index) = positions[row].get(&col_k) {
                    factors[update_index] -= multiplier * factors[upper_entry];
                }
            }
        }

        let Some(&row_diagonal_index) = positions[row].get(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        if factors[row_diagonal_index].abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<f64>::new();
    l_indptr.push(0);

    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<f64>::new();
    u_indptr.push(0);

    for row in 0..n {
        let mut l_row = Vec::<(usize, f64)>::new();
        l_row.push((row, 1.0));
        let mut u_row = Vec::<(usize, f64)>::new();

        for (&col, &value) in matrix.indices[matrix.indptr[row]..matrix.indptr[row + 1]]
            .iter()
            .zip(factors[matrix.indptr[row]..matrix.indptr[row + 1]].iter())
        {
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

/// Apply an ILU(0) preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`ilu0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ilu0_preconditioner(
    factorization: &ILU0Factorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    if factorization.l.nrows != factorization.l.ncols
        || factorization.u.nrows != factorization.u.ncols
        || factorization.l.nrows != factorization.u.nrows
    {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != factorization.l.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let n = rhs.len();
    let mut intermediate = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut sum = rhs[row];
        for entry in factorization.l.indptr[row]..factorization.l.indptr[row + 1] {
            let col = factorization.l.indices[entry];
            if col < row {
                sum -= factorization.l.data[entry] * intermediate[col];
            }
        }
        intermediate[row] = sum;
    }

    let mut output = Array1::<f64>::zeros(n);
    for row_reverse in 0..n {
        let row = n - 1 - row_reverse;
        let mut sum = intermediate[row];
        let mut diagonal = None;
        for entry in factorization.u.indptr[row]..factorization.u.indptr[row + 1] {
            let col = factorization.u.indices[entry];
            let value = factorization.u.data[entry];
            if col == row {
                diagonal = Some(value);
            } else if col > row {
                sum -= value * output[col];
            }
        }

        let Some(diagonal) = diagonal else {
            return Err(SparseError::SingularMatrix);
        };
        if diagonal.abs() <= DEFAULT_TOLERANCE {
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
pub fn matmat_dense(matrix: &CsrMatrix, dense: &Array2<f64>) -> Result<Array2<f64>, SparseError> {
    let mut output = Array2::<f64>::zeros((matrix.nrows, dense.ncols()));
    matmat_dense_into(matrix, dense, &mut output)?;
    Ok(output)
}

/// Compute sparse-dense matrix multiplication `Y = A B` into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_dense_into(
    matrix: &CsrMatrix,
    dense: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), SparseError> {
    if dense.nrows() != matrix.ncols || output.dim() != (matrix.nrows, dense.ncols()) {
        return Err(SparseError::DimensionMismatch);
    }

    output.fill(0.0);
    for row in 0..matrix.nrows {
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let value = matrix.data[entry];
            for dense_col in 0..dense.ncols() {
                output[[row, dense_col]] += value * dense[[col, dense_col]];
            }
        }
    }
    Ok(())
}

fn sparse_row_dot(
    left_indices: &[usize],
    left_data: &[f64],
    right_indices: &[usize],
    right_data: &[f64],
) -> f64 {
    let mut left_ptr = 0_usize;
    let mut right_ptr = 0_usize;
    let mut sum = 0.0_f64;

    while left_ptr < left_indices.len() && right_ptr < right_indices.len() {
        match left_indices[left_ptr].cmp(&right_indices[right_ptr]) {
            std::cmp::Ordering::Equal => {
                sum += left_data[left_ptr] * right_data[right_ptr];
                left_ptr += 1;
                right_ptr += 1;
            }
            std::cmp::Ordering::Less => {
                left_ptr += 1;
            }
            std::cmp::Ordering::Greater => {
                right_ptr += 1;
            }
        }
    }

    sum
}

/// Compute sparse-sparse matrix multiplication `C = A B` in CSR format.
///
/// # Errors
/// Returns an error if dimensions are incompatible or sparse structure is invalid.
pub fn matmat_sparse(left: &CsrMatrix, right: &CsrMatrix) -> Result<CsrMatrix, SparseError> {
    if left.ncols != right.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let right_transposed = transpose(right)?;

    let mut indptr = Vec::<usize>::with_capacity(left.nrows + 1);
    let mut indices = Vec::<usize>::new();
    let mut data = Vec::<f64>::new();
    indptr.push(0);

    for row in 0..left.nrows {
        let left_start = left.indptr[row];
        let left_end = left.indptr[row + 1];
        let left_indices = &left.indices[left_start..left_end];
        let left_data = &left.data[left_start..left_end];

        for col in 0..right.ncols {
            let right_start = right_transposed.indptr[col];
            let right_end = right_transposed.indptr[col + 1];
            let right_indices = &right_transposed.indices[right_start..right_end];
            let right_data = &right_transposed.data[right_start..right_end];

            let value = sparse_row_dot(left_indices, left_data, right_indices, right_data);
            if value.abs() > DEFAULT_TOLERANCE {
                indices.push(col);
                data.push(value);
            }
        }

        indptr.push(indices.len());
    }

    CsrMatrix::new(left.nrows, right.ncols, indptr, indices, data)
}

/// Compute batched sparse matrix-vector products.
///
/// Inputs are row vectors with shape `(batch, ncols)` and output is `(batch, nrows)`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec(
    matrix: &CsrMatrix,
    batch_vectors: &Array2<f64>,
) -> Result<Array2<f64>, SparseError> {
    let mut output = Array2::<f64>::zeros((batch_vectors.nrows(), matrix.nrows));
    batched_matvec_into(matrix, batch_vectors, &mut output)?;
    Ok(output)
}

/// Compute batched sparse matrix-vector products into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matvec_into(
    matrix: &CsrMatrix,
    batch_vectors: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), SparseError> {
    if batch_vectors.ncols() != matrix.ncols
        || output.dim() != (batch_vectors.nrows(), matrix.nrows)
    {
        return Err(SparseError::DimensionMismatch);
    }

    output.fill(0.0);
    for batch in 0..batch_vectors.nrows() {
        for row in 0..matrix.nrows {
            let mut sum = 0.0_f64;
            for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
                sum += matrix.data[entry] * batch_vectors[[batch, matrix.indices[entry]]];
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
pub fn jacobi_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
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
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);

    let mut diagonal = Array1::<f64>::zeros(n);
    for row in 0..n {
        let start = matrix.indptr[row];
        let end = matrix.indptr[row + 1];
        for entry in start..end {
            if matrix.indices[entry] == row {
                diagonal[row] = matrix.data[entry];
                break;
            }
        }
        if diagonal[row].abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
    }

    let mut x = Array1::<f64>::zeros(n);
    let mut x_next = Array1::<f64>::zeros(n);

    for _ in 0..max_iterations.max(1) {
        for row in 0..n {
            let start = matrix.indptr[row];
            let end = matrix.indptr[row + 1];
            let mut off_diagonal = 0.0_f64;
            for entry in start..end {
                let col = matrix.indices[entry];
                if col != row {
                    off_diagonal += matrix.data[entry] * x[col];
                }
            }
            x_next[row] = (rhs[row] - off_diagonal) / diagonal[row];
        }

        let mut delta_inf = 0.0_f64;
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
pub fn gauss_seidel_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
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
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let mut x = Array1::<f64>::zeros(n);

    for _ in 0..max_iterations.max(1) {
        let previous = x.clone();
        for row in 0..n {
            let start = matrix.indptr[row];
            let end = matrix.indptr[row + 1];

            let mut diagonal = 0.0_f64;
            let mut sum = 0.0_f64;
            for entry in start..end {
                let col = matrix.indices[entry];
                let value = matrix.data[entry];
                if col == row {
                    diagonal = value;
                } else {
                    sum += value * x[col];
                }
            }

            if diagonal.abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }

            x[row] = (rhs[row] - sum) / diagonal;
        }

        let mut delta_inf = 0.0_f64;
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
pub fn conjugate_gradient_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let mut x = Array1::<f64>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut residual_norm_sq = dot(&residual, &residual)?;

    if residual_norm_sq.sqrt() <= tolerance {
        return Ok(x);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec(matrix, &direction)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= DEFAULT_TOLERANCE {
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
pub fn pcg_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let preconditioner = jacobi_preconditioner(matrix)?;
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let mut solution = Array1::<f64>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut preconditioned_residual = apply_jacobi_preconditioner(&preconditioner, &residual)?;
    let mut direction = preconditioned_residual.clone();
    let mut rho = dot(&residual, &preconditioned_residual)?;

    if rho.sqrt() <= tolerance {
        return Ok(solution);
    }

    for _ in 0..max_iterations.max(1) {
        let matrix_direction = matvec(matrix, &direction)?;
        let denominator = dot(&direction, &matrix_direction)?;
        if denominator.abs() <= DEFAULT_TOLERANCE {
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

/// Solve sparse linear system `A x = b` with `BiCGSTAB` iteration.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, singular breakdown, or non-convergence.
pub fn bicgstab_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let dimension = rhs.len();
    let mut solution = Array1::<f64>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut krylov_vector = Array1::<f64>::zeros(dimension);
    let mut search_direction = Array1::<f64>::zeros(dimension);

    let residual_norm = dot(&residual, &residual)?.sqrt();
    if residual_norm <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        krylov_vector = matvec(matrix, &search_direction)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= DEFAULT_TOLERANCE {
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

        let transformed_auxiliary = matvec(matrix, &auxiliary_residual)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= DEFAULT_TOLERANCE {
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
pub fn bicgstab_ilu0_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let factorization = ilu0_factor(matrix)?;
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let dimension = rhs.len();
    let mut solution = Array1::<f64>::zeros(dimension);
    let mut residual = rhs.clone();
    let residual_shadow = residual.clone();
    let mut rho_prev = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut krylov_vector = Array1::<f64>::zeros(dimension);
    let mut search_direction = Array1::<f64>::zeros(dimension);

    if dot(&residual, &residual)?.sqrt() <= tolerance {
        return Ok(solution);
    }

    for iteration in 0..max_iterations.max(1) {
        let rho = dot(&residual_shadow, &residual)?;
        if rho.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }

        if iteration == 0 {
            search_direction.assign(&residual);
        } else {
            if omega.abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..dimension {
                search_direction[i] =
                    residual[i] + beta * (search_direction[i] - omega * krylov_vector[i]);
            }
        }

        let preconditioned_search = apply_ilu0_preconditioner(&factorization, &search_direction)?;
        krylov_vector = matvec(matrix, &preconditioned_search)?;
        let denominator = dot(&residual_shadow, &krylov_vector)?;
        if denominator.abs() <= DEFAULT_TOLERANCE {
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
            apply_ilu0_preconditioner(&factorization, &auxiliary_residual)?;
        let transformed_auxiliary = matvec(matrix, &preconditioned_auxiliary)?;
        let transformed_norm_sq = dot(&transformed_auxiliary, &transformed_auxiliary)?;
        if transformed_norm_sq.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        omega = dot(&transformed_auxiliary, &auxiliary_residual)? / transformed_norm_sq;
        if omega.abs() <= DEFAULT_TOLERANCE {
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

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr1};

    use super::*;

    fn toy_matrix() -> CsrMatrix {
        // [4 1 0]
        // [1 3 1]
        // [0 1 2]
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap()
    }

    fn csr_to_dense(matrix: &CsrMatrix) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((matrix.nrows, matrix.ncols));
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
        let vector = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = matvec(&matrix, &vector).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 10.0).abs() < 1e-12);
        assert!((y[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn jacobi_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = jacobi_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_invalid_structure() {
        let result = CsrMatrix::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(matches!(result, Err(SparseError::InvalidStructure)));
    }

    #[test]
    fn coo_to_csr_roundtrip_matvec() {
        let coo = CooMatrix::new(3, 3, vec![0, 0, 1, 1, 1, 2, 2], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let csr = coo.to_csr().unwrap();
        let vector = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = matvec(&csr, &vector).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-12);
        assert!((y[1] - 10.0).abs() < 1e-12);
        assert!((y[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn gauss_seidel_solves_diagonally_dominant_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = gauss_seidel_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
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

        let vector = arr1(&[1.0_f64, 2.0, 3.0]);
        let reference_product = matvec(&matrix, &vector).unwrap();
        let converted_product = matvec_csc(&csc, &vector).unwrap();
        for i in 0..vector.len() {
            assert!((reference_product[i] - converted_product[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_dense_matmat_matches_expected() {
        let matrix = toy_matrix();
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = matmat_dense(&matrix, &dense).unwrap();
        assert_eq!(y.dim(), (3, 2));
        assert!((y[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((y[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((y[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((y[[1, 1]] - 4.0).abs() < 1e-12);
        assert!((y[[2, 0]] - 2.0).abs() < 1e-12);
        assert!((y[[2, 1]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn batched_matvec_matches_single_path() {
        let matrix = toy_matrix();
        let batch = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]).unwrap();
        let batched = batched_matvec(&matrix, &batch).unwrap();
        assert_eq!(batched.dim(), (2, 3));
        let first = matvec(&matrix, &arr1(&[1.0, 2.0, 3.0])).unwrap();
        let second = matvec(&matrix, &arr1(&[3.0, 2.0, 1.0])).unwrap();
        for i in 0..3 {
            assert!((batched[[0, i]] - first[i]).abs() < 1e-12);
            assert!((batched[[1, i]] - second[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn conjugate_gradient_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = conjugate_gradient_solve(&matrix, &rhs, 1e-10, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn jacobi_preconditioner_builds_and_applies() {
        let matrix = toy_matrix();
        let preconditioner = jacobi_preconditioner(&matrix).unwrap();
        assert_eq!(preconditioner.inverse_diagonal.len(), 3);
        assert!((preconditioner.inverse_diagonal[0] - 0.25).abs() < 1e-12);
        assert!((preconditioner.inverse_diagonal[1] - (1.0 / 3.0)).abs() < 1e-12);
        assert!((preconditioner.inverse_diagonal[2] - 0.5).abs() < 1e-12);

        let rhs = arr1(&[4.0_f64, 6.0, 2.0]);
        let transformed = apply_jacobi_preconditioner(&preconditioner, &rhs).unwrap();
        assert!((transformed[0] - 1.0).abs() < 1e-12);
        assert!((transformed[1] - 2.0).abs() < 1e-12);
        assert!((transformed[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pcg_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = pcg_solve(&matrix, &rhs, 1e-10, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn sparse_sparse_matmat_matches_dense_result() {
        let matrix = toy_matrix();
        let product = matmat_sparse(&matrix, &matrix).unwrap();
        let vector = arr1(&[1.0_f64, 2.0, 3.0]);

        let lhs = matvec(&matrix, &vector).unwrap();
        let expected = matvec(&matrix, &lhs).unwrap();
        let observed = matvec(&product, &vector).unwrap();
        for i in 0..vector.len() {
            assert!((expected[i] - observed[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_system() {
        // [4 1 0]
        // [2 3 1]
        // [0 1 2]
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = bicgstab_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
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
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn apply_ilu0_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ilu0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = apply_ilu0_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn bicgstab_ilu0_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = bicgstab_ilu0_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }
}

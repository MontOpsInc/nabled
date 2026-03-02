//! Sparse matrix primitives and iterative solves over CSR/CSC/COO structures.

use std::collections::{BTreeMap, BTreeSet, HashMap};

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

/// Incomplete LU with threshold/drop tolerance (ILUT) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILUTFactorization {
    /// Unit-lower factor.
    pub l: CsrMatrix,
    /// Upper factor.
    pub u: CsrMatrix,
}

/// Incomplete LU with level-of-fill (ILU(k)) sparse factorization.
#[derive(Debug, Clone, PartialEq)]
pub struct ILUKFactorization {
    /// Unit-lower factor.
    pub l:             CsrMatrix,
    /// Upper factor.
    pub u:             CsrMatrix,
    /// Requested level-of-fill used during construction.
    pub level_of_fill: usize,
}

/// Configuration for ILUT-based sparse factorization and solves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ILUTConfig {
    /// Drop entries with absolute magnitude less than or equal to this value.
    pub drop_tolerance: f64,
    /// Maximum number of retained off-diagonal entries per row in each factor.
    pub max_fill:       usize,
}

impl ILUTConfig {
    /// Conservative profile prioritizing sparsity.
    #[must_use]
    pub const fn conservative() -> Self { Self { drop_tolerance: 1e-6, max_fill: 8 } }

    /// Balanced profile for general sparse workloads.
    #[must_use]
    pub const fn balanced() -> Self { Self { drop_tolerance: 1e-8, max_fill: 16 } }

    /// Aggressive profile prioritizing preconditioner quality.
    #[must_use]
    pub const fn aggressive() -> Self { Self { drop_tolerance: 1e-10, max_fill: 32 } }

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
        Self { drop_tolerance: 1e-8, max_fill: fill.min(dimension.max(1)) }
    }
}

impl Default for ILUTConfig {
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
pub struct IC0Factorization {
    /// Lower-triangular factor with diagonal terms.
    pub l:           CsrMatrix,
    /// Cached transpose of `l` for backward substitution.
    pub l_transpose: CsrMatrix,
}

/// Incomplete LDL(0) sparse factorization for symmetric sparse systems.
#[derive(Debug, Clone, PartialEq)]
pub struct ILDL0Factorization {
    /// Unit-lower factor.
    pub l:           CsrMatrix,
    /// Diagonal factor.
    pub d:           Array1<f64>,
    /// Cached transpose of `l` for backward substitution.
    pub l_transpose: CsrMatrix,
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

fn is_symmetric_from_positions(
    matrix: &CsrMatrix,
    positions: &[HashMap<usize, usize>],
    tolerance: f64,
) -> bool {
    for row in 0..matrix.nrows {
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let value = matrix.data[entry];
            let Some(&mirror_entry) = positions[col].get(&row) else {
                return false;
            };
            if (value - matrix.data[mirror_entry]).abs() > tolerance {
                return false;
            }
        }
    }
    true
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

fn retain_strongest_entries(entries: &mut Vec<(usize, f64)>, max_entries: usize) {
    if entries.len() <= max_entries {
        entries.sort_unstable_by_key(|&(col, _)| col);
        return;
    }
    entries.sort_unstable_by(|left, right| right.1.abs().total_cmp(&left.1.abs()));
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
pub fn ilut_factor(
    matrix: &CsrMatrix,
    drop_tolerance: f64,
    max_fill: usize,
) -> Result<ILUTFactorization, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let tolerance = drop_tolerance.max(0.0);
    let mut lower_rows = Vec::<Vec<(usize, f64)>>::with_capacity(n);
    let mut upper_rows = Vec::<Vec<(usize, f64)>>::with_capacity(n);
    let mut upper_diagonal = vec![0.0_f64; n];

    for row in 0..n {
        let mut working = BTreeMap::<usize, f64>::new();
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let value = matrix.data[entry];
            if col == row || value.abs() > tolerance {
                let _ = working.insert(col, value);
            }
        }

        let mut lower_candidates = Vec::<(usize, f64)>::new();
        let lower_columns = working.keys().copied().filter(|&col| col < row).collect::<Vec<_>>();
        for col_j in lower_columns {
            let Some(value) = working.remove(&col_j) else {
                continue;
            };
            if value.abs() <= tolerance {
                continue;
            }

            let diagonal = upper_diagonal[col_j];
            if diagonal.abs() <= DEFAULT_TOLERANCE {
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
                    working.get(&col_k).copied().unwrap_or(0.0) - multiplier * upper_value;
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
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
        lower_row.push((row, 1.0));
        lower_row.sort_unstable_by_key(|&(col, _)| col);

        let mut upper_row = Vec::<(usize, f64)>::with_capacity(upper_candidates.len() + 1);
        upper_row.push((row, diagonal));
        upper_row.extend(upper_candidates);
        upper_row.sort_unstable_by_key(|&(col, _)| col);

        lower_rows.push(lower_row);
        upper_rows.push(upper_row);
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
pub fn ilut_factor_with_config(
    matrix: &CsrMatrix,
    config: ILUTConfig,
) -> Result<ILUTFactorization, SparseError> {
    ilut_factor(matrix, config.drop_tolerance, config.max_fill)
}

fn iluk_initial_row_state(
    matrix: &CsrMatrix,
    row: usize,
) -> (BTreeMap<usize, f64>, HashMap<usize, usize>) {
    let mut values = BTreeMap::<usize, f64>::new();
    let mut levels = HashMap::<usize, usize>::new();
    for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
        let col = matrix.indices[entry];
        let value = matrix.data[entry];
        let _ = values.insert(col, value);
        let _ = levels.insert(col, 0);
    }
    (values, levels)
}

type SparseRowEntries = Vec<(usize, f64)>;
type SparseUpperRowWithLevel = Vec<(usize, f64, usize)>;
type IlukRowFactors = (SparseRowEntries, SparseRowEntries, SparseUpperRowWithLevel, f64);

#[allow(clippy::many_single_char_names)]
fn iluk_eliminate_row_entries(
    row: usize,
    level_of_fill: usize,
    values: &mut BTreeMap<usize, f64>,
    levels: &mut HashMap<usize, usize>,
    upper_rows_with_levels: &[Vec<(usize, f64, usize)>],
    upper_diagonal: &[f64],
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }

        let multiplier = value_ij / diagonal;
        if multiplier.abs() <= DEFAULT_TOLERANCE {
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

            let updated = values.get(&col_k).copied().unwrap_or(0.0_f64) - multiplier * upper_value;
            if updated.abs() <= DEFAULT_TOLERANCE {
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

fn iluk_finalize_row_entries(
    row: usize,
    level_of_fill: usize,
    values: &BTreeMap<usize, f64>,
    levels: &HashMap<usize, usize>,
) -> Result<IlukRowFactors, SparseError> {
    let Some(diagonal) = values.get(&row).copied() else {
        return Err(SparseError::SingularMatrix);
    };
    if diagonal.abs() <= DEFAULT_TOLERANCE {
        return Err(SparseError::SingularMatrix);
    }

    let mut lower_row = vec![(row, 1.0)];
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
pub fn iluk_factor(
    matrix: &CsrMatrix,
    level_of_fill: usize,
) -> Result<ILUKFactorization, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let mut lower_rows = Vec::<Vec<(usize, f64)>>::with_capacity(n);
    let mut upper_rows = Vec::<Vec<(usize, f64)>>::with_capacity(n);
    let mut upper_rows_with_levels = vec![Vec::<(usize, f64, usize)>::new(); n];
    let mut upper_diagonal = vec![0.0_f64; n];

    for row in 0..n {
        let (mut values, mut levels) = iluk_initial_row_state(matrix, row);
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
    let mut l_data = Vec::<f64>::new();
    l_indptr.push(0);
    let mut u_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut u_indices = Vec::<usize>::new();
    let mut u_data = Vec::<f64>::new();
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
pub fn iluk_factor_with_config(
    matrix: &CsrMatrix,
    config: ILUKConfig,
) -> Result<ILUKFactorization, SparseError> {
    iluk_factor(matrix, config.level_of_fill)
}

fn apply_lu_preconditioner(
    lower: &CsrMatrix,
    upper: &CsrMatrix,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    if lower.nrows != lower.ncols || upper.nrows != upper.ncols || lower.nrows != upper.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.len() != lower.nrows {
        return Err(SparseError::DimensionMismatch);
    }

    let n = rhs.len();
    let mut intermediate = Array1::<f64>::zeros(n);
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

    let mut output = Array1::<f64>::zeros(n);
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
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
pub fn apply_ilu0_preconditioner(
    factorization: &ILU0Factorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    apply_lu_preconditioner(&factorization.l, &factorization.u, rhs)
}

/// Apply an ILUT preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`ilut_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ilut_preconditioner(
    factorization: &ILUTFactorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
    apply_lu_preconditioner(&factorization.l, &factorization.u, rhs)
}

/// Apply an ILU(k) preconditioner to a dense vector.
///
/// Solves `L U x = rhs` where `L` and `U` come from [`iluk_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_iluk_preconditioner(
    factorization: &ILUKFactorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
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
pub fn ic0_factor(matrix: &CsrMatrix) -> Result<IC0Factorization, SparseError> {
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
            if diagonal.abs() <= DEFAULT_TOLERANCE {
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
        if diagonal <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        factors[row_diagonal_index] = diagonal.sqrt();
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<f64>::new();
    l_indptr.push(0);

    for row in 0..n {
        let mut row_entries = Vec::<(usize, f64)>::new();
        for (&col, &value) in matrix.indices[matrix.indptr[row]..matrix.indptr[row + 1]]
            .iter()
            .zip(factors[matrix.indptr[row]..matrix.indptr[row + 1]].iter())
        {
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
    let l_transpose = transpose(&l)?;
    Ok(IC0Factorization { l, l_transpose })
}

/// Apply an IC(0) preconditioner to a dense vector.
///
/// Solves `L L^T x = rhs` where `L` comes from [`ic0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ic0_preconditioner(
    factorization: &IC0Factorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
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
    let mut intermediate = Array1::<f64>::zeros(n);
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        intermediate[row] = sum / diagonal;
    }

    let mut output = Array1::<f64>::zeros(n);
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
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
pub fn ildl0_factor(matrix: &CsrMatrix) -> Result<ILDL0Factorization, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }

    let n = matrix.nrows;
    let positions = row_positions(matrix);
    if !is_symmetric_from_positions(matrix, &positions, DEFAULT_TOLERANCE) {
        return Err(SparseError::DimensionMismatch);
    }

    let mut factors = matrix.data.clone();
    let mut diagonal = Array1::<f64>::zeros(n);

    for row in 0..n {
        let mut lower_entries = Vec::<(usize, usize)>::new();
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
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
            if d_j.abs() <= DEFAULT_TOLERANCE {
                return Err(SparseError::SingularMatrix);
            }
            factors[row_col_entry] = sum / d_j;
        }

        let Some(&row_diagonal_index) = positions[row].get(&row) else {
            return Err(SparseError::SingularMatrix);
        };
        let mut d_i = factors[row_diagonal_index];
        for &(_, row_col_entry) in &lower_entries {
            let col_k = matrix.indices[row_col_entry];
            let l_ik = factors[row_col_entry];
            d_i -= l_ik * l_ik * diagonal[col_k];
        }
        if d_i.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        diagonal[row] = d_i;
        factors[row_diagonal_index] = d_i;
    }

    let mut l_indptr = Vec::<usize>::with_capacity(n + 1);
    let mut l_indices = Vec::<usize>::new();
    let mut l_data = Vec::<f64>::new();
    l_indptr.push(0);

    for row in 0..n {
        let mut row_entries = Vec::<(usize, f64)>::new();
        row_entries.push((row, 1.0));
        for (&col, &value) in matrix.indices[matrix.indptr[row]..matrix.indptr[row + 1]]
            .iter()
            .zip(factors[matrix.indptr[row]..matrix.indptr[row + 1]].iter())
        {
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
    let l_transpose = transpose(&l)?;
    Ok(ILDL0Factorization { l, d: diagonal, l_transpose })
}

/// Apply an ILDL(0) preconditioner to a dense vector.
///
/// Solves `L D L^T x = rhs` where factors come from [`ildl0_factor`].
///
/// # Errors
/// Returns an error if dimensions are incompatible or factors are singular.
pub fn apply_ildl0_preconditioner(
    factorization: &ILDL0Factorization,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, SparseError> {
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
    let mut intermediate = Array1::<f64>::zeros(n);
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
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        intermediate[row] = sum / diagonal;
    }

    let mut scaled = Array1::<f64>::zeros(n);
    for row in 0..n {
        let diagonal = factorization.d[row];
        if diagonal.abs() <= DEFAULT_TOLERANCE {
            return Err(SparseError::SingularMatrix);
        }
        scaled[row] = intermediate[row] / diagonal;
    }

    let mut output = Array1::<f64>::zeros(n);
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

/// Solve sparse linear system `A x = b` with IC(0)-preconditioned conjugate gradient.
///
/// This routine assumes an SPD matrix `A`.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn pcg_ic0_solve(
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

    let factorization = ic0_factor(matrix)?;
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);
    let mut solution = Array1::<f64>::zeros(matrix.ncols);
    let mut residual = rhs.clone();
    let mut preconditioned_residual = apply_ic0_preconditioner(&factorization, &residual)?;
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
pub fn gmres_ilu0_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ilu0_factor(matrix)?;
    gmres_ilu0_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilu0_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILU0Factorization,
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

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);

    let mut basis = Array2::<f64>::zeros((n, m + 1));
    let mut hessenberg = Array2::<f64>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ilu0_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<f64>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<f64>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec(matrix, &vj)?;
        let mut w = apply_ilu0_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = 0.0_f64;
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

    let mut h = Array2::<f64>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<f64>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y =
        crate::lu::solve(&normal_matrix, &normal_rhs).map_err(|_| SparseError::SingularMatrix)?;

    let mut solution = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut sum = 0.0_f64;
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec(matrix, &solution)?;
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
pub fn gmres_ilut_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    drop_tolerance: f64,
    max_fill: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ilut_factor(matrix, drop_tolerance, max_fill)?;
    gmres_ilut_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ilut_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUTFactorization,
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

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);

    let mut basis = Array2::<f64>::zeros((n, m + 1));
    let mut hessenberg = Array2::<f64>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ilut_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<f64>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<f64>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec(matrix, &vj)?;
        let mut w = apply_ilut_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = 0.0_f64;
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

    let mut h = Array2::<f64>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<f64>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y =
        crate::lu::solve(&normal_matrix, &normal_rhs).map_err(|_| SparseError::SingularMatrix)?;

    let mut solution = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut sum = 0.0_f64;
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec(matrix, &solution)?;
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
pub fn gmres_ilut_solve_with_config(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    config: ILUTConfig,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ilut_factor_with_config(matrix, config)?;
    gmres_ilut_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = iluk_factor(matrix, level_of_fill)?;
    gmres_iluk_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_iluk_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUKFactorization,
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

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);

    let mut basis = Array2::<f64>::zeros((n, m + 1));
    let mut hessenberg = Array2::<f64>::zeros((m + 1, m));

    let preconditioned_rhs = apply_iluk_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<f64>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<f64>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec(matrix, &vj)?;
        let mut w = apply_iluk_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = 0.0_f64;
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

    let mut h = Array2::<f64>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<f64>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y =
        crate::lu::solve(&normal_matrix, &normal_rhs).map_err(|_| SparseError::SingularMatrix)?;

    let mut solution = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut sum = 0.0_f64;
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec(matrix, &solution)?;
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
pub fn gmres_iluk_solve_with_config(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<f64>, SparseError> {
    let factorization = iluk_factor_with_config(matrix, config)?;
    gmres_iluk_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ildl0_factor(matrix)?;
    gmres_ildl0_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned GMRES.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
#[allow(clippy::many_single_char_names)]
pub fn gmres_ildl0_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILDL0Factorization,
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

    let n = rhs.len();
    let m = n.min(max_iterations.max(1));
    let tolerance = tolerance.max(DEFAULT_TOLERANCE);

    let mut basis = Array2::<f64>::zeros((n, m + 1));
    let mut hessenberg = Array2::<f64>::zeros((m + 1, m));

    let preconditioned_rhs = apply_ildl0_preconditioner(factorization, rhs)?;
    let beta = dot(&preconditioned_rhs, &preconditioned_rhs)?.sqrt();
    if beta <= tolerance {
        return Ok(Array1::<f64>::zeros(n));
    }
    for row in 0..n {
        basis[[row, 0]] = preconditioned_rhs[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut vj = Array1::<f64>::zeros(n);
        for row in 0..n {
            vj[row] = basis[[row, j]];
        }

        let av = matvec(matrix, &vj)?;
        let mut w = apply_ildl0_preconditioner(factorization, &av)?;

        for i in 0..=j {
            let mut hij = 0.0_f64;
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

    let mut h = Array2::<f64>::zeros((effective_m + 1, effective_m));
    for row in 0..=effective_m {
        for col in 0..effective_m {
            h[[row, col]] = hessenberg[[row, col]];
        }
    }
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<f64>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y =
        crate::lu::solve(&normal_matrix, &normal_rhs).map_err(|_| SparseError::SingularMatrix)?;

    let mut solution = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut sum = 0.0_f64;
        for col in 0..effective_m {
            sum += basis[[row, col]] * y[col];
        }
        solution[row] = sum;
    }

    let residual = rhs - &matvec(matrix, &solution)?;
    if dot(&residual, &residual)?.sqrt() <= tolerance {
        Ok(solution)
    } else {
        Err(SparseError::MaxIterationsExceeded)
    }
}

fn solve_multiple_rhs_with_solver(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    mut solve_column: impl FnMut(&Array1<f64>) -> Result<Array1<f64>, SparseError>,
) -> Result<Array2<f64>, SparseError> {
    if matrix.nrows != matrix.ncols {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.nrows() != matrix.nrows {
        return Err(SparseError::DimensionMismatch);
    }
    if rhs.is_empty() {
        return Err(SparseError::EmptyInput);
    }

    let mut output = Array2::<f64>::zeros((matrix.ncols, rhs.ncols()));
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
    let factorization = ilu0_factor(matrix)?;
    bicgstab_ilu0_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILU0Factorization,
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

        let preconditioned_search = apply_ilu0_preconditioner(factorization, &search_direction)?;
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
            apply_ilu0_preconditioner(factorization, &auxiliary_residual)?;
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

/// Solve sparse linear systems `A X = B` with ILU(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilu0_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILU0Factorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        bicgstab_ilu0_solve_with_factorization(
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
pub fn bicgstab_ilut_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    drop_tolerance: f64,
    max_fill: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ilut_factor(matrix, drop_tolerance, max_fill)?;
    bicgstab_ilut_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILUT-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUTFactorization,
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

        let preconditioned_search = apply_ilut_preconditioner(factorization, &search_direction)?;
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
            apply_ilut_preconditioner(factorization, &auxiliary_residual)?;
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

/// Solve sparse linear systems `A X = B` with ILUT-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ilut_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUTFactorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        bicgstab_ilut_solve_with_factorization(
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
pub fn bicgstab_iluk_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUKFactorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        bicgstab_iluk_solve_with_factorization(
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
pub fn bicgstab_ilut_solve_with_config(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    config: ILUTConfig,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ilut_factor_with_config(matrix, config)?;
    bicgstab_ilut_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// This routine supports general non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    level_of_fill: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = iluk_factor(matrix, level_of_fill)?;
    bicgstab_iluk_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUKFactorization,
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

        let preconditioned_search = apply_iluk_preconditioner(factorization, &search_direction)?;
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
            apply_iluk_preconditioner(factorization, &auxiliary_residual)?;
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

/// Solve sparse linear system `A x = b` with ILU(k)-preconditioned `BiCGSTAB`.
///
/// Uses an [`ILUKConfig`] profile for factorization parameters.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_iluk_solve_with_config(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    config: ILUKConfig,
) -> Result<Array1<f64>, SparseError> {
    let factorization = iluk_factor_with_config(matrix, config)?;
    bicgstab_iluk_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// This routine assumes a square symmetric matrix for ILDL(0) factorization.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Array1<f64>, SparseError> {
    let factorization = ildl0_factor(matrix)?;
    bicgstab_ildl0_solve_with_factorization(matrix, rhs, tolerance, max_iterations, &factorization)
}

/// Solve sparse linear system `A x = b` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization to avoid repeated setup for multiple RHS solves.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILDL0Factorization,
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

        let preconditioned_search = apply_ildl0_preconditioner(factorization, &search_direction)?;
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
            apply_ildl0_preconditioner(factorization, &auxiliary_residual)?;
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

/// Solve sparse linear systems `A X = B` with ILDL(0)-preconditioned `BiCGSTAB`.
///
/// Uses a precomputed factorization and solves each right-hand side column independently.
///
/// # Errors
/// Returns an error for invalid dimensions, factorization breakdown, or non-convergence.
pub fn bicgstab_ildl0_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILDL0Factorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        bicgstab_ildl0_solve_with_factorization(
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
pub fn gmres_ilu0_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILU0Factorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        gmres_ilu0_solve_with_factorization(
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
pub fn gmres_ilut_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUTFactorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        gmres_ilut_solve_with_factorization(
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
pub fn gmres_iluk_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILUKFactorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        gmres_iluk_solve_with_factorization(
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
pub fn gmres_ildl0_solve_multiple_with_factorization(
    matrix: &CsrMatrix,
    rhs: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
    factorization: &ILDL0Factorization,
) -> Result<Array2<f64>, SparseError> {
    solve_multiple_rhs_with_solver(matrix, rhs, |rhs_column| {
        gmres_ildl0_solve_with_factorization(
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

    fn symmetric_indefinite_matrix() -> CsrMatrix {
        // [4 1 0]
        // [1 0 1]
        // [0 1 3]
        CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0,
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
    fn ilut_factorization_reconstructs_toy_matrix() {
        let matrix = toy_matrix();
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();

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
    fn apply_ilut_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0]);
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
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn apply_iluk_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = apply_iluk_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
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
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn apply_ic0_preconditioner_rejects_bad_dimensions() {
        let matrix = toy_matrix();
        let factorization = ic0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0]);
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
                assert!((reconstructed[[row, col]] - expected[[row, col]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn apply_ildl0_preconditioner_rejects_bad_dimensions() {
        let matrix = symmetric_indefinite_matrix();
        let factorization = ildl0_factor(&matrix).unwrap();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = apply_ildl0_preconditioner(&factorization, &rhs);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn ildl0_factorization_rejects_nonsymmetric_input() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let result = ildl0_factor(&matrix);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn pcg_ic0_solves_spd_system() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0, 3.0]);
        let solution = pcg_ic0_solve(&matrix, &rhs, 1e-10, 2000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
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

    #[test]
    fn bicgstab_ilu0_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ilu0_factor(&matrix).unwrap();

        let direct = bicgstab_ilu0_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reused =
            bicgstab_ilu0_solve_with_factorization(&matrix, &rhs, 1e-10, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn bicgstab_ilut_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = bicgstab_ilut_solve(&matrix, &rhs, 1e-10, 5000, 0.0, 8).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bicgstab_ilut_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution =
            bicgstab_ilut_solve_with_config(&matrix, &rhs, 1e-10, 5000, ILUTConfig::balanced())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bicgstab_ilut_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();

        let direct = bicgstab_ilut_solve(&matrix, &rhs, 1e-10, 5000, 0.0, 8).unwrap();
        let reused =
            bicgstab_ilut_solve_with_factorization(&matrix, &rhs, 1e-10, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn bicgstab_iluk_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = bicgstab_iluk_solve(&matrix, &rhs, 1e-10, 5000, 1).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bicgstab_iluk_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution =
            bicgstab_iluk_solve_with_config(&matrix, &rhs, 1e-10, 5000, ILUKConfig::balanced())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bicgstab_iluk_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = iluk_factor(&matrix, 1).unwrap();

        let direct = bicgstab_iluk_solve(&matrix, &rhs, 1e-10, 5000, 1).unwrap();
        let reused =
            bicgstab_iluk_solve_with_factorization(&matrix, &rhs, 1e-10, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn bicgstab_ildl0_solves_symmetric_indefinite_system() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = bicgstab_ildl0_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bicgstab_ildl0_with_factorization_matches_direct() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ildl0_factor(&matrix).unwrap();

        let direct = bicgstab_ildl0_solve(&matrix, &rhs, 1e-10, 5000).unwrap();
        let reused =
            bicgstab_ildl0_solve_with_factorization(&matrix, &rhs, 1e-10, 5000, &factorization)
                .unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn gmres_ilu0_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = gmres_ilu0_solve(&matrix, &rhs, 1e-10, 10).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_ilu0_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ilu0_factor(&matrix).unwrap();

        let direct = gmres_ilu0_solve(&matrix, &rhs, 1e-10, 10).unwrap();
        let reused =
            gmres_ilu0_solve_with_factorization(&matrix, &rhs, 1e-10, 10, &factorization).unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn gmres_ilut_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = gmres_ilut_solve(&matrix, &rhs, 1e-10, 10, 0.0, 8).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_ilut_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();

        let direct = gmres_ilut_solve(&matrix, &rhs, 1e-10, 10, 0.0, 8).unwrap();
        let reused =
            gmres_ilut_solve_with_factorization(&matrix, &rhs, 1e-10, 10, &factorization).unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn gmres_iluk_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = gmres_iluk_solve(&matrix, &rhs, 1e-10, 10, 1).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_iluk_with_factorization_matches_direct() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = iluk_factor(&matrix, 1).unwrap();

        let direct = gmres_iluk_solve(&matrix, &rhs, 1e-10, 10, 1).unwrap();
        let reused =
            gmres_iluk_solve_with_factorization(&matrix, &rhs, 1e-10, 10, &factorization).unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn gmres_iluk_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution =
            gmres_iluk_solve_with_config(&matrix, &rhs, 1e-10, 10, ILUKConfig::aggressive())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_ildl0_solves_symmetric_indefinite_system() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution = gmres_ildl0_solve(&matrix, &rhs, 1e-10, 32).unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gmres_ildl0_with_factorization_matches_direct() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let factorization = ildl0_factor(&matrix).unwrap();

        let direct = gmres_ildl0_solve(&matrix, &rhs, 1e-10, 32).unwrap();
        let reused =
            gmres_ildl0_solve_with_factorization(&matrix, &rhs, 1e-10, 32, &factorization).unwrap();

        for i in 0..rhs.len() {
            assert!((direct[i] - reused[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn gmres_ilut_with_config_solves_nonsymmetric_system() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = arr1(&[1.0_f64, -2.0, 3.0]);
        let solution =
            gmres_ilut_solve_with_config(&matrix, &rhs, 1e-10, 10, ILUTConfig::aggressive())
                .unwrap();
        let reconstructed = matvec(&matrix, &solution).unwrap();
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ilut_config_profiles_are_ordered() {
        let conservative = ILUTConfig::conservative();
        let balanced = ILUTConfig::balanced();
        let aggressive = ILUTConfig::aggressive();
        assert!(conservative.drop_tolerance >= balanced.drop_tolerance);
        assert!(balanced.drop_tolerance >= aggressive.drop_tolerance);
        assert!(conservative.max_fill <= balanced.max_fill);
        assert!(balanced.max_fill <= aggressive.max_fill);

        let small = ILUTConfig::for_dimension(16);
        let medium = ILUTConfig::for_dimension(128);
        let large = ILUTConfig::for_dimension(2048);
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
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = gmres_ilut_solve(&matrix, &rhs, 1e-8, 10, 0.0, 8);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_iluk_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = gmres_iluk_solve(&matrix, &rhs, 1e-8, 10, 1);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn bicgstab_iluk_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = bicgstab_iluk_solve(&matrix, &rhs, 1e-8, 10, 1);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_ilu0_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = gmres_ilu0_solve(&matrix, &rhs, 1e-8, 10);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn gmres_ildl0_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = arr1(&[1.0_f64, 2.0]);
        let result = gmres_ildl0_solve(&matrix, &rhs, 1e-8, 10);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }

    #[test]
    fn bicgstab_ilu0_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, -2.0, 0.5, 3.0, -1.0]).unwrap();
        let factorization = ilu0_factor(&matrix).unwrap();
        let multi = bicgstab_ilu0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ilu0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn bicgstab_ilut_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, -2.0, 2.0, 3.0, 0.0]).unwrap();
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();
        let multi = bicgstab_ilut_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ilut_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn bicgstab_iluk_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, -2.0, 2.0, 3.0, 0.0]).unwrap();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let multi = bicgstab_iluk_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_iluk_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn bicgstab_ildl0_multi_rhs_matches_single_column_reuse() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, -2.0, 1.5, 3.0, -1.0]).unwrap();
        let factorization = ildl0_factor(&matrix).unwrap();
        let multi = bicgstab_ildl0_solve_multiple_with_factorization(
            &matrix,
            &rhs,
            1e-10,
            5000,
            &factorization,
        )
        .unwrap();

        for col in 0..rhs.ncols() {
            let single = bicgstab_ildl0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                5000,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gmres_ilu0_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, -2.0, 0.5, 3.0, -1.0]).unwrap();
        let factorization = ilu0_factor(&matrix).unwrap();
        let multi =
            gmres_ilu0_solve_multiple_with_factorization(&matrix, &rhs, 1e-10, 32, &factorization)
                .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ilu0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gmres_ilut_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, -2.0, 2.0, 3.0, 0.0]).unwrap();
        let factorization = ilut_factor(&matrix, 0.0, 8).unwrap();
        let multi =
            gmres_ilut_solve_multiple_with_factorization(&matrix, &rhs, 1e-10, 32, &factorization)
                .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ilut_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gmres_iluk_multi_rhs_matches_single_column_reuse() {
        let matrix = CsrMatrix::new(3, 3, vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2], vec![
            4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ])
        .unwrap();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, -2.0, 2.0, 3.0, 0.0]).unwrap();
        let factorization = iluk_factor(&matrix, 1).unwrap();
        let multi =
            gmres_iluk_solve_multiple_with_factorization(&matrix, &rhs, 1e-10, 32, &factorization)
                .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_iluk_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn gmres_ildl0_multi_rhs_matches_single_column_reuse() {
        let matrix = symmetric_indefinite_matrix();
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, -2.0, 1.5, 3.0, -1.0]).unwrap();
        let factorization = ildl0_factor(&matrix).unwrap();
        let multi =
            gmres_ildl0_solve_multiple_with_factorization(&matrix, &rhs, 1e-10, 32, &factorization)
                .unwrap();

        for col in 0..rhs.ncols() {
            let single = gmres_ildl0_solve_with_factorization(
                &matrix,
                &rhs.column(col).to_owned(),
                1e-10,
                32,
                &factorization,
            )
            .unwrap();
            for row in 0..rhs.nrows() {
                assert!((multi[[row, col]] - single[row]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn multi_rhs_rejects_dimension_mismatch() {
        let matrix = toy_matrix();
        let rhs = Array2::zeros((2, 2));
        let factorization = ilu0_factor(&matrix).unwrap();
        let result =
            gmres_ilu0_solve_multiple_with_factorization(&matrix, &rhs, 1e-8, 10, &factorization);
        assert!(matches!(result, Err(SparseError::DimensionMismatch)));
    }
}

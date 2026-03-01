//! Compile-time backend contracts for future accelerator/distributed kernels.

use std::fmt;

use ndarray::{Array2, ArrayView2, s};

/// Backend category for compile-time kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU backend.
    Cpu,
    /// GPU backend placeholder.
    Cuda,
    /// Distributed backend placeholder.
    Distributed,
}

/// Error type for backend orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorError {
    /// Selected backend is not currently available.
    UnsupportedBackend(BackendKind),
    /// Invalid chunking policy.
    InvalidChunkSize,
}

impl fmt::Display for AcceleratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorError::UnsupportedBackend(kind) => {
                write!(f, "backend {kind:?} is not currently available")
            }
            AcceleratorError::InvalidChunkSize => write!(f, "chunk size must be greater than zero"),
        }
    }
}

impl std::error::Error for AcceleratorError {}

/// Compile-time backend contract.
pub trait ComputeBackend {
    /// Backend kind.
    const KIND: BackendKind;
}

/// CPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    const KIND: BackendKind = BackendKind::Cpu;
}

/// CUDA backend placeholder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    const KIND: BackendKind = BackendKind::Cuda;
}

/// Distributed backend placeholder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DistributedBackend;

impl ComputeBackend for DistributedBackend {
    const KIND: BackendKind = BackendKind::Distributed;
}

/// Execute a closure with compile-time backend selection.
///
/// # Errors
/// Returns an error if the selected backend is not currently supported.
pub fn execute<B, T, F>(operation: F) -> Result<T, AcceleratorError>
where
    B: ComputeBackend,
    F: FnOnce() -> T,
{
    match B::KIND {
        BackendKind::Cpu => Ok(operation()),
        BackendKind::Cuda | BackendKind::Distributed => {
            Err(AcceleratorError::UnsupportedBackend(B::KIND))
        }
    }
}

/// Apply a CPU closure over row chunks.
///
/// This provides a deterministic chunking contract for future distributed
/// execution paths without introducing runtime backend switching.
///
/// # Errors
/// Returns an error for invalid chunking policy.
pub fn for_each_row_chunk(
    matrix: &Array2<f64>,
    chunk_rows: usize,
    mut operation: impl FnMut(ArrayView2<'_, f64>),
) -> Result<(), AcceleratorError> {
    if chunk_rows == 0 {
        return Err(AcceleratorError::InvalidChunkSize);
    }
    let mut row = 0_usize;
    while row < matrix.nrows() {
        let end = (row + chunk_rows).min(matrix.nrows());
        operation(matrix.slice(s![row..end, ..]));
        row = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn cpu_backend_executes_operation() {
        let value = execute::<CpuBackend, _, _>(|| 2 + 3).unwrap();
        assert_eq!(value, 5);
    }

    #[test]
    fn unsupported_backends_return_error() {
        let cuda = execute::<CudaBackend, _, _>(|| 1);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));

        let distributed = execute::<DistributedBackend, _, _>(|| 1);
        assert!(matches!(
            distributed,
            Err(AcceleratorError::UnsupportedBackend(BackendKind::Distributed))
        ));
    }

    #[test]
    fn chunking_visits_all_rows() {
        let matrix =
            Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                .unwrap();

        let mut seen_rows = 0_usize;
        for_each_row_chunk(&matrix, 2, |chunk| {
            seen_rows += chunk.nrows();
        })
        .unwrap();
        assert_eq!(seen_rows, matrix.nrows());
    }

    #[test]
    fn chunking_rejects_invalid_chunk_size() {
        let matrix = Array2::<f64>::zeros((2, 2));
        let result = for_each_row_chunk(&matrix, 0, |_chunk| {});
        assert!(matches!(result, Err(AcceleratorError::InvalidChunkSize)));
    }
}

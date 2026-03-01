//! Compile-time backend contracts for future accelerator/distributed kernels.

use std::{fmt, thread};

use ndarray::{Array2, ArrayView2, s};
#[cfg(feature = "accelerator-rayon")]
use rayon::prelude::*;

/// Backend category for compile-time kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU backend.
    Cpu,
    /// GPU backend (not yet implemented).
    Cuda,
    /// Distributed CPU-sharded backend.
    Distributed,
}

/// Error type for backend orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorError {
    /// Selected backend is not currently available.
    UnsupportedBackend(BackendKind),
    /// Invalid chunking policy.
    InvalidChunkSize,
    /// Invalid distributed worker count.
    InvalidWorkerCount,
    /// Matrix dimensions are incompatible.
    DimensionMismatch,
    /// Optional accelerator feature was not enabled at compile time.
    FeatureNotEnabled,
    /// A distributed worker panicked while executing a kernel.
    WorkerPanicked,
}

impl fmt::Display for AcceleratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorError::UnsupportedBackend(kind) => {
                write!(f, "backend {kind:?} is not currently available")
            }
            AcceleratorError::InvalidChunkSize => write!(f, "chunk size must be greater than zero"),
            AcceleratorError::InvalidWorkerCount => {
                write!(f, "worker count must be greater than zero")
            }
            AcceleratorError::DimensionMismatch => {
                write!(f, "matrix dimensions are incompatible")
            }
            AcceleratorError::FeatureNotEnabled => {
                write!(f, "feature `accelerator-rayon` is not enabled")
            }
            AcceleratorError::WorkerPanicked => {
                write!(f, "distributed worker panicked")
            }
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

/// Distributed backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DistributedBackend;

impl ComputeBackend for DistributedBackend {
    const KIND: BackendKind = BackendKind::Distributed;
}

/// Distributed execution configuration for row-sharded kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedConfig {
    /// Number of worker threads.
    pub workers:    usize,
    /// Rows per work chunk.
    pub chunk_rows: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self { Self { workers: 4, chunk_rows: 64 } }
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
        BackendKind::Cpu | BackendKind::Distributed => Ok(operation()),
        BackendKind::Cuda => Err(AcceleratorError::UnsupportedBackend(B::KIND)),
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

/// Compute matrix-matrix product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_serial(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<f64>::zeros((left.nrows(), right.ncols()));
    for row in 0..left.nrows() {
        for inner in 0..left.ncols() {
            let lhs = left[[row, inner]];
            for col in 0..right.ncols() {
                output[[row, col]] += lhs * right[[inner, col]];
            }
        }
    }
    Ok(output)
}

/// Compute matrix-matrix product with row-sharded distributed CPU execution.
///
/// Work is split into row chunks and scheduled over a fixed number of workers.
///
/// # Errors
/// Returns an error if dimensions are incompatible, worker/chunk config is invalid,
/// or if a worker panics.
pub fn matmat_distributed(
    left: &Array2<f64>,
    right: &Array2<f64>,
    config: DistributedConfig,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    if config.workers == 0 {
        return Err(AcceleratorError::InvalidWorkerCount);
    }
    if config.chunk_rows == 0 {
        return Err(AcceleratorError::InvalidChunkSize);
    }

    let rows = left.nrows();
    let cols = right.ncols();
    let mut chunks = Vec::new();
    let mut start_row = 0_usize;
    while start_row < rows {
        let end_row = (start_row + config.chunk_rows).min(rows);
        chunks.push((start_row, end_row));
        start_row = end_row;
    }

    if chunks.is_empty() {
        return Ok(Array2::<f64>::zeros((rows, cols)));
    }

    let worker_count = config.workers.min(chunks.len());
    let mut partials = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let left_ref = left;
            let right_ref = right;
            let chunks_ref = &chunks;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                let mut chunk_id = worker_id;
                while chunk_id < chunks_ref.len() {
                    let (start, end) = chunks_ref[chunk_id];
                    let block_rows = end - start;
                    let mut block = Array2::<f64>::zeros((block_rows, cols));
                    for local_row in 0..block_rows {
                        let row = start + local_row;
                        for inner in 0..left_ref.ncols() {
                            let lhs = left_ref[[row, inner]];
                            for col in 0..cols {
                                block[[local_row, col]] += lhs * right_ref[[inner, col]];
                            }
                        }
                    }
                    local.push((start, block));
                    chunk_id += worker_count;
                }
                local
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(worker_chunks) => partials.extend(worker_chunks),
                Err(_) => return Err(AcceleratorError::WorkerPanicked),
            }
        }
        Ok::<(), AcceleratorError>(())
    })?;

    let mut output = Array2::<f64>::zeros((rows, cols));
    for (start, block) in partials {
        let end = start + block.nrows();
        output.slice_mut(s![start..end, ..]).assign(&block);
    }
    Ok(output)
}

/// Compute matrix-matrix product using compile-time backend dispatch.
///
/// - `CpuBackend`: serial kernel
/// - `DistributedBackend`: row-sharded distributed kernel with default config
/// - `CudaBackend`: currently unsupported
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matmat_with_backend<B: ComputeBackend>(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    match B::KIND {
        BackendKind::Cpu => matmat_serial(left, right),
        BackendKind::Distributed => matmat_distributed(left, right, DistributedConfig::default()),
        BackendKind::Cuda => Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda)),
    }
}

/// Compute matrix-matrix product using feature-gated accelerated kernel.
///
/// When `accelerator-rayon` is enabled, rows are computed in parallel.
/// Otherwise, this returns [`AcceleratorError::FeatureNotEnabled`].
///
/// # Errors
/// Returns an error for incompatible dimensions or if accelerator feature is disabled.
pub fn matmat_accelerated(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    #[cfg(feature = "accelerator-rayon")]
    {
        let cols = right.ncols();
        let rows = left.nrows();
        let inner = left.ncols();
        let row_results = (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut out_row = vec![0.0_f64; cols];
                for k in 0..inner {
                    let lhs = left[[row, k]];
                    for col in 0..cols {
                        out_row[col] += lhs * right[[k, col]];
                    }
                }
                out_row
            })
            .collect::<Vec<_>>();

        let mut output = Array2::<f64>::zeros((rows, cols));
        for (row, row_values) in row_results.into_iter().enumerate() {
            for (col, value) in row_values.into_iter().enumerate() {
                output[[row, col]] = value;
            }
        }
        Ok(output)
    }

    #[cfg(not(feature = "accelerator-rayon"))]
    {
        let _ = left;
        let _ = right;
        Err(AcceleratorError::FeatureNotEnabled)
    }
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
    fn distributed_backend_executes_operation() {
        let value = execute::<DistributedBackend, _, _>(|| 7 * 6).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn cuda_backend_returns_error() {
        let cuda = execute::<CudaBackend, _, _>(|| 1);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
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

    #[test]
    fn serial_matmat_matches_expected() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let output = matmat_serial(&left, &right).unwrap();
        assert_eq!(output.dim(), (2, 2));
        assert!((output[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((output[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((output[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn distributed_matmat_matches_serial() {
        let left = Array2::from_shape_vec((5, 4), vec![
            1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 1.0, -1.0, 3.0, 1.0, 0.0, 2.0, 2.0,
            -1.0, 1.0, 0.0,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((4, 3), vec![
            1.0, 0.0, 2.0, 2.0, 1.0, -1.0, 1.0, 3.0, 0.0, -1.0, 2.0, 1.0,
        ])
        .unwrap();

        let serial = matmat_serial(&left, &right).unwrap();
        let distributed =
            matmat_distributed(&left, &right, DistributedConfig { workers: 3, chunk_rows: 2 })
                .unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - distributed[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_matmat_rejects_invalid_config() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);
        let invalid_workers =
            matmat_distributed(&left, &right, DistributedConfig { workers: 0, chunk_rows: 1 });
        assert!(matches!(invalid_workers, Err(AcceleratorError::InvalidWorkerCount)));

        let invalid_chunks =
            matmat_distributed(&left, &right, DistributedConfig { workers: 1, chunk_rows: 0 });
        assert!(matches!(invalid_chunks, Err(AcceleratorError::InvalidChunkSize)));
    }

    #[test]
    fn backend_dispatch_selects_expected_kernel() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let serial = matmat_with_backend::<CpuBackend>(&left, &right).unwrap();
        let distributed = matmat_with_backend::<DistributedBackend>(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - distributed[[row, col]]).abs() < 1e-12);
            }
        }

        let cuda = matmat_with_backend::<CudaBackend>(&left, &right);
        assert!(matches!(cuda, Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))));
    }

    #[test]
    fn matmat_rejects_dimension_mismatch() {
        let left = Array2::<f64>::zeros((2, 3));
        let right = Array2::<f64>::zeros((2, 2));
        assert!(matches!(matmat_serial(&left, &right), Err(AcceleratorError::DimensionMismatch)));
        assert!(matches!(
            matmat_accelerated(&left, &right),
            Err(AcceleratorError::DimensionMismatch)
        ));
    }

    #[cfg(not(feature = "accelerator-rayon"))]
    #[test]
    fn accelerated_matmat_requires_feature() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);
        let result = matmat_accelerated(&left, &right);
        assert!(matches!(result, Err(AcceleratorError::FeatureNotEnabled)));
    }

    #[cfg(feature = "accelerator-rayon")]
    #[test]
    fn accelerated_matmat_matches_serial() {
        let left = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 0.0, 1.0, -1.0,
        ])
        .unwrap();
        let right =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0, -1.0, 2.0]).unwrap();
        let serial = matmat_serial(&left, &right).unwrap();
        let accelerated = matmat_accelerated(&left, &right).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - accelerated[[row, col]]).abs() < 1e-12);
            }
        }
    }
}

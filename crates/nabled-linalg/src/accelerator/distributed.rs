use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use ndarray::{Array2, s};

use super::backends::AcceleratorError;

/// Distributed execution configuration for row-sharded kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedConfig {
    /// Number of worker threads.
    pub workers:    usize,
    /// Rows per work chunk.
    pub chunk_rows: usize,
    /// Scheduling policy across workers.
    pub schedule:   DistributedSchedule,
}

/// Scheduling policy for distributed CPU kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributedSchedule {
    /// Static worker partitioning by strided chunk id.
    Static,
    /// Dynamic queue-based chunk stealing.
    Dynamic,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self { workers: 4, chunk_rows: 64, schedule: DistributedSchedule::Static }
    }
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
    let partials = match config.schedule {
        DistributedSchedule::Static => {
            matmat_distributed_static(left, right, cols, &chunks, worker_count)?
        }
        DistributedSchedule::Dynamic => {
            matmat_distributed_dynamic(left, right, cols, &chunks, worker_count)?
        }
    };

    let mut output = Array2::<f64>::zeros((rows, cols));
    for (start, block) in partials {
        let end = start + block.nrows();
        output.slice_mut(s![start..end, ..]).assign(&block);
    }
    Ok(output)
}

fn compute_chunk_matmat(
    left: &Array2<f64>,
    right: &Array2<f64>,
    start: usize,
    end: usize,
    cols: usize,
) -> Array2<f64> {
    let mut block = Array2::<f64>::zeros((end - start, cols));
    for local_row in 0..(end - start) {
        let row = start + local_row;
        for inner in 0..left.ncols() {
            let lhs = left[[row, inner]];
            for col in 0..cols {
                block[[local_row, col]] += lhs * right[[inner, col]];
            }
        }
    }
    block
}

fn matmat_distributed_static(
    left: &Array2<f64>,
    right: &Array2<f64>,
    cols: usize,
    chunks: &[(usize, usize)],
    worker_count: usize,
) -> Result<Vec<(usize, Array2<f64>)>, AcceleratorError> {
    let mut partials = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let left_ref = left;
            let right_ref = right;
            let chunks_ref = chunks;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                let mut chunk_id = worker_id;
                while chunk_id < chunks_ref.len() {
                    let (start, end) = chunks_ref[chunk_id];
                    local
                        .push((start, compute_chunk_matmat(left_ref, right_ref, start, end, cols)));
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
    Ok(partials)
}

fn matmat_distributed_dynamic(
    left: &Array2<f64>,
    right: &Array2<f64>,
    cols: usize,
    chunks: &[(usize, usize)],
    worker_count: usize,
) -> Result<Vec<(usize, Array2<f64>)>, AcceleratorError> {
    let mut partials = Vec::new();
    let next_chunk = Arc::new(AtomicUsize::new(0));
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _worker_id in 0..worker_count {
            let left_ref = left;
            let right_ref = right;
            let chunks_ref = chunks;
            let next_chunk_ref = Arc::clone(&next_chunk);
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let chunk_id = next_chunk_ref.fetch_add(1, Ordering::Relaxed);
                    if chunk_id >= chunks_ref.len() {
                        break;
                    }
                    let (start, end) = chunks_ref[chunk_id];
                    local
                        .push((start, compute_chunk_matmat(left_ref, right_ref, start, end, cols)));
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
    Ok(partials)
}

/// Compute matrix-matrix product with tiled distributed CPU execution.
///
/// Work is split into `(tile_rows, tile_cols)` output tiles and scheduled
/// over a fixed number of workers.
///
/// # Errors
/// Returns an error if dimensions are incompatible, worker/tile config is invalid,
/// or if a worker panics.
pub fn matmat_distributed_tiled(
    left: &Array2<f64>,
    right: &Array2<f64>,
    workers: usize,
    tile_rows: usize,
    tile_cols: usize,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    if workers == 0 {
        return Err(AcceleratorError::InvalidWorkerCount);
    }
    if tile_rows == 0 || tile_cols == 0 {
        return Err(AcceleratorError::InvalidTileSize);
    }

    let rows = left.nrows();
    let cols = right.ncols();
    let inner = left.ncols();
    let mut tiles = Vec::new();
    let mut row_start = 0_usize;
    while row_start < rows {
        let row_end = (row_start + tile_rows).min(rows);
        let mut col_start = 0_usize;
        while col_start < cols {
            let col_end = (col_start + tile_cols).min(cols);
            tiles.push((row_start, row_end, col_start, col_end));
            col_start = col_end;
        }
        row_start = row_end;
    }

    if tiles.is_empty() {
        return Ok(Array2::<f64>::zeros((rows, cols)));
    }

    let worker_count = workers.min(tiles.len());
    let mut partials = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let left_ref = left;
            let right_ref = right;
            let tiles_ref = &tiles;
            handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                let mut tile_id = worker_id;
                while tile_id < tiles_ref.len() {
                    let (r0, r1, c0, c1) = tiles_ref[tile_id];
                    let mut block = Array2::<f64>::zeros((r1 - r0, c1 - c0));
                    for local_row in 0..(r1 - r0) {
                        let row = r0 + local_row;
                        for k in 0..inner {
                            let lhs = left_ref[[row, k]];
                            for local_col in 0..(c1 - c0) {
                                let col = c0 + local_col;
                                block[[local_row, local_col]] += lhs * right_ref[[k, col]];
                            }
                        }
                    }
                    local.push((r0, c0, block));
                    tile_id += worker_count;
                }
                local
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(worker_tiles) => partials.extend(worker_tiles),
                Err(_) => return Err(AcceleratorError::WorkerPanicked),
            }
        }
        Ok::<(), AcceleratorError>(())
    })?;

    let mut output = Array2::<f64>::zeros((rows, cols));
    for (row_start, col_start, block) in partials {
        let row_end = row_start + block.nrows();
        let col_end = col_start + block.ncols();
        output.slice_mut(s![row_start..row_end, col_start..col_end]).assign(&block);
    }
    Ok(output)
}

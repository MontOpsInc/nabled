//! Compile-time backend contracts for future accelerator/distributed kernels.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fmt, thread};

use ndarray::{Array2, ArrayView2, s};
#[cfg(feature = "accelerator-rayon")]
use rayon::prelude::*;
#[cfg(feature = "accelerator-wgpu")]
use wgpu::util::DeviceExt;

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
    /// Invalid tile geometry for tiled distributed kernels.
    InvalidTileSize,
    /// Matrix dimensions are incompatible.
    DimensionMismatch,
    /// Optional accelerator feature was not enabled at compile time.
    FeatureNotEnabled,
    /// A distributed worker panicked while executing a kernel.
    WorkerPanicked,
    /// No suitable GPU device was found.
    DeviceUnavailable,
    /// GPU kernel execution failed.
    KernelExecutionFailed,
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
            AcceleratorError::InvalidTileSize => {
                write!(f, "tile dimensions must be greater than zero")
            }
            AcceleratorError::DimensionMismatch => {
                write!(f, "matrix dimensions are incompatible")
            }
            AcceleratorError::FeatureNotEnabled => {
                write!(f, "requested accelerator feature is not enabled")
            }
            AcceleratorError::WorkerPanicked => {
                write!(f, "distributed worker panicked")
            }
            AcceleratorError::DeviceUnavailable => write!(f, "no suitable GPU device is available"),
            AcceleratorError::KernelExecutionFailed => {
                write!(f, "GPU kernel execution failed")
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

fn matmat_serial_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<f32>::zeros((left.nrows(), right.ncols()));
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

/// Compute matrix-matrix product using compile-time backend dispatch for `f32`.
///
/// This exposes GPU execution via [`CudaBackend`] when `accelerator-wgpu` is enabled.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matmat_with_backend_f32<B: ComputeBackend>(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    match B::KIND {
        BackendKind::Cpu | BackendKind::Distributed => matmat_serial_f32(left, right),
        BackendKind::Cuda => matmat_gpu_f32(left, right),
    }
}

#[cfg(feature = "accelerator-wgpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMatMulParams {
    rows:  u32,
    cols:  u32,
    inner: u32,
    _pad:  u32,
}

#[cfg(feature = "accelerator-wgpu")]
const WGSL_MATMUL: &str = r"
struct Params {
    rows: u32,
    cols: u32,
    inner: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.inner; k = k + 1u) {
        let lhs = left[row * params.inner + k];
        let rhs = right[k * params.cols + col];
        acc = acc + lhs * rhs;
    }
    out[row * params.cols + col] = acc;
}
";

#[cfg(feature = "accelerator-wgpu")]
struct GpuBuffers {
    output_buffer:   wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer:  wgpu::Buffer,
    bind_group:      wgpu::BindGroup,
    rows_u32:        u32,
    cols_u32:        u32,
}

#[cfg(feature = "accelerator-wgpu")]
fn output_size_bytes(rows: usize, cols: usize) -> Result<u64, AcceleratorError> {
    use std::mem::size_of;

    let bytes = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(size_of::<f32>()))
        .ok_or(AcceleratorError::KernelExecutionFailed)?;
    u64::try_from(bytes).map_err(|_| AcceleratorError::KernelExecutionFailed)
}

#[cfg(feature = "accelerator-wgpu")]
fn request_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue), AcceleratorError> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|_| AcceleratorError::DeviceUnavailable)?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .map_err(|_| AcceleratorError::DeviceUnavailable)
}

#[cfg(feature = "accelerator-wgpu")]
fn create_pipeline_and_layout(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("nabled.gpu.matmul"),
        source: wgpu::ShaderSource::Wgsl(WGSL_MATMUL.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some("nabled.gpu.bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count:      None,
            },
            wgpu::BindGroupLayoutEntry {
                binding:    1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count:      None,
            },
            wgpu::BindGroupLayoutEntry {
                binding:    2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count:      None,
            },
            wgpu::BindGroupLayoutEntry {
                binding:    3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count:      None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("nabled.gpu.pipeline_layout"),
        bind_group_layouts:   &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:               Some("nabled.gpu.pipeline"),
        layout:              Some(&pipeline_layout),
        module:              &shader,
        entry_point:         Some("main"),
        cache:               None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    (pipeline, bind_group_layout)
}

#[cfg(feature = "accelerator-wgpu")]
fn create_buffers_and_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<GpuBuffers, AcceleratorError> {
    let rows = left.nrows();
    let cols = right.ncols();
    let inner = left.ncols();
    let output_size = output_size_bytes(rows, cols)?;
    let rows_u32 = u32::try_from(rows).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let cols_u32 = u32::try_from(cols).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let inner_u32 = u32::try_from(inner).map_err(|_| AcceleratorError::KernelExecutionFailed)?;

    let left_data = left.iter().copied().collect::<Vec<_>>();
    let right_data = right.iter().copied().collect::<Vec<_>>();
    let params = GpuMatMulParams { rows: rows_u32, cols: cols_u32, inner: inner_u32, _pad: 0 };

    let left_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.left"),
        contents: bytemuck::cast_slice(&left_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let right_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.right"),
        contents: bytemuck::cast_slice(&right_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("nabled.gpu.output"),
        size:               output_size,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("nabled.gpu.readback"),
        size:               output_size,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("nabled.gpu.bind_group"),
        layout:  bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: left_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: right_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
        ],
    });

    Ok(GpuBuffers {
        output_buffer,
        readback_buffer,
        _params_buffer: params_buffer,
        bind_group,
        rows_u32,
        cols_u32,
    })
}

#[cfg(feature = "accelerator-wgpu")]
fn dispatch_gpu_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &GpuBuffers,
) -> Result<(), AcceleratorError> {
    let rows =
        usize::try_from(buffers.rows_u32).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let cols =
        usize::try_from(buffers.cols_u32).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let output_size = output_size_bytes(rows, cols)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("nabled.gpu.encoder"),
    });
    {
        let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute.set_pipeline(pipeline);
        compute.set_bind_group(0, &buffers.bind_group, &[]);
        let workgroups_x = buffers.cols_u32.div_ceil(16);
        let workgroups_y = buffers.rows_u32.div_ceil(16);
        compute.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    encoder.copy_buffer_to_buffer(
        &buffers.output_buffer,
        0,
        &buffers.readback_buffer,
        0,
        output_size,
    );
    let _submission_index = queue.submit(Some(encoder.finish()));

    let readback_slice = buffers.readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _poll_status = device.poll(wgpu::PollType::wait());
    match receiver.recv() {
        Ok(Ok(())) => Ok(()),
        _ => Err(AcceleratorError::KernelExecutionFailed),
    }
}

#[cfg(feature = "accelerator-wgpu")]
fn read_gpu_output(
    readback_buffer: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>, AcceleratorError> {
    let mapped = readback_buffer.slice(..).get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    readback_buffer.unmap();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|_| AcceleratorError::KernelExecutionFailed)
}

#[cfg(feature = "accelerator-wgpu")]
fn matmat_gpu_f32_wgpu(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    let (device, queue) = request_wgpu_device()?;
    let (pipeline, bind_group_layout) = create_pipeline_and_layout(&device);
    let buffers = create_buffers_and_bind_group(&device, &bind_group_layout, left, right)?;
    dispatch_gpu_kernel(&device, &queue, &pipeline, &buffers)?;
    read_gpu_output(&buffers.readback_buffer, left.nrows(), right.ncols())
}

/// Compute matrix-matrix product on GPU for `f32` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, or kernel failures.
pub fn matmat_gpu_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    #[cfg(feature = "accelerator-wgpu")]
    {
        matmat_gpu_f32_wgpu(left, right)
    }

    #[cfg(not(feature = "accelerator-wgpu"))]
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
        let distributed = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Static,
        })
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
        let invalid_workers = matmat_distributed(&left, &right, DistributedConfig {
            workers:    0,
            chunk_rows: 1,
            schedule:   DistributedSchedule::Static,
        });
        assert!(matches!(invalid_workers, Err(AcceleratorError::InvalidWorkerCount)));

        let invalid_chunks = matmat_distributed(&left, &right, DistributedConfig {
            workers:    1,
            chunk_rows: 0,
            schedule:   DistributedSchedule::Static,
        });
        assert!(matches!(invalid_chunks, Err(AcceleratorError::InvalidChunkSize)));
    }

    #[test]
    fn distributed_dynamic_matches_static() {
        let left =
            Array2::from_shape_vec((8, 5), (0..40).map(|value| f64::from(value) * 0.125).collect())
                .unwrap();
        let right =
            Array2::from_shape_vec((5, 7), (0..35).map(|value| f64::from(value) * -0.25).collect())
                .unwrap();

        let static_result = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Static,
        })
        .unwrap();
        let dynamic_result = matmat_distributed(&left, &right, DistributedConfig {
            workers:    3,
            chunk_rows: 2,
            schedule:   DistributedSchedule::Dynamic,
        })
        .unwrap();
        for row in 0..static_result.nrows() {
            for col in 0..static_result.ncols() {
                assert!((static_result[[row, col]] - dynamic_result[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_tiled_matmat_matches_serial() {
        let left =
            Array2::from_shape_vec((7, 5), (0..35).map(|v| f64::from(v) * 0.25).collect()).unwrap();
        let right =
            Array2::from_shape_vec((5, 6), (0..30).map(|v| f64::from(v) * -0.5).collect()).unwrap();

        let serial = matmat_serial(&left, &right).unwrap();
        let tiled = matmat_distributed_tiled(&left, &right, 3, 2, 3).unwrap();
        for row in 0..serial.nrows() {
            for col in 0..serial.ncols() {
                assert!((serial[[row, col]] - tiled[[row, col]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn distributed_tiled_matmat_rejects_invalid_config() {
        let left = Array2::<f64>::eye(2);
        let right = Array2::<f64>::eye(2);

        let invalid_workers = matmat_distributed_tiled(&left, &right, 0, 1, 1);
        assert!(matches!(invalid_workers, Err(AcceleratorError::InvalidWorkerCount)));

        let invalid_rows = matmat_distributed_tiled(&left, &right, 1, 0, 1);
        assert!(matches!(invalid_rows, Err(AcceleratorError::InvalidTileSize)));

        let invalid_cols = matmat_distributed_tiled(&left, &right, 1, 1, 0);
        assert!(matches!(invalid_cols, Err(AcceleratorError::InvalidTileSize)));
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

    #[cfg(not(feature = "accelerator-wgpu"))]
    #[test]
    fn gpu_matmat_requires_feature() {
        let left = Array2::<f32>::eye(2);
        let right = Array2::<f32>::eye(2);
        let result = matmat_gpu_f32(&left, &right);
        assert!(matches!(result, Err(AcceleratorError::FeatureNotEnabled)));
    }

    #[cfg(feature = "accelerator-wgpu")]
    #[test]
    fn gpu_matmat_matches_cpu_or_reports_unavailable_device() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let cpu = matmat_with_backend_f32::<CpuBackend>(&left, &right).unwrap();
        match matmat_with_backend_f32::<CudaBackend>(&left, &right) {
            Ok(gpu) => {
                for row in 0..cpu.nrows() {
                    for col in 0..cpu.ncols() {
                        assert!((cpu[[row, col]] - gpu[[row, col]]).abs() < 1e-4);
                    }
                }
            }
            Err(error) => assert!(matches!(error, AcceleratorError::DeviceUnavailable)),
        }
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

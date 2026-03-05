#[cfg(feature = "accelerator-wgpu")]
use std::mem::size_of;
#[cfg(feature = "accelerator-wgpu")]
use std::sync::OnceLock;

use ndarray::{Array1, Array2, Array3, ArrayD, ArrayView2, Axis, IxDyn, s};
#[cfg(feature = "accelerator-wgpu")]
use wgpu::util::DeviceExt;

use super::backends::AcceleratorError;
#[cfg(feature = "accelerator-wgpu")]
use super::backends::BackendKind;

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
const WGSL_MATMUL_F32: &str = r"
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
const WGSL_MATMUL_F64: &str = r"
struct Params {
    rows: u32,
    cols: u32,
    inner: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> left: array<f64>;
@group(0) @binding(1) var<storage, read> right: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    var acc: f64 = 0.0;
    for (var k: u32 = 0u; k < params.inner; k = k + 1u) {
        let lhs = left[row * params.inner + k];
        let rhs = right[k * params.cols + col];
        acc = acc + lhs * rhs;
    }
    out[row * params.cols + col] = acc;
}
";

#[cfg(feature = "accelerator-wgpu")]
struct WgpuRuntime {
    device:            wgpu::Device,
    queue:             wgpu::Queue,
    pipeline_f32:      wgpu::ComputePipeline,
    pipeline_f64:      Option<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "accelerator-wgpu")]
struct GpuBuffers {
    output_buffer:   wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer:  wgpu::Buffer,
    bind_group:      wgpu::BindGroup,
    elem_size:       usize,
    rows_u32:        u32,
    cols_u32:        u32,
}

#[cfg(feature = "accelerator-wgpu")]
fn output_size_bytes<T>(rows: usize, cols: usize) -> Result<u64, AcceleratorError> {
    use std::mem::size_of;

    let bytes = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(size_of::<T>()))
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
fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

#[cfg(feature = "accelerator-wgpu")]
fn create_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label: &'static str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("nabled.gpu.pipeline_layout"),
        bind_group_layouts:   &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:               Some(label),
        layout:              Some(&pipeline_layout),
        module:              &shader,
        entry_point:         Some("main"),
        cache:               None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

#[cfg(feature = "accelerator-wgpu")]
fn create_f64_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> Option<wgpu::ComputePipeline> {
    if !device.features().contains(wgpu::Features::SHADER_F64) {
        return None;
    }

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline =
        create_pipeline(device, bind_group_layout, WGSL_MATMUL_F64, "nabled.gpu.pipeline_f64");
    let _poll_status = device.poll(wgpu::PollType::wait());

    match pollster::block_on(device.pop_error_scope()) {
        Some(_error) => None,
        None => Some(pipeline),
    }
}

#[cfg(feature = "accelerator-wgpu")]
fn init_wgpu_runtime() -> Result<WgpuRuntime, AcceleratorError> {
    let (device, queue) = request_wgpu_device()?;
    let bind_group_layout = create_bind_group_layout(&device);
    let pipeline_f32 =
        create_pipeline(&device, &bind_group_layout, WGSL_MATMUL_F32, "nabled.gpu.pipeline_f32");
    let pipeline_f64 = create_f64_pipeline(&device, &bind_group_layout);
    Ok(WgpuRuntime { device, queue, pipeline_f32, pipeline_f64, bind_group_layout })
}

#[cfg(feature = "accelerator-wgpu")]
fn wgpu_runtime() -> Result<&'static WgpuRuntime, AcceleratorError> {
    static RUNTIME: OnceLock<Result<WgpuRuntime, AcceleratorError>> = OnceLock::new();
    match RUNTIME.get_or_init(init_wgpu_runtime) {
        Ok(runtime) => Ok(runtime),
        Err(error) => Err(*error),
    }
}

#[cfg(feature = "accelerator-wgpu")]
fn matrix_data<T>(matrix: &ArrayView2<'_, T>) -> Vec<T>
where
    T: Copy,
{
    matrix
        .as_slice_memory_order()
        .map_or_else(|| matrix.iter().copied().collect::<Vec<_>>(), <[T]>::to_vec)
}

#[cfg(feature = "accelerator-wgpu")]
fn create_buffers_and_bind_group<T>(
    runtime: &WgpuRuntime,
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<GpuBuffers, AcceleratorError>
where
    T: Copy + bytemuck::Pod,
{
    let rows = left.nrows();
    let cols = right.ncols();
    let inner = left.ncols();
    let output_size = output_size_bytes::<T>(rows, cols)?;
    let rows_u32 = u32::try_from(rows).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let cols_u32 = u32::try_from(cols).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let inner_u32 = u32::try_from(inner).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let params = GpuMatMulParams { rows: rows_u32, cols: cols_u32, inner: inner_u32, _pad: 0 };
    let left_data = matrix_data(left);
    let right_data = matrix_data(right);

    let left_buffer = runtime.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.left"),
        contents: bytemuck::cast_slice(&left_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let right_buffer = runtime.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.right"),
        contents: bytemuck::cast_slice(&right_data),
        usage:    wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("nabled.gpu.output"),
        size:               output_size,
        usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params_buffer = runtime.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("nabled.gpu.params"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
    });
    let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label:              Some("nabled.gpu.readback"),
        size:               output_size,
        usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = runtime.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("nabled.gpu.bind_group"),
        layout:  &runtime.bind_group_layout,
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
        elem_size: size_of::<T>(),
        rows_u32,
        cols_u32,
    })
}

#[cfg(feature = "accelerator-wgpu")]
fn dispatch_gpu_kernel(
    runtime: &WgpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    buffers: &GpuBuffers,
) -> Result<(), AcceleratorError> {
    let rows =
        usize::try_from(buffers.rows_u32).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let cols =
        usize::try_from(buffers.cols_u32).map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let output_size = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(buffers.elem_size))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AcceleratorError::KernelExecutionFailed)?;
    let mut encoder = runtime.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
    let _submission_index = runtime.queue.submit(Some(encoder.finish()));

    let readback_slice = buffers.readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _poll_status = runtime.device.poll(wgpu::PollType::wait());
    match receiver.recv() {
        Ok(Ok(())) => Ok(()),
        _ => Err(AcceleratorError::KernelExecutionFailed),
    }
}

#[cfg(feature = "accelerator-wgpu")]
fn read_gpu_output<T>(
    readback_buffer: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Result<Array2<T>, AcceleratorError>
where
    T: bytemuck::Pod,
{
    let mapped = readback_buffer.slice(..).get_mapped_range();
    let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    readback_buffer.unmap();
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|_| AcceleratorError::KernelExecutionFailed)
}

#[cfg(feature = "accelerator-wgpu")]
fn matmat_gpu_f32_wgpu(
    left: &ArrayView2<'_, f32>,
    right: &ArrayView2<'_, f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    let runtime = wgpu_runtime()?;
    let buffers = create_buffers_and_bind_group(runtime, left, right)?;
    dispatch_gpu_kernel(runtime, &runtime.pipeline_f32, &buffers)?;
    read_gpu_output(&buffers.readback_buffer, left.nrows(), right.ncols())
}

#[cfg(feature = "accelerator-wgpu")]
fn matmat_gpu_f64_wgpu(
    left: &ArrayView2<'_, f64>,
    right: &ArrayView2<'_, f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    let runtime = wgpu_runtime()?;
    let Some(pipeline) = runtime.pipeline_f64.as_ref() else {
        return Err(AcceleratorError::UnsupportedBackend(BackendKind::Gpu));
    };
    let buffers = create_buffers_and_bind_group(runtime, left, right)?;
    dispatch_gpu_kernel(runtime, pipeline, &buffers)?;
    read_gpu_output(&buffers.readback_buffer, left.nrows(), right.ncols())
}

fn matmat_gpu_f32_impl(
    left: &ArrayView2<'_, f32>,
    right: &ArrayView2<'_, f32>,
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

fn matmat_gpu_f64_impl(
    left: &ArrayView2<'_, f64>,
    right: &ArrayView2<'_, f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    #[cfg(feature = "accelerator-wgpu")]
    {
        matmat_gpu_f64_wgpu(left, right)
    }

    #[cfg(not(feature = "accelerator-wgpu"))]
    {
        let _ = left;
        let _ = right;
        Err(AcceleratorError::FeatureNotEnabled)
    }
}

fn shape_product(dims: &[usize]) -> usize { dims.iter().copied().product::<usize>().max(1) }

fn uncontracted_axes(ndim: usize, contracted_axis: usize) -> Result<Vec<usize>, AcceleratorError> {
    if contracted_axis >= ndim {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok((0..ndim).filter(|axis| *axis != contracted_axis).collect())
}

fn row_norms_squared<T>(matrix: &Array2<T>) -> Array1<T>
where
    T: num_traits::Float,
{
    let mut norms = Array1::<T>::zeros(matrix.nrows());
    for row in 0..matrix.nrows() {
        let mut sq_sum = T::zero();
        for col in 0..matrix.ncols() {
            let value = matrix[[row, col]];
            sq_sum = sq_sum + value * value;
        }
        norms[row] = sq_sum;
    }
    norms
}

/// Compute matrix-matrix product on GPU for `f32` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn matmat_gpu_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    matmat_gpu_f32_impl(&left.view(), &right.view())
}

/// Compute matrix-matrix product on GPU for `f64` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn matmat_gpu_f64(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    matmat_gpu_f64_impl(&left.view(), &right.view())
}

/// Compute matrix-vector product on GPU for `f32` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn matvec_gpu_f32(
    matrix: &Array2<f32>,
    vector: &Array1<f32>,
) -> Result<Array1<f32>, AcceleratorError> {
    if matrix.ncols() != vector.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let right = vector.view().insert_axis(Axis(1)).to_owned();
    let product = matmat_gpu_f32_impl(&matrix.view(), &right.view())?;
    Ok(product.column(0).to_owned())
}

/// Compute matrix-vector product on GPU for `f64` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn matvec_gpu_f64(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError> {
    if matrix.ncols() != vector.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let right = vector.view().insert_axis(Axis(1)).to_owned();
    let product = matmat_gpu_f64_impl(&matrix.view(), &right.view())?;
    Ok(product.column(0).to_owned())
}

/// Compute batched matrix-matrix products on GPU for `f32` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn batched_matmat_gpu_f32(
    left_batches: &Array3<f32>,
    right_batches: &Array3<f32>,
) -> Result<Array3<f32>, AcceleratorError> {
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let (batch_count, rows, _) = left_batches.dim();
    let cols = right_batches.dim().2;
    let mut output = Array3::<f32>::zeros((batch_count, rows, cols));

    for batch in 0..batch_count {
        let left = left_batches.slice(s![batch, .., ..]);
        let right = right_batches.slice(s![batch, .., ..]);
        let product = matmat_gpu_f32_impl(&left, &right)?;
        output.slice_mut(s![batch, .., ..]).assign(&product);
    }

    Ok(output)
}

/// Compute batched matrix-matrix products on GPU for `f64` inputs using `wgpu`.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn batched_matmat_gpu_f64(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
) -> Result<Array3<f64>, AcceleratorError> {
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let (batch_count, rows, _) = left_batches.dim();
    let cols = right_batches.dim().2;
    let mut output = Array3::<f64>::zeros((batch_count, rows, cols));

    for batch in 0..batch_count {
        let left = left_batches.slice(s![batch, .., ..]);
        let right = right_batches.slice(s![batch, .., ..]);
        let product = matmat_gpu_f64_impl(&left, &right)?;
        output.slice_mut(s![batch, .., ..]).assign(&product);
    }

    Ok(output)
}

/// Compute row-batch by matrix products on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn batched_row_matvec_gpu_f32(
    batch_vectors: &Array2<f32>,
    matrix: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if batch_vectors.ncols() != matrix.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let matrix_t = matrix.t().to_owned();
    matmat_gpu_f32_impl(&batch_vectors.view(), &matrix_t.view())
}

/// Compute row-batch by matrix products on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn batched_row_matvec_gpu_f64(
    batch_vectors: &Array2<f64>,
    matrix: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if batch_vectors.ncols() != matrix.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let matrix_t = matrix.t().to_owned();
    matmat_gpu_f64_impl(&batch_vectors.view(), &matrix_t.view())
}

/// Compute vector dot product on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn dot_gpu_f32(left: &Array1<f32>, right: &Array1<f32>) -> Result<f32, AcceleratorError> {
    if left.len() != right.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let left_row = left.view().insert_axis(Axis(0)).to_owned();
    let right_col = right.view().insert_axis(Axis(1)).to_owned();
    let product = matmat_gpu_f32_impl(&left_row.view(), &right_col.view())?;
    Ok(product[[0, 0]])
}

/// Compute vector dot product on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn dot_gpu_f64(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
    if left.len() != right.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let left_row = left.view().insert_axis(Axis(0)).to_owned();
    let right_col = right.view().insert_axis(Axis(1)).to_owned();
    let product = matmat_gpu_f64_impl(&left_row.view(), &right_col.view())?;
    Ok(product[[0, 0]])
}

/// Compute pairwise L2 distance matrix on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn pairwise_l2_gpu_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let cross = matmat_gpu_f32_impl(&left.view(), &right.t())?;
    let left_norms = row_norms_squared(left);
    let right_norms = row_norms_squared(right);
    let mut output = Array2::<f32>::zeros((left.nrows(), right.nrows()));

    for left_row in 0..left.nrows() {
        for right_row in 0..right.nrows() {
            let sq_distance = (left_norms[left_row] + right_norms[right_row]
                - 2.0_f32 * cross[[left_row, right_row]])
            .max(0.0_f32);
            output[[left_row, right_row]] = sq_distance.sqrt();
        }
    }
    Ok(output)
}

/// Compute pairwise L2 distance matrix on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn pairwise_l2_gpu_f64(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let cross = matmat_gpu_f64_impl(&left.view(), &right.t())?;
    let left_norms = row_norms_squared(left);
    let right_norms = row_norms_squared(right);
    let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));

    for left_row in 0..left.nrows() {
        for right_row in 0..right.nrows() {
            let sq_distance = (left_norms[left_row] + right_norms[right_row]
                - 2.0_f64 * cross[[left_row, right_row]])
            .max(0.0_f64);
            output[[left_row, right_row]] = sq_distance.sqrt();
        }
    }
    Ok(output)
}

/// Compute pairwise cosine similarity matrix on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, zero-norm
/// rows, or `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn pairwise_cosine_gpu_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let cross = matmat_gpu_f32_impl(&left.view(), &right.t())?;
    let left_norms = row_norms_squared(left).mapv(f32::sqrt);
    let right_norms = row_norms_squared(right).mapv(f32::sqrt);

    if left_norms.iter().any(|norm| *norm <= f32::EPSILON)
        || right_norms.iter().any(|norm| *norm <= f32::EPSILON)
    {
        return Err(AcceleratorError::KernelExecutionFailed);
    }

    let mut output = Array2::<f32>::zeros((left.nrows(), right.nrows()));
    for left_row in 0..left.nrows() {
        for right_row in 0..right.nrows() {
            output[[left_row, right_row]] =
                cross[[left_row, right_row]] / (left_norms[left_row] * right_norms[right_row]);
        }
    }
    Ok(output)
}

/// Compute pairwise cosine similarity matrix on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, zero-norm rows, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn pairwise_cosine_gpu_f64(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let cross = matmat_gpu_f64_impl(&left.view(), &right.t())?;
    let left_norms = row_norms_squared(left).mapv(f64::sqrt);
    let right_norms = row_norms_squared(right).mapv(f64::sqrt);

    if left_norms.iter().any(|norm| *norm <= f64::EPSILON)
        || right_norms.iter().any(|norm| *norm <= f64::EPSILON)
    {
        return Err(AcceleratorError::KernelExecutionFailed);
    }

    let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    for left_row in 0..left.nrows() {
        for right_row in 0..right.nrows() {
            output[[left_row, right_row]] =
                cross[[left_row, right_row]] / (left_norms[left_row] * right_norms[right_row]);
        }
    }
    Ok(output)
}

/// Compute N-D batched matrix multiplication over the last two axes on GPU for `f32` using
/// per-batch `wgpu` matmul kernels.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn tensor_batched_matmul_last_two_gpu_f32(
    left: &ArrayD<f32>,
    right: &ArrayD<f32>,
) -> Result<ArrayD<f32>, AcceleratorError> {
    if left.ndim() < 2 || right.ndim() < 2 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let batch_ndim = left.ndim() - 2;
    if left.ndim() != right.ndim()
        || left.shape()[..batch_ndim] != right.shape()[..batch_ndim]
        || left.shape()[left.ndim() - 1] != right.shape()[right.ndim() - 2]
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let rows = left.shape()[left.ndim() - 2];
    let inner = left.shape()[left.ndim() - 1];
    let cols = right.shape()[right.ndim() - 1];
    let batch_count = shape_product(&left.shape()[..batch_ndim]);
    let left_standard = left.as_standard_layout().to_owned();
    let right_standard = right.as_standard_layout().to_owned();
    let left_3d = left_standard
        .view()
        .into_shape_with_order((batch_count, rows, inner))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let right_3d = right_standard
        .view()
        .into_shape_with_order((batch_count, inner, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    let mut output_shape = left.shape()[..batch_ndim].to_vec();
    output_shape.push(rows);
    output_shape.push(cols);
    let mut output = ArrayD::<f32>::zeros(IxDyn(&output_shape));
    let mut output_3d = output
        .view_mut()
        .into_shape_with_order((batch_count, rows, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    for batch in 0..batch_count {
        let left_batch = left_3d.slice(s![batch, .., ..]);
        let right_batch = right_3d.slice(s![batch, .., ..]);
        let product = matmat_gpu_f32_impl(&left_batch, &right_batch)?;
        output_3d.slice_mut(s![batch, .., ..]).assign(&product);
    }

    Ok(output)
}

/// Compute N-D batched matrix multiplication over the last two axes on GPU for `f64` using
/// per-batch `wgpu` matmul kernels.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn tensor_batched_matmul_last_two_gpu_f64(
    left: &ArrayD<f64>,
    right: &ArrayD<f64>,
) -> Result<ArrayD<f64>, AcceleratorError> {
    if left.ndim() < 2 || right.ndim() < 2 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let batch_ndim = left.ndim() - 2;
    if left.ndim() != right.ndim()
        || left.shape()[..batch_ndim] != right.shape()[..batch_ndim]
        || left.shape()[left.ndim() - 1] != right.shape()[right.ndim() - 2]
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let rows = left.shape()[left.ndim() - 2];
    let inner = left.shape()[left.ndim() - 1];
    let cols = right.shape()[right.ndim() - 1];
    let batch_count = shape_product(&left.shape()[..batch_ndim]);
    let left_standard = left.as_standard_layout().to_owned();
    let right_standard = right.as_standard_layout().to_owned();
    let left_3d = left_standard
        .view()
        .into_shape_with_order((batch_count, rows, inner))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let right_3d = right_standard
        .view()
        .into_shape_with_order((batch_count, inner, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    let mut output_shape = left.shape()[..batch_ndim].to_vec();
    output_shape.push(rows);
    output_shape.push(cols);
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    let mut output_3d = output
        .view_mut()
        .into_shape_with_order((batch_count, rows, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    for batch in 0..batch_count {
        let left_batch = left_3d.slice(s![batch, .., ..]);
        let right_batch = right_3d.slice(s![batch, .., ..]);
        let product = matmat_gpu_f64_impl(&left_batch, &right_batch)?;
        output_3d.slice_mut(s![batch, .., ..]).assign(&product);
    }

    Ok(output)
}

/// Compute tensor contraction over one axis on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn tensor_contract_axes_gpu_f32(
    left: &ArrayD<f32>,
    right: &ArrayD<f32>,
    left_axis: usize,
    right_axis: usize,
) -> Result<ArrayD<f32>, AcceleratorError> {
    if left.ndim() == 0 || right.ndim() == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }
    if left.shape()[left_axis] != right.shape()[right_axis] {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let left_free_axes = uncontracted_axes(left.ndim(), left_axis)?;
    let right_free_axes = uncontracted_axes(right.ndim(), right_axis)?;
    let contract_size = left.shape()[left_axis];
    let left_outer =
        shape_product(&left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>());
    let right_outer =
        shape_product(&right_free_axes.iter().map(|axis| right.shape()[*axis]).collect::<Vec<_>>());

    let mut left_order = left_free_axes.clone();
    left_order.push(left_axis);
    let mut right_order = vec![right_axis];
    right_order.extend(right_free_axes.iter().copied());

    let left_2d = left
        .view()
        .permuted_axes(left_order)
        .to_owned()
        .into_shape_with_order((left_outer, contract_size))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let right_2d = right
        .view()
        .permuted_axes(right_order)
        .to_owned()
        .into_shape_with_order((contract_size, right_outer))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let output_2d = matmat_gpu_f32_impl(&left_2d.view(), &right_2d.view())?;

    let mut output_shape =
        left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>();
    output_shape.extend(right_free_axes.iter().map(|axis| right.shape()[*axis]));
    output_2d
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| AcceleratorError::DimensionMismatch)
}

/// Compute tensor contraction over one axis on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn tensor_contract_axes_gpu_f64(
    left: &ArrayD<f64>,
    right: &ArrayD<f64>,
    left_axis: usize,
    right_axis: usize,
) -> Result<ArrayD<f64>, AcceleratorError> {
    if left.ndim() == 0 || right.ndim() == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }
    if left.shape()[left_axis] != right.shape()[right_axis] {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let left_free_axes = uncontracted_axes(left.ndim(), left_axis)?;
    let right_free_axes = uncontracted_axes(right.ndim(), right_axis)?;
    let contract_size = left.shape()[left_axis];
    let left_outer =
        shape_product(&left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>());
    let right_outer =
        shape_product(&right_free_axes.iter().map(|axis| right.shape()[*axis]).collect::<Vec<_>>());

    let mut left_order = left_free_axes.clone();
    left_order.push(left_axis);
    let mut right_order = vec![right_axis];
    right_order.extend(right_free_axes.iter().copied());

    let left_2d = left
        .view()
        .permuted_axes(left_order)
        .to_owned()
        .into_shape_with_order((left_outer, contract_size))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let right_2d = right
        .view()
        .permuted_axes(right_order)
        .to_owned()
        .into_shape_with_order((contract_size, right_outer))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let output_2d = matmat_gpu_f64_impl(&left_2d.view(), &right_2d.view())?;

    let mut output_shape =
        left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>();
    output_shape.extend(right_free_axes.iter().map(|axis| right.shape()[*axis]));
    output_2d
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| AcceleratorError::DimensionMismatch)
}

/// Compute tensor reduction over the last axis on GPU for `f32` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
pub fn tensor_sum_last_axis_gpu_f32(input: &ArrayD<f32>) -> Result<ArrayD<f32>, AcceleratorError> {
    if input.ndim() == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let Some(last_axis) = input.shape().last().copied() else {
        return Err(AcceleratorError::DimensionMismatch);
    };
    if last_axis == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let outer = input.len() / last_axis;
    let matrix = input
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((outer, last_axis))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let ones = Array2::<f32>::from_elem((last_axis, 1), 1.0_f32);
    let reduced = matmat_gpu_f32_impl(&matrix.view(), &ones.view())?;

    let mut output_shape = input.shape().to_vec();
    let _ = output_shape.pop();
    reduced
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| AcceleratorError::DimensionMismatch)
}

/// Compute tensor reduction over the last axis on GPU for `f64` inputs.
///
/// # Errors
/// Returns an error for incompatible dimensions, unavailable device, unsupported `f64` shader
/// support, kernel failures, or `AcceleratorError::FeatureNotEnabled` when
/// `accelerator-wgpu` is disabled.
pub fn tensor_sum_last_axis_gpu_f64(input: &ArrayD<f64>) -> Result<ArrayD<f64>, AcceleratorError> {
    if input.ndim() == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }
    let Some(last_axis) = input.shape().last().copied() else {
        return Err(AcceleratorError::DimensionMismatch);
    };
    if last_axis == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let outer = input.len() / last_axis;
    let matrix = input
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((outer, last_axis))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let ones = Array2::<f64>::from_elem((last_axis, 1), 1.0_f64);
    let reduced = matmat_gpu_f64_impl(&matrix.view(), &ones.view())?;

    let mut output_shape = input.shape().to_vec();
    let _ = output_shape.pop();
    reduced
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| AcceleratorError::DimensionMismatch)
}

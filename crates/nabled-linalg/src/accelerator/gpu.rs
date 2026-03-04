use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn, s};
#[cfg(feature = "accelerator-wgpu")]
use wgpu::util::DeviceExt;

use super::backends::AcceleratorError;

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
/// Returns an error for incompatible dimensions, unavailable device, kernel failures, or
/// `AcceleratorError::FeatureNotEnabled` when `accelerator-wgpu` is disabled.
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

    let right = Array2::from_shape_vec((vector.len(), 1), vector.to_vec())
        .map_err(|_| AcceleratorError::KernelExecutionFailed)?;
    let product = matmat_gpu_f32(matrix, &right)?;
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

    let (batch, rows, _) = left_batches.dim();
    let cols = right_batches.dim().2;
    let mut output = Array3::<f32>::zeros((batch, rows, cols));

    for b in 0..batch {
        let left = left_batches.slice(s![b, .., ..]).to_owned();
        let right = right_batches.slice(s![b, .., ..]).to_owned();
        let product = matmat_gpu_f32(&left, &right)?;
        output.slice_mut(s![b, .., ..]).assign(&product);
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
    let batches = left.shape()[..batch_ndim].iter().copied().product::<usize>().max(1);

    let left_standard = left.as_standard_layout().to_owned();
    let right_standard = right.as_standard_layout().to_owned();
    let left_3d = left_standard
        .view()
        .into_shape_with_order((batches, rows, inner))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;
    let right_3d = right_standard
        .view()
        .into_shape_with_order((batches, inner, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    let mut output_shape = left.shape()[..batch_ndim].to_vec();
    output_shape.push(rows);
    output_shape.push(cols);
    let mut output = ArrayD::<f32>::zeros(IxDyn(&output_shape));
    let mut output_3d = output
        .view_mut()
        .into_shape_with_order((batches, rows, cols))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    for batch in 0..batches {
        let left_batch = left_3d.slice(s![batch, .., ..]).to_owned();
        let right_batch = right_3d.slice(s![batch, .., ..]).to_owned();
        let product = matmat_gpu_f32(&left_batch, &right_batch)?;
        output_3d.slice_mut(s![batch, .., ..]).assign(&product);
    }

    Ok(output)
}

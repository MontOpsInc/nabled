//! Heavy GPU burn/profile probe for large dense and complex tensor workloads.
//!
//! Run with:
//! `WGPU_BACKEND=vulkan cargo +stable test -p nabled --test gpu_burn_profile --no-default-features
//! --features accelerator-wgpu -- --ignored --nocapture`

#![cfg(all(feature = "accelerator-wgpu", feature = "linalg"))]

use std::time::Instant;

use nabled::linalg::accelerator::backends::{AcceleratorError, CpuBackend};
use nabled::linalg::accelerator::dispatch::tensor_batched_matmul_last_two_with_backend;
use nabled::linalg::accelerator::gpu::{
    matmat_gpu_f32, tensor_batched_matmul_last_two_gpu_complex64,
};
use ndarray::{Array2, ArrayD, IxDyn};
use num_complex::Complex64;

const MATMUL_SIZE: usize = 1536;
const MATMUL_ITERS: usize = 16;
const TENSOR_BATCH: usize = 8;
const TENSOR_M: usize = 192;
const TENSOR_K: usize = 192;
const TENSOR_N: usize = 192;
const TENSOR_ITERS: usize = 8;

fn read_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn usize_to_f64(value: usize) -> f64 {
    let narrowed = u32::try_from(value).expect("probe dimension should fit into u32");
    f64::from(narrowed)
}

fn deterministic_matrix(rows: usize, cols: usize, seed: f32) -> Array2<f32> {
    let mut value = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        value = (value * std::f32::consts::SQRT_2 + 0.271_828_f32).fract();
        data.push(value - 0.5_f32);
    }
    Array2::from_shape_vec((rows, cols), data).expect("matrix shape should match")
}

fn deterministic_complex_tensor(shape: &[usize], seed: f64) -> ArrayD<Complex64> {
    let mut value = seed;
    let total = shape.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        value = (value * 1.324_717_957_2_f64 + 0.141_421_356_2_f64).fract();
        let real = value - 0.5_f64;
        value = (value * 1.220_744_084_6_f64 + 0.577_350_269_1_f64).fract();
        let imag = value - 0.5_f64;
        data.push(Complex64::new(real, imag));
    }
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("tensor shape should match")
}

fn max_abs_diff_f32(left: &Array2<f32>, right: &Array2<f32>) -> f32 {
    left.iter().zip(right.iter()).map(|(lhs, rhs)| (lhs - rhs).abs()).fold(0.0_f32, f32::max)
}

fn max_abs_diff_complex(left: &ArrayD<Complex64>, right: &ArrayD<Complex64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "manual heavy GPU burn/profile probe"]
fn gpu_burn_dense_and_complex_tensor() {
    let n = read_env_usize("NABLED_GPU_BURN_MATMUL_SIZE", MATMUL_SIZE);
    let matmul_iters = read_env_usize("NABLED_GPU_BURN_MATMUL_ITERS", MATMUL_ITERS);
    let batch = read_env_usize("NABLED_GPU_BURN_TENSOR_BATCH", TENSOR_BATCH);
    let m = read_env_usize("NABLED_GPU_BURN_TENSOR_M", TENSOR_M);
    let k = read_env_usize("NABLED_GPU_BURN_TENSOR_K", TENSOR_K);
    let p = read_env_usize("NABLED_GPU_BURN_TENSOR_N", TENSOR_N);
    let tensor_iters = read_env_usize("NABLED_GPU_BURN_TENSOR_ITERS", TENSOR_ITERS);

    eprintln!(
        "gpu burn probe starting: dense={n}x{n} iters={matmul_iters}, complex_batch={batch} \
         [{m}x{k}]x[{k}x{p}] iters={tensor_iters}"
    );

    let left = deterministic_matrix(n, n, 0.123_456_f32);
    let right = deterministic_matrix(n, n, 0.654_321_f32);

    let warmup = match matmat_gpu_f32(&left, &right) {
        Ok(value) => value,
        Err(AcceleratorError::DeviceUnavailable) => {
            eprintln!("gpu burn probe skipped: no usable GPU device available");
            return;
        }
        Err(other) => panic!("gpu dense warmup failed: {other:?}"),
    };

    let cpu_dense = left.dot(&right);
    let warmup_diff = max_abs_diff_f32(&cpu_dense, &warmup);
    eprintln!("dense warmup diff(cpu,gpu)={warmup_diff:.6e}");
    assert!(warmup_diff < 2.5e-3_f32);

    let dense_start = Instant::now();
    let mut dense_last = warmup;
    for _ in 0..matmul_iters {
        dense_last = matmat_gpu_f32(&left, &right).expect("gpu dense matmul should succeed");
    }
    let dense_elapsed = dense_start.elapsed();
    let dense_diff = max_abs_diff_f32(&cpu_dense, &dense_last);
    let dense_seconds = dense_elapsed.as_secs_f64();
    let dense_operation_flops = 2.0_f64 * usize_to_f64(n).powi(3) * usize_to_f64(matmul_iters);
    let dense_throughput_gflops = if dense_seconds > 0.0 {
        dense_operation_flops / dense_seconds / 1.0e9
    } else {
        f64::INFINITY
    };
    eprintln!(
        "dense done: elapsed={dense_seconds:.3}s, approx_gflops={dense_throughput_gflops:.2}, \
         diff(cpu,last_gpu)={dense_diff:.6e}"
    );
    assert!(dense_diff < 2.5e-3_f32);

    let left_complex = deterministic_complex_tensor(&[batch, m, k], 0.333_333_f64);
    let right_complex = deterministic_complex_tensor(&[batch, k, p], 0.777_777_f64);

    let cpu_complex = tensor_batched_matmul_last_two_with_backend::<CpuBackend, Complex64>(
        &left_complex,
        &right_complex,
    )
    .expect("cpu complex tensor matmul should succeed");

    let warmup_complex =
        tensor_batched_matmul_last_two_gpu_complex64(&left_complex, &right_complex)
            .expect("gpu complex tensor matmul warmup should succeed");
    let warmup_complex_diff = max_abs_diff_complex(&cpu_complex, &warmup_complex);
    eprintln!("complex warmup diff(cpu,gpu)={warmup_complex_diff:.6e}");
    assert!(warmup_complex_diff < 2.5e-6_f64);

    let complex_start = Instant::now();
    let mut complex_last = warmup_complex;
    for _ in 0..tensor_iters {
        complex_last = tensor_batched_matmul_last_two_gpu_complex64(&left_complex, &right_complex)
            .expect("gpu complex tensor matmul should succeed");
    }
    let complex_elapsed = complex_start.elapsed();
    let complex_diff = max_abs_diff_complex(&cpu_complex, &complex_last);
    eprintln!(
        "complex done: elapsed={:.3}s, diff(cpu,last_gpu)={:.6e}",
        complex_elapsed.as_secs_f64(),
        complex_diff
    );
    assert!(complex_diff < 2.5e-6_f64);
}

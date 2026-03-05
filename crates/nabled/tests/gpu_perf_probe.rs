//! Manual GPU performance probe over one heavy matrix-multiply workload.
//!
//! Run with:
//! `WGPU_BACKEND=metal cargo +stable test -p nabled --test gpu_perf_probe --no-default-features
//! --features accelerator-wgpu -- --ignored --nocapture`

#![cfg(feature = "accelerator-wgpu")]

use std::time::{Duration, Instant};

use nabled::linalg::accelerator::backends::{AcceleratorError, CpuBackend, GpuBackend};
use nabled::linalg::accelerator::dispatch::matmat_with_backend;
use nabled::linalg::accelerator::gpu::matmat_gpu_f32;
use ndarray::Array2;

const MATRIX_SIZE: usize = 768;
const TIMED_ITERS: usize = 8;

fn read_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn deterministic_matrix(rows: usize, cols: usize, seed: f32) -> Array2<f32> {
    let mut value = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        value = (value * 1.618_034_f32 + 0.271_828_f32).fract();
        data.push(value - 0.5_f32);
    }
    Array2::from_shape_vec((rows, cols), data).expect("deterministic matrix shape should match")
}

fn max_abs_diff(left: &Array2<f32>, right: &Array2<f32>) -> f32 {
    left.iter().zip(right.iter()).map(|(lhs, rhs)| (lhs - rhs).abs()).fold(0.0_f32, f32::max)
}

fn run_cpu_timed(
    left: &Array2<f32>,
    right: &Array2<f32>,
    timed_iters: usize,
) -> (Array2<f32>, Duration) {
    let start = Instant::now();
    let mut last = Array2::<f32>::zeros((left.nrows(), right.ncols()));
    for _ in 0..timed_iters {
        last = matmat_with_backend::<CpuBackend, f32>(left, right)
            .expect("cpu backend matmat should succeed");
    }
    (last, start.elapsed())
}

fn run_gpu_timed(
    left: &Array2<f32>,
    right: &Array2<f32>,
    timed_iters: usize,
) -> (Array2<f32>, Duration) {
    let start = Instant::now();
    let mut last = Array2::<f32>::zeros((left.nrows(), right.ncols()));
    for _ in 0..timed_iters {
        // Direct GPU call avoids backend fallback and verifies actual GPU path behavior.
        last = matmat_gpu_f32(left, right).expect(
            "direct wgpu matmat failed; ensure a GPU device is available and accelerator-wgpu is \
             enabled",
        );
    }
    (last, start.elapsed())
}

#[test]
#[ignore = "manual performance probe; intentionally heavy and requires local GPU"]
fn gpu_vs_cpu_matmat_probe() {
    let matrix_size = read_env_usize("NABLED_GPU_PROBE_SIZE", MATRIX_SIZE);
    let timed_iters = read_env_usize("NABLED_GPU_PROBE_ITERS", TIMED_ITERS);

    let left = deterministic_matrix(matrix_size, matrix_size, 0.123_456_f32);
    let right = deterministic_matrix(matrix_size, matrix_size, 0.654_321_f32);

    eprintln!("gpu perf probe starting: size={matrix_size}x{matrix_size}, iters={timed_iters}");

    let cpu_baseline = matmat_with_backend::<CpuBackend, f32>(&left, &right)
        .expect("cpu baseline matmat should succeed");
    let gpu_warmup = match matmat_gpu_f32(&left, &right) {
        Ok(output) => output,
        Err(AcceleratorError::DeviceUnavailable) => {
            eprintln!("gpu perf probe skipped: no usable GPU device available in this environment");
            return;
        }
        Err(other) => {
            panic!("gpu warmup failed; verify local GPU and accelerator-wgpu feature: {other:?}")
        }
    };

    let warmup_diff = max_abs_diff(&cpu_baseline, &gpu_warmup);
    eprintln!("warmup max_abs_diff(cpu,gpu)={warmup_diff:.6e}");

    let (cpu_last, cpu_elapsed) = run_cpu_timed(&left, &right, timed_iters);
    let (gpu_last, gpu_elapsed) = run_gpu_timed(&left, &right, timed_iters);

    let cpu_gpu_diff = max_abs_diff(&cpu_last, &gpu_last);
    let gpu_dispatch = matmat_with_backend::<GpuBackend, f32>(&left, &right)
        .expect("gpu backend dispatch matmat should succeed");
    let gpu_dispatch_diff = max_abs_diff(&gpu_last, &gpu_dispatch);

    let cpu_secs = cpu_elapsed.as_secs_f64();
    let gpu_secs = gpu_elapsed.as_secs_f64();
    let speedup = if gpu_secs > 0.0 { cpu_secs / gpu_secs } else { f64::INFINITY };

    eprintln!(
        "cpu total={cpu_secs:.3}s, gpu total={gpu_secs:.3}s, speedup={speedup:.3}x, \
         diff(cpu,gpu)={cpu_gpu_diff:.6e}, diff(gpu,dsp)={gpu_dispatch_diff:.6e}"
    );

    assert!(cpu_gpu_diff < 1.0e-3_f32, "cpu vs gpu max abs diff too large: {cpu_gpu_diff}");
    assert!(
        gpu_dispatch_diff < 1.0e-6_f32,
        "direct gpu vs gpu-backend-dispatch max abs diff too large: {gpu_dispatch_diff}"
    );
}

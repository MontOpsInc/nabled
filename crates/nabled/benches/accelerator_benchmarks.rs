use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::linalg::accelerator;
#[cfg(feature = "accelerator-wgpu")]
use nabled::linalg::accelerator::backends::{CpuBackend, GpuBackend};
#[cfg(feature = "accelerator-wgpu")]
use nabled::linalg::accelerator::dispatch::{
    batched_matmat_with_backend, matmat_with_backend, matvec_with_backend, pairwise_l2_with_backend,
};
use ndarray::Array2;
#[cfg(feature = "accelerator-wgpu")]
use ndarray::{Array1, Array3};
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

#[cfg(feature = "accelerator-wgpu")]
fn random_matrix_f32(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0_f32..1.0_f32)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

#[cfg(feature = "accelerator-wgpu")]
fn random_vector_f32(size: usize) -> Array1<f32> {
    let mut rng = rand::rng();
    let values = (0..size).map(|_| rng.random_range(-1.0_f32..1.0_f32)).collect::<Vec<_>>();
    Array1::from_vec(values)
}

#[cfg(feature = "accelerator-wgpu")]
fn random_batches_f32(batch: usize, rows: usize, cols: usize) -> Array3<f32> {
    let mut rng = rand::rng();
    let values =
        (0..batch * rows * cols).map(|_| rng.random_range(-1.0_f32..1.0_f32)).collect::<Vec<_>>();
    Array3::from_shape_vec((batch, rows, cols), values).expect("shape should match")
}

fn manual_matmat(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.ncols()));
    for row in 0..left.nrows() {
        for inner in 0..left.ncols() {
            let lhs = left[[row, inner]];
            for col in 0..right.ncols() {
                output[[row, col]] += lhs * right[[inner, col]];
            }
        }
    }
    output
}

fn benchmark_cpu_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerator_nabled_ndarray");
    for size in [64_usize, 128, 192] {
        let left = random_matrix(size, size);
        let right = random_matrix(size, size);
        let id = format!("square-{size}x{size}");
        _ = group.bench_with_input(BenchmarkId::new("matmat_serial", &id), &size, |bench, _| {
            bench.iter(|| accelerator::cpu::matmat_serial(black_box(&left), black_box(&right)));
        });

        #[cfg(feature = "accelerator-rayon")]
        {
            _ = group.bench_with_input(
                BenchmarkId::new("matmat_accelerated", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        accelerator::cpu::matmat_accelerated(black_box(&left), black_box(&right))
                    });
                },
            );
        }
    }
    group.finish();
}

fn benchmark_manual_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerator_competitor_manual");
    for size in [64_usize, 128, 192] {
        let left = random_matrix(size, size);
        let right = random_matrix(size, size);
        let id = format!("square-{size}x{size}");
        _ = group.bench_with_input(BenchmarkId::new("matmat_manual", &id), &size, |bench, _| {
            bench.iter(|| {
                drop(black_box(manual_matmat(black_box(&left), black_box(&right))));
            });
        });
    }
    group.finish();
}

#[cfg(feature = "accelerator-wgpu")]
fn benchmark_gpu_cpu_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerator_nabled_gpu_cpu_f32");
    for size in [64_usize, 128, 192] {
        let left = random_matrix_f32(size, size);
        let right = random_matrix_f32(size, size);
        let vector = random_vector_f32(size);
        let batch_left = random_batches_f32(4, size, size);
        let batch_right = random_batches_f32(4, size, size);
        let id = format!("square-{size}x{size}");

        _ = group.bench_with_input(BenchmarkId::new("matmat_cpu_f32", &id), &size, |bench, _| {
            bench.iter(|| {
                let result =
                    matmat_with_backend::<CpuBackend, f32>(black_box(&left), black_box(&right))
                        .expect("cpu matmat should succeed");
                drop(black_box(result));
            });
        });
        _ = group.bench_with_input(BenchmarkId::new("matvec_cpu_f32", &id), &size, |bench, _| {
            bench.iter(|| {
                let result =
                    matvec_with_backend::<CpuBackend, f32>(black_box(&left), black_box(&vector))
                        .expect("cpu matvec should succeed");
                drop(black_box(result));
            });
        });
        _ = group.bench_with_input(
            BenchmarkId::new("pairwise_l2_cpu_f32", &id),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = pairwise_l2_with_backend::<CpuBackend, f32>(
                        black_box(&left),
                        black_box(&right),
                    )
                    .expect("cpu pairwise_l2 should succeed");
                    drop(black_box(result));
                });
            },
        );
        _ = group.bench_with_input(
            BenchmarkId::new("batched_matmat_cpu_f32", &id),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = batched_matmat_with_backend::<CpuBackend, f32>(
                        black_box(&batch_left),
                        black_box(&batch_right),
                    )
                    .expect("cpu batched_matmat should succeed");
                    drop(black_box(result));
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "accelerator-wgpu")]
fn benchmark_gpu_wgpu_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerator_nabled_gpu_wgpu_f32");
    for size in [64_usize, 128, 192] {
        let left = random_matrix_f32(size, size);
        let right = random_matrix_f32(size, size);
        let vector = random_vector_f32(size);
        let batch_left = random_batches_f32(4, size, size);
        let batch_right = random_batches_f32(4, size, size);
        let id = format!("square-{size}x{size}");

        _ = group.bench_with_input(BenchmarkId::new("matmat_gpu_f32", &id), &size, |bench, _| {
            bench.iter(|| {
                let result =
                    matmat_with_backend::<GpuBackend, f32>(black_box(&left), black_box(&right))
                        .expect("gpu matmat should succeed");
                drop(black_box(result));
            });
        });
        _ = group.bench_with_input(BenchmarkId::new("matvec_gpu_f32", &id), &size, |bench, _| {
            bench.iter(|| {
                let result =
                    matvec_with_backend::<GpuBackend, f32>(black_box(&left), black_box(&vector))
                        .expect("gpu matvec should succeed");
                drop(black_box(result));
            });
        });
        _ = group.bench_with_input(
            BenchmarkId::new("pairwise_l2_gpu_f32", &id),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = pairwise_l2_with_backend::<GpuBackend, f32>(
                        black_box(&left),
                        black_box(&right),
                    )
                    .expect("gpu pairwise_l2 should succeed");
                    drop(black_box(result));
                });
            },
        );
        _ = group.bench_with_input(
            BenchmarkId::new("batched_matmat_gpu_f32", &id),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let result = batched_matmat_with_backend::<GpuBackend, f32>(
                        black_box(&batch_left),
                        black_box(&batch_right),
                    )
                    .expect("gpu batched_matmat should succeed");
                    drop(black_box(result));
                });
            },
        );
    }
    group.finish();
}

fn benchmark_accelerator(c: &mut Criterion) {
    benchmark_cpu_group(c);
    benchmark_manual_group(c);

    #[cfg(feature = "accelerator-wgpu")]
    {
        benchmark_gpu_cpu_group(c);
        benchmark_gpu_wgpu_group(c);
    }
}

criterion_group!(benches, benchmark_accelerator);
criterion_main!(benches);

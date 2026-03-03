use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::linalg::matrix;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn random_vector(size: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let values = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array1::from_vec(values)
}

fn random_batched_matrices(batch: usize, rows: usize, cols: usize) -> Array3<f64> {
    let mut rng = rand::rng();
    let values = (0..batch * rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array3::from_shape_vec((batch, rows, cols), values).expect("shape should match")
}

fn benchmark_matrix(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("matrix_nabled_ndarray");
        for size in [128_usize, 256, 512] {
            let matrix_value = random_matrix(size, size);
            let rhs_vector = random_vector(size);
            let rhs_matrix = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(BenchmarkId::new("matvec", &id), &size, |bench, _| {
                bench.iter(|| matrix::matvec(black_box(&matrix_value), black_box(&rhs_vector)));
            });

            _ = group.bench_with_input(BenchmarkId::new("matmat", &id), &size, |bench, _| {
                bench.iter(|| matrix::matmat(black_box(&matrix_value), black_box(&rhs_matrix)));
            });
        }

        for size in [64_usize, 128] {
            let left = random_batched_matrices(16, size, size);
            let right = random_batched_matrices(16, size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(
                BenchmarkId::new("batched_matmat", &id),
                &size,
                |bench, _| {
                    bench.iter(|| matrix::batched_matmat(black_box(&left), black_box(&right)));
                },
            );
        }
        group.finish();
    }

    {
        let mut competitor_group = c.benchmark_group("matrix_competitor_ndarray");
        for size in [128_usize, 256, 512] {
            let matrix_value = random_matrix(size, size);
            let rhs_vector = random_vector(size);
            let rhs_matrix = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("matvec_dot", &id),
                &size,
                |bench, _| {
                    bench.iter(|| matrix_value.dot(black_box(&rhs_vector)));
                },
            );

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("matmat_dot", &id),
                &size,
                |bench, _| {
                    bench.iter(|| matrix_value.dot(black_box(&rhs_matrix)));
                },
            );
        }

        for size in [64_usize, 128] {
            let left = random_batched_matrices(16, size, size);
            let right = random_batched_matrices(16, size, size);
            let id = format!("square-{size}x{size}");

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("batched_matmat_naive", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let products = left
                            .axis_iter(Axis(0))
                            .zip(right.axis_iter(Axis(0)))
                            .map(|(left_matrix, right_matrix)| left_matrix.dot(&right_matrix))
                            .collect::<Vec<_>>();
                        drop(black_box(products));
                    });
                },
            );
        }
        competitor_group.finish();
    }
}

criterion_group!(benches, benchmark_matrix);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::Mat;
use nabled::linalg::svd;
use ndarray::Array2;
use num_complex::Complex64;
use rand::RngExt;

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn generate_random_complex_matrix(rows: usize, cols: usize) -> Array2<Complex64> {
    let mut rng = rand::rng();
    let data: Vec<Complex64> = (0..rows * cols)
        .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn ndarray_to_faer(matrix: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| matrix[[i, j]])
}

fn benchmark_ndarray_svd(c: &mut Criterion) {
    let sizes = [32_usize, 64, 96];

    {
        let mut group = c.benchmark_group("svd_nabled_ndarray");
        for size in sizes {
            let matrix = generate_random_matrix(size, size);

            _ = group.bench_with_input(BenchmarkId::new("full_svd", size), &size, |b, _| {
                b.iter(|| svd::decompose(black_box(&matrix)));
            });

            _ = group.bench_with_input(BenchmarkId::new("truncated_svd", size), &size, |b, _| {
                b.iter(|| svd::decompose_truncated(black_box(&matrix), black_box(size / 2)));
            });
        }

        for size in [16_usize, 32] {
            let matrix = generate_random_complex_matrix(size, size);
            _ = group.bench_with_input(
                BenchmarkId::new("full_svd_complex", size),
                &size,
                |b, _| {
                    b.iter(|| svd::decompose_complex(black_box(&matrix)));
                },
            );
        }

        group.finish();
    }

    {
        let mut competitor = c.benchmark_group("svd_competitor_faer_direct");
        for size in sizes {
            let matrix = generate_random_matrix(size, size);
            let matrix_faer = ndarray_to_faer(&matrix);

            _ = competitor.bench_with_input(BenchmarkId::new("full_svd", size), &size, |b, _| {
                b.iter(|| matrix_faer.as_ref().svd());
            });

            _ = competitor.bench_with_input(
                BenchmarkId::new("truncated_svd", size),
                &size,
                |b, _| {
                    b.iter(|| matrix_faer.as_ref().thin_svd());
                },
            );
        }
        competitor.finish();
    }
}

criterion_group!(benches, benchmark_ndarray_svd);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::qr::{QRConfig, ndarray_qr};
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn generate_random_vector(len: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array1::from_vec(data)
}

fn benchmark_ndarray_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_nabled_ndarray");
    let config = QRConfig::default();
    let sizes = [32_usize, 64, 96];

    for size in sizes {
        let matrix = generate_random_matrix(size, size);
        let rhs = generate_random_vector(size);

        _ = group.bench_with_input(BenchmarkId::new("qr", size), &size, |b, _| {
            b.iter(|| ndarray_qr::compute_qr(black_box(&matrix), black_box(&config)));
        });

        _ = group.bench_with_input(BenchmarkId::new("least_squares", size), &size, |b, _| {
            b.iter(|| {
                ndarray_qr::solve_least_squares(
                    black_box(&matrix),
                    black_box(&rhs),
                    black_box(&config),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_qr);
criterion_main!(benches);

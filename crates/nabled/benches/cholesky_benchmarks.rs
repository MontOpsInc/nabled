use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::cholesky::ndarray_cholesky;
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_spd_matrix(size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..size * size).map(|_| rng.random_range(-1.0..1.0)).collect();
    let base = Array2::from_shape_vec((size, size), data).expect("shape should match data length");
    let mut spd = base.t().dot(&base);
    for i in 0..size {
        spd[[i, i]] += 1.0;
    }
    spd
}

fn generate_random_vector(size: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array1::from_vec(data)
}

fn benchmark_ndarray_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_nabled_ndarray");
    let sizes = [16_usize, 32, 64];

    for size in sizes {
        let matrix = generate_spd_matrix(size);
        let rhs = generate_random_vector(size);

        _ = group.bench_with_input(BenchmarkId::new("decompose", size), &size, |b, _| {
            b.iter(|| ndarray_cholesky::decompose(black_box(&matrix)));
        });

        _ = group.bench_with_input(BenchmarkId::new("solve", size), &size, |b, _| {
            b.iter(|| ndarray_cholesky::solve(black_box(&matrix), black_box(&rhs)));
        });

        _ = group.bench_with_input(BenchmarkId::new("inverse", size), &size, |b, _| {
            b.iter(|| ndarray_cholesky::inverse(black_box(&matrix)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_cholesky);
criterion_main!(benches);

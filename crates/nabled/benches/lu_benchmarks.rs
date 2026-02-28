use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::lu::ndarray_lu;
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_well_conditioned_matrix(size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..size * size).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut matrix =
        Array2::from_shape_vec((size, size), data).expect("shape should match data length");
    let diagonal_bias = u32::try_from(size).map_or(f64::from(u32::MAX), f64::from);
    for i in 0..size {
        matrix[[i, i]] += diagonal_bias;
    }
    matrix
}

fn generate_random_vector(size: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array1::from_vec(data)
}

fn benchmark_ndarray_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_nabled_ndarray");
    let sizes = [32_usize, 64, 96];

    for size in sizes {
        let matrix = generate_well_conditioned_matrix(size);
        let rhs = generate_random_vector(size);

        _ = group.bench_with_input(BenchmarkId::new("decompose", size), &size, |b, _| {
            b.iter(|| ndarray_lu::decompose(black_box(&matrix)));
        });

        _ = group.bench_with_input(BenchmarkId::new("solve", size), &size, |b, _| {
            b.iter(|| ndarray_lu::solve(black_box(&matrix), black_box(&rhs)));
        });

        _ = group.bench_with_input(BenchmarkId::new("determinant", size), &size, |b, _| {
            b.iter(|| ndarray_lu::determinant(black_box(&matrix)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_lu);
criterion_main!(benches);

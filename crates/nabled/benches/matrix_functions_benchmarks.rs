use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::matrix_functions;
use ndarray::Array2;
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

fn benchmark_ndarray_matrix_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_functions_nabled_ndarray");
    let sizes = [16_usize, 24, 32];

    for size in sizes {
        let matrix = generate_spd_matrix(size);

        _ = group.bench_with_input(BenchmarkId::new("matrix_exp_eigen", size), &size, |b, _| {
            b.iter(|| matrix_functions::matrix_exp_eigen(black_box(&matrix)));
        });

        _ = group.bench_with_input(BenchmarkId::new("matrix_log_eigen", size), &size, |b, _| {
            b.iter(|| matrix_functions::matrix_log_eigen(black_box(&matrix)));
        });

        _ = group.bench_with_input(BenchmarkId::new("matrix_power_half", size), &size, |b, _| {
            b.iter(|| matrix_functions::matrix_power(black_box(&matrix), black_box(0.5)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_matrix_functions);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::eigen;
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

fn benchmark_ndarray_eigen(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigen_nabled_ndarray");
    let sizes = [16_usize, 32, 48];

    for size in sizes {
        let a = generate_spd_matrix(size);
        let b = generate_spd_matrix(size);

        _ = group.bench_with_input(BenchmarkId::new("symmetric", size), &size, |bench, _| {
            bench.iter(|| eigen::symmetric(black_box(&a)));
        });

        _ = group.bench_with_input(BenchmarkId::new("generalized", size), &size, |bench, _| {
            bench.iter(|| eigen::generalized(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_eigen);
criterion_main!(benches);

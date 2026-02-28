use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::svd::ndarray_svd;
use ndarray::Array2;
use rand::RngExt;

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn benchmark_ndarray_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_nabled_ndarray");
    let sizes = [32_usize, 64, 96];

    for size in sizes {
        let matrix = generate_random_matrix(size, size);

        _ = group.bench_with_input(BenchmarkId::new("full_svd", size), &size, |b, _| {
            b.iter(|| ndarray_svd::compute_svd(black_box(&matrix)));
        });

        _ = group.bench_with_input(BenchmarkId::new("truncated_svd", size), &size, |b, _| {
            b.iter(|| ndarray_svd::compute_truncated_svd(black_box(&matrix), black_box(size / 2)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_svd);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::schur;
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..size * size).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((size, size), values).expect("shape should match")
}

fn benchmark_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("schur_nabled_ndarray");

    for size in [16_usize, 32, 48] {
        let matrix = random_matrix(size);
        let id = format!("square-{size}x{size}");
        _ = group.bench_with_input(BenchmarkId::new("compute_schur", &id), &size, |bench, _| {
            bench.iter(|| schur::compute_schur(black_box(&matrix)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_schur);
criterion_main!(benches);

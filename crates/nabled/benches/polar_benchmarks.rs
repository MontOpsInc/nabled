use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::polar;
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn benchmark_polar(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_nabled_ndarray");

    for size in [16_usize, 32, 48] {
        let matrix = random_matrix(size, size);
        let id = format!("square-{size}x{size}");
        _ = group.bench_with_input(BenchmarkId::new("compute_polar", &id), &size, |bench, _| {
            bench.iter(|| polar::compute_polar(black_box(&matrix)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_polar);
criterion_main!(benches);

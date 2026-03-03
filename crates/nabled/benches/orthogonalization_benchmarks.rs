use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::linalg::orthogonalization;
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn benchmark_orthogonalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonalization_nabled_ndarray");

    for (rows, cols) in [(64_usize, 32_usize), (128, 64), (192, 96)] {
        let matrix = random_matrix(rows, cols);
        let id = format!("tall-{rows}x{cols}");

        _ = group.bench_with_input(BenchmarkId::new("gram_schmidt", &id), &rows, |bench, _| {
            bench.iter(|| orthogonalization::gram_schmidt(black_box(&matrix)));
        });

        _ = group.bench_with_input(
            BenchmarkId::new("gram_schmidt_classic", &id),
            &rows,
            |bench, _| {
                bench.iter(|| orthogonalization::gram_schmidt_classic(black_box(&matrix)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_orthogonalization);
criterion_main!(benches);

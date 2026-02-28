use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::sylvester::ndarray_sylvester;
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn make_stable_square(size: usize) -> Array2<f64> {
    let mut matrix = random_matrix(size, size);
    for i in 0..size {
        matrix[[i, i]] += 5.0;
    }
    matrix
}

fn benchmark_sylvester(c: &mut Criterion) {
    let mut group = c.benchmark_group("sylvester_nabled_ndarray");

    for size in [8_usize, 16, 24] {
        let matrix_a = make_stable_square(size);
        let matrix_b = make_stable_square(size);
        let matrix_c = random_matrix(size, size);
        let id = format!("square-{size}x{size}");

        _ = group.bench_with_input(BenchmarkId::new("solve_sylvester", &id), &size, |bench, _| {
            bench.iter(|| {
                ndarray_sylvester::solve_sylvester(
                    black_box(&matrix_a),
                    black_box(&matrix_b),
                    black_box(&matrix_c),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_sylvester);
criterion_main!(benches);

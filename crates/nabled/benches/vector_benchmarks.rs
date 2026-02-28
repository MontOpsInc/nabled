use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::vector::{PairwiseCosineWorkspace, ndarray_vector};
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_random_vector(size: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array1::from_vec(data)
}

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn benchmark_ndarray_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_nabled_ndarray");

    for size in [128_usize, 256, 512] {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);

        _ = group.bench_with_input(
            BenchmarkId::new("cosine_similarity", size),
            &size,
            |bench, _| {
                bench.iter(|| ndarray_vector::cosine_similarity(black_box(&a), black_box(&b)));
            },
        );

        _ = group.bench_with_input(BenchmarkId::new("dot", size), &size, |bench, _| {
            bench.iter(|| ndarray_vector::dot(black_box(&a), black_box(&b)));
        });
    }

    for rows in [32_usize, 64] {
        let cols = 128_usize;
        let left = generate_random_matrix(rows, cols);
        let right = generate_random_matrix(rows, cols);
        let mut l2_out = Array2::<f64>::zeros((rows, rows));
        let mut cosine_out = Array2::<f64>::zeros((rows, rows));
        let mut workspace = PairwiseCosineWorkspace::default();
        let id = format!("square-{rows}x{rows}");

        _ = group.bench_with_input(BenchmarkId::new("pairwise_l2_into", &id), &rows, |bench, _| {
            bench.iter(|| {
                ndarray_vector::pairwise_l2_distance_into(
                    black_box(&left),
                    black_box(&right),
                    black_box(&mut l2_out),
                )
            });
        });

        _ = group.bench_with_input(
            BenchmarkId::new("pairwise_cosine_ws_into", &id),
            &rows,
            |bench, _| {
                bench.iter(|| {
                    ndarray_vector::pairwise_cosine_similarity_with_workspace_into(
                        black_box(&left),
                        black_box(&right),
                        black_box(&mut cosine_out),
                        black_box(&mut workspace),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_vector);
criterion_main!(benches);

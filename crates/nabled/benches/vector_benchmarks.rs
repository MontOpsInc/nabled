use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::vector::{self as vector, PairwiseCosineWorkspace};
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

fn cosine_similarity_ndarray_baseline(a: &Array1<f64>, b: &Array1<f64>) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    let denom = norm_a * norm_b;
    if denom <= f64::EPSILON {
        return None;
    }
    Some(dot / denom)
}

fn pairwise_l2_naive(left: &Array2<f64>, right: &Array2<f64>, output: &mut Array2<f64>) {
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut sum = 0.0_f64;
            for k in 0..left.ncols() {
                let delta = left[[i, k]] - right[[j, k]];
                sum += delta * delta;
            }
            output[[i, j]] = sum.sqrt();
        }
    }
}

fn bench_nabled_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_nabled_ndarray");
    for size in [128_usize, 256, 512] {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);

        _ = group.bench_with_input(
            BenchmarkId::new("cosine_similarity", size),
            &size,
            |bench, _| {
                bench.iter(|| vector::cosine_similarity(black_box(&a), black_box(&b)));
            },
        );

        _ = group.bench_with_input(BenchmarkId::new("dot", size), &size, |bench, _| {
            bench.iter(|| vector::dot(black_box(&a), black_box(&b)));
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
                vector::pairwise_l2_distance_into(
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
                    vector::pairwise_cosine_similarity_with_workspace_into(
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

fn bench_competitor_vector(c: &mut Criterion) {
    let mut competitor_group = c.benchmark_group("vector_competitor_ndarray");
    for size in [128_usize, 256, 512] {
        let a = generate_random_vector(size);
        let b = generate_random_vector(size);

        _ = competitor_group.bench_with_input(
            BenchmarkId::new("cosine_similarity", size),
            &size,
            |bench, _| {
                bench.iter(|| cosine_similarity_ndarray_baseline(black_box(&a), black_box(&b)));
            },
        );

        _ = competitor_group.bench_with_input(BenchmarkId::new("dot", size), &size, |bench, _| {
            bench.iter(|| {
                if a.len() != b.len() || a.is_empty() {
                    return None;
                }
                Some(a.dot(black_box(&b)))
            });
        });
    }

    for rows in [32_usize, 64] {
        let cols = 128_usize;
        let left = generate_random_matrix(rows, cols);
        let right = generate_random_matrix(rows, cols);
        let mut l2_competitor_out = Array2::<f64>::zeros((rows, rows));
        let id = format!("square-{rows}x{rows}");

        _ = competitor_group.bench_with_input(
            BenchmarkId::new("pairwise_l2_naive", &id),
            &rows,
            |bench, _| {
                bench.iter(|| {
                    pairwise_l2_naive(
                        black_box(&left),
                        black_box(&right),
                        black_box(&mut l2_competitor_out),
                    );
                });
            },
        );
    }

    competitor_group.finish();
}

fn benchmark_ndarray_vector(c: &mut Criterion) {
    bench_nabled_vector(c);
    bench_competitor_vector(c);
}

criterion_group!(benches, benchmark_ndarray_vector);
criterion_main!(benches);

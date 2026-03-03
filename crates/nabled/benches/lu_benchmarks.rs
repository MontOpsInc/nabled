use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::Mat;
use faer::linalg::solvers::Solve as _;
use nabled::linalg::lu;
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

fn ndarray_to_faer(matrix: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| matrix[[i, j]])
}

fn vector_to_faer_col(vector: &Array1<f64>) -> Mat<f64> {
    Mat::from_fn(vector.len(), 1, |i, _| vector[i])
}

fn benchmark_ndarray_lu(c: &mut Criterion) {
    let sizes = [32_usize, 64, 96];

    {
        let mut group = c.benchmark_group("lu_nabled_ndarray");
        for size in sizes {
            let matrix = generate_well_conditioned_matrix(size);
            let rhs = generate_random_vector(size);

            _ = group.bench_with_input(BenchmarkId::new("decompose", size), &size, |b, _| {
                b.iter(|| lu::decompose(black_box(&matrix)));
            });

            _ = group.bench_with_input(BenchmarkId::new("solve", size), &size, |b, _| {
                b.iter(|| lu::solve(black_box(&matrix), black_box(&rhs)));
            });

            _ = group.bench_with_input(BenchmarkId::new("determinant", size), &size, |b, _| {
                b.iter(|| lu::determinant(black_box(&matrix)));
            });
        }
        group.finish();
    }

    {
        let mut competitor = c.benchmark_group("lu_competitor_faer_direct");
        for size in sizes {
            let matrix = generate_well_conditioned_matrix(size);
            let rhs = generate_random_vector(size);
            let matrix_faer = ndarray_to_faer(&matrix);
            let rhs_faer = vector_to_faer_col(&rhs);

            _ = competitor.bench_with_input(BenchmarkId::new("decompose", size), &size, |b, _| {
                b.iter(|| matrix_faer.as_ref().partial_piv_lu());
            });

            _ = competitor.bench_with_input(BenchmarkId::new("solve", size), &size, |b, _| {
                b.iter(|| {
                    let decomposition = matrix_faer.as_ref().partial_piv_lu();
                    decomposition.solve(rhs_faer.as_ref())
                });
            });

            _ = competitor.bench_with_input(
                BenchmarkId::new("determinant", size),
                &size,
                |b, _| {
                    b.iter(|| matrix_faer.as_ref().determinant());
                },
            );
        }
        competitor.finish();
    }
}

criterion_group!(benches, benchmark_ndarray_lu);
criterion_main!(benches);

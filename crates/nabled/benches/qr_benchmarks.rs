use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::Mat;
use faer::linalg::solvers::SolveLstsq as _;
use nabled::linalg::qr::{self as qr, QRConfig};
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

fn generate_random_vector(len: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();
    Array1::from_vec(data)
}

fn ndarray_to_faer(matrix: &Array2<f64>) -> Mat<f64> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| matrix[[i, j]])
}

fn vector_to_faer_col(vector: &Array1<f64>) -> Mat<f64> {
    Mat::from_fn(vector.len(), 1, |i, _| vector[i])
}

fn benchmark_ndarray_qr(c: &mut Criterion) {
    let config = QRConfig::default();
    let sizes = [32_usize, 64, 96];

    {
        let mut group = c.benchmark_group("qr_nabled_ndarray");
        for size in sizes {
            let matrix = generate_random_matrix(size, size);
            let rhs = generate_random_vector(size);

            _ = group.bench_with_input(BenchmarkId::new("qr", size), &size, |b, _| {
                b.iter(|| qr::decompose(black_box(&matrix), black_box(&config)));
            });

            _ = group.bench_with_input(BenchmarkId::new("least_squares", size), &size, |b, _| {
                b.iter(|| {
                    qr::solve_least_squares(black_box(&matrix), black_box(&rhs), black_box(&config))
                });
            });
        }
        group.finish();
    }

    {
        let mut competitor = c.benchmark_group("qr_competitor_faer_direct");
        for size in sizes {
            let matrix = generate_random_matrix(size, size);
            let rhs = generate_random_vector(size);
            let matrix_faer = ndarray_to_faer(&matrix);
            let rhs_faer = vector_to_faer_col(&rhs);

            _ = competitor.bench_with_input(BenchmarkId::new("qr", size), &size, |b, _| {
                b.iter(|| matrix_faer.as_ref().qr());
            });
            _ = competitor.bench_with_input(
                BenchmarkId::new("least_squares", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let decomposition = matrix_faer.as_ref().qr();
                        decomposition.solve_lstsq(rhs_faer.as_ref())
                    });
                },
            );
        }
        competitor.finish();
    }
}

criterion_group!(benches, benchmark_ndarray_qr);
criterion_main!(benches);

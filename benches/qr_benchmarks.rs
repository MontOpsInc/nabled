use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::qr::{QRConfig, nalgebra_qr, ndarray_qr};
use nalgebra::linalg::ColPivQR;
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use rand::RngExt;

#[derive(Clone, Copy)]
enum MatrixShape {
    Square,
    TallSkinny,
}

impl MatrixShape {
    fn label(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::TallSkinny => "tall",
        }
    }

    fn dims(self, size: usize) -> (usize, usize) {
        match self {
            Self::Square => (size, size),
            Self::TallSkinny => (size * 2, size),
        }
    }
}

fn generate_matrix_pair(rows: usize, cols: usize) -> (DMatrix<f64>, Array2<f64>) {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        data.push(rng.random_range(-1.0..1.0));
    }

    let nalg = DMatrix::from_row_slice(rows, cols, &data);
    let nd = Array2::from_shape_vec((rows, cols), data)
        .expect("matrix dimensions should match generated data length");
    (nalg, nd)
}

fn generate_rhs(rows: usize) -> DVector<f64> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(rows);
    for _ in 0..rows {
        data.push(rng.random_range(-1.0..1.0));
    }
    DVector::from_vec(data)
}

fn reconstruction_error(reference: &DMatrix<f64>, reconstructed: &DMatrix<f64>) -> f64 {
    (reconstructed - reference).norm() / reference.norm().max(f64::EPSILON)
}

fn assert_nabled_nalgebra_qr_correct(matrix: &DMatrix<f64>, config: &QRConfig<f64>) {
    let qr = nalgebra_qr::compute_qr(matrix, config).expect("nabled nalgebra QR should succeed");
    let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
    let err = reconstruction_error(matrix, &reconstructed);
    assert!(err < 1e-8, "nabled nalgebra QR reconstruction error too high: {err}");
}

fn assert_nalgebra_direct_qr_correct(matrix: &DMatrix<f64>) {
    let qr = matrix.clone().qr();
    let reconstructed = qr.q() * qr.r();
    let err = reconstruction_error(matrix, &reconstructed);
    assert!(err < 1e-8, "direct nalgebra QR reconstruction error too high: {err}");
}

fn assert_nabled_pivoted_qr_correct(matrix: &DMatrix<f64>, config: &QRConfig<f64>) {
    let qr = nalgebra_qr::compute_qr_with_pivoting(matrix, config)
        .expect("nabled pivoted nalgebra QR should succeed");
    let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
    let err = reconstruction_error(matrix, &reconstructed);
    assert!(err < 1e-8, "nabled pivoted QR reconstruction error too high: {err}");
}

fn assert_nalgebra_direct_pivoted_qr_correct(matrix: &DMatrix<f64>) {
    let col_piv_qr = ColPivQR::new(matrix.clone());
    let mut p = DMatrix::identity(matrix.ncols(), matrix.ncols());
    col_piv_qr.p().permute_columns(&mut p);
    let reconstructed = col_piv_qr.q() * col_piv_qr.r() * p.transpose();
    let err = reconstruction_error(matrix, &reconstructed);
    assert!(err < 1e-8, "direct nalgebra pivoted QR reconstruction error too high: {err}");
}

fn benchmark_nabled_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_nabled");
    let config = QRConfig::default();
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let (nalg_matrix, nd_matrix) = generate_matrix_pair(rows, cols);
            let rhs = generate_rhs(rows);

            assert_nabled_nalgebra_qr_correct(&nalg_matrix, &config);
            assert_nabled_pivoted_qr_correct(&nalg_matrix, &config);
            drop(
                ndarray_qr::compute_qr(&nd_matrix, &config)
                    .expect("nabled ndarray QR should succeed for benchmark input"),
            );
            drop(
                ndarray_qr::solve_least_squares(
                    &nd_matrix,
                    &ndarray::Array1::from_vec(rhs.as_slice().to_vec()),
                    &config,
                )
                .expect("nabled ndarray least-squares should succeed for benchmark input"),
            );

            let id = format!("{}-{rows}x{cols}", shape.label());

            _ = group.bench_with_input(BenchmarkId::new("nalgebra_qr", &id), &size, |b, _| {
                b.iter(|| nalgebra_qr::compute_qr(black_box(&nalg_matrix), black_box(&config)));
            });

            _ = group.bench_with_input(
                BenchmarkId::new("nalgebra_qr_pivoted", &id),
                &size,
                |b, _| {
                    b.iter(|| {
                        nalgebra_qr::compute_qr_with_pivoting(
                            black_box(&nalg_matrix),
                            black_box(&config),
                        )
                    });
                },
            );

            _ = group.bench_with_input(BenchmarkId::new("ndarray_qr", &id), &size, |b, _| {
                b.iter(|| ndarray_qr::compute_qr(black_box(&nd_matrix), black_box(&config)));
            });

            _ = group.bench_with_input(
                BenchmarkId::new("nalgebra_least_squares", &id),
                &size,
                |b, _| {
                    b.iter(|| {
                        nalgebra_qr::solve_least_squares(
                            black_box(&nalg_matrix),
                            black_box(&rhs),
                            black_box(&config),
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_nalgebra_qr_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_competitor_nalgebra_direct");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let (matrix, _) = generate_matrix_pair(rows, cols);
            assert_nalgebra_direct_qr_correct(&matrix);
            assert_nalgebra_direct_pivoted_qr_correct(&matrix);

            let id = format!("{}-{rows}x{cols}", shape.label());

            _ = group.bench_with_input(BenchmarkId::new("qr", &id), &size, |b, _| {
                b.iter(|| black_box(matrix.clone()).qr());
            });

            _ = group.bench_with_input(BenchmarkId::new("qr_pivoted", &id), &size, |b, _| {
                b.iter(|| ColPivQR::new(black_box(matrix.clone())));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_nabled_qr, benchmark_nalgebra_qr_competitor);
criterion_main!(benches);

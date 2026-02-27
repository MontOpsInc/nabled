use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::MatRef;
use nabled::qr::{QRConfig, nalgebra_qr, ndarray_qr};
use nalgebra::linalg::ColPivQR;
use nalgebra::{DMatrix, DVector};
#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
use nalgebra_lapack::QR as NalgebraLapackQr;
#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
use nalgebra_lapack::qr::QrDecomposition;
use ndarray::Array2;
#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
use ndarray_linalg::QR as NdarrayLinalgQr;
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
    let data = generate_matrix_data(rows, cols);
    let nalg = DMatrix::from_row_slice(rows, cols, &data);
    let nd = Array2::from_shape_vec((rows, cols), data)
        .expect("matrix dimensions should match generated data length");
    (nalg, nd)
}

fn generate_matrix_data(rows: usize, cols: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        data.push(rng.random_range(-1.0..1.0));
    }
    data
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

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
fn frobenius_norm_ndarray(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
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

fn faer_to_nalgebra(matrix: MatRef<'_, f64>) -> DMatrix<f64> {
    let (rows, cols) = matrix.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(matrix[(i, j)]);
        }
    }
    DMatrix::from_row_slice(rows, cols, &data)
}

fn assert_faer_direct_qr_correct(data: &[f64], rows: usize, cols: usize) {
    let matrix = MatRef::from_row_major_slice(data, rows, cols);
    let qr = matrix.qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R();
    let reference = DMatrix::from_row_slice(rows, cols, data);
    let reconstructed = faer_to_nalgebra(q.as_ref()) * faer_to_nalgebra(r);
    let err = reconstruction_error(&reference, &reconstructed);
    assert!(err < 1e-8, "direct faer QR reconstruction error too high: {err}");
}

fn assert_faer_direct_pivoted_qr_correct(data: &[f64], rows: usize, cols: usize) {
    let matrix = MatRef::from_row_major_slice(data, rows, cols);
    let col_piv_qr = matrix.col_piv_qr();
    let r = col_piv_qr.thin_R();
    assert_eq!(r.ncols(), cols, "faer pivoted QR R columns should match input");
    assert!(r.nrows() <= rows, "faer pivoted QR R rows should be bounded by input rows");
    for i in 0..r.nrows() {
        for j in 0..r.ncols() {
            assert!(r[(i, j)].is_finite(), "faer pivoted QR produced non-finite R entry");
        }
    }
}

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
fn assert_nalgebra_lapack_qr_correct(matrix: &DMatrix<f64>) {
    let qr = NalgebraLapackQr::new(matrix.clone()).expect("nalgebra-lapack QR should succeed");
    let reconstructed = qr.q() * qr.r();
    let err = reconstruction_error(matrix, &reconstructed);
    assert!(err < 1e-8, "nalgebra-lapack QR reconstruction error too high: {err}");
}

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
fn assert_ndarray_linalg_qr_correct(matrix: &Array2<f64>) {
    let (q, r) = matrix.view().qr().expect("ndarray-linalg QR should succeed");
    let reconstructed = q.dot(&r);
    let diff = &reconstructed - matrix;
    let err = frobenius_norm_ndarray(&diff) / frobenius_norm_ndarray(matrix).max(f64::EPSILON);
    assert!(err < 1e-8, "ndarray-linalg QR reconstruction error too high: {err}");
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

fn benchmark_faer_qr_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_competitor_faer_direct");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let data = generate_matrix_data(rows, cols);
            assert_faer_direct_qr_correct(&data, rows, cols);
            assert_faer_direct_pivoted_qr_correct(&data, rows, cols);

            let id = format!("{}-{rows}x{cols}", shape.label());

            _ = group.bench_with_input(BenchmarkId::new("qr", &id), &size, |b, _| {
                b.iter(|| {
                    MatRef::from_row_major_slice(black_box(data.as_slice()), rows, cols).qr()
                });
            });

            _ = group.bench_with_input(BenchmarkId::new("qr_pivoted", &id), &size, |b, _| {
                b.iter(|| {
                    MatRef::from_row_major_slice(black_box(data.as_slice()), rows, cols)
                        .col_piv_qr()
                });
            });
        }
    }

    group.finish();
}

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
fn benchmark_nalgebra_lapack_qr_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_competitor_nalgebra_lapack");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let (matrix, _) = generate_matrix_pair(rows, cols);
            assert_nalgebra_lapack_qr_correct(&matrix);
            let id = format!("{}-{rows}x{cols}", shape.label());

            _ = group.bench_with_input(BenchmarkId::new("qr", &id), &size, |b, _| {
                b.iter(|| NalgebraLapackQr::new(black_box(matrix.clone())));
            });
        }
    }

    group.finish();
}

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
fn benchmark_ndarray_linalg_qr_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_competitor_ndarray_linalg");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let (_, matrix) = generate_matrix_pair(rows, cols);
            assert_ndarray_linalg_qr_correct(&matrix);
            let id = format!("{}-{rows}x{cols}", shape.label());

            _ = group.bench_with_input(BenchmarkId::new("qr", &id), &size, |b, _| {
                b.iter(|| matrix.view().qr());
            });
        }
    }

    group.finish();
}

#[cfg(all(feature = "lapack-competitors", target_os = "linux"))]
criterion_group!(
    benches,
    benchmark_nabled_qr,
    benchmark_nalgebra_qr_competitor,
    benchmark_faer_qr_competitor,
    benchmark_nalgebra_lapack_qr_competitor,
    benchmark_ndarray_linalg_qr_competitor
);

#[cfg(not(all(feature = "lapack-competitors", target_os = "linux")))]
criterion_group!(
    benches,
    benchmark_nabled_qr,
    benchmark_nalgebra_qr_competitor,
    benchmark_faer_qr_competitor
);
criterion_main!(benches);

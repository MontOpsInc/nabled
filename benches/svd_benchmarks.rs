use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::MatRef;
use nabled::svd::{nalgebra_svd, ndarray_svd};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::RngExt;

#[derive(Clone, Copy)]
enum MatrixShape {
    Square,
    TallSkinny,
    WideShort,
}

impl MatrixShape {
    fn label(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::TallSkinny => "tall",
            Self::WideShort => "wide",
        }
    }

    fn dims(self, size: usize) -> (usize, usize) {
        match self {
            Self::Square => (size, size),
            Self::TallSkinny => (size * 2, size),
            Self::WideShort => (size, size * 2),
        }
    }
}

fn generate_random_matrix_nalgebra(rows: usize, cols: usize) -> DMatrix<f64> {
    let data = generate_random_matrix_data(rows, cols);
    DMatrix::from_row_slice(rows, cols, &data)
}

fn generate_random_matrix_ndarray(rows: usize, cols: usize) -> Array2<f64> {
    let data = generate_random_matrix_data(rows, cols);
    Array2::from_shape_vec((rows, cols), data).expect("random matrix dimensions should match data")
}

fn generate_random_matrix_data(rows: usize, cols: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(rows * cols);

    for _ in 0..rows * cols {
        data.push(rng.random_range(-1.0..1.0));
    }

    data
}

fn frobenius_norm_ndarray(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn assert_nabled_nalgebra_svd_correct(matrix: &DMatrix<f64>) {
    let svd = nalgebra_svd::compute_svd(matrix).expect("nabled nalgebra SVD should succeed");
    let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
    let err = (&reconstructed - matrix).norm() / matrix.norm().max(f64::EPSILON);
    assert!(err < 1e-8, "nabled nalgebra SVD reconstruction error too high: {err}");
}

fn assert_nalgebra_direct_svd_correct(matrix: &DMatrix<f64>) {
    let svd = matrix.clone().svd(true, true);
    let (Some(u), Some(vt)) = (svd.u, svd.v_t) else {
        panic!("nalgebra direct SVD failed to produce U or V^T");
    };
    let sigma = DMatrix::from_diagonal(&svd.singular_values);
    let reconstructed = u * sigma * vt;
    let err = (&reconstructed - matrix).norm() / matrix.norm().max(f64::EPSILON);
    assert!(err < 1e-8, "nalgebra direct SVD reconstruction error too high: {err}");
}

fn assert_nabled_ndarray_svd_correct(matrix: &Array2<f64>) {
    let svd = ndarray_svd::compute_svd(matrix).expect("nabled ndarray SVD should succeed");
    let reconstructed = ndarray_svd::reconstruct_matrix(&svd);
    let diff = &reconstructed - matrix;
    let err = frobenius_norm_ndarray(&diff) / frobenius_norm_ndarray(matrix).max(f64::EPSILON);
    assert!(err < 1e-8, "nabled ndarray SVD reconstruction error too high: {err}");
}

fn assert_faer_direct_svd_correct(data: &[f64], rows: usize, cols: usize) {
    let faer_matrix = MatRef::from_row_major_slice(data, rows, cols);
    let faer_singular_values =
        faer_matrix.singular_values().expect("faer direct singular values should succeed");

    let nalgebra_singular_values =
        DMatrix::from_row_slice(rows, cols, data).svd(false, false).singular_values;
    assert_eq!(
        faer_singular_values.len(),
        nalgebra_singular_values.len(),
        "faer and nalgebra singular-value lengths should match"
    );

    for (faer_value, nalgebra_value) in
        faer_singular_values.iter().zip(nalgebra_singular_values.iter())
    {
        let diff = (faer_value - nalgebra_value).abs();
        assert!(
            diff < 1e-8,
            "faer singular value diverges from nalgebra baseline: {faer_value} vs {nalgebra_value}"
        );
    }
}

fn benchmark_nalgebra_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_nabled_nalgebra");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny, MatrixShape::WideShort];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let matrix = generate_random_matrix_nalgebra(rows, cols);
            assert_nabled_nalgebra_svd_correct(&matrix);

            _ = group.bench_with_input(
                BenchmarkId::new("full_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| b.iter(|| nalgebra_svd::compute_svd(black_box(&matrix))),
            );

            let k = 5_usize.min(rows.min(cols));
            _ = group.bench_with_input(
                BenchmarkId::new("truncated_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| b.iter(|| nalgebra_svd::compute_truncated_svd(black_box(&matrix), k)),
            );
        }
    }

    group.finish();
}

fn benchmark_ndarray_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_nabled_ndarray");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny, MatrixShape::WideShort];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let matrix = generate_random_matrix_ndarray(rows, cols);
            assert_nabled_ndarray_svd_correct(&matrix);

            _ = group.bench_with_input(
                BenchmarkId::new("full_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| b.iter(|| ndarray_svd::compute_svd(black_box(&matrix))),
            );

            let k = 5_usize.min(rows.min(cols));
            _ = group.bench_with_input(
                BenchmarkId::new("truncated_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| b.iter(|| ndarray_svd::compute_truncated_svd(black_box(&matrix), k)),
            );
        }
    }

    group.finish();
}

fn benchmark_nalgebra_direct_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_competitor_nalgebra_direct");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny, MatrixShape::WideShort];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let matrix = generate_random_matrix_nalgebra(rows, cols);
            assert_nalgebra_direct_svd_correct(&matrix);

            _ = group.bench_with_input(
                BenchmarkId::new("full_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| b.iter(|| black_box(matrix.clone()).svd(true, true)),
            );
        }
    }

    group.finish();
}

fn benchmark_faer_direct_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_competitor_faer_direct");
    let sizes = [16, 64, 128];
    let shapes = [MatrixShape::Square, MatrixShape::TallSkinny, MatrixShape::WideShort];

    for size in sizes {
        for shape in shapes {
            let (rows, cols) = shape.dims(size);
            let data = generate_random_matrix_data(rows, cols);
            assert_faer_direct_svd_correct(&data, rows, cols);

            _ = group.bench_with_input(
                BenchmarkId::new("full_svd", format!("{}-{rows}x{cols}", shape.label())),
                &size,
                |b, _| {
                    b.iter(|| {
                        MatRef::from_row_major_slice(black_box(data.as_slice()), rows, cols).svd()
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_nalgebra_svd,
    benchmark_ndarray_svd,
    benchmark_nalgebra_direct_competitor,
    benchmark_faer_direct_competitor
);
criterion_main!(benches);

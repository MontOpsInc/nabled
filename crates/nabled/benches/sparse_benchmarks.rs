use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::sparse::{self as sparse, CsrMatrix};
use ndarray::{Array1, Array2};
use rand::RngExt;

fn make_diagonally_dominant_tridiagonal(size: usize) -> CsrMatrix {
    let mut indptr = Vec::with_capacity(size + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);

    for row in 0..size {
        if row > 0 {
            indices.push(row - 1);
            data.push(-1.0);
        }
        indices.push(row);
        data.push(4.0);
        if row + 1 < size {
            indices.push(row + 1);
            data.push(-1.0);
        }
        indptr.push(indices.len());
    }

    CsrMatrix::new(size, size, indptr, indices, data).expect("valid tridiagonal CSR")
}

fn random_vector(size: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let values = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array1::from_vec(values)
}

fn csr_to_dense(matrix: &CsrMatrix) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((matrix.nrows, matrix.ncols));
    for row in 0..matrix.nrows {
        let start = matrix.indptr[row];
        let end = matrix.indptr[row + 1];
        for idx in start..end {
            dense[[row, matrix.indices[idx]]] = matrix.data[idx];
        }
    }
    dense
}

fn benchmark_sparse(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("sparse_nabled_ndarray");
        for size in [128_usize, 256, 512] {
            let matrix = make_diagonally_dominant_tridiagonal(size);
            let rhs = random_vector(size);
            let mut output = Array1::<f64>::zeros(size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(BenchmarkId::new("csr_matvec", &id), &size, |bench, _| {
                bench.iter(|| sparse::matvec(black_box(&matrix), black_box(&rhs)));
            });

            _ = group.bench_with_input(
                BenchmarkId::new("csr_matvec_into", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        sparse::matvec_into(
                            black_box(&matrix),
                            black_box(&rhs),
                            black_box(&mut output),
                        )
                    });
                },
            );

            _ = group.bench_with_input(BenchmarkId::new("jacobi_solve", &id), &size, |bench, _| {
                bench.iter(|| {
                    sparse::jacobi_solve(
                        black_box(&matrix),
                        black_box(&rhs),
                        black_box(1e-8),
                        black_box(10_000),
                    )
                });
            });
        }
        group.finish();
    }

    {
        let mut competitor_group = c.benchmark_group("sparse_competitor_ndarray");
        for size in [128_usize, 256, 512] {
            let matrix = make_diagonally_dominant_tridiagonal(size);
            let dense_matrix = csr_to_dense(&matrix);
            let rhs = random_vector(size);
            let id = format!("square-{size}x{size}");

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("dense_matvec", &id),
                &size,
                |bench, _| {
                    bench.iter(|| dense_matrix.dot(black_box(&rhs)));
                },
            );
        }
        competitor_group.finish();
    }
}

criterion_group!(benches, benchmark_sparse);
criterion_main!(benches);

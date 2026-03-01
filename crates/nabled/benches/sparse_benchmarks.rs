use std::hint::black_box;

use criterion::measurement::WallTime;
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

fn benchmark_sparse_solvers(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
) {
    let mut output = Array1::<f64>::zeros(size);
    let zero_fill_factorization = sparse::ilu0_factor(matrix).expect("ilu0 factorization");
    let threshold_factorization = sparse::ilut_factor(matrix, 0.0, 16).expect("ilut factorization");

    benchmark_sparse_core(group, id, size, matrix, rhs, &mut output);
    benchmark_sparse_bicgstab(
        group,
        id,
        size,
        matrix,
        rhs,
        &zero_fill_factorization,
        &threshold_factorization,
    );
    benchmark_sparse_gmres(
        group,
        id,
        size,
        matrix,
        rhs,
        &zero_fill_factorization,
        &threshold_factorization,
    );
}

fn benchmark_sparse_core(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    mut output: &mut Array1<f64>,
) {
    _ = group.bench_with_input(BenchmarkId::new("csr_matvec", id), &size, |bench, _| {
        bench.iter(|| sparse::matvec(black_box(matrix), black_box(rhs)));
    });

    _ = group.bench_with_input(BenchmarkId::new("csr_matvec_into", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::matvec_into(black_box(matrix), black_box(rhs), black_box(&mut output))
        });
    });

    _ = group.bench_with_input(BenchmarkId::new("jacobi_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::jacobi_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(10_000),
            )
        });
    });

    _ = group.bench_with_input(BenchmarkId::new("pcg_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::pcg_solve(black_box(matrix), black_box(rhs), black_box(1e-8), black_box(10_000))
        });
    });
}

fn benchmark_sparse_bicgstab(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    zero_fill_factorization: &sparse::ILU0Factorization,
    threshold_factorization: &sparse::ILUTFactorization,
) {
    _ = group.bench_with_input(BenchmarkId::new("pcg_ic0_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::pcg_ic0_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(10_000),
            )
        });
    });

    _ = group.bench_with_input(BenchmarkId::new("bicgstab_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::bicgstab_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(10_000),
            )
        });
    });

    _ = group.bench_with_input(BenchmarkId::new("bicgstab_ilu0_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::bicgstab_ilu0_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(10_000),
            )
        });
    });

    _ = group.bench_with_input(
        BenchmarkId::new("bicgstab_ilu0_solve_reuse", id),
        &size,
        |bench, _| {
            bench.iter(|| {
                sparse::bicgstab_ilu0_solve_with_factorization(
                    black_box(matrix),
                    black_box(rhs),
                    black_box(1e-8),
                    black_box(10_000),
                    black_box(zero_fill_factorization),
                )
            });
        },
    );

    _ = group.bench_with_input(BenchmarkId::new("bicgstab_ilut_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::bicgstab_ilut_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(10_000),
                black_box(0.0),
                black_box(16),
            )
        });
    });

    _ = group.bench_with_input(
        BenchmarkId::new("bicgstab_ilut_solve_reuse", id),
        &size,
        |bench, _| {
            bench.iter(|| {
                sparse::bicgstab_ilut_solve_with_factorization(
                    black_box(matrix),
                    black_box(rhs),
                    black_box(1e-8),
                    black_box(10_000),
                    black_box(threshold_factorization),
                )
            });
        },
    );
}

fn benchmark_sparse_gmres(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    zero_fill_factorization: &sparse::ILU0Factorization,
    threshold_factorization: &sparse::ILUTFactorization,
) {
    _ = group.bench_with_input(BenchmarkId::new("gmres_ilu0_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::gmres_ilu0_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(128),
            )
        });
    });

    _ = group.bench_with_input(
        BenchmarkId::new("gmres_ilu0_solve_reuse", id),
        &size,
        |bench, _| {
            bench.iter(|| {
                sparse::gmres_ilu0_solve_with_factorization(
                    black_box(matrix),
                    black_box(rhs),
                    black_box(1e-8),
                    black_box(128),
                    black_box(zero_fill_factorization),
                )
            });
        },
    );

    _ = group.bench_with_input(BenchmarkId::new("gmres_ilut_solve", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::gmres_ilut_solve(
                black_box(matrix),
                black_box(rhs),
                black_box(1e-8),
                black_box(128),
                black_box(0.0),
                black_box(16),
            )
        });
    });

    _ = group.bench_with_input(
        BenchmarkId::new("gmres_ilut_solve_reuse", id),
        &size,
        |bench, _| {
            bench.iter(|| {
                sparse::gmres_ilut_solve_with_factorization(
                    black_box(matrix),
                    black_box(rhs),
                    black_box(1e-8),
                    black_box(128),
                    black_box(threshold_factorization),
                )
            });
        },
    );
}

fn benchmark_sparse_matmul(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    dense_rhs: &Array2<f64>,
) {
    _ = group.bench_with_input(BenchmarkId::new("csr_matmat_dense", id), &size, |bench, _| {
        bench.iter(|| sparse::matmat_dense(black_box(matrix), black_box(dense_rhs)));
    });

    _ = group.bench_with_input(BenchmarkId::new("csr_matmat_sparse", id), &size, |bench, _| {
        bench.iter(|| sparse::matmat_sparse(black_box(matrix), black_box(matrix)));
    });
}

fn benchmark_sparse_nabled(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_nabled_ndarray");
    for size in [128_usize, 256, 512] {
        let matrix = make_diagonally_dominant_tridiagonal(size);
        let rhs = random_vector(size);
        let dense_rhs = Array2::<f64>::ones((size, 8));
        let id = format!("square-{size}x{size}");
        benchmark_sparse_solvers(&mut group, &id, size, &matrix, &rhs);
        benchmark_sparse_matmul(&mut group, &id, size, &matrix, &dense_rhs);
    }
    group.finish();
}

fn benchmark_sparse_competitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_competitor_ndarray");
    for size in [128_usize, 256, 512] {
        let matrix = make_diagonally_dominant_tridiagonal(size);
        let dense_matrix = csr_to_dense(&matrix);
        let rhs = random_vector(size);
        let id = format!("square-{size}x{size}");

        _ = group.bench_with_input(BenchmarkId::new("dense_matvec", &id), &size, |bench, _| {
            bench.iter(|| dense_matrix.dot(black_box(&rhs)));
        });
    }
    group.finish();
}

fn benchmark_sparse(c: &mut Criterion) {
    benchmark_sparse_nabled(c);
    benchmark_sparse_competitor(c);
}

criterion_group!(benches, benchmark_sparse);
criterion_main!(benches);

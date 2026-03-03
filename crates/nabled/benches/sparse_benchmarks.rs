use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::linalg::sparse::{self as sparse, CsrMatrix};
use ndarray::{Array1, Array2};
use rand::RngExt;

struct SparseFactorizations<'a> {
    direct:    &'a sparse::SparseLUFactorization,
    zero_fill: &'a sparse::ILU0Factorization,
    threshold: &'a sparse::ILUTFactorization,
    level:     &'a sparse::ILUKFactorization,
    symmetric: &'a sparse::ILDL0Factorization,
}

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
    let rhs_matrix = Array2::<f64>::ones((size, 4));
    let direct_factorization = sparse::sparse_lu_factor(matrix).expect("sparse lu factorization");
    let zero_fill_factorization = sparse::ilu0_factor(matrix).expect("ilu0 factorization");
    let threshold_factorization = sparse::ilut_factor(matrix, 0.0, 16).expect("ilut factorization");
    let level_factorization = sparse::iluk_factor(matrix, 1).expect("iluk factorization");
    let symmetric_factorization = sparse::ildl0_factor(matrix).expect("ildl0 factorization");
    let factorizations = SparseFactorizations {
        direct:    &direct_factorization,
        zero_fill: &zero_fill_factorization,
        threshold: &threshold_factorization,
        level:     &level_factorization,
        symmetric: &symmetric_factorization,
    };

    benchmark_sparse_core(group, id, size, matrix, rhs, &mut output, &factorizations);
    benchmark_sparse_bicgstab(group, id, size, matrix, rhs, &rhs_matrix, &factorizations);
    benchmark_sparse_gmres(group, id, size, matrix, rhs, &rhs_matrix, &factorizations);
}

fn benchmark_sparse_core(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    mut output: &mut Array1<f64>,
    factorizations: &SparseFactorizations<'_>,
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

    _ = group.bench_with_input(BenchmarkId::new("sparse_lu_solve", id), &size, |bench, _| {
        bench.iter(|| sparse::sparse_lu_solve(black_box(matrix), black_box(rhs)));
    });

    _ = group.bench_with_input(BenchmarkId::new("sparse_lu_solve_reuse", id), &size, |bench, _| {
        bench.iter(|| {
            sparse::sparse_lu_solve_with_factorization(
                black_box(matrix),
                black_box(rhs),
                black_box(factorizations.direct),
            )
        });
    });
}

fn bench_sparse_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    label: &str,
    mut run: impl FnMut() -> T,
) {
    _ = group.bench_with_input(BenchmarkId::new(label, id), &size, |bench, _| {
        bench.iter(&mut run);
    });
}

fn benchmark_sparse_bicgstab(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    rhs_matrix: &Array2<f64>,
    factorizations: &SparseFactorizations<'_>,
) {
    benchmark_sparse_bicgstab_single_rhs(group, id, size, matrix, rhs, factorizations);
    benchmark_sparse_bicgstab_multi_rhs(group, id, size, matrix, rhs_matrix, factorizations);
}

fn benchmark_sparse_bicgstab_single_rhs(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    factorizations: &SparseFactorizations<'_>,
) {
    bench_sparse_case(group, id, size, "pcg_ic0_solve", || {
        sparse::pcg_ic0_solve(black_box(matrix), black_box(rhs), black_box(1e-8), black_box(10_000))
    });
    bench_sparse_case(group, id, size, "bicgstab_solve", || {
        sparse::bicgstab_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ilu0_solve", || {
        sparse::bicgstab_ilu0_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ilu0_solve_reuse", || {
        sparse::bicgstab_ilu0_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.zero_fill),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ilut_solve", || {
        sparse::bicgstab_ilut_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(0.0),
            black_box(16),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ilut_solve_reuse", || {
        sparse::bicgstab_ilut_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.threshold),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_iluk_solve", || {
        sparse::bicgstab_iluk_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(1),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_iluk_solve_reuse", || {
        sparse::bicgstab_iluk_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.level),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ildl0_solve", || {
        sparse::bicgstab_ildl0_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ildl0_solve_reuse", || {
        sparse::bicgstab_ildl0_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.symmetric),
        )
    });
}

fn benchmark_sparse_bicgstab_multi_rhs(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs_matrix: &Array2<f64>,
    factorizations: &SparseFactorizations<'_>,
) {
    bench_sparse_case(group, id, size, "bicgstab_ilu0_solve_multi_reuse", || {
        sparse::bicgstab_ilu0_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.zero_fill),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_iluk_solve_multi_reuse", || {
        sparse::bicgstab_iluk_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.level),
        )
    });
    bench_sparse_case(group, id, size, "bicgstab_ildl0_solve_multi_reuse", || {
        sparse::bicgstab_ildl0_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(10_000),
            black_box(factorizations.symmetric),
        )
    });
}

fn benchmark_sparse_gmres(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    id: &str,
    size: usize,
    matrix: &CsrMatrix,
    rhs: &Array1<f64>,
    rhs_matrix: &Array2<f64>,
    factorizations: &SparseFactorizations<'_>,
) {
    bench_sparse_case(group, id, size, "gmres_ilu0_solve", || {
        sparse::gmres_ilu0_solve(black_box(matrix), black_box(rhs), black_box(1e-8), black_box(128))
    });
    bench_sparse_case(group, id, size, "gmres_ilu0_solve_reuse", || {
        sparse::gmres_ilu0_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.zero_fill),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ilut_solve", || {
        sparse::gmres_ilut_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(0.0),
            black_box(16),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ilut_solve_reuse", || {
        sparse::gmres_ilut_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.threshold),
        )
    });
    bench_sparse_case(group, id, size, "gmres_iluk_solve", || {
        sparse::gmres_iluk_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(1),
        )
    });
    bench_sparse_case(group, id, size, "gmres_iluk_solve_reuse", || {
        sparse::gmres_iluk_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.level),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ildl0_solve", || {
        sparse::gmres_ildl0_solve(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ildl0_solve_reuse", || {
        sparse::gmres_ildl0_solve_with_factorization(
            black_box(matrix),
            black_box(rhs),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.symmetric),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ilu0_solve_multi_reuse", || {
        sparse::gmres_ilu0_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.zero_fill),
        )
    });
    bench_sparse_case(group, id, size, "gmres_iluk_solve_multi_reuse", || {
        sparse::gmres_iluk_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.level),
        )
    });
    bench_sparse_case(group, id, size, "gmres_ildl0_solve_multi_reuse", || {
        sparse::gmres_ildl0_solve_multiple_with_factorization(
            black_box(matrix),
            black_box(rhs_matrix),
            black_box(1e-8),
            black_box(128),
            black_box(factorizations.symmetric),
        )
    });
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

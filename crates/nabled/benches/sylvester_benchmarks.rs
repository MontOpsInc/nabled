use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::sylvester;
use ndarray::{Array1, Array2};
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn make_stable_square(size: usize) -> Array2<f64> {
    let mut matrix = random_matrix(size, size);
    for i in 0..size {
        matrix[[i, i]] += 5.0;
    }
    matrix
}

fn solve_linear_system_gaussian(
    coefficients: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Option<Array1<f64>> {
    let n = coefficients.nrows();
    if coefficients.ncols() != n || rhs.len() != n {
        return None;
    }

    let mut a = coefficients.clone();
    let mut b = rhs.clone();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = a[[k, k]].abs();
        for row in (k + 1)..n {
            let candidate = a[[row, k]].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= 1e-12 {
            return None;
        }

        if pivot_row != k {
            let top = a.row(k).to_owned();
            let bottom = a.row(pivot_row).to_owned();
            a.row_mut(k).assign(&bottom);
            a.row_mut(pivot_row).assign(&top);
            b.swap(k, pivot_row);
        }

        let pivot = a[[k, k]];
        for row in (k + 1)..n {
            let factor = a[[row, k]] / pivot;
            a[[row, k]] = 0.0;
            for col in (k + 1)..n {
                a[[row, col]] -= factor * a[[k, col]];
            }
            b[row] -= factor * b[k];
        }
    }

    let mut solution = Array1::<f64>::zeros(n);
    for rev in 0..n {
        let row = n - 1 - rev;
        let mut sum = b[row];
        for col in (row + 1)..n {
            sum -= a[[row, col]] * solution[col];
        }
        let diagonal = a[[row, row]];
        if diagonal.abs() <= 1e-12 {
            return None;
        }
        solution[row] = sum / diagonal;
    }

    Some(solution)
}

fn solve_sylvester_manual_kronecker(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
    matrix_c: &Array2<f64>,
) -> Option<Array2<f64>> {
    let n = matrix_a.nrows();
    let m = matrix_b.nrows();
    if matrix_a.ncols() != n
        || matrix_b.ncols() != m
        || matrix_c.nrows() != n
        || matrix_c.ncols() != m
    {
        return None;
    }

    let dim = n * m;
    let mut coefficients = Array2::<f64>::zeros((dim, dim));
    let mut rhs = Array1::<f64>::zeros(dim);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                coefficients[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                coefficients[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    let vectorized = solve_linear_system_gaussian(&coefficients, &rhs)?;
    let mut solution = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            solution[[i, j]] = vectorized[i * m + j];
        }
    }

    Some(solution)
}

fn benchmark_sylvester(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("sylvester_nabled_ndarray");
        for size in [8_usize, 16, 24] {
            let matrix_a = make_stable_square(size);
            let matrix_b = make_stable_square(size);
            let matrix_c = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(
                BenchmarkId::new("solve_sylvester", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        sylvester::solve_sylvester(
                            black_box(&matrix_a),
                            black_box(&matrix_b),
                            black_box(&matrix_c),
                        )
                    });
                },
            );
        }
        group.finish();
    }

    {
        let mut competitor = c.benchmark_group("sylvester_competitor_manual");
        for size in [4_usize, 8, 12] {
            let matrix_a = make_stable_square(size);
            let matrix_b = make_stable_square(size);
            let matrix_c = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = competitor.bench_with_input(
                BenchmarkId::new("solve_sylvester_kronecker", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        solve_sylvester_manual_kronecker(
                            black_box(&matrix_a),
                            black_box(&matrix_b),
                            black_box(&matrix_c),
                        )
                    });
                },
            );
        }
        competitor.finish();
    }
}

criterion_group!(benches, benchmark_sylvester);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::schur;
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..size * size).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((size, size), values).expect("shape should match")
}

fn classical_qr(matrix: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (rows, cols) = matrix.dim();
    let mut q = Array2::<f64>::zeros((rows, cols));
    let mut r = Array2::<f64>::zeros((cols, cols));

    for j in 0..cols {
        let mut v = matrix.column(j).to_owned();
        for i in 0..j {
            let q_i = q.column(i);
            let projection = q_i.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f64>();
            r[[i, j]] = projection;
            let correction = q_i.mapv(|value| value * projection);
            v = &v - &correction;
        }

        let norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            continue;
        }
        r[[j, j]] = norm;
        let normalized = v.mapv(|value| value / norm);
        q.column_mut(j).assign(&normalized);
    }

    (q, r)
}

fn manual_schur_qr_iterations(
    matrix: &Array2<f64>,
    max_iterations: usize,
) -> (Array2<f64>, Array2<f64>) {
    let size = matrix.nrows();
    let mut t = matrix.clone();
    let mut q_total = Array2::<f64>::eye(size);

    for _ in 0..max_iterations {
        let (q, r) = classical_qr(&t);
        t = r.dot(&q);
        q_total = q_total.dot(&q);
    }

    (q_total, t)
}

fn benchmark_schur(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("schur_nabled_ndarray");
        for size in [16_usize, 32, 48] {
            let matrix = random_matrix(size);
            let id = format!("square-{size}x{size}");
            _ = group.bench_with_input(
                BenchmarkId::new("compute_schur", &id),
                &size,
                |bench, _| {
                    bench.iter(|| schur::compute_schur(black_box(&matrix)));
                },
            );
        }
        group.finish();
    }

    {
        let mut competitor = c.benchmark_group("schur_competitor_manual");
        for size in [16_usize, 32] {
            let matrix = random_matrix(size);
            let id = format!("square-{size}x{size}");
            _ = competitor.bench_with_input(
                BenchmarkId::new("manual_qr_iteration", &id),
                &size,
                |bench, _| {
                    bench.iter(|| manual_schur_qr_iterations(black_box(&matrix), black_box(96)));
                },
            );
        }
        competitor.finish();
    }
}

criterion_group!(benches, benchmark_schur);
criterion_main!(benches);

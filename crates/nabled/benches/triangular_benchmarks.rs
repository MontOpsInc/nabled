use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::triangular::ndarray_triangular;
use ndarray::{Array1, Array2};
use rand::RngExt;

fn generate_random_data(len: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(rng.random_range(-1.0..1.0));
    }
    data
}

fn generate_lower_triangular(size: usize) -> Array2<f64> {
    let values = generate_random_data(size * size);
    let mut matrix = Array2::zeros((size, size));

    for i in 0..size {
        for j in 0..=i {
            let value = values[i * size + j];
            matrix[[i, j]] = if i == j { value.abs() + 1.0 } else { value };
        }
    }

    matrix
}

fn generate_upper_triangular(size: usize) -> Array2<f64> {
    generate_lower_triangular(size).t().to_owned()
}

fn generate_rhs(size: usize) -> Array1<f64> { Array1::from_vec(generate_random_data(size)) }

fn assert_ndarray_triangular_correct(lower: &Array2<f64>, upper: &Array2<f64>, rhs: &Array1<f64>) {
    let x_lower = ndarray_triangular::solve_lower(lower, rhs)
        .expect("nabled ndarray lower solve should work");
    let x_upper = ndarray_triangular::solve_upper(upper, rhs)
        .expect("nabled ndarray upper solve should work");

    let lower_residual = lower.dot(&x_lower) - rhs;
    let upper_residual = upper.dot(&x_upper) - rhs;
    let rhs_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt().max(f64::EPSILON);
    let lower_err = lower_residual.iter().map(|value| value * value).sum::<f64>().sqrt() / rhs_norm;
    let upper_err = upper_residual.iter().map(|value| value * value).sum::<f64>().sqrt() / rhs_norm;

    assert!(lower_err < 1e-8, "ndarray lower triangular residual too high: {lower_err}");
    assert!(upper_err < 1e-8, "ndarray upper triangular residual too high: {upper_err}");
}

fn benchmark_ndarray_triangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangular_nabled_ndarray");
    let sizes = [16, 64, 128];

    for size in sizes {
        let lower = generate_lower_triangular(size);
        let upper = generate_upper_triangular(size);
        let rhs = generate_rhs(size);
        let id = format!("square-{size}x{size}");

        assert_ndarray_triangular_correct(&lower, &upper, &rhs);

        _ = group.bench_with_input(BenchmarkId::new("solve_lower", &id), &size, |b, _| {
            b.iter(|| ndarray_triangular::solve_lower(black_box(&lower), black_box(&rhs)));
        });

        _ = group.bench_with_input(BenchmarkId::new("solve_upper", &id), &size, |b, _| {
            b.iter(|| ndarray_triangular::solve_upper(black_box(&upper), black_box(&rhs)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_ndarray_triangular);
criterion_main!(benches);

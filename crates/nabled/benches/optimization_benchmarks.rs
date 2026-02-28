use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::optimization::ndarray_optimization;
use nabled::{AdamConfig, SGDConfig};
use ndarray::Array1;
use rand::RngExt;

fn random_start(dim: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let values = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect::<Vec<_>>();
    Array1::from_vec(values)
}

fn quadratic_objective(x: &Array1<f64>) -> f64 { x.iter().map(|value| value * value).sum() }

fn quadratic_gradient(x: &Array1<f64>) -> Array1<f64> { x.mapv(|value| 2.0 * value) }

fn benchmark_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_nabled_ndarray");

    for dim in [64_usize, 128, 256] {
        let start = random_start(dim);
        let id = format!("vector-{dim}x1");

        let sgd_config =
            SGDConfig { learning_rate: 1e-2, max_iterations: 200, tolerance: 1e-8 };
        _ = group.bench_with_input(BenchmarkId::new("gradient_descent", &id), &dim, |bench, _| {
            bench.iter(|| {
                ndarray_optimization::gradient_descent(
                    black_box(&start),
                    black_box(quadratic_objective),
                    black_box(quadratic_gradient),
                    black_box(&sgd_config),
                )
            });
        });

        let adam_config = AdamConfig {
            learning_rate:  1e-2,
            beta1:          0.9,
            beta2:          0.999,
            epsilon:        1e-8,
            max_iterations: 200,
            tolerance:      1e-8,
        };
        _ = group.bench_with_input(BenchmarkId::new("adam", &id), &dim, |bench, _| {
            bench.iter(|| {
                ndarray_optimization::adam(
                    black_box(&start),
                    black_box(quadratic_objective),
                    black_box(quadratic_gradient),
                    black_box(&adam_config),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_optimization);
criterion_main!(benches);

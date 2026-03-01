use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::{AdamConfig, SGDConfig, optimization};
use ndarray::Array1;
use rand::RngExt;

fn random_start(dim: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    let values = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect::<Vec<_>>();
    Array1::from_vec(values)
}

fn quadratic_objective(x: &Array1<f64>) -> f64 { x.iter().map(|value| value * value).sum() }

fn quadratic_gradient(x: &Array1<f64>) -> Array1<f64> { x.mapv(|value| 2.0 * value) }

fn gradient_descent_manual(
    start: &Array1<f64>,
    learning_rate: f64,
    max_iterations: usize,
) -> Array1<f64> {
    let mut x = start.clone();
    for _ in 0..max_iterations {
        let grad = quadratic_gradient(&x);
        x = &x - &(learning_rate * &grad);
    }
    x
}

fn adam_manual(start: &Array1<f64>, config: &AdamConfig) -> Array1<f64> {
    let mut x = start.clone();
    let mut m = Array1::<f64>::zeros(start.len());
    let mut v = Array1::<f64>::zeros(start.len());
    let mut beta1_t = 1.0_f64;
    let mut beta2_t = 1.0_f64;

    for _ in 0..config.max_iterations {
        let grad = quadratic_gradient(&x);
        m = config.beta1 * &m + (1.0 - config.beta1) * &grad;
        v = config.beta2 * &v + (1.0 - config.beta2) * grad.mapv(|value| value * value);

        beta1_t *= config.beta1;
        beta2_t *= config.beta2;
        let m_hat = &m / (1.0 - beta1_t);
        let v_hat = &v / (1.0 - beta2_t);

        let step = m_hat / v_hat.mapv(|value| value.sqrt() + config.epsilon);
        x = &x - &(config.learning_rate * step);
    }

    x
}

fn benchmark_optimization(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("optimization_nabled_ndarray");
        for dim in [64_usize, 128, 256] {
            let start = random_start(dim);
            let id = format!("vector-{dim}x1");

            let sgd_config =
                SGDConfig { learning_rate: 1e-2, max_iterations: 200, tolerance: 1e-8 };
            _ = group.bench_with_input(
                BenchmarkId::new("gradient_descent", &id),
                &dim,
                |bench, _| {
                    bench.iter(|| {
                        optimization::gradient_descent(
                            black_box(&start),
                            black_box(quadratic_objective),
                            black_box(quadratic_gradient),
                            black_box(&sgd_config),
                        )
                    });
                },
            );

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
                    optimization::adam(
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

    {
        let mut competitor_group = c.benchmark_group("optimization_competitor_manual");
        for dim in [64_usize, 128, 256] {
            let start = random_start(dim);
            let id = format!("vector-{dim}x1");
            let sgd_config =
                SGDConfig { learning_rate: 1e-2, max_iterations: 200, tolerance: 1e-8 };
            let adam_config = AdamConfig {
                learning_rate:  1e-2,
                beta1:          0.9,
                beta2:          0.999,
                epsilon:        1e-8,
                max_iterations: 200,
                tolerance:      1e-8,
            };

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("gradient_descent_manual", &id),
                &dim,
                |bench, _| {
                    bench.iter(|| {
                        gradient_descent_manual(
                            black_box(&start),
                            black_box(sgd_config.learning_rate),
                            black_box(sgd_config.max_iterations),
                        )
                    });
                },
            );

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("adam_manual", &id),
                &dim,
                |bench, _| {
                    bench.iter(|| adam_manual(black_box(&start), black_box(&adam_config)));
                },
            );
        }
        competitor_group.finish();
    }
}

criterion_group!(benches, benchmark_optimization);
criterion_main!(benches);

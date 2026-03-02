use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::{DistributedConfig, accelerator};
use ndarray::Array2;
use rand::RngExt;

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((rows, cols), values).expect("shape should match")
}

fn manual_matmat(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.ncols()));
    for row in 0..left.nrows() {
        for inner in 0..left.ncols() {
            let lhs = left[[row, inner]];
            for col in 0..right.ncols() {
                output[[row, col]] += lhs * right[[inner, col]];
            }
        }
    }
    output
}

fn benchmark_accelerator(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("accelerator_nabled_ndarray");
        for size in [64_usize, 128, 192] {
            let left = random_matrix(size, size);
            let right = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(
                BenchmarkId::new("matmat_serial", &id),
                &size,
                |bench, _| {
                    bench.iter(|| accelerator::matmat_serial(black_box(&left), black_box(&right)));
                },
            );

            _ = group.bench_with_input(
                BenchmarkId::new("matmat_distributed", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        accelerator::matmat_distributed(
                            black_box(&left),
                            black_box(&right),
                            DistributedConfig { workers: 4, chunk_rows: 32 },
                        )
                    });
                },
            );

            _ = group.bench_with_input(
                BenchmarkId::new("matmat_distributed_tiled", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        accelerator::matmat_distributed_tiled(
                            black_box(&left),
                            black_box(&right),
                            4,
                            32,
                            32,
                        )
                    });
                },
            );

            #[cfg(feature = "accelerator-rayon")]
            {
                _ = group.bench_with_input(
                    BenchmarkId::new("matmat_accelerated", &id),
                    &size,
                    |bench, _| {
                        bench.iter(|| {
                            accelerator::matmat_accelerated(black_box(&left), black_box(&right))
                        });
                    },
                );
            }
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group("accelerator_competitor_manual");
        for size in [64_usize, 128, 192] {
            let left = random_matrix(size, size);
            let right = random_matrix(size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(
                BenchmarkId::new("matmat_manual", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        drop(black_box(manual_matmat(black_box(&left), black_box(&right))));
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, benchmark_accelerator);
criterion_main!(benches);

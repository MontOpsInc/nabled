use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled::tensor;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use rand::RngExt;

fn random_cube(batch: usize, rows: usize, cols: usize) -> Array3<f64> {
    let mut rng = rand::rng();
    let values = (0..batch * rows * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array3::from_shape_vec((batch, rows, cols), values).expect("shape should match")
}

fn random_batch_vectors(batch: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let values = (0..batch * cols).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    Array2::from_shape_vec((batch, cols), values).expect("shape should match")
}

fn random_tensor(shape: &[usize]) -> ArrayD<f64> {
    let total_size = shape.iter().product::<usize>();
    let mut rng = rand::rng();
    let values = (0..total_size).map(|_| rng.random_range(-1.0..1.0)).collect::<Vec<_>>();
    ArrayD::from_shape_vec(IxDyn(shape), values).expect("shape should match")
}

fn benchmark_tensor(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("tensor_nabled_ndarray");
        for size in [32_usize, 64, 128] {
            let cube = random_cube(16, size, size);
            let vectors = random_batch_vectors(16, size);
            let right = random_cube(16, size, size);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(BenchmarkId::new("cube_matvec", &id), &size, |bench, _| {
                bench.iter(|| tensor::cube_matvec(black_box(&cube), black_box(&vectors)));
            });

            _ = group.bench_with_input(BenchmarkId::new("cube_matmat", &id), &size, |bench, _| {
                bench.iter(|| tensor::cube_matmat(black_box(&cube), black_box(&right)));
            });
        }

        for size in [32_usize, 64] {
            let tensor_value = random_tensor(&[8, size, size]);
            let left_contract = random_tensor(&[4, size, size / 2]);
            let right_contract = random_tensor(&[3, size / 2, size]);
            let left_batched = random_tensor(&[4, 4, size, size / 2]);
            let right_batched = random_tensor(&[4, 4, size / 2, size]);
            let id = format!("square-{size}x{size}");

            _ = group.bench_with_input(
                BenchmarkId::new("sum_last_axis", &id),
                &size,
                |bench, _| {
                    bench.iter(|| tensor::sum_last_axis(black_box(&tensor_value)));
                },
            );

            _ = group.bench_with_input(
                BenchmarkId::new("batched_dot_last_axis", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        tensor::batched_dot_last_axis(
                            black_box(&tensor_value),
                            black_box(&tensor_value),
                        )
                    });
                },
            );

            _ = group.bench_with_input(
                BenchmarkId::new("contract_axes", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        tensor::contract_axes(
                            black_box(&left_contract),
                            black_box(&right_contract),
                            black_box(&[2]),
                            black_box(&[1]),
                        )
                    });
                },
            );

            _ = group.bench_with_input(
                BenchmarkId::new("batched_matmul_last_two", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        tensor::batched_matmul_last_two(
                            black_box(&left_batched),
                            black_box(&right_batched),
                        )
                    });
                },
            );
        }

        group.finish();
    }

    {
        let mut competitor_group = c.benchmark_group("tensor_competitor_manual");
        for size in [32_usize, 64, 128] {
            let cube = random_cube(16, size, size);
            let vectors = random_batch_vectors(16, size);
            let id = format!("square-{size}x{size}");

            _ = competitor_group.bench_with_input(
                BenchmarkId::new("cube_matvec_naive", &id),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let mut output = Array2::<f64>::zeros((cube.dim().0, cube.dim().1));
                        for batch in 0..cube.dim().0 {
                            for row in 0..cube.dim().1 {
                                let mut sum = 0.0_f64;
                                for col in 0..cube.dim().2 {
                                    sum += cube[[batch, row, col]] * vectors[[batch, col]];
                                }
                                output[[batch, row]] = sum;
                            }
                        }
                        drop(black_box(output));
                    });
                },
            );
        }
        competitor_group.finish();
    }
}

criterion_group!(benches, benchmark_tensor);
criterion_main!(benches);

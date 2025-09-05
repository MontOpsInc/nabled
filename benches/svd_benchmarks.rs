use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_linalg::svd::{nalgebra_svd, ndarray_svd};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::Rng;

fn generate_random_matrix_nalgebra(size: usize) -> DMatrix<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    
    for _ in 0..size * size {
        data.push(rng.gen_range(-1.0..1.0));
    }
    
    DMatrix::from_row_slice(size, size, &data)
}

fn generate_random_matrix_ndarray(size: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    
    for _ in 0..size * size {
        data.push(rng.gen_range(-1.0..1.0));
    }
    
    Array2::from_shape_vec((size, size), data).unwrap()
}

fn benchmark_nalgebra_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("nalgebra_svd");
    
    for size in [10, 50, 100, 200].iter() {
        let matrix = generate_random_matrix_nalgebra(*size);
        
        group.bench_with_input(BenchmarkId::new("full_svd", size), size, |b, _| {
            b.iter(|| {
                nalgebra_svd::compute_svd(black_box(&matrix))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("truncated_svd_k=5", size), size, |b, _| {
            b.iter(|| {
                nalgebra_svd::compute_truncated_svd(black_box(&matrix), 5)
            })
        });
    }
    
    group.finish();
}

fn benchmark_ndarray_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_svd");
    
    for size in [10, 50, 100, 200].iter() {
        let matrix = generate_random_matrix_ndarray(*size);
        
        group.bench_with_input(BenchmarkId::new("full_svd", size), size, |b, _| {
            b.iter(|| {
                ndarray_svd::compute_svd(black_box(&matrix))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("truncated_svd_k=5", size), size, |b, _| {
            b.iter(|| {
                ndarray_svd::compute_truncated_svd(black_box(&matrix), 5)
            })
        });
    }
    
    group.finish();
}

fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_comparison");
    
    for size in [10, 50, 100].iter() {
        let nalgebra_matrix = generate_random_matrix_nalgebra(*size);
        let ndarray_matrix = generate_random_matrix_ndarray(*size);
        
        group.bench_with_input(BenchmarkId::new("nalgebra", size), size, |b, _| {
            b.iter(|| {
                nalgebra_svd::compute_svd(black_box(&nalgebra_matrix))
            })
        });
        
        group.bench_with_input(BenchmarkId::new("ndarray", size), size, |b, _| {
            b.iter(|| {
                ndarray_svd::compute_svd(black_box(&ndarray_matrix))
            })
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_nalgebra_svd, benchmark_ndarray_svd, benchmark_comparison);
criterion_main!(benches);

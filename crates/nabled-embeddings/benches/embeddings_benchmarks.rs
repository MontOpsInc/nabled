use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nabled_embeddings::{Metric, query_corpus_scores, rerank};
use ndarray::{Array1, Array2};

/// Deterministic pseudo-random matrix so benches are reproducible without a RNG dependency.
// The shifted value fits in 53 bits, so the f64 cast is exact for bench data generation.
#[allow(clippy::cast_precision_loss)]
fn synthetic_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
        bits.mul_add(2.0, -1.0)
    };
    let data: Vec<f64> = (0..rows * cols).map(|_| next()).collect();
    Array2::from_shape_vec((rows, cols), data).expect("shape matches data length")
}

fn bench_query_corpus_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings_query_corpus_scores");
    let dim = 256_usize;
    let queries = synthetic_matrix(8, dim, 1);
    for n_corpus in [256_usize, 1024, 4096] {
        let corpus = synthetic_matrix(n_corpus, dim, 2);
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let id = format!("{metric:?}-n{n_corpus}");
            _ = group.bench_with_input(
                BenchmarkId::new("scores", &id),
                &n_corpus,
                |bench, _| {
                    bench.iter(|| {
                        query_corpus_scores(black_box(&queries), black_box(&corpus), metric)
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_rerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings_rerank");
    let dim = 256_usize;
    let query: Array1<f64> = synthetic_matrix(1, dim, 3).row(0).to_owned();
    for n_candidates in [256_usize, 1024, 4096] {
        let candidates = synthetic_matrix(n_candidates, dim, 4);
        let id = format!("cosine-n{n_candidates}-k10");
        _ = group.bench_with_input(
            BenchmarkId::new("rerank", &id),
            &n_candidates,
            |bench, _| {
                bench.iter(|| {
                    rerank(
                        black_box(&query.view()),
                        black_box(&candidates.view()),
                        10,
                        Metric::Cosine,
                    )
                });
            },
        );
    }
    group.finish();
}

fn benchmark_embeddings(c: &mut Criterion) {
    bench_query_corpus_scores(c);
    bench_rerank(c);
}

criterion_group!(benches, benchmark_embeddings);
criterion_main!(benches);

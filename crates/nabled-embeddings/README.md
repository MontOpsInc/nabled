# nabled-embeddings

A lightweight, ndarray-native, Arrow-zero-copy compute and rerank layer for embedding vectors -
bring vectors from any model, compute exactly, deploy anywhere.

`nabled-embeddings` is the exact rerank/compute step that sits **next to** a vector store, not a
vector database. It composes `nabled-linalg` vector kernels and `nabled-ml` PCA into the
post-embedding numerics a retrieval pipeline needs.

## Install

```toml
[dependencies]
nabled-embeddings = "0.0.11"
```

## Key modules

1. `normalize`: row-wise L2 normalization (`normalize_rows` / `_view` / `_into`).
2. `similarity`: `query_corpus_scores` plus the `Metric` enum (`Cosine`, `Dot`, `L2`).
3. `topk` / `rerank`: direction-aware partial-selection `top_k`, the `rerank` entrypoint, and the
   id-carrying `rerank_with_ids` / `batch_rerank_with_ids` (`NeighborWithId` + `attach_ids`).
4. `knn`: exact `brute_force_knn` for small corpora, evaluation, and golden tests.
5. `cache`: `CorpusWorkspace` — build-once / query-many corpus with cached cosine norms.
6. `metrics`: offline ranking quality (`recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`,
   `ndcg_at_k`).
7. `mmr`: Maximal Marginal Relevance diversification (`mmr`, `lambda` in `[0, 1]`).
8. `quantize`: int8 row quantization (`QuantizedMatrix`, `quantize_rows`, `dequantize`,
   `query_corpus_scores_quantized`).
9. `compress`: PCA fit + `compress` projection for dimensionality reduction.

## Why use it

- **Lightweight:** pure Rust + ndarray, minimal dependencies, no C++/FAISS/CUDA build, embeddable.
- **Efficient in-pipeline:** zero-copy Arrow/Lance ingress at the facade, view-based APIs, `*_into`
  allocation control, and no Python interpreter in the Rust path.
- **Exact, BLAS-backed** scoring with partial-selection top-k.

This crate does **not** claim to be faster than FAISS; exact brute-force scoring is memory/BLAS
bound and competitive at parity, not dominant. Any performance number should come from the
criterion bench (`benches/embeddings_benchmarks.rs`), not assertion.

## Bring-your-own-vectors

Any encoder's dense float vectors plug in unchanged (OpenAI, Cohere, Sentence-BERT, CLIP, custom)
because the crate depends only on shape and dtype, never the model. Two rules the math cannot
enforce:

1. Query and corpus must come from the **same model** and share the same `dim`.
2. Pick the `Metric` to match how the model was trained. Dot on un-normalized vectors favors
   larger-norm rows by design.

## Metric selection

| Metric   | Polarity         | Delegates to                       | Use when |
| -------- | ---------------- | ---------------------------------- | -------- |
| `Cosine` | higher is better | `vector::pairwise_cosine_similarity` | default; angle-only similarity |
| `Dot`    | higher is better | `matrix::matmat(queries, corpusᵀ)`   | MIPS-style models; normalized inputs (= cosine) |
| `L2`     | lower is better  | `vector::pairwise_l2_distance`       | Euclidean models; same ranking as normalized cosine |

## Crate graph

- **Depends on:** `nabled-core`, `nabled-linalg`, `nabled-ml`.
- **Used by:** facade `nabled` via the `embeddings` feature; `pynabled` Python bindings.

## Quick example

```rust
use nabled_embeddings::{Metric, rerank};
use ndarray::arr2;

let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
let query = arr2(&[[1.0_f64, 0.0]]);
let top = rerank(&query.row(0), &corpus.view(), 2, Metric::Cosine)?;
assert_eq!(top[0].index, 0);
# Ok::<(), nabled_embeddings::EmbeddingError>(())
```

Switch metrics by passing `Metric::Dot` (maximum inner product) or `Metric::L2` (Euclidean
distance); selection (`top_k`) follows each metric's ranking polarity automatically.

## Retrieval extensions

**Rerank with ids** — thread recall-stage global ids through the exact rerank (single + batch). The
batch path composes `brute_force_knn` (there is intentionally no bare `batch_rerank`); the new value
is id mapping.

```rust
use nabled_embeddings::{Metric, rerank_with_ids};
use ndarray::{arr1, arr2};

let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
let query = arr1(&[1.0_f64, 0.0]);
let ids = [100_i64, 200, 300];
let top = rerank_with_ids(&query.view(), &candidates.view(), &ids, 2, Metric::Cosine)?;
assert_eq!(top[0].id, 100);
# Ok::<(), nabled_embeddings::EmbeddingError>(())
```

**Corpus workspace reuse** — `CorpusWorkspace` precomputes the corpus's metric state (cosine row
norms) once and reuses it across many queries against a static corpus; it owns the prepared corpus.

```rust
use nabled_embeddings::{CorpusWorkspace, Metric};
use ndarray::arr2;

let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
let ws = CorpusWorkspace::build(&corpus.view(), Metric::Cosine)?;
let query = arr2(&[[1.0_f64, 0.0]]);
let top = ws.rerank_with(&query.row(0), 2)?;
# Ok::<(), nabled_embeddings::EmbeddingError>(())
```

**MMR** — diversity-aware rerank; `lambda = 1.0` reproduces plain `rerank`, lower values diversify.

```rust
use nabled_embeddings::{Metric, mmr};
use ndarray::{arr1, arr2};

let candidates = arr2(&[[1.0_f64, 0.0], [0.99, 0.01], [0.0, 1.0]]);
let query = arr1(&[1.0_f64, 0.0]);
let diversified = mmr(&query.view(), &candidates.view(), 2, 0.5, Metric::Cosine)?;
# Ok::<(), nabled_embeddings::EmbeddingError>(())
```

**int8 quantization** — per-row symmetric int8 (`scale = amax / 127`), ~4× smaller storage; scoring
is dequantize-then-existing-kernel, so results approximate the `f32` path within tolerance.

```rust
use nabled_embeddings::{Metric, quantize_rows};
use ndarray::arr2;

let corpus = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
let q = quantize_rows(&corpus.view())?;
let queries = quantize_rows(&arr2(&[[1.0_f32, 0.0]]).view())?;
let scores = queries.query_corpus_scores_quantized(&q, Metric::Cosine)?;
# Ok::<(), nabled_embeddings::EmbeddingError>(())
```

**Evaluation metrics** — pure ranking math over retrieved-id lists vs ground-truth sets (binary
relevance): `recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `ndcg_at_k`.

**Arrow-native rerank wrappers** — when embeddings live in Arrow columns, the `nabled` facade
exposes `arrow_query_corpus_scores` / `arrow_rerank` / `arrow_normalize_rows` /
`arrow_brute_force_knn` (feature `embeddings` + `arrow`), zero-copy from `FixedSizeListArray`. See
`docs/EMBEDDINGS.md`.

## Performance

Benchmark before claiming. Methodology, reproduction steps, and the honest exact-vs-exact framing
against `numpy` cosine + argsort live in `docs/BENCHMARKS.md`. Numbers must come from the criterion
bench (`benches/embeddings_benchmarks.rs`), not assertion; no "faster than FAISS" claims without
disclosed measurements.

## Non-goals

- No ANN index (HNSW/IVF) - use a vector store for recall, then rerank candidates here.
- No storage or persistence.
- No model inference, tokenizers, or ONNX/candle.

## Optional features

1. `blas`, `lapack-provider`: forwarded to `nabled-linalg`/`nabled-ml`.
2. `openblas-system`, `openblas-static`, `netlib-system`, `netlib-static`, `magma-system`.

## Docs

1. API docs: <https://docs.rs/nabled-embeddings>
2. Workspace repo: <https://github.com/MontOpsInc/nabled>
3. Domain guide: `docs/EMBEDDINGS.md`
4. Facade feature: `nabled` with `embeddings`

# Embeddings Benchmarks

This document describes **how** to benchmark `nabled-embeddings` and how to position the results
honestly. It deliberately contains **no measured numbers**: per the crate's positioning rule, every
public performance claim must come from a reproducible benchmark on disclosed hardware, not from
assertion. Placeholder tables are provided below for you to fill in from your own runs.

## Positioning rule (read first)

`nabled-embeddings` is the **exact** rerank / compute step that sits next to a vector store, not an
ANN engine. Exact brute-force scoring is memory- and BLAS-bound: it is expected to be *competitive
at parity*, not dominant. There is **no "faster than FAISS" claim** anywhere, and none may be added
without published benchmark numbers, the exact hardware/version disclosure below, and an
apples-to-apples methodology. ANN engines (FAISS/HNSW/IVF) solve a different problem (approximate
recall over large corpora); the honest comparison target for this crate is an **exact** baseline
such as `numpy` cosine + `argsort` over the same candidate set.

## What the benches cover

The criterion suite lives in
[`crates/nabled-embeddings/benches/embeddings_benchmarks.rs`](../crates/nabled-embeddings/benches/embeddings_benchmarks.rs)
and uses a deterministic, seedless pseudo-random matrix generator so runs are reproducible without
an RNG dependency. Current groups:

| Bench group | What it measures | Shapes |
| ----------- | ---------------- | ------ |
| `embeddings_query_corpus_scores` | full score matrix over `Metric { Cosine, Dot, L2 }` | `dim = 256`, `queries = 8`, `n_corpus ∈ {256, 1024, 4096}` |
| `embeddings_rerank` | single-query score + top-`k` (`k = 10`) | `dim = 256`, `n_candidates ∈ {256, 1024, 4096}` |
| `embeddings_corpus_workspace_reuse` | stateless recompute vs `CorpusWorkspace` reuse | `dim = 256`, `n_corpus = 4096`, `n_queries = 32` |
| `embeddings_quantize_rows` | int8 row-quantization throughput | `dim = 256`, `n_rows ∈ {1024, 4096}` |

## How to reproduce

```bash
# All embeddings benches (release, criterion):
cargo bench -p nabled-embeddings --bench embeddings_benchmarks

# A single group (criterion filter), e.g. the workspace-reuse comparison:
cargo bench -p nabled-embeddings --bench embeddings_benchmarks -- corpus_workspace_reuse
```

For BLAS-backed numbers, build with a provider feature (matching the workspace quality gates), e.g.:

```bash
cargo bench -p nabled-embeddings --bench embeddings_benchmarks --features openblas-system
```

Criterion writes detailed reports under `target/criterion/`. Record the **median** (and report the
interval) for each shape, not a single sample.

## Comparison methodology: vs `numpy` cosine + argsort

For an honest exact-vs-exact comparison, benchmark the same operation in NumPy over the **same**
candidate set, dtype, and `k`. Keep the comparison apples-to-apples:

1. Use identical `f32` inputs and the same metric semantics (cosine = normalize then dot).
2. Use the same `k` and a partial selection on the NumPy side (`np.argpartition` + a small sort of
   the retained slice) — comparing against a full `np.argsort` over-penalizes NumPy and inflates any
   nabled advantage. Report both if you want to show the selection effect.
3. Warm up, then time the steady state; exclude one-time allocation/setup unless that is explicitly
   what you are measuring.
4. Pin threads consistently (BLAS thread count, `rayon` pool) across both sides and disclose them.

Sketch of the NumPy exact baseline (for cosine + top-`k`):

```python
import numpy as np

def numpy_cosine_topk(query, corpus, k):
    q = query / np.linalg.norm(query)
    c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    scores = c @ q
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
```

## Hardware / version disclosure (fill in per run)

Every reported number must be accompanied by this disclosure block:

| Field | Value |
| ----- | ----- |
| CPU (model, cores/threads) | (pending) |
| RAM | (pending) |
| OS / kernel | (pending) |
| Rust toolchain (`rustc -V`) | (pending) |
| `nabled` / `nabled-embeddings` version | (pending) |
| BLAS provider + version (e.g. OpenBLAS) | (pending) |
| BLAS thread count / `rayon` threads | (pending) |
| NumPy version + BLAS backend (comparison) | (pending) |
| criterion version | (pending) |

## Results (placeholders — fill from your runs)

> The cells below are intentionally empty. Do **not** populate them with estimates; only paste
> measured medians from a disclosed run.

### `query_corpus_scores` (median per call)

| Metric | `n_corpus = 256` | `1024` | `4096` |
| ------ | --------------- | ------ | ------ |
| Cosine | (pending) | (pending) | (pending) |
| Dot | (pending) | (pending) | (pending) |
| L2 | (pending) | (pending) | (pending) |

### `rerank` (single query, `k = 10`, median per call)

| `n_candidates = 256` | `1024` | `4096` |
| -------------------- | ------ | ------ |
| (pending) | (pending) | (pending) |

### Corpus workspace reuse (32 queries vs 4096×256 corpus, median per batch)

| Path | Median |
| ---- | ------ |
| stateless (recompute corpus norms per query) | (pending) |
| `CorpusWorkspace` (norms cached once) | (pending) |

### int8 quantization throughput (median per call)

| `n_rows = 1024` | `4096` |
| --------------- | ------ |
| (pending) | (pending) |

### Exact-vs-exact: nabled vs NumPy cosine + argpartition

| Shape | nabled (median) | NumPy (median) | Notes |
| ----- | --------------- | -------------- | ----- |
| (pending) | (pending) | (pending) | same dtype / `k` / threads |

## Reporting checklist

- [ ] Disclosure block filled in completely.
- [ ] Medians (with interval) recorded, not single samples.
- [ ] Comparison baseline is **exact** (NumPy cosine + partial selection), not an ANN engine.
- [ ] No "faster than FAISS" framing.
- [ ] Same dtype, `k`, metric semantics, and thread settings on both sides.

# Release v0.0.11

## Embeddings (`nabled-embeddings`)

First crates.io release of the embedding retrieval compute crate and facade/Python wiring (landed
on `main` after the `v0.0.10` tag; `0.0.10` on crates.io did not include this crate).

**Rust (`nabled-embeddings` / `nabled` feature `embeddings`):**

- Row normalization, `query_corpus_scores` over `Metric { Cosine, Dot, L2 }`, direction-aware
  `top_k` / `rerank`, exact `brute_force_knn`, PCA `fit_pca` / `compress`
- Id-aware `rerank_with_ids` / `batch_rerank_with_ids`
- `CorpusWorkspace` for build-once / query-many cosine corpus reuse
- Offline eval metrics: `recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `ndcg_at_k`
- MMR diversity rerank and per-row int8 quantization (`QuantizedMatrix`)
- Facade Arrow wrappers (`embeddings` + `arrow`): `arrow_query_corpus_scores`, `arrow_rerank`,
  `arrow_normalize_rows`, `arrow_brute_force_knn`

**Python (`pynabled.embeddings`, default wheels):**

- `normalize_rows`, `query_corpus_scores`, `rerank`, `brute_force_knn`, `compress_pca`
- `rerank_with_ids`, `batch_rerank_with_ids`, `CorpusWorkspace`, eval metrics, `mmr`, quantization
- Arrow helpers under `pynabled.arrow`
- Example: `python/examples/embeddings/lance_rerank.py` (LanceDB is example-only)

See `docs/EMBEDDINGS.md`, `docs/BENCHMARKS.md`, and `crates/nabled-embeddings/README.md`.

```toml
nabled = { version = "0.0.11", features = ["embeddings"] }
nabled-embeddings = "0.0.11"
```

## Release automation

- Post-publish co-owner ensure via `scripts/crates_io_ensure_coowners.sh` in `release.yml`
- `prepare-release` now bumps `nabled-embeddings` workspace dependency pins

## Maintainer actions

1. `just tag-release 0.0.11` after this PR merges
2. `just tag-pypi-release`
3. Verify Owners on crates.io if the co-owner step logs any failures

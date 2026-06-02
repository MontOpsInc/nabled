# Release v0.0.10

## Embeddings (`nabled-embeddings`)

New publishable crate and opt-in facade feature for embedding retrieval compute — the exact
rerank step next to a vector store, not a vector database (no ANN index, storage, or model
inference).

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
nabled = { version = "0.0.10", features = ["embeddings"] }
# or standalone:
nabled-embeddings = "0.0.10"
```

## Release automation

- Post-publish co-owner ensure via `scripts/crates_io_ensure_coowners.sh` in `release.yml`
  (adds `NiklausParcell` on all workspace crates idempotently)
- Manual repair workflow reuses the same script (`crates-io-add-owner`)

## Physical AI crates.io publish fix

- Expanded README and `description` metadata for all six Physical AI domain crates
- Hardened `.github/workflows/release.yml` with rate-limit pacing, publish resume, 429 retry, and
  split GitHub Release job
- Updated `docs/PUBLISH_CHECKLIST.md` with publish order, rate limits, and co-ownership policy

## Maintainer actions

1. `just tag-release 0.0.10` (publishes crates including `nabled-embeddings`; co-owners ensured
   automatically)
2. After Release workflow completes, `just tag-pypi-release`
3. Verify Owners on crates.io (George + NiklausParcell) if the co-owner step logged any failures

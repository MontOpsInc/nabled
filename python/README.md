# pynabled

`pynabled` is the Python package for `nabled`, an ndarray-native numerical library for dense
linear algebra, decompositions, sparse workflows, tensor routines, and selected ML/statistics
operations.

## Install

Install the published wheel:

```bash
pip install pynabled
```

`pynabled` requires Python 3.10+ and NumPy.

## Optional features

Published PyPI wheels use the default feature set.

Default wheels include the Rust `arrow` feature. Python Arrow workflows still require `pyarrow`
at runtime.

Optional provider and accelerator support are source-build workflows that use the same Cargo
feature names as the Rust facade:

- `openblas-system`
- `openblas-static`
- `netlib-system`
- `netlib-static`
- `magma-system`
- `accelerator-rayon`
- `accelerator-wgpu`

For `pip` / `uv` source builds, the package exposes friendly build settings:

```bash
PYNABLED_PROVIDER="openblas-system magma-system" python -m pip install --no-binary pynabled pynabled
PYNABLED_ACCELERATORS=rayon python -m pip install --no-binary pynabled pynabled
```

You can pass the same knobs through frontend config settings with
`pynabled-provider`, `pynabled-accelerators`, and `pynabled-features`.

Builds can be inspected at runtime with `pynabled.build_features()`.

Build guide:
<https://github.com/MontOpsInc/nabled/blob/main/BUILD.md>

## Quick example

```python
import numpy as np
import pynabled

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
result = pynabled.svd_decompose(a)

print(result.singular_values)
```

## What it includes

- Dense vector and matrix kernels over `numpy.ndarray`
- Decompositions including SVD, QR, LU, Cholesky, eigen, Schur, and polar
- Sparse matrix carriers and solver/preconditioner workflows
- Tensor decomposition and reconstruction helpers
- PCA, regression, iterative solvers, Jacobian helpers, and optimization configs
- `pynabled.embeddings`: a lightweight, ndarray-native, Arrow-zero-copy compute and rerank layer
  for embedding vectors (see below)
- Optional Arrow and `ndarrow` interop when built with `arrow`

## Embeddings

`pynabled.embeddings` is a lightweight, ndarray-native, Arrow-zero-copy compute and rerank layer
for embedding vectors — bring vectors from any model, compute exactly, deploy anywhere. It is the
exact rerank/compute step that sits **next to** a vector store (e.g. LanceDB), not a vector
database: no ANN index, no storage, no model inference. Bring dense float vectors from any encoder
(OpenAI, Cohere, Sentence-BERT, CLIP, custom) and compute exact scores here. Two rules the math
cannot enforce: query and corpus must be the **same model and `dim`**, and you pick the metric to
match how the model was trained (`"cosine"` default, `"dot"` for MIPS-style models, `"l2"` where
applicable; dot on un-normalized vectors favors larger-norm rows by design).

The headline pattern is "ANN narrows, nabled reranks exactly," over a pure Arrow interchange so the
same entrypoint reranks any batch producer's top-N candidates (LanceDB shown):

```python
import lance              # example-only: pip install lance
import numpy as np
import pynabled

query = ...               # np.ndarray, shape (dim,), from your encoder
dataset = lance.dataset("corpus.lance")
candidates = dataset.to_table(nearest={"column": "vector", "q": query, "k": 50})
cand_vecs = np.stack([np.asarray(v, np.float32) for v in candidates["vector"].to_pylist()])

# The SAME entrypoint reranks candidates from any Arrow batch producer.
top = pynabled.embeddings.rerank(query, cand_vecs, k=10, metric="cosine")
print(top.indices, top.scores)
```

`lance` and `sentence-transformers` are **example-only** dependencies; they are not required by the
package or its tests. A runnable end-to-end script lives at
`python/examples/embeddings/lance_rerank.py`. The full embeddings surface is `normalize_rows`,
`query_corpus_scores`, `rerank`, `brute_force_knn`, and `compress_pca`, plus id-carrying
`rerank_with_ids` / `batch_rerank_with_ids` (returning `IdNeighbors`), a build-once/query-many
`CorpusWorkspace`, offline eval metrics (`recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`,
`ndcg_at_k`), MMR diversity rerank (`mmr`), and int8 quantization (`quantize_rows`, `dequantize`,
`query_corpus_scores_quantized`, `QuantizedMatrix`). Arrow-native rerank wrappers ship under
`pynabled.arrow`. See `docs/EMBEDDINGS.md` for the full guide and `docs/BENCHMARKS.md` for
benchmarking methodology.

## API and behavior

`numpy.ndarray` is the canonical CPU array carrier. Where the Rust API admits borrowed views,
`pynabled` preserves that contract at the Python boundary instead of forcing extra copies.

Structured results are returned as typed Python objects such as `SvdResult`, `QrResult`,
`LuResult`, `PcaResult`, and tensor-specific result types.

Some higher-level convenience APIs still materialize owned arrays internally where the current
Rust API shape requires it. Callback-driven Jacobian and optimization helpers also cross back into
Python on each evaluation, so they do not have the same performance contract as the direct
array-in/array-out kernels.

## Project links

- Repository: <https://github.com/MontOpsInc/nabled>
- Python package docs: <https://github.com/MontOpsInc/nabled/blob/main/python/README.md>
- Build guide: <https://github.com/MontOpsInc/nabled/blob/main/BUILD.md>
- Changelog: <https://github.com/MontOpsInc/nabled/blob/main/CHANGELOG.md>

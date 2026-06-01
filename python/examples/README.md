# pynabled Examples

End-to-end examples using pynabled.

## Structure

- **pca/** – Principal component analysis
  - `iris_pca.py` – Simple Iris PCA with 2D projection and explained variance
  - `pca_full_analysis.py` – Full PCA pipeline on synthetic data: scree plot, cumulative variance, reconstruction error, 2D projection. Uses make_classification (unit-scale features) for reliable SVD convergence.

- **regression/** – Linear regression
  - `linear_regression.py` – Fit on diabetes dataset, predicted vs actual, residual plot

- **svd/** – SVD compression
  - `svd_compression.py` – Low-rank approximation of a 16×16 image, reconstruction error vs rank, visual comparison

- **similarity/** – Similarity search
  - `similarity_search.py` – k-NN via pairwise cosine similarity on digit embeddings

- **sparse/** – Sparse linear solvers
  - `sparse_solve.py` – PCG solve for sparse SPD system (diagonal example)

- **arrow/** – PyArrow bridge (requires a default build and `pyarrow`)
  - `arrow_svd.py` – SVD on PyArrow FixedSizeListArray matrix, round-trip verification

- **embeddings/** – Embedding retrieval (requires a default build)
  - `lance_rerank.py` – LanceDB ANN → nabled exact rerank over a pure Arrow interchange. Uses the **example-only** packages `lance` (required) and `sentence-transformers` (optional). These are NOT part of the crate graph, `pyproject.toml`, or the CI/python-quality gate.

## Requirements

- numpy, pynabled, scikit-learn, scipy, matplotlib
- For arrow examples: pyarrow and a default pynabled build
- For `embeddings/lance_rerank.py` only: `pip install lance sentence-transformers` (example-only, never required by the package or its tests)

## Run

From repo root:

```bash
python python/examples/pca/iris_pca.py
python python/examples/pca/pca_full_analysis.py
python python/examples/regression/linear_regression.py
python python/examples/svd/svd_compression.py
python python/examples/similarity/similarity_search.py
python python/examples/sparse/sparse_solve.py
```

Arrow examples (install `pyarrow` first):

```bash
python python/examples/arrow/arrow_svd.py
```

Embeddings example (install the example-only deps first):

```bash
pip install lance sentence-transformers
python python/examples/embeddings/lance_rerank.py
```

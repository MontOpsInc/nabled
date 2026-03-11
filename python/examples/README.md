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

## Requirements

- numpy, pynabled, scikit-learn, scipy, matplotlib

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

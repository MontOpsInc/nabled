//! Lightweight, ndarray-native, Arrow-zero-copy compute and rerank layer for embedding vectors.
//!
//! Bring vectors from any model, compute exactly, deploy anywhere. `nabled-embeddings` composes
//! existing `nabled-linalg` vector kernels and `nabled-ml` PCA into the post-embedding numerics a
//! retrieval pipeline needs: normalize, score, rerank, exact brute-force kNN, and PCA compression.
//!
//! # What this is (and is not)
//!
//! This is the **exact rerank / compute layer that sits next to a vector store**, not a vector
//! database. It is deliberately:
//!
//! - **bring-your-own-vectors:** it accepts `Array2<T>` / `ArrayView2<T>` (`f32`/`f64` via
//!   [`NabledReal`](nabled_core::scalar::NabledReal)). Any encoder's dense float vectors plug in
//!   unchanged because the crate depends only on shape and dtype, never the model.
//! - **numerics-only:** no model inference, no tokenizers, no ONNX/candle.
//! - **Arrow-free in the core:** Arrow/Lance ingress lives at the `nabled::arrow` facade and in
//!   `pynabled`, never here.
//!
//! Non-goals: it is **not** an ANN/vector-search engine (no HNSW/IVF index), **not** storage, and
//! **not** a FAISS/HNSW replacement. Use an ANN index for recall, then rerank its candidates here.
//!
//! # Two rules the math cannot enforce
//!
//! 1. Query and corpus must come from the **same model** and share the same `dim`.
//! 2. You pick the [`Metric`] to match how the model was trained: [`Metric::Cosine`] (the default),
//!    [`Metric::Dot`] for maximum-inner-product (MIPS) models, or [`Metric::L2`] where applicable.
//!    Dot on un-normalized vectors favors larger-norm rows by design.
//!
//! # Feature flags
//!
//! 1. `blas`: forwards BLAS acceleration to `nabled-linalg`/`nabled-ml`.
//! 2. `lapack-provider`: enables provider-dispatched PCA paths.
//! 3. `openblas-system`/`openblas-static`/`netlib-system`/`netlib-static`/`magma-system`: provider
//!    backends forwarded to the lower crates.
//!
//! # Quick example
//!
//! ```rust
//! use nabled_embeddings::{Metric, rerank};
//! use ndarray::arr2;
//!
//! let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
//! let query = arr2(&[[1.0_f64, 0.0]]);
//! let top = rerank(&query.row(0), &corpus.view(), 2, Metric::Cosine)?;
//! assert_eq!(top[0].index, 0);
//! # Ok::<(), nabled_embeddings::EmbeddingError>(())
//! ```

pub mod compress;
pub mod error;
pub mod knn;
pub mod normalize;
pub mod rerank;
pub mod similarity;
pub mod topk;

pub use compress::{PcaModel, compress, compress_into, compress_view, fit_pca};
pub use error::EmbeddingError;
pub use knn::brute_force_knn;
pub use normalize::{normalize_rows, normalize_rows_into, normalize_rows_view};
pub use rerank::rerank;
pub use similarity::{
    Metric, query_corpus_scores, query_corpus_scores_into, query_corpus_scores_view,
};
pub use topk::{Neighbor, top_k};

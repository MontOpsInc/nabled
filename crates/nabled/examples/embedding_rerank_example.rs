//! Exact rerank over a tiny corpus, before and after PCA compression.
//!
//! Run with: `cargo run -p nabled --features embeddings --example embedding_rerank_example`.

use nabled::embeddings::{Metric, compress, fit_pca, rerank};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embeddings = Array2::from_shape_vec((6, 5), vec![
        0.8, 0.1, 0.0, 0.4, 0.2, // doc 0
        0.7, 0.2, 0.1, 0.3, 0.1, // doc 1
        0.1, 0.9, 0.2, 0.0, 0.1, // doc 2
        0.0, 0.8, 0.3, 0.1, 0.2, // doc 3
        0.2, 0.1, 0.9, 0.7, 0.6, // doc 4
        0.1, 0.0, 0.8, 0.6, 0.7, // doc 5
    ])?;
    let query = Array2::from_shape_vec((1, 5), vec![0.75, 0.2, 0.1, 0.35, 0.15])?;

    let top_full = rerank(&query.row(0), &embeddings.view(), 3, Metric::Cosine)?;
    println!("Top-3 docs in original space:");
    for neighbor in &top_full {
        println!("  doc {} (cosine {:.4})", neighbor.index, neighbor.score);
    }

    let pca_model = fit_pca(&embeddings, 3)?;
    let compressed_embeddings = compress(&embeddings, &pca_model);
    let compressed_query = compress(&query, &pca_model);

    let top_compressed =
        rerank(&compressed_query.row(0), &compressed_embeddings.view(), 3, Metric::Cosine)?;
    println!("Top-3 docs after PCA compression:");
    for neighbor in &top_compressed {
        println!("  doc {} (cosine {:.4})", neighbor.index, neighbor.score);
    }

    Ok(())
}

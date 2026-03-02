use std::cmp::Ordering;

use nabled::{pca, vector};
use ndarray::Array2;

fn top_k_indices(scores: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed = scores.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| right.partial_cmp(left).unwrap_or(Ordering::Equal));
    indexed.truncate(k.min(indexed.len()));
    indexed
}

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

    let full_scores = vector::pairwise_cosine_similarity(&query, &embeddings)?;
    let top_full = top_k_indices(full_scores.row(0).as_slice().ok_or("non-contiguous row")?, 3);
    println!("Top-3 docs in original space: {top_full:?}");

    let pca_model = pca::compute_pca(&embeddings, Some(3))?;
    let compressed_embeddings = pca::transform(&embeddings, &pca_model);
    let compressed_query = pca::transform(&query, &pca_model);

    let compressed_scores =
        vector::pairwise_cosine_similarity(&compressed_query, &compressed_embeddings)?;
    let top_compressed =
        top_k_indices(compressed_scores.row(0).as_slice().ok_or("non-contiguous row")?, 3);
    println!("Top-3 docs after PCA compression: {top_compressed:?}");

    Ok(())
}

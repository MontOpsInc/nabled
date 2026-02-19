//! Example: SVD with Apache Arrow

use ndarray::Array2;
use rust_linalg::arrow::conversions::ndarray_to_record_batch;
use rust_linalg::arrow::svd;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Arrow SVD Example ===\n");

    // Create matrix and convert to RecordBatch
    let matrix = Array2::from_shape_vec((3, 3), vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ])?;
    let batch = ndarray_to_record_batch(&matrix)?;

    println!("Input matrix (3x3):");
    println!("  columns: {}", batch.num_columns());
    println!("  rows: {}", batch.num_rows());

    // Compute SVD
    let svd_result = svd::compute_svd(&batch)?;
    println!("\nSingular values: {:?}", {
        let v: Vec<f64> = svd_result
            .singular_values
            .iter()
            .filter_map(|x| x)
            .collect();
        v
    });
    println!("Condition number: {}", svd::condition_number(&svd_result));
    println!("Matrix rank: {}", svd::matrix_rank(&svd_result, None));

    // Truncated SVD
    let truncated = svd::compute_truncated_svd(&batch, 2)?;
    println!("\nTruncated SVD (k=2) singular values: {:?}", {
        let v: Vec<f64> = truncated
            .singular_values
            .iter()
            .filter_map(|x| x)
            .collect();
        v
    });

    // Reconstruct
    let reconstructed = svd::reconstruct_matrix(&svd_result)?;
    println!("\nReconstruction: {}x{} matrix", reconstructed.num_rows(), reconstructed.num_columns());

    Ok(())
}

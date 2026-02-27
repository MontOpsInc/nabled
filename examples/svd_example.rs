//! Example usage of the SVD implementations

use nabled::svd::{nalgebra_svd, ndarray_svd};
use nalgebra::DMatrix;
use ndarray::Array2;

fn main() {
    println!("=== SVD Example ===");

    // Create a test matrix
    let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    // Test with nalgebra
    println!("\n--- Nalgebra SVD ---");
    let nalgebra_matrix = DMatrix::from_row_slice(3, 3, &matrix_data);
    println!("Original matrix:\n{nalgebra_matrix}");

    match nalgebra_svd::compute_svd(&nalgebra_matrix) {
        Ok(svd) => {
            println!("Singular values: {:?}", svd.singular_values);
            println!("Condition number: {}", nalgebra_svd::condition_number(&svd));
            println!("Matrix rank: {}", nalgebra_svd::matrix_rank(&svd, None));

            // Test truncated SVD
            let truncated = nalgebra_svd::compute_truncated_svd(&nalgebra_matrix, 2).unwrap();
            println!("Truncated SVD (k=2) singular values: {:?}", truncated.singular_values);
        }
        Err(e) => println!("Error: {e}"),
    }

    // Test with ndarray
    println!("\n--- Ndarray SVD ---");
    let ndarray_matrix = Array2::from_shape_vec((3, 3), matrix_data).unwrap();
    println!("Original matrix:\n{ndarray_matrix:?}");

    match ndarray_svd::compute_svd(&ndarray_matrix) {
        Ok(svd) => {
            println!("Singular values: {:?}", svd.singular_values);
            println!("Condition number: {}", ndarray_svd::condition_number(&svd));
            println!("Matrix rank: {}", ndarray_svd::matrix_rank(&svd, None));

            // Test truncated SVD
            let truncated = ndarray_svd::compute_truncated_svd(&ndarray_matrix, 2).unwrap();
            println!("Truncated SVD (k=2) singular values: {:?}", truncated.singular_values);
        }
        Err(e) => println!("Error: {e}"),
    }

    // Test matrix reconstruction
    println!("\n--- Matrix Reconstruction Test ---");
    let test_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let svd = nalgebra_svd::compute_svd(&test_matrix).unwrap();
    let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);

    println!("Original:\n{test_matrix}");
    println!("Reconstructed:\n{reconstructed}");
    println!("Difference norm: {}", (test_matrix - reconstructed).norm());
}

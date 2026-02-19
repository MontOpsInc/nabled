//! Example: QR decomposition and least squares with Apache Arrow

use arrow::array::Float64Array;
use ndarray::Array2;
use rust_linalg::arrow::conversions::ndarray_to_record_batch;
use rust_linalg::arrow::qr;
use rust_linalg::qr::QRConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Arrow QR Example ===\n");

    // Create matrix and convert to RecordBatch
    let a_matrix = Array2::from_shape_vec((4, 2), vec![
        1.0, 1.0,
        1.0, 2.0,
        1.0, 3.0,
        1.0, 4.0,
    ])?;
    let a_batch = ndarray_to_record_batch(&a_matrix)?;

    // Create RHS vector as Float64Array
    let b = Float64Array::from(vec![2.0, 3.0, 4.0, 5.0]);

    println!("Least squares: min ||Ax - b||");
    println!("  A: {}x{}", a_batch.num_rows(), a_batch.num_columns());
    println!("  b: {} elements", b.len());

    let config = QRConfig::default();
    let x = qr::solve_least_squares(&a_batch, &b, &config)?;
    println!("\nSolution x: {:?}", x.iter().collect::<Vec<_>>());

    // QR decomposition
    let qr_matrix = Array2::from_shape_vec((3, 3), vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    ])?;
    let qr_batch = ndarray_to_record_batch(&qr_matrix)?;
    let qr_result = qr::compute_qr(&qr_batch, &config)?;
    println!("\nQR decomposition rank: {}", qr_result.rank);

    Ok(())
}

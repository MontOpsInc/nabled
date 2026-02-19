//! # QR Decomposition Example
//!
//! This example demonstrates various QR decomposition operations using the rust-linalg library.
//! It shows how to:
//! - Perform basic QR decomposition
//! - Use reduced QR decomposition
//! - Solve least squares problems
//! - Handle both nalgebra and ndarray matrices

use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use rust_linalg::{nalgebra_qr, ndarray_qr, QRConfig, QRError};

fn main() -> Result<(), QRError> {
    println!("🔢 QR Decomposition Examples\n");

    // Example 1: Basic QR Decomposition with Nalgebra
    println!("📊 Example 1: Basic QR Decomposition (Nalgebra)");
    let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

    println!("Original matrix A:");
    println!("{}", matrix);

    let config = QRConfig::default();
    let qr = nalgebra_qr::compute_qr(&matrix, &config)?;

    println!("\nQ matrix (orthogonal):");
    println!("{}", qr.q);

    println!("\nR matrix (upper triangular):");
    println!("{}", qr.r);

    println!("\nMatrix rank: {}", qr.rank);

    // Verify reconstruction
    let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
    println!("\nReconstruction A = Q * R:");
    println!("{}", reconstructed);

    // Verify Q is orthogonal (Q^T * Q = I)
    let qt_q = qr.q.transpose() * &qr.q;
    println!("\nQ^T * Q (should be identity):");
    println!("{}", qt_q);

    println!("\n{}\n", "=".repeat(60));

    // Example 2: Rectangular Matrix QR Decomposition
    println!("📐 Example 2: Rectangular Matrix QR Decomposition");
    let rectangular = DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    println!("Original 4×3 matrix:");
    println!("{}", rectangular);

    let qr_rect = nalgebra_qr::compute_qr(&rectangular, &config)?;

    println!("\nQ matrix (4×3):");
    println!("{}", qr_rect.q);

    println!("\nR matrix (3×3):");
    println!("{}", qr_rect.r);

    println!("\n{}\n", "=".repeat(60));

    // Example 3: Reduced QR Decomposition
    println!("🎯 Example 3: Reduced QR Decomposition");
    let reduced_qr = nalgebra_qr::compute_reduced_qr(&rectangular, &config)?;

    println!("Reduced Q matrix (4×3):");
    println!("{}", reduced_qr.q);

    println!("\nReduced R matrix (3×3):");
    println!("{}", reduced_qr.r);

    // Verify Q^T * Q = I for reduced Q
    let qt_q_reduced = reduced_qr.q.transpose() * &reduced_qr.q;
    println!("\nQ^T * Q for reduced Q (should be 3×3 identity):");
    println!("{}", qt_q_reduced);

    println!("\n{}\n", "=".repeat(60));

    // Example 4: Least Squares Problem
    println!("📈 Example 4: Least Squares Problem");
    // Solve Ax = b where A is overdetermined
    let a = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

    let b = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

    println!("Coefficient matrix A (4×2):");
    println!("{}", a);

    println!("\nRight-hand side b:");
    println!("{}", b);

    let x = nalgebra_qr::solve_least_squares(&a, &b, &config)?;

    println!("\nLeast squares solution x:");
    println!("{}", x);

    // Verify the solution
    let residual = &a * &x - &b;
    let residual_norm = residual.norm();
    println!("\nResidual norm ||Ax - b||: {:.6}", residual_norm);

    println!("\n{}\n", "=".repeat(60));

    // Example 5: QR with Ndarray
    println!("🔢 Example 5: QR Decomposition with Ndarray");
    let ndarray_matrix =
        Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]).unwrap();

    println!("Original ndarray matrix:");
    println!("{:?}", ndarray_matrix);

    let qr_ndarray = ndarray_qr::compute_qr(&ndarray_matrix, &config)?;

    println!("\nQ matrix:");
    println!("{}", qr_ndarray.q);

    println!("\nR matrix:");
    println!("{}", qr_ndarray.r);

    println!("\n{}\n", "=".repeat(60));

    // Example 6: Condition Number
    println!("📊 Example 6: Matrix Condition Number");
    let well_conditioned = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

    let ill_conditioned = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0001]);

    let qr_well = nalgebra_qr::compute_qr(&well_conditioned, &config)?;
    let qr_ill = nalgebra_qr::compute_qr(&ill_conditioned, &config)?;

    let cond_well = nalgebra_qr::condition_number(&qr_well);
    let cond_ill = nalgebra_qr::condition_number(&qr_ill);

    println!("Well-conditioned matrix condition number: {:.6}", cond_well);
    println!("Ill-conditioned matrix condition number: {:.6}", cond_ill);

    println!("\n{}\n", "=".repeat(60));

    // Example 7: Error Handling
    println!("⚠️  Example 7: Error Handling");

    // Test with empty matrix
    let empty_matrix = DMatrix::<f64>::zeros(0, 0);
    match nalgebra_qr::compute_qr(&empty_matrix, &config) {
        Err(QRError::EmptyMatrix) => println!("✅ Correctly caught empty matrix error"),
        _ => println!("❌ Expected empty matrix error"),
    }

    // Test with invalid configuration
    let invalid_config = QRConfig {
        rank_tolerance: -1.0,
        max_iterations: 100,
        use_pivoting: false,
    };

    match nalgebra_qr::compute_qr(&matrix, &invalid_config) {
        Err(QRError::InvalidInput(_)) => println!("✅ Correctly caught invalid input error"),
        _ => println!("❌ Expected invalid input error"),
    }

    println!("\n🎉 All QR decomposition examples completed successfully!");

    Ok(())
}

//! Cholesky decomposition: solve Ax = b for positive-definite A

use nabled::cholesky;
use nalgebra::{DMatrix, DVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cholesky Decomposition Example ===\n");

    let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
    let a = &l * l.transpose();
    let b = DVector::from_vec(vec![1.0, 2.0]);

    let x = cholesky::nalgebra_cholesky::solve(&a, &b)?;
    println!("Solve Ax = b (A is symmetric positive-definite):");
    println!("  A = {a}");
    println!("  b = {b}");
    println!("  x = {x}");

    let decomp = cholesky::nalgebra_cholesky::compute_cholesky(&a)?;
    println!("\nCholesky factor L:");
    println!("  L = {}", decomp.l);
    println!("  L * L^T = {}", &decomp.l * decomp.l.transpose());

    Ok(())
}

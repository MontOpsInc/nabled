//! LU decomposition: solve Ax = b and compute inverse

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use rust_linalg::lu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LU Decomposition Example ===\n");

    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = DVector::from_vec(vec![5.0, 6.0]);

    let x = lu::nalgebra_lu::solve(&a, &b)?;
    println!("Solve Ax = b:");
    println!("  A = {}", a);
    println!("  b = {}", b);
    println!("  x = {}", x);

    let inv = lu::nalgebra_lu::inverse(&a)?;
    println!("\nInverse of A:");
    println!("  A^(-1) = {}", inv);
    println!("  A * A^(-1) = {}", &a * &inv);

    println!("\n=== Ndarray version ===");
    let a_nd = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    let b_nd = Array1::from_vec(vec![5.0, 6.0]);
    let x_nd = lu::ndarray_lu::solve(&a_nd, &b_nd)?;
    println!("  x = {:?}", x_nd);

    Ok(())
}

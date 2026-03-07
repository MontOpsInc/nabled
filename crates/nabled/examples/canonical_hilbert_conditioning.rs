//! Canonical Hilbert-matrix conditioning demo.
//!
//! This mirrors a classic numerical-linear-algebra textbook example:
//! Hilbert matrices are SPD but notoriously ill-conditioned.

use std::num::TryFromIntError;

use nabled::linalg::{cholesky, lu, svd};
use ndarray::{Array1, Array2};

fn hilbert(n: usize) -> Result<Array2<f64>, TryFromIntError> {
    let mut matrix = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let denominator = f64::from(u32::try_from(i + j + 1)?);
            matrix[[i, j]] = 1.0_f64 / denominator;
        }
    }
    Ok(matrix)
}

fn l2_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn residual_l2(matrix: &Array2<f64>, solution: &Array1<f64>, rhs: &Array1<f64>) -> f64 {
    let residual = matrix.dot(solution) - rhs;
    l2_norm(&residual)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 10_usize;
    let matrix = hilbert(n)?;
    let rhs = Array1::from_elem(n, 1.0_f64);

    let svd_result = svd::decompose(&matrix)?;
    let condition_number = svd::condition_number(&svd_result);

    let lu_solution = lu::solve(&matrix, &rhs)?;
    let cholesky_solution = cholesky::solve(&matrix, &rhs)?;

    let lu_residual = residual_l2(&matrix, &lu_solution, &rhs);
    let cholesky_residual = residual_l2(&matrix, &cholesky_solution, &rhs);

    println!("Hilbert({n}) conditioning showcase");
    println!("cond_2(H)                = {condition_number:.6e}");
    println!("||H x_lu - b||_2         = {lu_residual:.6e}");
    println!("||H x_cholesky - b||_2   = {cholesky_residual:.6e}");
    println!(
        "max |x_lu - x_cholesky|   = {:.6e}",
        lu_solution
            .iter()
            .zip(cholesky_solution.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max)
    );

    Ok(())
}

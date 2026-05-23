//! Canonical Sylvester/Lyapunov residual demo.
//!
//! Textbook form:
//! - Sylvester: `A X + X B = C`
//! - Continuous Lyapunov: `A P + P A^T + Q = 0`

use nabled::linalg::sylvester;
use ndarray::Array2;

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix_a = Array2::from_shape_vec((3, 3), vec![
        -1.0_f64, 0.2_f64, 0.0_f64, 0.0_f64, -2.0_f64, 0.3_f64, 0.0_f64, 0.0_f64, -3.0_f64,
    ])?;
    let matrix_b = Array2::from_shape_vec((2, 2), vec![-0.5_f64, 0.0_f64, 0.1_f64, -1.5_f64])?;
    let matrix_c = Array2::from_shape_vec((3, 2), vec![
        1.0_f64, 2.0_f64, -0.5_f64, 0.25_f64, 0.0_f64, 1.5_f64,
    ])?;

    let sylvester_solution = sylvester::solve_sylvester(&matrix_a, &matrix_b, &matrix_c)?;
    let sylvester_residual =
        matrix_a.dot(&sylvester_solution) + sylvester_solution.dot(&matrix_b) - &matrix_c;

    let process_noise = Array2::eye(3);
    let lyapunov_solution = sylvester::solve_lyapunov(&matrix_a, &process_noise)?;
    let lyapunov_residual =
        matrix_a.dot(&lyapunov_solution) + lyapunov_solution.dot(&matrix_a.t()) + process_noise;

    println!("Sylvester/Lyapunov textbook residual showcase");
    println!("||A X + X B - C||_F      = {:.6e}", frobenius_norm(&sylvester_residual));
    println!("||A P + P A^T + Q||_F    = {:.6e}", frobenius_norm(&lyapunov_residual));

    Ok(())
}

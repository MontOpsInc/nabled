//! Example usage of matrix functions (exponential, logarithm, power)

use nalgebra::DMatrix;
use ndarray::Array2;
use rust_linalg::matrix_functions::{
    MatrixFunctionError, nalgebra_matrix_functions, ndarray_matrix_functions,
};

fn main() -> Result<(), MatrixFunctionError> {
    println!("=== Matrix Functions Example ===");

    // Test matrix exponential
    println!("\n--- Matrix Exponential ---");

    // Identity matrix: exp(I) = e * I
    let identity = DMatrix::<f64>::identity(3, 3);
    println!("Identity matrix:\n{}", identity);

    let exp_identity = nalgebra_matrix_functions::matrix_exp_eigen(&identity)?;
    println!("exp(I):\n{}", exp_identity);
    println!("Expected: e * I = {} * I", std::f64::consts::E);

    // Zero matrix: exp(0) = I
    let zero = DMatrix::<f64>::zeros(2, 2);
    println!("\nZero matrix:\n{}", zero);

    let exp_zero = nalgebra_matrix_functions::matrix_exp_eigen(&zero)?;
    println!("exp(0):\n{}", exp_zero);
    println!("Expected: I");

    // Diagonal matrix
    let diagonal = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
    println!("\nDiagonal matrix:\n{}", diagonal);

    let exp_diagonal = nalgebra_matrix_functions::matrix_exp_eigen(&diagonal)?;
    println!("exp(diagonal):\n{}", exp_diagonal);
    println!(
        "Expected: diag(e, e²) = diag({}, {})",
        std::f64::consts::E,
        std::f64::consts::E * std::f64::consts::E
    );

    // Test matrix logarithm
    println!("\n--- Matrix Logarithm ---");

    // Identity matrix: log(I) = 0
    let log_identity = nalgebra_matrix_functions::matrix_log_eigen(&identity)?;
    println!("log(I):\n{}", log_identity);
    println!("Expected: 0 matrix");

    // Positive definite matrix
    let pos_def = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    println!("\nPositive definite matrix:\n{}", pos_def);

    let log_pos_def = nalgebra_matrix_functions::matrix_log_eigen(&pos_def)?;
    println!("log(pos_def):\n{}", log_pos_def);

    // Test exp(log(A)) ≈ A
    let exp_log_pos_def = nalgebra_matrix_functions::matrix_exp_eigen(&log_pos_def)?;
    println!("exp(log(pos_def)):\n{}", exp_log_pos_def);
    println!("Difference from original: {}", (exp_log_pos_def - pos_def).norm());

    // Test matrix power
    println!("\n--- Matrix Power ---");

    let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]);
    println!("Matrix:\n{}", matrix);

    // A^0.5 (square root)
    let sqrt_matrix = nalgebra_matrix_functions::matrix_power(&matrix, 0.5)?;
    println!("A^0.5:\n{}", sqrt_matrix);

    // A^2
    let square_matrix = nalgebra_matrix_functions::matrix_power(&matrix, 2.0)?;
    println!("A^2:\n{}", square_matrix);

    // Verify: (A^0.5)^2 ≈ A
    let sqrt_squared = nalgebra_matrix_functions::matrix_power(&sqrt_matrix, 2.0)?;
    println!("(A^0.5)^2:\n{}", sqrt_squared);
    println!("Difference from A: {}", (sqrt_squared - matrix).norm());

    // Test with ndarray
    println!("\n--- Ndarray Matrix Functions ---");

    let ndarray_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    println!("Ndarray matrix:\n{:?}", ndarray_matrix);

    let exp_ndarray = ndarray_matrix_functions::matrix_exp_eigen(&ndarray_matrix)?;
    println!("exp(ndarray_matrix):\n{:?}", exp_ndarray);

    let log_ndarray = ndarray_matrix_functions::matrix_log_eigen(&ndarray_matrix)?;
    println!("log(ndarray_matrix):\n{:?}", log_ndarray);

    // Test error handling
    println!("\n--- Error Handling ---");

    // Non-square matrix
    let non_square = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    match nalgebra_matrix_functions::matrix_exp_eigen(&non_square) {
        Err(MatrixFunctionError::NotSquare) => {
            println!("✓ Correctly caught non-square matrix error")
        }
        _ => println!("✗ Failed to catch non-square matrix error"),
    }

    // Matrix with negative eigenvalues for logarithm
    let negative_eigen = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
    match nalgebra_matrix_functions::matrix_log_eigen(&negative_eigen) {
        Err(MatrixFunctionError::NegativeEigenvalues) => {
            println!("✓ Correctly caught negative eigenvalues error")
        }
        _ => println!("✗ Failed to catch negative eigenvalues error"),
    }

    println!("\n=== Matrix Functions Example Complete ===");
    Ok(())
}

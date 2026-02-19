//! Linear regression: least squares fit with R-squared

use nalgebra::{DMatrix, DVector};
use rust_linalg::regression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Linear Regression Example ===\n");

    let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = DVector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let result = regression::nalgebra_regression::linear_regression(&x, &y, true)?;

    println!("Fit y = beta_0 + beta_1 * x");
    println!("  Coefficients: {:?}", result.coefficients.as_slice());
    println!("  R-squared: {:.6}", result.r_squared);
    println!("  Fitted values: {:?}", result.fitted_values.as_slice());
    println!("  Residuals: {:?}", result.residuals.as_slice());

    Ok(())
}

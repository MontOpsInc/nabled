//! Complex Jacobian Example
//!
//! This example demonstrates how to use the complex derivative functions
//! in the rust-linalg library.

use nalgebra::DVector;
use num_complex::Complex;
use rust_linalg::jacobian::{JacobianConfig, JacobianError, complex_jacobian};

fn main() -> Result<(), JacobianError> {
    println!("=== Complex Jacobian Computation Examples ===\n");

    // Example 1: Complex function f(z) = z²
    println!("1. Complex function f(z) = z²");
    let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
        let mut result = DVector::zeros(x.len());
        for i in 0..x.len() {
            result[i] = x[i] * x[i];
        }
        Ok(result)
    };

    let x = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    println!("   Point: [{}, {}]", x[0], x[1]);

    let config = JacobianConfig::default();
    let jacobian = complex_jacobian::numerical_jacobian(&f, &x, &config)?;
    println!("   Jacobian:");
    println!("   {}", jacobian);
    println!("   Expected: [[2.0, 0.0], [0.0, 4.0]]\n");

    // Example 2: Complex gradient
    println!("2. Complex gradient of f(z) = z₁² + z₂²");
    let f_grad = |x: &DVector<Complex<f64>>| -> Result<Complex<f64>, String> {
        let mut sum = Complex::new(0.0, 0.0);
        for i in 0..x.len() {
            sum = sum + x[i] * x[i];
        }
        Ok(sum)
    };

    let x_grad = DVector::from_vec(vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]);
    println!("   Point: [{}, {}]", x_grad[0], x_grad[1]);

    let gradient = complex_jacobian::numerical_gradient(&f_grad, &x_grad, &config)?;
    println!("   Gradient: {}", gradient);
    println!("   Expected: [6.0, 8.0]\n");

    // Example 3: Complex Hessian
    println!("3. Complex Hessian of f(z) = z₁² + z₂²");
    let x_hess = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    println!("   Point: [{}, {}]", x_hess[0], x_hess[1]);

    let hessian = complex_jacobian::numerical_hessian(&f_grad, &x_hess, &config)?;
    println!("   Hessian:");
    println!("   {}", hessian);
    println!("   Expected: [[2.0, 0.0], [0.0, 2.0]]\n");

    // Example 4: Error handling
    println!("4. Error handling example");
    let error_f = |_x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
        Err("Simulated function error".to_string())
    };

    let x_error = DVector::from_vec(vec![Complex::new(1.0, 0.0)]);
    match complex_jacobian::numerical_jacobian(&error_f, &x_error, &config) {
        Err(e) => println!("   Caught expected error: {}", e),
        Ok(_) => println!("   Unexpected success"),
    }

    println!("\n=== All complex derivative examples completed successfully! ===");
    Ok(())
}

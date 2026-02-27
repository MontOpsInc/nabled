//! # Jacobian Computation Example
//!
//! This example demonstrates how to use the Jacobian computation functions
//! for both nalgebra and ndarray.

use nabled::{JacobianConfig, JacobianError, nalgebra_jacobian, ndarray_jacobian};
use nalgebra::DVector;
use ndarray::Array1;

fn main() -> Result<(), JacobianError> {
    println!("=== Jacobian Computation Examples ===\n");

    // Example 1: Simple quadratic function with nalgebra
    println!("1. Nalgebra - Quadratic function f(x,y) = [x², y²]");
    let f_quad = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
        Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
    };

    let x1 = DVector::from_vec(vec![2.0, 3.0]);
    let config = JacobianConfig::default();

    let jacobian1 = nalgebra_jacobian::numerical_jacobian(&f_quad, &x1, &config)?;
    println!("   Point: [2.0, 3.0]");
    println!("   Jacobian (forward differences):");
    println!("   {jacobian1}");

    let jacobian1_central = nalgebra_jacobian::numerical_jacobian_central(&f_quad, &x1, &config)?;
    println!("   Jacobian (central differences):");
    println!("   {jacobian1_central}");
    println!("   Expected: [[4.0, 0.0], [0.0, 6.0]]\n");

    // Example 2: Gradient computation (scalar function)
    println!("2. Nalgebra - Gradient of f(x,y) = x² + y²");
    let f_scalar = |x: &DVector<f64>| -> Result<f64, String> { Ok(x[0] * x[0] + x[1] * x[1]) };

    let x2 = DVector::from_vec(vec![3.0, 4.0]);
    let gradient = nalgebra_jacobian::numerical_gradient(&f_scalar, &x2, &config)?;
    println!("   Point: [3.0, 4.0]");
    println!("   Gradient: {gradient}");
    println!("   Expected: [6.0, 8.0]\n");

    // Example 3: Hessian computation
    println!("3. Nalgebra - Hessian of f(x,y) = x³ + y³");
    let f_cubic =
        |x: &DVector<f64>| -> Result<f64, String> { Ok(x[0] * x[0] * x[0] + x[1] * x[1] * x[1]) };

    let x3 = DVector::from_vec(vec![2.0, 3.0]);
    let hessian = nalgebra_jacobian::numerical_hessian(&f_cubic, &x3, &config)?;
    println!("   Point: [2.0, 3.0]");
    println!("   Hessian:");
    println!("   {hessian}");
    println!("   Expected: [[12.0, 0.0], [0.0, 18.0]]\n");

    // Example 4: More complex function with ndarray
    println!("4. Ndarray - Complex function f(x,y,z) = [x²+y, y²+z, z²+x]");
    let f_complex = |x: &Array1<f64>| -> Result<Array1<f64>, String> {
        Ok(Array1::from_vec(vec![x[0] * x[0] + x[1], x[1] * x[1] + x[2], x[2] * x[2] + x[0]]))
    };

    let x4 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let jacobian4 = ndarray_jacobian::numerical_jacobian(&f_complex, &x4, &config)?;
    println!("   Point: [1.0, 2.0, 3.0]");
    println!("   Jacobian:");
    for i in 0..jacobian4.nrows() {
        print!("   [");
        for j in 0..jacobian4.ncols() {
            print!("{:8.4}", jacobian4[(i, j)]);
            if j < jacobian4.ncols() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
    println!("   Expected: [[2.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 6.0]]\n");

    // Example 5: Gradient with ndarray
    println!("5. Ndarray - Gradient of f(x,y,z) = x² + y² + z²");
    let f_sphere =
        |x: &Array1<f64>| -> Result<f64, String> { Ok(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) };

    let x5 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let gradient5 = ndarray_jacobian::numerical_gradient(&f_sphere, &x5, &config)?;
    println!("   Point: [1.0, 2.0, 3.0]");
    println!("   Gradient: {:?}", gradient5.to_vec());
    println!("   Expected: [2.0, 4.0, 6.0]\n");

    // Example 6: Error handling
    println!("6. Error handling example");
    let f_error = |_x: &DVector<f64>| -> Result<DVector<f64>, String> {
        Err("Simulated function error".to_string())
    };

    let x6 = DVector::from_vec(vec![1.0, 2.0]);
    match nalgebra_jacobian::numerical_jacobian(&f_error, &x6, &config) {
        Err(JacobianError::FunctionError(msg)) => {
            println!("   Caught expected error: {msg}");
        }
        _ => {
            println!("   Unexpected result");
        }
    }

    // Example 7: Custom configuration
    println!("\n7. Custom configuration with different step size");
    let custom_config = JacobianConfig::<f64> { step_size: 1e-8, ..JacobianConfig::default() };

    let jacobian_precise = nalgebra_jacobian::numerical_jacobian(&f_quad, &x1, &custom_config)?;
    println!("   Point: [2.0, 3.0]");
    println!("   Jacobian (high precision):");
    println!("   {jacobian_precise}");

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

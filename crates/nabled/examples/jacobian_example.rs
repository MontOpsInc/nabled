use nabled::jacobian::ndarray_jacobian;
use nabled::{JacobianConfig, JacobianError};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JacobianConfig::default();

    let f_vec = |x: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
        Ok(Array1::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
    };
    let x = Array1::from_vec(vec![2.0, 3.0]);
    let jacobian = ndarray_jacobian::numerical_jacobian(&f_vec, &x, &config)?;
    println!("jacobian:\n{jacobian:?}");

    let f_scalar = |x: &Array1<f64>| -> Result<f64, JacobianError> {
        Ok(x.iter().map(|value| value * value).sum())
    };
    let gradient = ndarray_jacobian::numerical_gradient(&f_scalar, &x, &config)?;
    println!("gradient: {gradient:?}");

    let hessian = ndarray_jacobian::numerical_hessian(&f_scalar, &x, &config)?;
    println!("hessian:\n{hessian:?}");

    Ok(())
}

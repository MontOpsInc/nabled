use nabled::{JacobianConfig, JacobianError, jacobian};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = JacobianConfig::default();
    let x = Array1::from_vec(vec![1.0, 2.0]);

    let f = |input: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
        Ok(Array1::from_vec(vec![input[0] * input[0] + input[1], input[1] * input[1] + input[0]]))
    };

    let jacobian = jacobian::numerical_jacobian(&f, &x, &config)?;
    println!("jacobian:\n{jacobian:?}");

    Ok(())
}

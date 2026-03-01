use nabled::regression;
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

    let result = regression::linear_regression(&x, &y, true)?;
    println!("coefficients: {:?}", result.coefficients);

    Ok(())
}

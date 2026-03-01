use nabled::matrix_functions::{self as matrix_functions, MatrixFunctionError};
use ndarray::Array2;

fn run_examples() -> Result<(), MatrixFunctionError> {
    let identity = Array2::<f64>::eye(2);
    let exp_identity = matrix_functions::matrix_exp_eigen(&identity)?;
    println!("exp(I):\n{exp_identity:?}");

    let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let log_matrix = matrix_functions::matrix_log_eigen(&matrix)?;
    println!("log(A):\n{log_matrix:?}");

    let power = matrix_functions::matrix_power(&matrix, 0.5)?;
    println!("A^(1/2):\n{power:?}");

    Ok(())
}

fn main() {
    if let Err(error) = run_examples() {
        eprintln!("example failed: {error}");
        std::process::exit(1);
    }
}

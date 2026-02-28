use nabled::cholesky::ndarray_cholesky;
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0])?;
    let b = Array1::from_vec(vec![1.0, 1.0]);

    let x = ndarray_cholesky::solve(&a, &b)?;
    println!("solution: {x:?}");

    let decomp = ndarray_cholesky::decompose(&a)?;
    println!("L:\n{:?}", decomp.l);

    Ok(())
}

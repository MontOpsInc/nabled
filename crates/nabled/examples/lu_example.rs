use nabled::lu::ndarray_lu;
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    let b = Array1::from_vec(vec![5.0, 11.0]);

    let x = ndarray_lu::solve(&a, &b)?;
    println!("solution: {x:?}");

    let inv = ndarray_lu::inverse(&a)?;
    println!("inverse:\n{inv:?}");

    Ok(())
}

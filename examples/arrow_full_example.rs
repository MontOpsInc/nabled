//! Example: Full Arrow linear algebra workflow

use arrow::array::Float64Array;
use ndarray::Array2;
use rust_linalg::arrow::conversions::ndarray_to_record_batch;
use rust_linalg::arrow::{jacobian, matrix_functions, svd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Arrow Linear Algebra Full Example ===\n");

    // 1. SVD
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
    let batch = ndarray_to_record_batch(&matrix)?;
    let svd_result = svd::compute_svd(&batch)?;
    println!("SVD singular values: {:?}", {
        svd_result.singular_values.iter().filter_map(|x| x).collect::<Vec<_>>()
    });

    // 2. Matrix exponential
    let exp_batch = matrix_functions::matrix_exp_eigen(&batch)?;
    println!("Matrix exp computed: {}x{}", exp_batch.num_rows(), exp_batch.num_columns());

    // 3. Jacobian
    let x_vec = Float64Array::from(vec![2.0, 3.0]);
    let f = |x: &nalgebra::DVector<f64>| {
        Ok(nalgebra::DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
    };
    let jac = jacobian::numerical_jacobian(f, &x_vec, &Default::default())?;
    println!("Jacobian computed: {}x{}", jac.num_rows(), jac.num_columns());

    println!("\nAll Arrow linear algebra operations completed successfully.");
    Ok(())
}

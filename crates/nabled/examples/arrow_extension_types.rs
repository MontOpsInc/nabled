//! Demonstrates `nabled::arrow` workflows over explicit Arrow extension carriers.
//!
//! Run with:
//! `cargo run --package nabled --example arrow_extensions_example --features arrow`

use arrow_array::Float64Array;
use arrow_array::types::Float64Type;
use nabled::arrow::{matrix, sparse, tensor, vector};
use nabled::ndarrow::{
    AsNdarray, IntoArrow, array2_complex64_to_fixed_size_list, arrayd_to_fixed_shape_tensor,
    arrays_to_variable_shape_tensor, complex64_as_array_view1, complex64_as_array_view2,
    csr_batch_to_extension_array, csr_matrix_batch_iter, csr_to_extension_array,
    fixed_shape_tensor_as_array_viewd, fixed_size_list_as_array2, variable_shape_tensor_iter,
};
use ndarray::{ArrayD, IxDyn, array};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dense matrix (2D) represented as Arrow FixedSizeListArray via `into_arrow`.
    let dense = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]].into_arrow()?;
    // Standard dense vector RHS as primitive Float64Array (non-extension).
    let rhs = Float64Array::from(vec![1.0_f64, 2.0_f64]);

    // Non-extension Arrow vectors for generic vector dot product.
    let dot = vector::dot(
        &Float64Array::from(vec![1.0_f64, 1.0, 1.0]),
        &Float64Array::from(vec![2.0_f64, 2.0, 2.0]),
    )?;
    println!("dense vector dot = {dot}");

    // Dense matrix-vector multiply: input dense arrow matrix + dense vector.
    let matvec = matrix::matvec::<Float64Type>(&dense, &rhs)?;
    println!("matvec = {:?}", matvec.as_ndarray()?);

    // Dense matrix-matrix multiply using arrow-fixed-size-list-backed matrices.
    let dense_t = array![[1.0_f64, 3.0, 5.0], [2.0, 4.0, 6.0]].into_arrow()?;
    let matmat = matrix::matmat::<Float64Type>(&dense, &dense_t)?;
    let matmat_view = fixed_size_list_as_array2::<Float64Type>(&matmat)?;
    println!("matmat shape = {:?}", matmat_view.shape());

    // Complex dense matrix values represented as custom complex fixed-size-list extension arrays.
    let left = array![[Complex64::new(1.0, 1.0), Complex64::new(0.0, 2.0)], [
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 2.0)
    ]];
    let right = array![[Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0)], [
        Complex64::new(0.0, 2.0),
        Complex64::new(2.0, 0.0)
    ]];
    let left_complex = array2_complex64_to_fixed_size_list(left.clone())?;
    let right_complex = array2_complex64_to_fixed_size_list(right.clone())?;
    let (dot_field, dot_array) = vector::batched_dot_hermitian(&left_complex, &right_complex)?;
    let dot_view = complex64_as_array_view1(&dot_field, &dot_array)?;
    println!("complex batched_dot_hermitian = {:?}", dot_view.iter().copied().collect::<Vec<_>>());

    // Complex normalization using complex extension arrays -> complex ndarray view.
    let normalized = vector::batched_normalize_complex(&left_complex)?;
    let normalized_view = complex64_as_array_view2(&normalized)?;
    println!(
        "complex normalized row0 = {:?}",
        normalized_view.row(0).iter().copied().collect::<Vec<_>>()
    );

    // Fixed-shape tensor extension (ndarray ArrayD -> Arrow fixed-shape tensor extension).
    let fixed_shape =
        ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![3.0, 4.0, 0.0, 5.0, 8.0, 15.0, 7.0, 24.0])?;
    let (tensor_field, tensor_array) = arrayd_to_fixed_shape_tensor("tensor", fixed_shape)?;
    let (norm_field, norm_array) =
        tensor::l2_norm_last_axis::<Float64Type>(&tensor_field, &tensor_array)?;
    let norm_view = fixed_shape_tensor_as_array_viewd::<Float64Type>(&norm_field, &norm_array)?;
    println!("fixed-shape tensor l2 norm shape = {:?}", norm_view.shape());

    // Fixed-shape tensor axis permutation (still fixed-shape tensor extension).
    let (permuted_field, permuted_array) =
        tensor::permute_axes::<Float64Type>(&tensor_field, &tensor_array, &[1, 0, 2])?;
    let permuted_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&permuted_field, &permuted_array)?;
    println!("permuted fixed-shape tensor shape = {:?}", permuted_view.shape());

    // Variable-shape tensor rows packed into variable-shape tensor extension.
    let ragged_left = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![3.0_f64, 4.0, 0.0, 5.0])?;
    let ragged_right = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![8.0_f64, 15.0, 17.0])?;
    let (ragged_field, ragged_array) =
        arrays_to_variable_shape_tensor("ragged", vec![ragged_left, ragged_right], None)?;
    let (sum_field, sum_array) =
        tensor::sum_last_axis_variable::<Float64Type>(&ragged_field, &ragged_array)?;
    let mut sum_iter = variable_shape_tensor_iter::<Float64Type>(&sum_field, &sum_array)?;
    let first_sum = sum_iter.next().expect("at least one row")?.1;
    let second_sum = sum_iter.next().expect("second row")?.1;
    println!("ragged row sums = {:?}", vec![
        first_sum.iter().copied().collect::<Vec<_>>(),
        second_sum.iter().copied().collect::<Vec<_>>()
    ]);

    // Sparse CSR matrix in extension array form (CSR row pointers, col indices, nonzero values).
    let (csr_field, csr_matrix) =
        csr_to_extension_array("sparse", 4, vec![0_i32, 2, 3, 5], vec![0_u32, 2, 1, 0, 3], vec![
            1.0_f64, 5.0, 2.0, 3.0, 4.0,
        ])?;
    let csr_rhs = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let csr_matvec =
        sparse::matvec_csr_extension::<Float64Type>(&csr_field, &csr_matrix, &csr_rhs)?;
    println!("csr matvec = {:?}", csr_matvec.as_ndarray()?);

    // Batch of CSR matrices in extension form; iterated as per-matrix sparse views.
    let (batch_field, batch_matrices) = csr_batch_to_extension_array(
        "sparse_batch",
        vec![[2, 2], [1, 3]],
        vec![vec![0_i32, 1, 2], vec![0_i32, 2]],
        vec![vec![0_u32, 1], vec![0_u32, 2]],
        vec![vec![2.0_f64, 3.0], vec![1.0_f64, 4.0]],
    )?;
    let mut batch_iter = csr_matrix_batch_iter::<Float64Type>(&batch_field, &batch_matrices)?;
    if let Some(Ok((_, first_matrix))) = batch_iter.next() {
        println!(
            "batch row 0: {}x{} with {} nnz",
            first_matrix.nrows,
            first_matrix.ncols,
            first_matrix.nnz()
        );
    }

    Ok(())
}

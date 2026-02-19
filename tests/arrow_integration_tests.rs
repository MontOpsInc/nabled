//! Integration tests for Arrow linear algebra

use approx::assert_relative_eq;
use arrow::array::Float64Array;
use arrow::record_batch::RecordBatch;
use ndarray::Array2;
use rust_linalg::arrow::conversions::{ndarray_to_record_batch, record_batch_to_ndarray};
use rust_linalg::arrow::{matrix_functions, qr, svd};

fn make_3x3_batch() -> RecordBatch {
    let matrix = Array2::from_shape_vec((3, 3), vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ])
    .unwrap();
    ndarray_to_record_batch(&matrix).unwrap()
}

#[test]
fn test_arrow_svd_roundtrip() {
    let batch = make_3x3_batch();
    let svd_result = svd::compute_svd(&batch).unwrap();
    let reconstructed = svd::reconstruct_matrix(&svd_result).unwrap();

    let original = record_batch_to_ndarray(&batch).unwrap();
    let reconstructed_arr = record_batch_to_ndarray(&reconstructed).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(original[[i, j]], reconstructed_arr[[i, j]], epsilon = 1e-5);
        }
    }
}

#[test]
fn test_arrow_svd_condition_number() {
    let batch = make_3x3_batch();
    let svd_result = svd::compute_svd(&batch).unwrap();
    let cond = svd::condition_number(&svd_result);
    assert!(cond >= 1.0);
    // Rank-deficient matrices can have infinite condition number
}

#[test]
fn test_arrow_svd_matrix_rank() {
    let batch = make_3x3_batch();
    let svd_result = svd::compute_svd(&batch).unwrap();
    let rank = svd::matrix_rank(&svd_result, None);
    assert_eq!(rank, 2); // 3x3 matrix with rows in arithmetic progression has rank 2
}

#[test]
fn test_arrow_truncated_svd() {
    let batch = make_3x3_batch();
    let truncated = svd::compute_truncated_svd(&batch, 2).unwrap();
    let sv_count = truncated.singular_values.iter().filter_map(|x| x).count();
    assert_eq!(sv_count, 2);
}

#[test]
fn test_arrow_qr_least_squares() {
    let a = Array2::from_shape_vec((4, 2), vec![
        1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0,
    ])
    .unwrap();
    let a_batch = ndarray_to_record_batch(&a).unwrap();
    let b = Float64Array::from(vec![2.0, 3.0, 4.0, 5.0]);

    let x = qr::solve_least_squares(&a_batch, &b, &Default::default()).unwrap();
    let x_vals: Vec<f64> = x.iter().filter_map(|v| v).collect();
    assert_eq!(x_vals.len(), 2);
    // Solution to [1,1;1,2;1,3;1,4] * [x0;x1] = [2;3;4;5] is approximately [1, 1]
    assert_relative_eq!(x_vals[0], 1.0, epsilon = 0.1);
    assert_relative_eq!(x_vals[1], 1.0, epsilon = 0.1);
}

#[test]
fn test_arrow_matrix_exp() {
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let batch = ndarray_to_record_batch(&matrix).unwrap();
    let exp_batch = matrix_functions::matrix_exp_eigen(&batch).unwrap();
    assert_eq!(exp_batch.num_rows(), 2);
    assert_eq!(exp_batch.num_columns(), 2);
}

#[test]
fn test_arrow_conversion_roundtrip() {
    let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let batch = ndarray_to_record_batch(&matrix).unwrap();
    let back = record_batch_to_ndarray(&batch).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(matrix[[i, j]], back[[i, j]], epsilon = 1e-10);
        }
    }
}

//! Integration tests for ndarray-first APIs.

use approx::assert_relative_eq;
use nabled::IterativeConfig;
use nabled::iterative::ndarray_iterative;
use nabled::matrix_functions::ndarray_matrix_functions;
use nabled::orthogonalization::ndarray_orthogonalization;
use nabled::pca::ndarray_pca;
use nabled::regression::ndarray_regression;
use nabled::stats::ndarray_stats;
use nabled::svd::{SVDError, ndarray_svd};
use nabled::sylvester::ndarray_sylvester;
use nabled::triangular::ndarray_triangular;
use nabled::vector::{PairwiseCosineWorkspace, ndarray_vector};
use ndarray::{Array1, Array2};

#[test]
fn test_svd_identity_matrix() {
    let identity = Array2::<f64>::eye(3);
    let svd = ndarray_svd::decompose(&identity).unwrap();

    for &sv in &svd.singular_values {
        assert_relative_eq!(sv, 1.0, epsilon = 1e-10);
    }
    assert_relative_eq!(ndarray_svd::condition_number(&svd), 1.0, epsilon = 1e-10);
    assert_eq!(ndarray_svd::rank(&svd, None), 3);
}

#[test]
fn test_svd_reconstruction() {
    let matrix =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let svd = ndarray_svd::decompose(&matrix).unwrap();
    let reconstructed = ndarray_svd::reconstruct_matrix(&svd);

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            assert_relative_eq!(matrix[[i, j]], reconstructed[[i, j]], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_truncated_svd_and_errors() {
    let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let truncated = ndarray_svd::decompose_truncated(&matrix, 1).unwrap();
    assert_eq!(truncated.singular_values.len(), 1);

    let invalid = ndarray_svd::decompose_truncated(&matrix, 0);
    assert!(matches!(invalid, Err(SVDError::InvalidInput(_))));
}

#[test]
fn test_triangular_residual() {
    let lower =
        Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Array1::from_vec(vec![2.0, 5.0, 32.0]);
    let x = ndarray_triangular::solve_lower(&lower, &rhs).unwrap();
    let reconstructed = lower.dot(&x);

    for i in 0..3 {
        assert_relative_eq!(reconstructed[i], rhs[i], epsilon = 1e-10);
    }
}

#[test]
fn test_iterative_cg_matches_direct_system() {
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b = Array1::from_vec(vec![1.0, 2.0]);
    let x = ndarray_iterative::conjugate_gradient(&a, &b, &IterativeConfig::default()).unwrap();
    let reconstructed = a.dot(&x);
    assert_relative_eq!(reconstructed[0], b[0], epsilon = 1e-8);
    assert_relative_eq!(reconstructed[1], b[1], epsilon = 1e-8);
}

#[test]
fn test_stats_covariance() {
    let matrix =
        Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();
    let cov = ndarray_stats::covariance_matrix(&matrix).unwrap();
    assert_eq!(cov.dim(), (2, 2));
}

#[test]
fn test_matrix_function_roundtrip() {
    let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let log_matrix = ndarray_matrix_functions::matrix_log_eigen(&matrix).unwrap();
    let roundtrip = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(roundtrip[[i, j]], matrix[[i, j]], epsilon = 1e-6);
        }
    }
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_regression_pca_orthogonalization_and_sylvester() {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    let regression = ndarray_regression::linear_regression(&x, &y, true).unwrap();
    assert_relative_eq!(regression.coefficients[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(regression.coefficients[1], 2.0, epsilon = 1e-8);

    let pca_input = Array2::from_shape_vec((5, 3), vec![
        1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
    ])
    .unwrap();
    let pca = ndarray_pca::compute_pca(&pca_input, Some(3)).unwrap();
    let transformed = ndarray_pca::transform(&pca_input, &pca);
    let reconstructed = ndarray_pca::inverse_transform(&transformed, &pca);
    assert_eq!(reconstructed.dim(), pca_input.dim());

    let q = ndarray_orthogonalization::gram_schmidt_classic(&pca_input).unwrap();
    assert_eq!(q.nrows(), pca_input.nrows());

    let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
    let c = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let x = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();
    assert_eq!(x.dim(), (2, 2));
}

#[test]
fn test_vector_primitives_and_workspace_paths() {
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
    let dot = ndarray_vector::dot(&a, &b).unwrap();
    assert_relative_eq!(dot, 32.0, epsilon = 1e-10);

    let left = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let right = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    let mut cosine = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    let mut workspace = PairwiseCosineWorkspace::default();
    ndarray_vector::pairwise_cosine_similarity_with_workspace_into(
        &left,
        &right,
        &mut cosine,
        &mut workspace,
    )
    .unwrap();
    assert_relative_eq!(cosine[[0, 0]], 1.0, epsilon = 1e-10);
    assert_relative_eq!(cosine[[0, 1]], 0.0, epsilon = 1e-10);

    let mut l2 = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    ndarray_vector::pairwise_l2_distance_into(&left, &right, &mut l2).unwrap();
    assert_relative_eq!(l2[[0, 0]], 0.0, epsilon = 1e-10);
}

#![cfg(feature = "arrow")]

use approx::assert_abs_diff_eq;
use arrow_array::types::{Float32Type, Float64Type, UInt32Type};
use arrow_array::{Float64Array, ListArray};
use nabled::arrow::{
    batched, cholesky, eigen, iterative, jacobian, lu, matrix, matrix_functions, optimization,
    orthogonalization, pca, polar, qr, regression, schur, sparse, stats, svd, tensor, triangular,
    vector,
};
use nabled::linalg::eigen::NonsymmetricEigenConfig;
use nabled::linalg::qr::QRConfig;
use nabled::linalg::svd::PseudoInverseConfig;
use nabled::ml::iterative::IterativeConfig;
use nabled::ml::jacobian::{JacobianConfig, JacobianError};
use nabled::ml::optimization::{
    AdamConfig, BFGSConfig, LineSearchConfig, MomentumConfig, ProjectedGradientConfig,
    RMSPropConfig, SGDConfig,
};
use ndarray::{Array1, ArrayD, IxDyn, array};
use ndarrow::{
    AsNdarray, IntoArrow, arrayd_to_fixed_shape_tensor, csr_to_extension_array,
    fixed_shape_tensor_as_array_viewd, fixed_size_list_as_array2,
};

fn tensor_arrow_f64(
    name: &str,
    shape: &[usize],
    values: Vec<f64>,
) -> (arrow_schema::Field, arrow_array::FixedSizeListArray) {
    let tensor = ArrayD::from_shape_vec(IxDyn(shape), values).unwrap();
    arrayd_to_fixed_shape_tensor(name, tensor).unwrap()
}

#[test]
fn arrow_vector_workflows_cover_real_surface() {
    let left = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let right = Float64Array::from(vec![4.0, 5.0, 6.0]);

    let dot = vector::dot(&left, &right).unwrap();
    let cosine = vector::cosine_similarity(&left, &right).unwrap();
    let cosine_distance = vector::cosine_distance(&left, &right).unwrap();
    let norm = vector::l2_norm(&left).unwrap();

    assert_abs_diff_eq!(dot, 32.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(cosine, 0.974_631_846_197_076_2, epsilon = 1.0e-12);
    assert_abs_diff_eq!(cosine_distance, 0.025_368_153_802_923_787, epsilon = 1.0e-12);
    assert_abs_diff_eq!(norm, 14.0_f64.sqrt(), epsilon = 1.0e-12);

    let left_batch = array![[1.0_f64, 0.0], [1.0, 1.0]].into_arrow().unwrap();
    let right_batch = array![[0.0_f64, 1.0], [1.0, 1.0]].into_arrow().unwrap();

    let pairwise_l2 =
        vector::pairwise_l2_distance::<Float64Type>(&left_batch, &right_batch).unwrap();
    let pairwise_l2_view = fixed_size_list_as_array2::<Float64Type>(&pairwise_l2).unwrap();
    assert_abs_diff_eq!(pairwise_l2_view[[0, 0]], 2.0_f64.sqrt(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(pairwise_l2_view[[1, 1]], 0.0_f64, epsilon = 1.0e-12);

    let pairwise_cosine =
        vector::pairwise_cosine_similarity::<Float64Type>(&left_batch, &right_batch).unwrap();
    let pairwise_cosine_view = fixed_size_list_as_array2::<Float64Type>(&pairwise_cosine).unwrap();
    assert_abs_diff_eq!(pairwise_cosine_view[[0, 0]], 0.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(pairwise_cosine_view[[1, 1]], 1.0_f64, epsilon = 1.0e-12);

    let pairwise_cosine_distance =
        vector::pairwise_cosine_distance::<Float64Type>(&left_batch, &right_batch).unwrap();
    let pairwise_cosine_distance_view =
        fixed_size_list_as_array2::<Float64Type>(&pairwise_cosine_distance).unwrap();
    assert_abs_diff_eq!(pairwise_cosine_distance_view[[0, 0]], 1.0_f64, epsilon = 1.0e-12);

    let batched_dot = vector::batched_dot::<Float64Type>(&left_batch, &right_batch).unwrap();
    let batched_dot_view = batched_dot.as_ndarray().unwrap();
    assert_abs_diff_eq!(batched_dot_view[0], 0.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(batched_dot_view[1], 2.0_f64, epsilon = 1.0e-12);
}

#[test]
fn arrow_dense_matrix_kernels_cover_scalar_and_batched_paths() {
    let left = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let right = array![[5.0_f32, 6.0], [7.0, 8.0]];
    let vector = Array1::from_vec(vec![2.0_f32, -1.0]);

    let left_arrow = left.into_arrow().unwrap();
    let right_arrow = right.into_arrow().unwrap();
    let vector_arrow = vector.into_arrow().unwrap();

    let matvec = matrix::matvec::<Float32Type>(&left_arrow, &vector_arrow).unwrap();
    let matvec_view = matvec.as_ndarray().unwrap();
    assert_abs_diff_eq!(matvec_view[0], 0.0_f32, epsilon = 1.0e-6_f32);
    assert_abs_diff_eq!(matvec_view[1], 2.0_f32, epsilon = 1.0e-6_f32);

    let matmat = matrix::matmat::<Float32Type>(&left_arrow, &right_arrow).unwrap();
    let matmat_view = fixed_size_list_as_array2::<Float32Type>(&matmat).unwrap();
    assert_abs_diff_eq!(matmat_view[[0, 0]], 19.0_f32, epsilon = 1.0e-6_f32);
    assert_abs_diff_eq!(matmat_view[[1, 1]], 50.0_f32, epsilon = 1.0e-6_f32);

    let batch_vectors = array![[1.0_f32, 0.0], [1.0, 1.0]].into_arrow().unwrap();
    let transform = array![[2.0_f32, 1.0], [0.0, 3.0]].into_arrow().unwrap();
    let output = matrix::batched_row_matvec::<Float32Type>(&batch_vectors, &transform).unwrap();
    let output_view = fixed_size_list_as_array2::<Float32Type>(&output).unwrap();
    assert_abs_diff_eq!(output_view[[0, 0]], 2.0_f32, epsilon = 1.0e-6_f32);
    assert_abs_diff_eq!(output_view[[1, 1]], 3.0_f32, epsilon = 1.0e-6_f32);
}

#[test]
fn arrow_lu_and_cholesky_cover_solve_and_inverse_paths() {
    let lu_matrix = array![[4.0_f64, 7.0], [2.0, 6.0]];
    let rhs = Float64Array::from(vec![18.0, 14.0]);
    let lu_matrix_arrow = lu_matrix.clone().into_arrow().unwrap();

    let lu_solution = lu::solve_f64(&lu_matrix_arrow, &rhs).unwrap();
    let lu_view = lu_solution.as_ndarray().unwrap();
    assert_abs_diff_eq!(lu_view[0], 1.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(lu_view[1], 2.0_f64, epsilon = 1.0e-12);

    let lu_inverse = lu::inverse_f64(&lu_matrix_arrow).unwrap();
    let lu_inverse_view = fixed_size_list_as_array2::<Float64Type>(&lu_inverse).unwrap();
    assert_abs_diff_eq!(lu_inverse_view[[0, 0]], 0.6_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(lu_inverse_view[[1, 1]], 0.4_f64, epsilon = 1.0e-12);

    let determinant = lu::determinant_f64(&lu_matrix_arrow).unwrap();
    let log_determinant = lu::log_determinant_f64(&lu_matrix_arrow).unwrap();
    assert_abs_diff_eq!(determinant, 10.0_f64, epsilon = 1.0e-12);
    assert_eq!(log_determinant.sign, 1);
    assert_abs_diff_eq!(log_determinant.ln_abs_det, 10.0_f64.ln(), epsilon = 1.0e-12);

    let chol_matrix = array![[4.0_f64, 1.0], [1.0, 3.0]];
    let chol_rhs = Float64Array::from(vec![1.0, 2.0]);
    let chol_matrix_arrow = chol_matrix.into_arrow().unwrap();

    let chol_solution = cholesky::solve_f64(&chol_matrix_arrow, &chol_rhs).unwrap();
    let chol_view = chol_solution.as_ndarray().unwrap();
    assert_abs_diff_eq!(chol_view[0], 0.090_909_090_909_090_91, epsilon = 1.0e-12);
    assert_abs_diff_eq!(chol_view[1], 0.636_363_636_363_636_4, epsilon = 1.0e-12);

    let chol_inverse = cholesky::inverse_f64(&chol_matrix_arrow).unwrap();
    let chol_inverse_view = fixed_size_list_as_array2::<Float64Type>(&chol_inverse).unwrap();
    assert_abs_diff_eq!(chol_inverse_view[[0, 0]], 0.272_727_272_727_272_7, epsilon = 1.0e-12);
    assert_abs_diff_eq!(chol_inverse_view[[1, 1]], 0.363_636_363_636_363_65, epsilon = 1.0e-12);
}

#[test]
fn arrow_qr_triangular_and_svd_cover_linear_algebra_helpers() {
    let matrix = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let matrix_arrow = matrix.clone().into_arrow().unwrap();
    let config = QRConfig::<f64>::default();

    let qr_result = qr::decompose_f64(&matrix_arrow, &config).unwrap();
    let reconstructed = qr::reconstruct_f64(&qr_result).unwrap();
    let reconstructed_view = fixed_size_list_as_array2::<Float64Type>(&reconstructed).unwrap();
    assert_abs_diff_eq!(reconstructed_view[[0, 0]], 1.0_f64, epsilon = 1.0e-10);
    assert_abs_diff_eq!(reconstructed_view[[2, 1]], 6.0_f64, epsilon = 1.0e-10);

    let rhs = Float64Array::from(vec![1.0_f64, 0.0, 1.0]);
    let least_squares = qr::solve_least_squares_f64(&matrix_arrow, &rhs, &config).unwrap();
    let least_squares_view = least_squares.as_ndarray().unwrap();
    assert_eq!(least_squares_view.len(), 2);

    let lower = array![[2.0_f64, 0.0], [1.0, 3.0]].into_arrow().unwrap();
    let upper = array![[2.0_f64, 1.0], [0.0, 4.0]].into_arrow().unwrap();
    let rhs_matrix = array![[2.0_f64, 4.0], [8.0, 10.0]].into_arrow().unwrap();

    let lower_solved = triangular::solve_lower_matrix_f64(&lower, &rhs_matrix).unwrap();
    let lower_solved_view = fixed_size_list_as_array2::<Float64Type>(&lower_solved).unwrap();
    assert_abs_diff_eq!(lower_solved_view[[0, 0]], 1.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(lower_solved_view[[1, 1]], 8.0_f64 / 3.0, epsilon = 1.0e-12);

    let upper_solved = triangular::solve_upper_matrix_f64(&upper, &rhs_matrix).unwrap();
    let upper_solved_view = fixed_size_list_as_array2::<Float64Type>(&upper_solved).unwrap();
    assert_abs_diff_eq!(upper_solved_view[[1, 0]], 2.0_f64, epsilon = 1.0e-12);

    let svd_matrix = array![[1.0_f64, 0.0], [0.0, 0.0]].into_arrow().unwrap();
    let truncated = svd::decompose_truncated_f64(&svd_matrix, 1).unwrap();
    assert_eq!(truncated.singular_values.len(), 1);
    let with_tolerance = svd::decompose_with_tolerance_f64(&svd_matrix, 1.0e-12).unwrap();
    assert_eq!(with_tolerance.singular_values.len(), 2);

    let pseudo_inverse =
        svd::pseudo_inverse_f64(&svd_matrix, PseudoInverseConfig::default()).unwrap();
    let pseudo_inverse_view = fixed_size_list_as_array2::<Float64Type>(&pseudo_inverse).unwrap();
    assert_abs_diff_eq!(pseudo_inverse_view[[0, 0]], 1.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(pseudo_inverse_view[[1, 1]], 0.0_f64, epsilon = 1.0e-12);

    let null_space = svd::null_space_f64(&svd_matrix, Some(1.0e-12)).unwrap();
    let null_space_view = fixed_size_list_as_array2::<Float64Type>(&null_space).unwrap();
    assert_eq!(null_space_view.dim(), (2, 1));
    assert_abs_diff_eq!(null_space_view[[0, 0]], 0.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(null_space_view[[1, 0]].abs(), 1.0_f64, epsilon = 1.0e-12);
}

#[test]
fn arrow_eigen_schur_polar_orthogonalization_and_matrix_functions_work() {
    let symmetric = array![[2.0_f64, 0.0], [0.0, 5.0]].into_arrow().unwrap();
    let generalized_b = array![[1.0_f64, 0.0], [0.0, 1.0]].into_arrow().unwrap();
    let nonsymmetric = array![[1.0_f64, 2.0], [0.0, 3.0]].into_arrow().unwrap();

    let symmetric_result = eigen::symmetric_f64(&symmetric).unwrap();
    let mut eigenvalues = symmetric_result.eigenvalues.to_vec();
    eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    assert_abs_diff_eq!(eigenvalues[0], 2.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(eigenvalues[1], 5.0_f64, epsilon = 1.0e-12);

    let generalized_result = eigen::generalized_f64(&symmetric, &generalized_b).unwrap();
    assert_eq!(generalized_result.eigenvalues.len(), 2);

    let nonsymmetric_result = eigen::nonsymmetric_f64(&nonsymmetric).unwrap();
    assert_eq!(nonsymmetric_result.eigenvalues.len(), 2);

    let non_config = NonsymmetricEigenConfig::default();
    let (balanced, diagonal) = eigen::balance_nonsymmetric_f64(&nonsymmetric, &non_config).unwrap();
    let balanced_view = fixed_size_list_as_array2::<Float64Type>(&balanced).unwrap();
    assert_eq!(balanced_view.dim(), (2, 2));
    assert_eq!(diagonal.len(), 2);

    let bi = eigen::nonsymmetric_bi_f64(&nonsymmetric, &non_config).unwrap();
    assert_eq!(bi.right_eigenvectors.dim(), (2, 2));

    let schur_result = schur::compute_f64(&nonsymmetric).unwrap();
    assert_eq!(schur_result.t.dim(), (2, 2));

    let polar_input = array![[2.0_f64, 0.0], [0.0, 3.0]].into_arrow().unwrap();
    let polar_result = polar::compute_f64(&polar_input).unwrap();
    assert_abs_diff_eq!(polar_result.u[[0, 0]], 1.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(polar_result.p[[1, 1]], 3.0_f64, epsilon = 1.0e-12);

    let ortho_input = array![[1.0_f64, 1.0], [0.0, 1.0]].into_arrow().unwrap();
    let gram = orthogonalization::gram_schmidt_f64(&ortho_input).unwrap();
    let gram_view = fixed_size_list_as_array2::<Float64Type>(&gram).unwrap();
    let gram_gram = gram_view.t().dot(&gram_view);
    assert_abs_diff_eq!(gram_gram[[0, 0]], 1.0_f64, epsilon = 1.0e-10);
    assert_abs_diff_eq!(gram_gram[[1, 1]], 1.0_f64, epsilon = 1.0e-10);

    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]].into_arrow().unwrap();
    let exp_identity = matrix_functions::exp_f64(&identity, 16, 1.0e-12).unwrap();
    let exp_identity_view = fixed_size_list_as_array2::<Float64Type>(&exp_identity).unwrap();
    assert_abs_diff_eq!(exp_identity_view[[0, 0]], std::f64::consts::E, epsilon = 1.0e-10);

    let positive_diag = array![[1.0_f64, 0.0], [0.0, std::f64::consts::E]].into_arrow().unwrap();
    let log_eigen = matrix_functions::log_eigen_f64(&positive_diag).unwrap();
    let log_eigen_view = fixed_size_list_as_array2::<Float64Type>(&log_eigen).unwrap();
    assert_abs_diff_eq!(log_eigen_view[[1, 1]], 1.0_f64, epsilon = 1.0e-10);

    let log_svd = matrix_functions::log_svd_f64(&positive_diag).unwrap();
    let log_svd_view = fixed_size_list_as_array2::<Float64Type>(&log_svd).unwrap();
    assert_abs_diff_eq!(log_svd_view[[1, 1]], 1.0_f64, epsilon = 1.0e-10);

    let sqrt_diag = array![[4.0_f64, 0.0], [0.0, 9.0]].into_arrow().unwrap();
    let power = matrix_functions::power_f64(&sqrt_diag, 0.5).unwrap();
    let power_view = fixed_size_list_as_array2::<Float64Type>(&power).unwrap();
    assert_abs_diff_eq!(power_view[[0, 0]], 2.0_f64, epsilon = 1.0e-10);
    assert_abs_diff_eq!(power_view[[1, 1]], 3.0_f64, epsilon = 1.0e-10);

    let sign_input = array![[-2.0_f64, 0.0], [0.0, 3.0]].into_arrow().unwrap();
    let sign = matrix_functions::sign_f64(&sign_input).unwrap();
    let sign_view = fixed_size_list_as_array2::<Float64Type>(&sign).unwrap();
    assert_abs_diff_eq!(sign_view[[0, 0]], -1.0_f64, epsilon = 1.0e-10);
    assert_abs_diff_eq!(sign_view[[1, 1]], 1.0_f64, epsilon = 1.0e-10);
}

#[test]
fn arrow_iterative_pca_regression_and_stats_work() {
    let spd = array![[4.0_f64, 1.0], [1.0, 3.0]].into_arrow().unwrap();
    let cg_rhs = Float64Array::from(vec![1.0, 2.0]);
    let iter_config = IterativeConfig::<f64>::default();

    let cg = iterative::conjugate_gradient_f64(&spd, &cg_rhs, &iter_config).unwrap();
    let cg_view = cg.as_ndarray().unwrap();
    assert_abs_diff_eq!(cg_view[0], 1.0_f64 / 11.0, epsilon = 1.0e-10);
    assert_abs_diff_eq!(cg_view[1], 7.0_f64 / 11.0, epsilon = 1.0e-10);

    let general = array![[4.0_f64, 1.0], [2.0, 3.0]].into_arrow().unwrap();
    let gmres_rhs = Float64Array::from(vec![1.0, 2.0]);
    let gmres = iterative::gmres_f64(&general, &gmres_rhs, &iter_config).unwrap();
    let gmres_view = gmres.as_ndarray().unwrap();
    assert_abs_diff_eq!(gmres_view[0], 0.1_f64, epsilon = 1.0e-8);
    assert_abs_diff_eq!(gmres_view[1], 0.6_f64, epsilon = 1.0e-8);

    let pca_input =
        array![[1.0_f64, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]].into_arrow().unwrap();
    let pca_result = pca::compute_f64(&pca_input, Some(1)).unwrap();
    assert_eq!(pca_result.components.dim(), (1, 2));
    let projected = pca::transform_f64(&pca_input, &pca_result).unwrap();
    let reconstructed = pca::inverse_transform_f64(&projected, &pca_result).unwrap();
    let reconstructed_view = fixed_size_list_as_array2::<Float64Type>(&reconstructed).unwrap();
    assert_eq!(reconstructed_view.dim(), (4, 2));

    let x = array![[1.0_f64], [2.0], [3.0], [4.0]].into_arrow().unwrap();
    let y = Float64Array::from(vec![2.0, 4.0, 6.0, 8.0]);
    let regression_result = regression::linear_regression_f64(&x, &y, false).unwrap();
    assert_abs_diff_eq!(regression_result.coefficients[0], 2.0_f64, epsilon = 1.0e-10);

    let stats_input = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]].into_arrow().unwrap();
    let means = stats::column_means_f64(&stats_input).unwrap();
    let means_view = means.as_ndarray().unwrap();
    assert_abs_diff_eq!(means_view[0], 3.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(means_view[1], 4.0_f64, epsilon = 1.0e-12);

    let centered = stats::center_columns_f64(&stats_input).unwrap();
    let centered_view = fixed_size_list_as_array2::<Float64Type>(&centered).unwrap();
    assert_abs_diff_eq!(centered_view[[0, 0]], -2.0_f64, epsilon = 1.0e-12);

    let covariance = stats::covariance_matrix_f64(&stats_input).unwrap();
    let covariance_view = fixed_size_list_as_array2::<Float64Type>(&covariance).unwrap();
    assert_abs_diff_eq!(covariance_view[[0, 0]], 4.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(covariance_view[[0, 1]], 4.0_f64, epsilon = 1.0e-12);

    let correlation = stats::correlation_matrix_f64(&stats_input).unwrap();
    let correlation_view = fixed_size_list_as_array2::<Float64Type>(&correlation).unwrap();
    assert_abs_diff_eq!(correlation_view[[0, 1]], 1.0_f64, epsilon = 1.0e-12);
}

#[test]
fn arrow_jacobian_work() {
    let x = Float64Array::from(vec![1.0, 2.0]);
    let jacobian_config = JacobianConfig::<f64>::default();

    let vector_fn = |values: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
        Ok(array![values[0] + values[1], values[0] * values[1]])
    };
    let scalar_fn = |values: &Array1<f64>| -> Result<f64, JacobianError> {
        Ok(values[0] * values[0] + 3.0_f64 * values[1] * values[1])
    };

    let jac =
        jacobian::numerical_jacobian::<Float64Type, _>(&vector_fn, &x, &jacobian_config).unwrap();
    let jac_view = fixed_size_list_as_array2::<Float64Type>(&jac).unwrap();
    assert_abs_diff_eq!(jac_view[[0, 0]], 1.0_f64, epsilon = 1.0e-4);
    assert_abs_diff_eq!(jac_view[[0, 1]], 1.0_f64, epsilon = 1.0e-4);
    assert_abs_diff_eq!(jac_view[[1, 0]], 2.0_f64, epsilon = 1.0e-4);
    assert_abs_diff_eq!(jac_view[[1, 1]], 1.0_f64, epsilon = 1.0e-4);

    let jac_central =
        jacobian::numerical_jacobian_central::<Float64Type, _>(&vector_fn, &x, &jacobian_config)
            .unwrap();
    let jac_central_view = fixed_size_list_as_array2::<Float64Type>(&jac_central).unwrap();
    assert_abs_diff_eq!(jac_central_view[[1, 0]], 2.0_f64, epsilon = 1.0e-6);

    let gradient =
        jacobian::numerical_gradient::<Float64Type, _>(&scalar_fn, &x, &jacobian_config).unwrap();
    let gradient_view = gradient.as_ndarray().unwrap();
    assert_abs_diff_eq!(gradient_view[0], 2.0_f64, epsilon = 1.0e-4);
    assert_abs_diff_eq!(gradient_view[1], 12.0_f64, epsilon = 1.0e-3);

    let hessian =
        jacobian::numerical_hessian::<Float64Type, _>(&scalar_fn, &x, &jacobian_config).unwrap();
    let hessian_view = fixed_size_list_as_array2::<Float64Type>(&hessian).unwrap();
    assert_abs_diff_eq!(hessian_view[[0, 0]], 2.0_f64, epsilon = 1.0e-2);
    assert_abs_diff_eq!(hessian_view[[1, 1]], 6.0_f64, epsilon = 1.0e-2);
}

#[test]
fn arrow_optimization_work() {
    let point = Float64Array::from(vec![5.0]);
    let direction = Float64Array::from(vec![-4.0]);
    let objective = |values: &Array1<f64>| (values[0] - 3.0_f64).powi(2);
    let gradient_fn = |values: &Array1<f64>| array![2.0_f64 * (values[0] - 3.0_f64)];

    let line_search = optimization::backtracking_line_search::<Float64Type, _, _>(
        &point,
        &direction,
        objective,
        gradient_fn,
        &LineSearchConfig::<f64>::default(),
    )
    .unwrap();
    assert!(line_search.is_finite());
    assert!(line_search > 0.0_f64);

    let gd = optimization::gradient_descent::<Float64Type, _, _>(
        &point,
        objective,
        gradient_fn,
        &SGDConfig { learning_rate: 0.25, max_iterations: 128, tolerance: 1.0e-8 },
    )
    .unwrap();
    assert_abs_diff_eq!(gd.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-6);

    let adam =
        optimization::adam::<Float64Type, _, _>(&point, objective, gradient_fn, &AdamConfig {
            learning_rate:  0.2,
            beta1:          0.9,
            beta2:          0.999,
            epsilon:        1.0e-8,
            max_iterations: 4096,
            tolerance:      1.0e-6,
        })
        .unwrap();
    assert_abs_diff_eq!(adam.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-3);

    let momentum = optimization::momentum_descent::<Float64Type, _, _>(
        &point,
        objective,
        gradient_fn,
        &MomentumConfig {
            learning_rate:  0.05,
            momentum:       0.8,
            max_iterations: 256,
            tolerance:      1.0e-8,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(momentum.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-3);

    let rmsprop = optimization::rmsprop::<Float64Type, _, _>(
        &point,
        objective,
        gradient_fn,
        &RMSPropConfig {
            learning_rate:  0.1,
            rho:            0.9,
            epsilon:        1.0e-8,
            max_iterations: 4096,
            tolerance:      1.0e-6,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(rmsprop.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 5.0e-3);

    let projected = optimization::projected_gradient_descent_box::<Float64Type, _, _>(
        &point,
        objective,
        gradient_fn,
        &Float64Array::from(vec![2.5]),
        &Float64Array::from(vec![3.5]),
        &ProjectedGradientConfig {
            learning_rate:  0.25,
            max_iterations: 128,
            tolerance:      1.0e-8,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(projected.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-6);

    let sgd = optimization::stochastic_gradient_descent::<Float64Type, _>(
        &point,
        |values: &Array1<f64>, _iteration| array![2.0_f64 * (values[0] - 3.0_f64)],
        &SGDConfig { learning_rate: 0.25, max_iterations: 128, tolerance: 1.0e-8 },
    )
    .unwrap();
    assert_abs_diff_eq!(sgd.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-6);

    let bfgs =
        optimization::bfgs::<Float64Type, _, _>(&point, objective, gradient_fn, &BFGSConfig {
            step_size:           0.5,
            max_iterations:      64,
            tolerance:           1.0e-8,
            curvature_tolerance: 1.0e-12,
        })
        .unwrap();
    assert_abs_diff_eq!(bfgs.as_ndarray().unwrap()[0], 3.0_f64, epsilon = 1.0e-6);
}

#[test]
fn arrow_sparse_csr_columns_and_extension_work() {
    let indices = ListArray::from_iter_primitive::<UInt32Type, _, _>([
        Some(vec![Some(0_u32), Some(2)]),
        Some(vec![Some(1)]),
        Some(vec![Some(0), Some(3)]),
    ]);
    let values = ListArray::from_iter_primitive::<Float64Type, _, _>([
        Some(vec![Some(1.0_f64), Some(5.0)]),
        Some(vec![Some(2.0)]),
        Some(vec![Some(3.0), Some(4.0)]),
    ]);
    let vector = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);

    let matvec_columns =
        sparse::matvec_csr_columns::<Float64Type>(&indices, &values, 4, &vector).unwrap();
    let matvec_columns_view = matvec_columns.as_ndarray().unwrap();
    assert_abs_diff_eq!(matvec_columns_view[0], 16.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(matvec_columns_view[1], 4.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(matvec_columns_view[2], 19.0_f64, epsilon = 1.0e-12);

    let (field, extension) =
        csr_to_extension_array("sparse", 4, vec![0_i32, 2, 3, 5], vec![0_u32, 2, 1, 0, 3], vec![
            1.0_f64, 5.0, 2.0, 3.0, 4.0,
        ])
        .unwrap();
    let dense = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
    let dense_arrow = dense.into_arrow().unwrap();

    let matmat_extension =
        sparse::matmat_dense_csr_extension::<Float64Type>(&field, &extension, &dense_arrow)
            .unwrap();
    let matmat_extension_view =
        fixed_size_list_as_array2::<Float64Type>(&matmat_extension).unwrap();
    assert_abs_diff_eq!(matmat_extension_view[[0, 0]], 6.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(matmat_extension_view[[0, 1]], 5.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(matmat_extension_view[[1, 1]], 2.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(matmat_extension_view[[2, 0]], 11.0_f64, epsilon = 1.0e-12);

    let (spd_field, spd_extension) =
        csr_to_extension_array("spd", 2, vec![0_i32, 2, 4], vec![0_u32, 1, 0, 1], vec![
            4.0_f64, 1.0, 1.0, 3.0,
        ])
        .unwrap();
    let spd_rhs = Float64Array::from(vec![1.0, 2.0]);

    let sparse_lu =
        sparse::sparse_lu_solve_csr_extension::<Float64Type>(&spd_field, &spd_extension, &spd_rhs)
            .unwrap();
    let sparse_lu_view = sparse_lu.as_ndarray().unwrap();
    assert_abs_diff_eq!(sparse_lu_view[0], 1.0_f64 / 11.0, epsilon = 1.0e-10);
    assert_abs_diff_eq!(sparse_lu_view[1], 7.0_f64 / 11.0, epsilon = 1.0e-10);

    let jacobi = sparse::jacobi_solve_csr_extension::<Float64Type>(
        &spd_field,
        &spd_extension,
        &spd_rhs,
        1.0e-12,
        100,
    )
    .unwrap();
    let jacobi_view = jacobi.as_ndarray().unwrap();
    assert_abs_diff_eq!(jacobi_view[0], 1.0_f64 / 11.0, epsilon = 1.0e-6);
    assert_abs_diff_eq!(jacobi_view[1], 7.0_f64 / 11.0, epsilon = 1.0e-6);

    let cg = sparse::conjugate_gradient_solve_csr_extension::<Float64Type>(
        &spd_field,
        &spd_extension,
        &spd_rhs,
        1.0e-12,
        20,
    )
    .unwrap();
    let cg_view = cg.as_ndarray().unwrap();
    assert_abs_diff_eq!(cg_view[0], 1.0_f64 / 11.0, epsilon = 1.0e-10);
    assert_abs_diff_eq!(cg_view[1], 7.0_f64 / 11.0, epsilon = 1.0e-10);
}

#[test]
fn arrow_tensor_workflows_cover_higher_rank_paths() {
    let (field, array) =
        tensor_arrow_f64("tensor", &[2, 2, 2], vec![3.0, 4.0, 0.0, 5.0, 8.0, 15.0, 7.0, 24.0]);

    let (summed_field, summed_array) =
        tensor::sum_last_axis::<Float64Type>(&field, &array).unwrap();
    let summed_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&summed_field, &summed_array).unwrap();
    assert_eq!(summed_view.shape(), &[2, 2]);
    assert_abs_diff_eq!(summed_view[[0, 0]], 7.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(summed_view[[1, 1]], 31.0_f64, epsilon = 1.0e-12);

    let (norm_field, norm_array) =
        tensor::l2_norm_last_axis::<Float64Type>(&field, &array).unwrap();
    let norm_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&norm_field, &norm_array).unwrap();
    assert_eq!(norm_view.shape(), &[2, 2]);
    assert_abs_diff_eq!(norm_view[[0, 0]], 5.0_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(norm_view[[1, 1]], 25.0_f64, epsilon = 1.0e-12);

    let (_normalized_field, normalized_array) =
        tensor::normalize_last_axis::<Float64Type>(&field, &array).unwrap();
    let normalized_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&field, &normalized_array).unwrap();
    assert_abs_diff_eq!(normalized_view[[0, 0, 0]], 0.6_f64, epsilon = 1.0e-12);
    assert_abs_diff_eq!(normalized_view[[0, 0, 1]], 0.8_f64, epsilon = 1.0e-12);

    let (other_field, other_array) =
        tensor_arrow_f64("other", &[2, 2, 2], vec![1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let (batched_dot_field, batched_dot_array) =
        tensor::batched_dot_last_axis::<Float64Type>(&field, &array, &other_field, &other_array)
            .unwrap();
    let batched_dot_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&batched_dot_field, &batched_dot_array)
            .unwrap();
    assert_eq!(batched_dot_view.shape(), &[2, 2]);
    assert_abs_diff_eq!(batched_dot_view[[0, 0]], 7.0_f64, epsilon = 1.0e-12);

    let (permuted_field, permuted_array) =
        tensor::permute_axes::<Float64Type>(&field, &array, &[1, 0, 2]).unwrap();
    let permuted_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&permuted_field, &permuted_array).unwrap();
    assert_eq!(permuted_view.shape(), &[2, 2, 2]);
    assert_abs_diff_eq!(permuted_view[[1, 0, 0]], 0.0_f64, epsilon = 1.0e-12);

    let (contract_left_field, contract_left_array) = tensor_arrow_f64("left", &[2, 3, 2], vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    ]);
    let (contract_right_field, contract_right_array) =
        tensor_arrow_f64("right", &[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0]);
    let (contracted_field, contracted_array) = tensor::contract_axes::<Float64Type>(
        &contract_left_field,
        &contract_left_array,
        &contract_right_field,
        &contract_right_array,
        &[2],
        &[1],
    )
    .unwrap();
    let contracted_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&contracted_field, &contracted_array)
            .unwrap();
    assert_eq!(contracted_view.shape(), &[2, 3, 2, 2]);

    let (left_batch_field, left_batch_array) =
        tensor_arrow_f64("left_batch", &[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 2.0]);
    let (right_batch_field, right_batch_array) =
        tensor_arrow_f64("right_batch", &[2, 2, 2], vec![5.0, 6.0, 7.0, 8.0, 1.0, 0.0, 0.0, 1.0]);
    let (batched_mm_field, batched_mm_array) = tensor::batched_matmul_last_two::<Float64Type>(
        &left_batch_field,
        &left_batch_array,
        &right_batch_field,
        &right_batch_array,
    )
    .unwrap();
    let batched_mm_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&batched_mm_field, &batched_mm_array)
            .unwrap();
    assert_eq!(batched_mm_view.shape(), &[2, 2, 2]);
    assert_abs_diff_eq!(batched_mm_view[[0, 0, 0]], 19.0_f64, epsilon = 1.0e-12);

    let cube_vectors = array![[1.0_f64, 0.0], [1.0, 1.0]].into_arrow().unwrap();
    let cube_matvec =
        tensor::cube_matvec::<Float64Type>(&left_batch_field, &left_batch_array, &cube_vectors)
            .unwrap();
    let cube_matvec_view = fixed_size_list_as_array2::<Float64Type>(&cube_matvec).unwrap();
    assert_abs_diff_eq!(cube_matvec_view[[0, 0]], 1.0_f64, epsilon = 1.0e-12);

    let (cube_mm_field, cube_mm_array) = tensor::cube_matmat::<Float64Type>(
        &left_batch_field,
        &left_batch_array,
        &right_batch_field,
        &right_batch_array,
    )
    .unwrap();
    let cube_mm_view =
        fixed_shape_tensor_as_array_viewd::<Float64Type>(&cube_mm_field, &cube_mm_array).unwrap();
    assert_eq!(cube_mm_view.shape(), &[2, 2, 2]);
}

#[test]
fn arrow_batched_decomposition_wrappers_work() {
    let (field, array) =
        tensor_arrow_f64("batched", &[2, 2, 2], vec![4.0, 1.0, 1.0, 3.0, 2.0, 0.0, 0.0, 5.0]);
    let qr_config = QRConfig::<f64>::default();

    let qr_results = batched::qr_f64(&field, &array, &qr_config).unwrap();
    assert_eq!(qr_results.len(), 2);

    let svd_results = batched::svd_f64(&field, &array).unwrap();
    assert_eq!(svd_results.len(), 2);

    let lu_results = batched::lu_f64(&field, &array).unwrap();
    assert_eq!(lu_results.len(), 2);

    let cholesky_results = batched::cholesky_f64(&field, &array).unwrap();
    assert_eq!(cholesky_results.len(), 2);

    let eigen_results = batched::symmetric_eigen_f64(&field, &array).unwrap();
    assert_eq!(eigen_results.len(), 2);
}

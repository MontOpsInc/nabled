//! Physical AI integration scenarios S1–S8.

use approx::assert_relative_eq;
use nabled::control::lqr::{LqrResult, discrete_lqr};
use nabled::kinematics::chain::{ChainSpec, DhConvention, JointType};
use nabled::kinematics::fk::{end_effector_pose, fk_view};
use nabled::kinematics::ik::{IkConfig, inverse_kinematics_dls};
use nabled::kinematics::jacobian::{jacobian, jacobian_translation};
use nabled::ml::stats::rolling::{rolling_covariance, rolling_covariance_view};
use nabled::model::dh::to_chain_spec;
use nabled::model::urdf::from_urdf_file;
use nabled::sensor::kalman::{KalmanState, predict, update};
use ndarray::{arr1, arr2};

fn planar2r_chain() -> ChainSpec<f64> {
    ChainSpec::from_dh(
        DhConvention::Standard,
        vec![JointType::Revolute, JointType::Revolute],
        arr1(&[1.0, 1.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[0.0, 0.0]),
    )
    .expect("valid 2R chain")
}

/// S1: 2R planar FK + Jacobian vs reference.
#[test]
fn s1_planar_2r_fk_jacobian() {
    let chain = planar2r_chain();
    let q = arr1(&[0.3_f64, 0.5]);
    let pose = end_effector_pose(&chain, &q).expect("fk");
    let expected_x = q[0].cos() + (q[0] + q[1]).cos();
    let expected_y = q[0].sin() + (q[0] + q[1]).sin();
    assert_relative_eq!(pose.translation[0], expected_x, epsilon = 1e-10);
    assert_relative_eq!(pose.translation[1], expected_y, epsilon = 1e-10);

    let j = jacobian_translation(&chain, &q).expect("jacobian");
    let h = 1e-6;
    for col in 0..2 {
        let mut q_plus = q.clone();
        q_plus[col] += h;
        let pose_plus = fk_view(&chain, &q_plus.view()).unwrap();
        let mut q_minus = q.clone();
        q_minus[col] -= h;
        let pose_minus = fk_view(&chain, &q_minus.view()).unwrap();
        let deriv_x = (pose_plus.translation[0] - pose_minus.translation[0]) / (2.0 * h);
        let deriv_y = (pose_plus.translation[1] - pose_minus.translation[1]) / (2.0 * h);
        assert_relative_eq!(j[[0, col]], deriv_x, epsilon = 1e-5);
        assert_relative_eq!(j[[1, col]], deriv_y, epsilon = 1e-5);
    }
}

/// S2: 6-DOF DH arm FK + Jacobian.
#[test]
fn s2_six_dof_fk_jacobian() {
    let chain = ChainSpec::from_dh(
        DhConvention::Standard,
        vec![JointType::Revolute; 6],
        arr1(&[0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]),
        arr1(&[
            std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            0.0,
        ]),
        arr1(&[0.089_159, 0.0, 0.0, 0.433_07, 0.0, 0.0]),
        arr1(&[0.0; 6]),
    )
    .expect("valid 6-DOF chain");
    let q = arr1(&[0.0_f64, -0.3, 0.5, 0.2, -0.1, 0.4]);
    let pose = end_effector_pose(&chain, &q).expect("fk");
    assert!(pose.translation.iter().all(|v| v.is_finite()));
    let j = jacobian(&chain, &q).expect("jacobian");
    assert_eq!(j.nrows(), 6);
    assert_eq!(j.ncols(), 6);
}

/// S3: DLS IK to target SE(3) pose.
#[test]
fn s3_dls_ik_to_target_pose() {
    let chain = ChainSpec::from_dh(
        DhConvention::Standard,
        vec![JointType::Revolute; 6],
        arr1(&[0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]),
        arr1(&[
            std::f64::consts::FRAC_PI_2,
            0.0,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            0.0,
        ]),
        arr1(&[0.089_159, 0.0, 0.0, 0.433_07, 0.0, 0.0]),
        arr1(&[0.0; 6]),
    )
    .expect("valid 6-DOF chain");
    let q_target = arr1(&[0.2_f64, -0.3, 0.5, 0.1, -0.2, 0.4]);
    let target = fk_view(&chain, &q_target.view()).expect("fk");
    let q_init = q_target.clone();
    let q = inverse_kinematics_dls(&chain, &q_init, &target, &IkConfig::default()).expect("ik");
    let achieved = fk_view(&chain, &q.view()).expect("fk");
    let err = nabled::kinematics::pose_error(&achieved, &target).expect("pose error");
    assert!(err.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-2);
}

/// S4: URDF minimal → model → FK.
#[test]
fn s4_urdf_model_fk() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/physical_ai/planar2r.urdf");
    let model = from_urdf_file::<f64>(path).expect("urdf");
    let chain = to_chain_spec(&model).expect("chain");
    let q = arr1(&[0.2_f64, 0.3]);
    let pose = end_effector_pose(&chain, &q).expect("fk");
    assert!(pose.translation[0].is_finite());
}

/// S5: RNEA τ for known q, qd, qdd (dynamics stub).
#[test]
#[ignore = "dynamics RNEA reference not yet populated"]
fn s5_rnea_tau() {}

/// S6: Discrete LQR stabilizes double integrator.
#[test]
fn s6_discrete_lqr_double_integrator() {
    let dt = 0.1_f64;
    let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b = arr2(&[[0.0], [dt]]);
    let q = arr2(&[[10.0, 0.0], [0.0, 1.0]]);
    let r = arr2(&[[0.1]]);
    let LqrResult { gain, riccati: _ } = discrete_lqr(&a, &b, &q, &r).expect("lqr");
    let closed = &a - &b.dot(&gain);
    let eig = nabled::linalg::eigen::nonsymmetric(&closed).expect("eigen");
    for lambda in &eig.eigenvalues {
        let mag = (lambda.re * lambda.re + lambda.im * lambda.im).sqrt();
        assert!(mag < 1.0);
    }
}

/// S7: Linear Kalman fuses synthetic measurements.
#[test]
fn s7_linear_kalman_fusion() {
    let state = KalmanState { mean: arr1(&[0.0_f64]), covariance: arr2(&[[1.0]]) };
    let transition = arr2(&[[1.0]]);
    let process_cov = arr2(&[[0.01]]);
    let predicted = predict(&state, &transition.view(), &process_cov.view()).expect("predict");
    let observation = arr2(&[[1.0]]);
    let measurement_cov = arr2(&[[0.05]]);
    let measurement = arr1(&[1.0]);
    let updated =
        update(&predicted, &measurement.view(), &observation.view(), &measurement_cov.view())
            .expect("update");
    assert!(updated.mean[0] > 0.7);
    assert!(updated.covariance[[0, 0]] < 1.0);
}

/// S8: Rolling covariance across chunked columns.
#[test]
fn s8_rolling_covariance_chunked() {
    let matrix = arr2(&[[1.0_f64, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]);
    let window = 3;
    let cov_full = rolling_covariance(&matrix.view(), window);
    let chunk1 = rolling_covariance_view(&matrix.slice(ndarray::s![0..3, ..]), window);
    let chunk2 = rolling_covariance_view(&matrix.slice(ndarray::s![2..5, ..]), window);
    assert_relative_eq!(cov_full[[2, 0]], chunk1[[2, 0]], epsilon = 1e-10);
    assert_relative_eq!(cov_full[[4, 3]], chunk2[[2, 3]], epsilon = 1e-10);
}

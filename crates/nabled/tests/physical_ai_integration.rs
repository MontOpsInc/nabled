//! Physical AI integration scenarios S1–S22.

#![expect(clippy::many_single_char_names, clippy::cast_precision_loss)]

use approx::assert_relative_eq;
use nabled::control::dare::{dare_residual_norm, dare_solve};
use nabled::control::gramian::controllability_gramian;
use nabled::control::lqr::{LqrResult, discrete_lqr};
use nabled::control::observer::luenberger_gain;
use nabled::control::pole::place_poles;
use nabled::dynamics::config::DynamicsConfig;
use nabled::dynamics::fd::forward_dynamics_with_config;
use nabled::dynamics::rnea::rnea_with_config;
use nabled::kinematics::chain::{ChainSpec, DhConvention, JointType};
use nabled::kinematics::fk::{end_effector_pose, fk_view};
use nabled::kinematics::ik::{IkConfig, inverse_kinematics_dls};
use nabled::kinematics::jacobian::{jacobian, jacobian_translation};
use nabled::kinematics::tree::{end_effector_pose_tree, jacobian_tree};
use nabled::ml::stats::rolling::{rolling_covariance, rolling_covariance_view};
use nabled::model::dh::{extract_chain_spec, to_chain_spec};
use nabled::model::fixture::{load_planar2r_json, load_y_branch_json};
use nabled::model::urdf::from_urdf_file;
use nabled::sensor::camera::{PinholeIntrinsics, pinhole_project};
use nabled::sensor::ekf::{EkConfig, EkModel, ekf_predict, ekf_update};
use nabled::sensor::imu::strapdown_predict;
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
    let q_init = arr1(&[0.0; 6]);
    let config = IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() };
    let q = inverse_kinematics_dls(&chain, &q_init, &target, &config).expect("ik");
    let achieved = fk_view(&chain, &q.view()).expect("fk");
    let err = nabled::kinematics::pose_error(&achieved, &target).expect("pose error");
    assert!(err.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-3);
}

/// S4: URDF minimal → model → FK.
#[test]
fn s4_urdf_model_fk() {
    let fixture = load_planar2r_json().expect("fixture");
    let urdf_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/physical_ai/planar2r.urdf");
    let model = from_urdf_file::<f64>(urdf_path).expect("urdf");
    let chain = to_chain_spec(&model).expect("chain");

    for case in &fixture.cases {
        if case.ee_translation.as_ref().is_none_or(|t| t.len() != 3) {
            continue;
        }
        let ee = case.ee_translation.as_ref().expect("ee");
        let q = arr1(&case.q);
        let pose = end_effector_pose(&chain, &q).expect("fk");
        assert_relative_eq!(pose.translation[0], ee[0], epsilon = 1e-10);
        assert_relative_eq!(pose.translation[1], ee[1], epsilon = 1e-10);
        assert_relative_eq!(pose.translation[2], ee[2], epsilon = 1e-10);
    }
}

/// S5: RNEA τ vs forward-dynamics round-trip on planar 2R fixture.
#[test]
fn s5_rnea_tau() {
    let fixture = load_planar2r_json().expect("fixture");
    let model = fixture.to_robot_model::<f64>().expect("model");
    let chain = fixture.to_chain_spec::<f64>().expect("chain");
    let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
    let config = DynamicsConfig { gravity };
    let case =
        fixture.cases.iter().find(|c| c.qd.is_some() && c.qdd.is_some()).expect("dynamics case");
    let q = arr1(&case.q);
    let qd = arr1(case.qd.as_ref().expect("qd"));
    let qdd = arr1(case.qdd.as_ref().expect("qdd"));
    let tau = rnea_with_config(&model, &chain, &q, &qd, &qdd.view(), &config).expect("rnea");
    assert!(tau.iter().all(|v| v.is_finite()));
    let recovered =
        forward_dynamics_with_config(&model, &chain, &q, &qd, &tau.view(), &config).expect("fd");
    assert_relative_eq!(recovered, qdd, epsilon = 1e-6);
}

/// S5b: Forward dynamics round-trip on planar 2R fixture.
#[test]
fn s5b_forward_dynamics_round_trip() {
    let fixture = load_planar2r_json().expect("fixture");
    let model = fixture.to_robot_model::<f64>().expect("model");
    let chain = fixture.to_chain_spec::<f64>().expect("chain");
    let config = DynamicsConfig { gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]) };
    let q = arr1(&[0.2_f64, 0.4]);
    let qd = arr1(&[0.05_f64, -0.1]);
    let qdd = arr1(&[0.3_f64, 0.15]);
    let tau = rnea_with_config(&model, &chain, &q, &qd, &qdd.view(), &config).expect("rnea");
    let qdd_fd =
        forward_dynamics_with_config(&model, &chain, &q, &qd, &tau.view(), &config).expect("fd");
    assert_relative_eq!(qdd_fd, qdd, epsilon = 1e-6);
}

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

/// S12: sine at known bin → dominant frequency detection.
#[cfg(feature = "signal")]
#[test]
fn s12_dominant_frequency_sine_bin() {
    use nabled::linalg::signal::fft::{bin_to_hz, dominant_frequency};

    let n = 128;
    let bin = 7;
    let sample_rate = 480.0_f64;
    let freq = bin as f64 * sample_rate / n as f64;
    let signal: ndarray::Array1<f64> = ndarray::Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / sample_rate;
        (2.0 * std::f64::consts::PI * freq * t).sin()
    }));
    let peak_bin = dominant_frequency(&signal.view()).expect("dominant bin");
    assert_eq!(peak_bin, bin);
    assert_relative_eq!(
        bin_to_hz(peak_bin, n, sample_rate),
        freq,
        epsilon = sample_rate / n as f64
    );
}

/// S13: real FFT round-trip through facade.
#[cfg(feature = "signal")]
#[test]
fn s13_rfft_irfft_round_trip() {
    use nabled::linalg::signal::fft::{irfft, rfft};

    let signal = arr1(&[0.2_f64, -0.5, 1.0, 0.0, -1.0, 0.3, 0.7, -0.2]);
    let spectrum = rfft(&signal.view()).expect("rfft");
    let reconstructed = irfft(&spectrum).expect("irfft");
    for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
        assert_relative_eq!(orig, recon, epsilon = 1e-10);
    }
}

/// S14: full autocorrelation peaks at known period.
#[cfg(feature = "signal")]
#[test]
fn s14_autocorrelation_period_peak() {
    use nabled::linalg::signal::correlation::autocorrelation_full;

    let period = 10;
    let n = 80;
    let signal: ndarray::Array1<f64> = ndarray::Array1::from_iter(
        (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).cos()),
    );
    let acf = autocorrelation_full(&signal.view()).expect("autocorr");
    let (peak_lag, _) = acf
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("nonzero lag peak");
    assert_eq!(peak_lag, period);
}

/// S9: DARE solution satisfies algebraic residual.
#[test]
fn s9_dare_algebraic_residual() {
    let dt = 0.1_f64;
    let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b = arr2(&[[0.0], [dt]]);
    let q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let r = arr2(&[[1.0]]);
    let p = dare_solve(&a, &b, &q, &r).expect("dare");
    let norm = dare_residual_norm(&a, &b, &q, &r, &p).expect("residual");
    assert!(norm < 1e-8, "residual norm {norm}");
}

/// S10: Continuous gramian satisfies Lyapunov equation.
#[test]
fn s10_continuous_gramian_lyapunov() {
    let a = arr2(&[[-1.0, 0.0], [0.0, -2.0]]);
    let b = arr2(&[[1.0], [0.0]]);
    let w = controllability_gramian(&a, &b).expect("gramian");
    let bb = b.dot(&b.t());
    let residual = a.dot(&w) + w.dot(&a.t()) + bb;
    let norm: f64 = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-8, "residual norm {norm}");
}

/// S15: Luenberger observer tracks plant state.
#[test]
fn s15_luenberger_observer_tracks_state() {
    let dt = 0.05_f64;
    let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b = arr2(&[[0.0], [dt]]);
    let c = arr2(&[[1.0, 0.0]]);
    let l = luenberger_gain(&a, &c, &[-0.5, -0.6]).expect("observer gain");
    let mut x = arr1(&[1.0_f64, 0.5]);
    let mut x_hat = arr1(&[0.0_f64, 0.0]);
    let u = 0.2_f64;
    for _ in 0..80 {
        let y = c.dot(&x);
        x = a.dot(&x) + &(b.column(0).to_owned() * u);
        let innovation = &y - &c.dot(&x_hat);
        x_hat = a.dot(&x_hat)
            + &(b.column(0).to_owned() * u)
            + &(l.column(0).to_owned() * innovation[[0]]);
    }
    let err = (&x - &x_hat).mapv(|v| v * v).sum().sqrt();
    assert!(err < 1e-2, "observer error {err}");
}

/// S16: Pole placement assigns requested closed-loop eigenvalues.
#[test]
fn s16_pole_placement_double_integrator() {
    let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
    let b = arr2(&[[0.0], [1.0]]);
    let poles = [-1.0_f64, -2.0];
    let k = place_poles(&a, &b, &poles).expect("gain");
    let closed = &a - &b.dot(&k);
    let eig = nabled::linalg::eigen::nonsymmetric(&closed).expect("eigen");
    for &pole in &poles {
        let matched = eig
            .eigenvalues
            .iter()
            .any(|lambda| (lambda.re - pole).abs() < 1e-6 && lambda.im.abs() < 1e-6);
        assert!(matched, "missing pole {pole}");
    }
}

/// S21: LQR gain matches `(R + B' P B)⁻¹ B' P A`.
#[test]
fn s21_lqr_algebraic_consistency() {
    let dt = 0.1_f64;
    let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b = arr2(&[[0.0], [dt]]);
    let q = arr2(&[[10.0, 0.0], [0.0, 1.0]]);
    let r = arr2(&[[0.1]]);
    let LqrResult { gain, riccati: p } = discrete_lqr(&a, &b, &q, &r).expect("lqr");
    let bpb = b.t().dot(&p.dot(&b)) + &r;
    let bpb_inv = nabled::linalg::lu::inverse(&bpb).expect("inverse");
    let manual = bpb_inv.dot(&b.t()).dot(&p).dot(&a);
    assert_relative_eq!(gain, manual, epsilon = 1e-8);
}

/// S11: Multi-dimensional linear Kalman tracking.
#[test]
fn s11_multi_d_kalman_tracking() {
    let state =
        KalmanState { mean: arr1(&[0.0_f64, 0.0]), covariance: arr2(&[[1.0, 0.0], [0.0, 1.0]]) };
    let f = arr2(&[[1.0, 0.1], [0.0, 1.0]]);
    let q = arr2(&[[0.01, 0.0], [0.0, 0.01]]);
    let predicted = predict(&state, &f.view(), &q.view()).expect("predict");
    let h = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let r = arr2(&[[0.05, 0.0], [0.0, 0.05]]);
    let z = arr1(&[0.8_f64, 0.6]);
    let updated = update(&predicted, &z.view(), &h.view(), &r.view()).expect("update");
    assert!(updated.mean[0] > 0.4);
    assert!(updated.mean[1] > 0.3);
}

/// S17: EKF update improves scalar nonlinear estimate.
#[test]
fn s17_ekf_nonlinear_update() {
    let model = EkModel {
        predict_state: |x: &ndarray::ArrayView1<'_, f64>| arr1(&[x[0].cos()]),
        predict_jacobian: |x: &ndarray::ArrayView1<'_, f64>| arr2(&[[-x[0].sin()]]),
        measure: |x: &ndarray::ArrayView1<'_, f64>| arr1(&[x[0]]),
        measure_jacobian: |_: &ndarray::ArrayView1<'_, f64>| arr2(&[[1.0]]),
    };
    let config = EkConfig { process_noise: arr2(&[[0.01]]), measurement_noise: arr2(&[[0.05]]) };
    let state = KalmanState { mean: arr1(&[0.2_f64]), covariance: arr2(&[[1.0]]) };
    let predicted = ekf_predict(&state, &model, &config).expect("predict");
    let updated = ekf_update(&predicted, &arr1(&[0.9]).view(), &model, &config).expect("update");
    assert!(updated.mean[0] > state.mean[0]);
}

/// S18: Pinhole camera projection.
#[test]
fn s18_pinhole_camera_projection() {
    let intrinsics = PinholeIntrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
    let point = arr1(&[0.1_f64, 0.2, 1.0]);
    let uv = pinhole_project(&point.view(), &intrinsics).expect("project");
    assert_relative_eq!(uv[0], 370.0, epsilon = 1e-10);
    assert_relative_eq!(uv[1], 340.0, epsilon = 1e-10);
}

/// S19: IMU strapdown matches small-angle quaternion increment.
#[test]
fn s19_imu_strapdown_small_angle() {
    let q0 = arr1(&[1.0_f64, 0.0, 0.0, 0.0]);
    let gyro = arr1(&[0.0_f64, 0.0, 0.1]);
    let dt = 0.01;
    let q1 = strapdown_predict(&q0, &gyro, dt).expect("strapdown");
    let expected =
        nabled::linalg::geometry::quat::from_axis_angle(&nabled::linalg::geometry::AxisAngle {
            axis: [0.0, 0.0, 1.0],
            angle: 0.001,
        });
    assert_relative_eq!(q1[0], expected.w, epsilon = 1e-6);
    assert_relative_eq!(q1[3], expected.z, epsilon = 1e-6);
}

/// S20: Rolling covariance on synthetic innovation sequence.
#[test]
fn s20_rolling_covariance_innovations() {
    let innovations = arr2(&[[0.1_f64, -0.2], [0.0, 0.1], [-0.1, 0.0], [0.2, 0.1], [0.05, -0.05]]);
    let cov = rolling_covariance(&innovations.view(), 3);
    assert!(cov[[4, 0]].is_finite());
    assert!(cov[[4, 3]] > 0.0);
}

/// S22: Branched-tree FK/Jacobian vs Y-branch fixture; serial extract for dynamics.
#[test]
fn s22_y_branch_tree_fk() {
    let fixture = load_y_branch_json().expect("fixture");
    let urdf_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/physical_ai/Y_branch.urdf");
    let model = from_urdf_file::<f64>(urdf_path).expect("urdf");
    assert_eq!(model.dof(), 3);

    for case in &fixture.cases {
        let q = arr1(&case.q);
        let left = end_effector_pose_tree(&model, "base", "left_ee", &q).expect("left fk");
        let right = end_effector_pose_tree(&model, "base", "right_ee", &q).expect("right fk");
        assert_relative_eq!(left.translation[0], case.left_ee_translation[0], epsilon = 1e-10);
        assert_relative_eq!(left.translation[1], case.left_ee_translation[1], epsilon = 1e-10);
        assert_relative_eq!(left.translation[2], case.left_ee_translation[2], epsilon = 1e-10);
        assert_relative_eq!(right.translation[0], case.right_ee_translation[0], epsilon = 1e-10);
        assert_relative_eq!(right.translation[1], case.right_ee_translation[1], epsilon = 1e-10);
        assert_relative_eq!(right.translation[2], case.right_ee_translation[2], epsilon = 1e-10);

        let j_left = jacobian_tree(&model, "base", "left_ee", &q).expect("left jacobian");
        assert_eq!(j_left.nrows(), 6);
        assert_eq!(j_left.ncols(), 3);
        let h = 1e-6;
        for col in 0..3 {
            let mut q_plus = q.clone();
            q_plus[col] += h;
            let pose_plus = end_effector_pose_tree(&model, "base", "left_ee", &q_plus).unwrap();
            let mut q_minus = q.clone();
            q_minus[col] -= h;
            let pose_minus = end_effector_pose_tree(&model, "base", "left_ee", &q_minus).unwrap();
            let deriv_x = (pose_plus.translation[0] - pose_minus.translation[0]) / (2.0 * h);
            let deriv_y = (pose_plus.translation[1] - pose_minus.translation[1]) / (2.0 * h);
            assert_relative_eq!(j_left[[0, col]], deriv_x, epsilon = 1e-5);
            assert_relative_eq!(j_left[[1, col]], deriv_y, epsilon = 1e-5);
        }
    }

    let left_chain = extract_chain_spec(&model, "base", "left_ee").expect("left chain");
    assert_eq!(left_chain.num_joints(), 2);
    let q = arr1(&fixture.cases[0].q);
    let q_chain = arr1(&[q[0], q[1]]);
    let serial_pose = end_effector_pose(&left_chain, &q_chain).expect("serial fk");
    let tree_pose = end_effector_pose_tree(&model, "base", "left_ee", &q).expect("tree fk");
    assert_relative_eq!(serial_pose.translation, tree_pose.translation, epsilon = 1e-10);
}

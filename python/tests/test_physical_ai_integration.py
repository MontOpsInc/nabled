"""Physical AI integration scenarios S1–S23 (Python parity with Rust)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pynabled
from pynabled import control, dynamics, geometry, kinematics, model, sensor

FIXTURES = Path(__file__).resolve().parents[2] / "crates/nabled/tests/fixtures/physical_ai"

pytestmark = pytest.mark.filterwarnings("ignore")


def _planar2r_chain():
    return kinematics.ChainSpec.from_dh(
        ["revolute", "revolute"],
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    )


def test_s1_planar_2r_fk_jacobian():
    chain = _planar2r_chain()
    q = np.array([0.3, 0.5])
    pose = kinematics.end_effector_pose(chain, q)
    expected_x = np.cos(q[0]) + np.cos(q[0] + q[1])
    expected_y = np.sin(q[0]) + np.sin(q[0] + q[1])
    np.testing.assert_allclose(pose.translation[0], expected_x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pose.translation[1], expected_y, rtol=0, atol=1e-10)

    j = kinematics.jacobian_translation(chain, q)
    h = 1e-6
    for col in range(2):
        q_plus = q.copy()
        q_plus[col] += h
        q_minus = q.copy()
        q_minus[col] -= h
        rot_p, t_p = kinematics.fk(chain, q_plus)
        rot_m, t_m = kinematics.fk(chain, q_minus)
        deriv_x = (t_p[0] - t_m[0]) / (2 * h)
        deriv_y = (t_p[1] - t_m[1]) / (2 * h)
        np.testing.assert_allclose(j[0, col], deriv_x, rtol=0, atol=1e-5)
        np.testing.assert_allclose(j[1, col], deriv_y, rtol=0, atol=1e-5)


def test_s2_six_dof_fk_jacobian():
    fixture = model.load_six_dof_dh_fixture(str(FIXTURES / "six_dof_dh.json"))
    chain = fixture.to_chain_spec()
    q = np.array([0.0, -0.3, 0.5, 0.2, -0.1, 0.4])
    pose = kinematics.end_effector_pose(chain, q)
    assert np.all(np.isfinite(pose.translation))
    j = kinematics.jacobian(chain, q)
    assert j.shape == (6, 6)


def test_s3_dls_ik_to_target_pose():
    fixture = model.load_six_dof_dh_fixture(str(FIXTURES / "six_dof_dh.json"))
    chain = fixture.to_chain_spec()
    q_target = np.array([0.2, -0.3, 0.5, 0.1, -0.2, 0.4])
    rot, trans = kinematics.fk(chain, q_target)
    target = geometry.transform3_from_parts(rot, trans)
    q_init = np.zeros(6)
    config = kinematics.IkConfig(max_iterations=200, tolerance=1e-3)
    result = kinematics.inverse_kinematics_dls(chain, q_init, target, config)
    rot_a, trans_a = kinematics.fk(chain, result.q)
    achieved = geometry.transform3_from_parts(rot_a, trans_a)
    err = kinematics.pose_error(achieved, target)
    assert np.linalg.norm(err) < 1e-3


def test_s4_urdf_model_fk():
    fixture_data = json.loads((FIXTURES / "2r_planar.json").read_text())
    robot = model.from_urdf_file(str(FIXTURES / "planar2r.urdf"))
    chain = model.to_chain_spec(robot)
    for case in fixture_data["cases"]:
        ee = case.get("ee_translation")
        if ee is None or len(ee) != 3:
            continue
        q = np.array(case["q"])
        pose = kinematics.end_effector_pose(chain, q)
        np.testing.assert_allclose(pose.translation, ee, rtol=0, atol=1e-10)


def test_s5_rnea_tau():
    fixture_data = json.loads((FIXTURES / "2r_planar.json").read_text())
    fixture = model.load_planar2r_fixture(str(FIXTURES / "2r_planar.json"))
    robot = fixture.to_robot_model()
    chain = fixture.to_chain_spec()
    gravity = tuple(fixture.gravity or (0.0, -9.81, 0.0))
    config = dynamics.DynamicsConfig(gravity=gravity)
    case = next(c for c in fixture_data["cases"] if c.get("qd") and c.get("qdd"))
    q = np.array(case["q"])
    qd = np.array(case["qd"])
    qdd = np.array(case["qdd"])
    tau = dynamics.rnea(robot, chain, q, qd, qdd, config=config)
    assert np.all(np.isfinite(tau))
    recovered = dynamics.forward_dynamics(robot, chain, q, qd, tau, config=config)
    np.testing.assert_allclose(recovered, qdd, rtol=0, atol=1e-6)


def test_s5b_forward_dynamics_round_trip():
    fixture = model.load_planar2r_fixture(str(FIXTURES / "2r_planar.json"))
    robot = fixture.to_robot_model()
    chain = fixture.to_chain_spec()
    gravity = tuple(fixture.gravity or (0.0, -9.81, 0.0))
    config = dynamics.DynamicsConfig(gravity=gravity)
    q = np.array([0.2, 0.4])
    qd = np.array([0.05, -0.1])
    qdd = np.array([0.3, 0.15])
    tau = dynamics.rnea(robot, chain, q, qd, qdd, config=config)
    qdd_fd = dynamics.forward_dynamics(robot, chain, q, qd, tau, config=config)
    np.testing.assert_allclose(qdd_fd, qdd, rtol=0, atol=1e-6)


def test_s6_discrete_lqr_double_integrator():
    dt = 0.1
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.0], [dt]])
    q_mat = np.array([[10.0, 0.0], [0.0, 1.0]])
    r_mat = np.array([[0.1]])
    result = control.discrete_lqr(a, b, q_mat, r_mat)
    closed = a - b @ result.gain
    eig = pynabled.eigen_nonsymmetric(closed)
    for lam in eig.eigenvalues:
        assert abs(lam) < 1.0


def test_s7_linear_kalman_fusion():
    state = sensor.KalmanState(np.array([0.0]), np.array([[1.0]]))
    transition = np.array([[1.0]])
    process_cov = np.array([[0.01]])
    predicted = sensor.kalman_predict(state, transition, process_cov)
    observation = np.array([[1.0]])
    measurement_cov = np.array([[0.05]])
    measurement = np.array([1.0])
    updated = sensor.kalman_update(predicted, measurement, observation, measurement_cov)
    assert updated.mean[0] > 0.7
    assert updated.covariance[0, 0] < 1.0


def test_s8_rolling_covariance_chunked():
    matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    window = 3
    cov_full = pynabled.rolling_covariance(matrix, window)
    chunk1 = pynabled.rolling_covariance(matrix[0:3], window)
    chunk2 = pynabled.rolling_covariance(matrix[2:5], window)
    np.testing.assert_allclose(cov_full[2, 0], chunk1[2, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(cov_full[4, 3], chunk2[2, 3], rtol=0, atol=1e-10)


@pytest.mark.skipif(pynabled.signal is None, reason="signal feature not compiled")
def test_s12_dominant_frequency_sine_bin():
    from pynabled import signal

    n = 128
    bin_idx = 7
    sample_rate = 480.0
    freq = bin_idx * sample_rate / n
    t = np.arange(n) / sample_rate
    x = np.sin(2 * np.pi * freq * t)
    peak_bin = signal.dominant_frequency(x)
    assert peak_bin == bin_idx
    np.testing.assert_allclose(
        signal.bin_to_hz(peak_bin, n, sample_rate),
        freq,
        rtol=0,
        atol=sample_rate / n,
    )


@pytest.mark.skipif(pynabled.signal is None, reason="signal feature not compiled")
def test_s13_rfft_irfft_round_trip():
    from pynabled import signal

    x = np.array([0.2, -0.5, 1.0, 0.0, -1.0, 0.3, 0.7, -0.2])
    spectrum = signal.rfft(x)
    reconstructed = signal.irfft(spectrum)
    np.testing.assert_allclose(x, reconstructed, rtol=0, atol=1e-10)


@pytest.mark.skipif(pynabled.signal is None, reason="signal feature not compiled")
def test_s14_autocorrelation_period_peak():
    from pynabled import signal

    period = 10
    n = 80
    t = np.arange(n)
    x = np.cos(2 * np.pi * t / period)
    acf = signal.autocorrelation_full(x)
    peak_lag = int(np.argmax(acf[1:]) + 1)
    assert peak_lag == period


def test_s9_dare_algebraic_residual():
    dt = 0.1
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.0], [dt]])
    q_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    r_mat = np.array([[1.0]])
    p = control.dare_solve(a, b, q_mat, r_mat)
    norm = control.dare_residual_norm(a, b, q_mat, r_mat, p)
    assert norm < 1e-8


def test_s10_continuous_gramian_lyapunov():
    a = np.array([[-1.0, 0.0], [0.0, -2.0]])
    b = np.array([[1.0], [0.0]])
    w = control.controllability_gramian(a, b)
    bb = b @ b.T
    residual = a @ w + w @ a.T + bb
    assert np.linalg.norm(residual) < 1e-8


def test_s15_luenberger_observer_tracks_state():
    dt = 0.05
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.0], [dt]])
    c = np.array([[1.0, 0.0]])
    l = control.luenberger_gain(a, c, [-0.5, -0.6])
    x = np.array([1.0, 0.5])
    x_hat = np.array([0.0, 0.0])
    u = 0.2
    for _ in range(80):
        y = c @ x
        x = a @ x + (b[:, 0] * u)
        innovation = y - c @ x_hat
        x_hat = a @ x_hat + (b[:, 0] * u) + (l[:, 0] * innovation[0])
    err = np.linalg.norm(x - x_hat)
    assert err < 1e-2


def test_s16_pole_placement_double_integrator():
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0], [1.0]])
    poles = [-1.0, -2.0]
    k = control.place_poles(a, b, poles)
    closed = a - b @ k
    eig = pynabled.eigen_nonsymmetric(closed)
    for pole in poles:
        matched = any(abs(lam.real - pole) < 1e-6 and abs(lam.imag) < 1e-6 for lam in eig.eigenvalues)
        assert matched


def test_s21_lqr_algebraic_consistency():
    dt = 0.1
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.0], [dt]])
    q_mat = np.array([[10.0, 0.0], [0.0, 1.0]])
    r_mat = np.array([[0.1]])
    result = control.discrete_lqr(a, b, q_mat, r_mat)
    bpb = b.T @ result.riccati @ b + r_mat
    bpb_inv = pynabled.lu_inverse(bpb)
    manual = bpb_inv @ b.T @ result.riccati @ a
    np.testing.assert_allclose(result.gain, manual, rtol=0, atol=1e-8)


def test_s11_multi_d_kalman_tracking():
    state = sensor.KalmanState(np.array([0.0, 0.0]), np.eye(2))
    f = np.array([[1.0, 0.1], [0.0, 1.0]])
    q_mat = np.array([[0.01, 0.0], [0.0, 0.01]])
    predicted = sensor.kalman_predict(state, f, q_mat)
    h = np.eye(2)
    r_mat = np.array([[0.05, 0.0], [0.0, 0.05]])
    z = np.array([0.8, 0.6])
    updated = sensor.kalman_update(predicted, z, h, r_mat)
    assert updated.mean[0] > 0.4
    assert updated.mean[1] > 0.3


def test_s17_ekf_nonlinear_update():
    def predict_state(x):
        return np.array([np.cos(x[0])])

    def predict_jacobian(x):
        return np.array([[-np.sin(x[0])]])

    def measure(x):
        return np.array([x[0]])

    def measure_jacobian(_x):
        return np.array([[1.0]])

    state = sensor.KalmanState(np.array([0.2]), np.array([[1.0]]))
    predicted = sensor.ekf_predict(
        state,
        predict_state,
        predict_jacobian,
        np.array([[0.01]]),
    )
    updated = sensor.ekf_update(
        predicted,
        np.array([0.9]),
        measure,
        measure_jacobian,
        np.array([[0.05]]),
    )
    assert updated.mean[0] > state.mean[0]


def test_s18_pinhole_camera_projection():
    intrinsics = sensor.PinholeIntrinsics(500.0, 500.0, 320.0, 240.0)
    point = np.array([0.1, 0.2, 1.0])
    uv = sensor.pinhole_project(point, intrinsics)
    np.testing.assert_allclose(uv[0], 370.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(uv[1], 340.0, rtol=0, atol=1e-10)


def test_s19_imu_strapdown_small_angle():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = np.array([0.0, 0.0, 0.1])
    dt = 0.01
    q1 = sensor.strapdown_predict(q0, gyro, dt)
    expected = geometry.quat_from_axis_angle([0.0, 0.0, 1.0], 0.001)
    np.testing.assert_allclose(q1[0], expected[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(q1[3], expected[3], rtol=0, atol=1e-6)


def test_s20_rolling_covariance_innovations():
    innovations = np.array(
        [[0.1, -0.2], [0.0, 0.1], [-0.1, 0.0], [0.2, 0.1], [0.05, -0.05]],
    )
    cov = pynabled.rolling_covariance(innovations, 3)
    assert np.isfinite(cov[4, 0])
    assert cov[4, 3] > 0.0


def test_s22_y_branch_tree_fk():
    fixture_data = json.loads((FIXTURES / "y_branch.json").read_text())
    robot = model.from_urdf_file(str(FIXTURES / "Y_branch.urdf"))
    assert robot.dof == 3

    for case in fixture_data["cases"]:
        q = np.array(case["q"])
        left = kinematics.end_effector_pose_tree(robot, "base", "left_ee", q)
        right = kinematics.end_effector_pose_tree(robot, "base", "right_ee", q)
        np.testing.assert_allclose(left.translation, case["left_ee_translation"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(right.translation, case["right_ee_translation"], rtol=0, atol=1e-10)

        j_left = kinematics.jacobian_tree(robot, "base", "left_ee", q)
        assert j_left.shape == (6, 3)
        h = 1e-6
        for col in range(3):
            q_plus = q.copy()
            q_plus[col] += h
            q_minus = q.copy()
            q_minus[col] -= h
            pose_plus = kinematics.end_effector_pose_tree(robot, "base", "left_ee", q_plus)
            pose_minus = kinematics.end_effector_pose_tree(robot, "base", "left_ee", q_minus)
            deriv_x = (pose_plus.translation[0] - pose_minus.translation[0]) / (2 * h)
            deriv_y = (pose_plus.translation[1] - pose_minus.translation[1]) / (2 * h)
            np.testing.assert_allclose(j_left[0, col], deriv_x, rtol=0, atol=1e-5)
            np.testing.assert_allclose(j_left[1, col], deriv_y, rtol=0, atol=1e-5)

    left_chain = model.extract_chain_spec(robot, "base", "left_ee")
    assert left_chain.num_joints == 2
    q = np.array(fixture_data["cases"][0]["q"])
    q_chain = np.array([q[0], q[1]])
    serial_pose = kinematics.end_effector_pose(left_chain, q_chain)
    tree_pose = kinematics.end_effector_pose_tree(robot, "base", "left_ee", q)
    np.testing.assert_allclose(serial_pose.translation, tree_pose.translation, rtol=0, atol=1e-10)


def test_s23_y_branch_tree_ik():
    robot = model.from_urdf_file(str(FIXTURES / "Y_branch.urdf"))
    q_target = np.array([0.3, 0.5, -0.2])
    target = kinematics.end_effector_pose_tree(robot, "base", "left_ee", q_target)
    q_init = np.zeros(3)
    config = kinematics.IkConfig(max_iterations=300, tolerance=1e-3)
    result = kinematics.inverse_kinematics_tree_dls(
        robot, "base", "left_ee", q_init, target, config
    )
    assert result.q.shape == (3,)
    achieved = kinematics.end_effector_pose_tree(robot, "base", "left_ee", result.q)
    err = kinematics.pose_error(achieved, target)
    assert np.linalg.norm(err) < 1e-3

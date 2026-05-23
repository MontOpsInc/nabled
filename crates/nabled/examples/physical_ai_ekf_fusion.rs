//! Synthetic perception fusion: pinhole projection, EKF, and IMU strapdown.
//!
//! Run: `cargo run -p nabled --example physical_ai_ekf_fusion`
//!
//! Composes: constant-velocity landmark in the camera frame → `pinhole_project` measurements
//! fused by `ekf_predict`/`ekf_update`, with parallel `strapdown_predict` orientation.
//!
//! MT-A2: `nabled-sim` was not extracted — sensor fusion callbacks and camera geometry are
//! unrelated to rigid-body integration or batch IK orchestration.

use nabled::linalg::geometry::{AxisAngle, quat};
use nabled::sensor::camera::{PinholeIntrinsics, pinhole_jacobian, pinhole_project};
use nabled::sensor::ekf::{EkConfig, EkModel, ekf_predict, ekf_update};
use nabled::sensor::imu::strapdown_predict;
use nabled::sensor::kalman::KalmanState;
use ndarray::{Array1, Array2, ArrayView1, arr1, arr2, s};

const DT: f64 = 0.05;
const DEPTH: f64 = 2.0;
const INTRINSICS: PinholeIntrinsics<f64> =
    PinholeIntrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };

fn predict_state(state: &ArrayView1<'_, f64>) -> Array1<f64> {
    arr1(&[state[0] + state[2] * DT, state[1] + state[3] * DT, state[2], state[3]])
}

fn predict_jacobian(_state: &ArrayView1<'_, f64>) -> Array2<f64> {
    arr2(&[[1.0, 0.0, DT, 0.0], [0.0, 1.0, 0.0, DT], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
}

fn measure_state(state: &ArrayView1<'_, f64>) -> Array1<f64> {
    let point = arr1(&[state[0], state[1], DEPTH]);
    pinhole_project(&point.view(), &INTRINSICS).expect("valid depth")
}

fn measure_jacobian(state: &ArrayView1<'_, f64>) -> Array2<f64> {
    let point = arr1(&[state[0], state[1], DEPTH]);
    let h_point = pinhole_jacobian(&point.view(), &INTRINSICS).expect("valid depth");
    let mut h_state = Array2::zeros((2, 4));
    h_state.slice_mut(s![.., 0..2]).assign(&h_point.slice(s![.., 0..2]));
    h_state
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        EkModel { predict_state, predict_jacobian, measure: measure_state, measure_jacobian };
    let config = EkConfig {
        process_noise: arr2(&[
            [0.001, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.01],
        ]),
        measurement_noise: arr2(&[[4.0, 0.0], [0.0, 4.0]]),
    };

    let mut state = KalmanState {
        mean: arr1(&[0.0, 0.0, 0.4, 0.1]),
        covariance: arr2(&[
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ]),
    };
    let mut orientation = arr1(&[1.0, 0.0, 0.0, 0.0]);
    let gyro = arr1(&[0.0, 0.0, 0.08]);

    println!("step   true_x   est_x    uv_u     uv_v     |q|");
    for step in 0..=20 {
        let true_x = 0.4 * f64::from(step) * DT;
        let true_point = arr1(&[true_x, 0.15, DEPTH]);
        let uv = pinhole_project(&true_point.view(), &INTRINSICS)?;
        let noisy_uv = arr1(&[uv[0] + 1.5, uv[1] - 1.0]);

        if step % 5 == 0 {
            println!(
                "{step:3}    {true_x:.3}    {:.3}    {:.1}    {:.1}    {:.4}",
                state.mean[0], noisy_uv[0], noisy_uv[1], orientation[0]
            );
        }

        if step == 20 {
            break;
        }

        let predicted = ekf_predict(&state, &model, &config)?;
        state = ekf_update(&predicted, &noisy_uv.view(), &model, &config)?;
        orientation = strapdown_predict(&orientation, &gyro, DT)?;
    }

    let expected =
        quat::from_axis_angle(&AxisAngle { axis: [0.0, 0.0, 1.0], angle: 0.08 * DT * 20.0 });
    println!(
        "Final orientation w={:.4} (expected {:.4}), position estimate x={:.3}",
        orientation[0], expected.w, state.mean[0]
    );

    Ok(())
}

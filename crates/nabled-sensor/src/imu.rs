//! IMU strapdown integration.

use nabled_core::scalar::NabledReal;
use nabled_linalg::geometry::{Quat, quat};
use ndarray::{Array1, ArrayView1};

use crate::SensorError;

/// Integrate orientation with gyro `[wx, wy, wz]` over `dt` (quaternion `[w,x,y,z]`).
pub fn strapdown_predict<T: NabledReal>(
    orientation: &Array1<T>,
    gyro: &Array1<T>,
    dt: T,
) -> Result<Array1<T>, SensorError> {
    strapdown_predict_view(&orientation.view(), &gyro.view(), dt)
}

/// Integrate orientation with gyro from joint views (no copy on input).
pub fn strapdown_predict_view<T: NabledReal>(
    orientation: &ArrayView1<'_, T>,
    gyro: &ArrayView1<'_, T>,
    dt: T,
) -> Result<Array1<T>, SensorError> {
    let mut out = orientation.to_owned();
    strapdown_predict_view_into(orientation, gyro, dt, &mut out)?;
    Ok(out)
}

pub fn strapdown_predict_into<T: NabledReal>(
    orientation: &Array1<T>,
    gyro: &Array1<T>,
    dt: T,
    output: &mut Array1<T>,
) -> Result<(), SensorError> {
    strapdown_predict_view_into(&orientation.view(), &gyro.view(), dt, output)
}

/// Integrate orientation in-place from joint views and write to `output`.
pub fn strapdown_predict_view_into<T: NabledReal>(
    orientation: &ArrayView1<'_, T>,
    gyro: &ArrayView1<'_, T>,
    dt: T,
    output: &mut Array1<T>,
) -> Result<(), SensorError> {
    if orientation.len() != 4 || gyro.len() != 3 || output.len() != 4 {
        return Err(SensorError::DimensionMismatch);
    }
    let q = Quat { w: orientation[0], x: orientation[1], y: orientation[2], z: orientation[3] };
    let delta = quat::exp(&ndarray::arr1(&[gyro[0] * dt, gyro[1] * dt, gyro[2] * dt]).view());
    let next = quat::mul(&delta, &q);
    let normalized = quat::normalize(&next);
    output[0] = normalized.w;
    output[1] = normalized.x;
    output[2] = normalized.y;
    output[3] = normalized.z;
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_linalg::geometry::{AxisAngle, quat};

    use super::*;

    #[test]
    fn strapdown_matches_small_angle_exp() {
        let q0 = quat::identity::<f64>();
        let orientation = ndarray::arr1(&[q0.w, q0.x, q0.y, q0.z]);
        let gyro = ndarray::arr1(&[0.0_f64, 0.0, 0.1]);
        let dt = 0.01;
        let predicted = strapdown_predict(&orientation, &gyro, dt).unwrap();
        let expected = quat::from_axis_angle(&AxisAngle { axis: [0.0, 0.0, 1.0], angle: 0.001 });
        assert_relative_eq!(predicted[0], expected.w, epsilon = 1e-6);
        assert_relative_eq!(predicted[3], expected.z, epsilon = 1e-6);
    }
}

//! Pinhole camera model.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1};

use crate::SensorError;

/// Pinhole intrinsics `{ fx, fy, cx, cy }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeIntrinsics<T> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
}

impl<T: NabledReal> PinholeIntrinsics<T> {
    /// Build a 3×3 intrinsic matrix `K`.
    #[must_use]
    pub fn matrix(&self) -> Array2<T> {
        let mut k = Array2::<T>::zeros((3, 3));
        k[[0, 0]] = self.fx;
        k[[1, 1]] = self.fy;
        k[[0, 2]] = self.cx;
        k[[1, 2]] = self.cy;
        k[[2, 2]] = T::one();
        k
    }
}

pub fn pinhole_project<T: NabledReal>(
    point: &ArrayView1<'_, T>,
    intrinsics: &PinholeIntrinsics<T>,
) -> Result<Array1<T>, SensorError> {
    if point.len() != 3 {
        return Err(SensorError::DimensionMismatch);
    }
    if point[2] <= T::from_f64(1e-12).unwrap_or(T::zero()) {
        return Err(SensorError::InvalidInput("point behind camera".to_string()));
    }
    let u = intrinsics.fx * point[0] / point[2] + intrinsics.cx;
    let v = intrinsics.fy * point[1] / point[2] + intrinsics.cy;
    Ok(ndarray::arr1(&[u, v]))
}

pub fn pinhole_project_k<T: NabledReal>(
    point: &ArrayView1<'_, T>,
    intrinsics: &Array2<T>,
) -> Result<Array1<T>, SensorError> {
    if intrinsics.dim() != (3, 3) {
        return Err(SensorError::DimensionMismatch);
    }
    let k = PinholeIntrinsics {
        fx: intrinsics[[0, 0]],
        fy: intrinsics[[1, 1]],
        cx: intrinsics[[0, 2]],
        cy: intrinsics[[1, 2]],
    };
    pinhole_project(point, &k)
}

pub fn pinhole_jacobian<T: NabledReal>(
    point: &ArrayView1<'_, T>,
    intrinsics: &PinholeIntrinsics<T>,
) -> Result<Array2<T>, SensorError> {
    if point.len() != 3 {
        return Err(SensorError::DimensionMismatch);
    }
    let (x, y, z) = (point[0], point[1], point[2]);
    if z <= T::from_f64(1e-12).unwrap_or(T::zero()) {
        return Err(SensorError::InvalidInput("point behind camera".to_string()));
    }
    let z2 = z * z;
    let mut j = Array2::<T>::zeros((2, 3));
    j[[0, 0]] = intrinsics.fx / z;
    j[[0, 2]] = -intrinsics.fx * x / z2;
    j[[1, 1]] = intrinsics.fy / z;
    j[[1, 2]] = -intrinsics.fy * y / z2;
    Ok(j)
}

pub fn pinhole_jacobian_k<T: NabledReal>(
    point: &ArrayView1<'_, T>,
    intrinsics: &Array2<T>,
) -> Result<Array2<T>, SensorError> {
    if intrinsics.dim() != (3, 3) {
        return Err(SensorError::DimensionMismatch);
    }
    let k = PinholeIntrinsics {
        fx: intrinsics[[0, 0]],
        fy: intrinsics[[1, 1]],
        cx: intrinsics[[0, 2]],
        cy: intrinsics[[1, 2]],
    };
    pinhole_jacobian(point, &k)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn pinhole_project_centered() {
        let k = PinholeIntrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let p = ndarray::arr1(&[0.1_f64, 0.2, 1.0]);
        let uv = pinhole_project(&p.view(), &k).unwrap();
        assert_relative_eq!(uv[0], 370.0, epsilon = 1e-10);
        assert_relative_eq!(uv[1], 340.0, epsilon = 1e-10);
    }

    #[test]
    fn pinhole_jacobian_finite_diff() {
        let k = PinholeIntrinsics { fx: 400.0, fy: 400.0, cx: 0.0, cy: 0.0 };
        let p = ndarray::arr1(&[0.2_f64, -0.1, 2.0]);
        let j = pinhole_jacobian(&p.view(), &k).unwrap();
        let h = 1e-6;
        for col in 0..3 {
            let mut plus = p.clone();
            plus[col] += h;
            let mut minus = p.clone();
            minus[col] -= h;
            let u_plus = pinhole_project(&plus.view(), &k).unwrap();
            let u_minus = pinhole_project(&minus.view(), &k).unwrap();
            let deriv_u = (u_plus[0] - u_minus[0]) / (2.0 * h);
            let deriv_v = (u_plus[1] - u_minus[1]) / (2.0 * h);
            assert_relative_eq!(j[[0, col]], deriv_u, epsilon = 1e-5);
            assert_relative_eq!(j[[1, col]], deriv_v, epsilon = 1e-5);
        }
    }

    #[test]
    fn pinhole_rejects_degenerate_intrinsics_matrix() {
        let point = ndarray::arr1(&[0.1_f64, 0.2, 1.0]);
        let bad_k = ndarray::arr2(&[[500.0, 0.0], [0.0, 500.0]]);
        assert_eq!(
            pinhole_project_k(&point.view(), &bad_k),
            Err(SensorError::DimensionMismatch)
        );
        assert_eq!(
            pinhole_jacobian_k(&point.view(), &bad_k),
            Err(SensorError::DimensionMismatch)
        );
    }

    #[test]
    fn pinhole_rejects_point_behind_camera() {
        let k = PinholeIntrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let on_plane = ndarray::arr1(&[0.1_f64, 0.2, 0.0]);
        let near_plane = ndarray::arr1(&[0.1_f64, 0.2, 1e-15]);
        let err = SensorError::InvalidInput("point behind camera".to_string());
        assert_eq!(pinhole_project(&on_plane.view(), &k), Err(err.clone()));
        assert_eq!(pinhole_jacobian(&on_plane.view(), &k), Err(err.clone()));
        assert_eq!(pinhole_project(&near_plane.view(), &k), Err(err));
    }

    #[test]
    fn pinhole_rejects_wrong_point_dimension() {
        let k = PinholeIntrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let point = ndarray::arr1(&[1.0_f64, 2.0]);
        assert_eq!(
            pinhole_project(&point.view(), &k),
            Err(SensorError::DimensionMismatch)
        );
        assert_eq!(
            pinhole_jacobian(&point.view(), &k),
            Err(SensorError::DimensionMismatch)
        );
    }
}

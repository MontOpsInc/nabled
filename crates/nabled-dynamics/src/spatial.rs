//! Spatial inertia and joint motion subspaces.

use nabled_core::scalar::NabledReal;
use nabled_model::joint::{JointAxis, JointType};
use nabled_model::link::InertialSpec;
use ndarray::{Array1, Array2, ArrayView1};

/// Build spatial inertia from link inertial spec.
pub fn from_inertial_spec<T: NabledReal>(spec: &InertialSpec<T>) -> SpatialInertia<T> {
    SpatialInertia {
        mass:    spec.mass,
        com:     ndarray::arr1(&spec.com),
        inertia: spec.inertia.clone(),
    }
}

/// 6×6 spatial inertia matrix via the parallel-axis theorem.
pub fn spatial_inertia_6x6<T: NabledReal>(spec: &InertialSpec<T>) -> Array2<T> {
    let m = spec.mass;
    let c = ndarray::arr1(&spec.com);
    let i3 = &spec.inertia;
    let cx = nabled_linalg::geometry::so3::hat(&c.view());
    let mcx = cx.mapv(|v| v * m);
    let top_left = i3 + &mcx.t().dot(&cx);
    let top_right = mcx.t();
    let bottom_left = mcx.clone();
    let mut spatial = Array2::<T>::zeros((6, 6));
    for i in 0..3 {
        for j in 0..3 {
            spatial[[i, j]] = top_left[[i, j]];
            spatial[[i, j + 3]] = top_right[[i, j]];
            spatial[[i + 3, j]] = bottom_left[[i, j]];
            spatial[[i + 3, j + 3]] = if i == j { m } else { T::zero() };
        }
    }
    spatial
}

/// Apply spatial inertia to a motion vector: `f = I v`.
pub fn spatial_inertia_apply<T: NabledReal>(inertia: &Array2<T>, v: &ArrayView1<'_, T>) -> Array1<T> {
    inertia.dot(v)
}

/// Transform spatial inertia: `I' = Ad^T I Ad`.
pub fn spatial_inertia_transform<T: NabledReal>(
    inertia: &Array2<T>,
    adjoint: &Array2<T>,
) -> Array2<T> {
    adjoint.t().dot(inertia).dot(adjoint)
}

/// Joint motion subspace `S` (6-vector) for revolute/prismatic joints.
pub fn joint_motion_subspace<T: NabledReal>(joint_type: JointType, axis: JointAxis) -> Array1<T> {
    let unit = axis.unit_vector::<T>();
    let mut s = Array1::<T>::zeros(6);
    match joint_type {
        JointType::Revolute => {
            s[0] = unit[0];
            s[1] = unit[1];
            s[2] = unit[2];
        }
        JointType::Prismatic => {
            s[3] = unit[0];
            s[4] = unit[1];
            s[5] = unit[2];
        }
        JointType::Fixed => {}
    }
    s
}

/// Spatial gravity acceleration vector `[0; g]`.
pub fn spatial_gravity<T: NabledReal>(gravity: &[T; 3]) -> Array1<T> {
    let mut accel = Array1::<T>::zeros(6);
    accel[3] = gravity[0];
    accel[4] = gravity[1];
    accel[5] = gravity[2];
    accel
}

/// Apply motion cross product via geometry twist helper.
pub fn motion_cross_product<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Array1<T> {
    nabled_linalg::geometry::twist::motion_cross(a, b)
}

/// Apply force cross product via geometry twist helper.
pub fn force_cross_product<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Array1<T> {
    nabled_linalg::geometry::twist::force_cross(a, b)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialInertia<T> {
    /// Link mass.
    pub mass:    T,
    /// Center of mass.
    pub com:     Array1<T>,
    /// 3×3 inertia tensor.
    pub inertia: Array2<T>,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn point_mass_inertia_diagonal() {
        let spec = InertialSpec {
            mass:    2.0_f64,
            com:     [0.5, 0.0, 0.0],
            inertia: Array2::<f64>::zeros((3, 3)),
        };
        let i6 = spatial_inertia_6x6(&spec);
        assert_relative_eq!(i6[[5, 5]], 2.0, epsilon = 1e-12);
        assert_relative_eq!(i6[[4, 4]], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn spatial_inertia_apply_matches_hand_calc() {
        let spec = InertialSpec {
            mass:    1.0_f64,
            com:     [0.0, 0.0, 0.0],
            inertia: Array2::<f64>::eye(3),
        };
        let i6 = spatial_inertia_6x6(&spec);
        let v = ndarray::arr1(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let f = spatial_inertia_apply(&i6, &v.view());
        assert_relative_eq!(f[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn revolute_z_subspace() {
        let s = joint_motion_subspace::<f64>(JointType::Revolute, JointAxis::Z);
        assert_relative_eq!(s[2], 1.0, epsilon = 1e-12);
    }
}

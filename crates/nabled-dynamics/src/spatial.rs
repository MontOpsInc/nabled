//! Spatial inertia utilities.

use nabled_core::scalar::NabledReal;
use nabled_model::link::InertialSpec;
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialInertia<T> {
    /// Link mass.
    pub mass:    T,
    /// Center of mass.
    pub com:     Array1<T>,
    /// 3×3 inertia tensor.
    pub inertia: Array2<T>,
}

/// Build spatial inertia from link inertial spec.
pub fn from_inertial_spec<T: NabledReal>(spec: &InertialSpec<T>) -> SpatialInertia<T> {
    SpatialInertia {
        mass:    spec.mass,
        com:     ndarray::arr1(&spec.com),
        inertia: spec.inertia.clone(),
    }
}

/// Apply motion cross product via geometry twist helper.
pub fn motion_cross_product<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Array1<T> {
    nabled_linalg::geometry::twist::motion_cross(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_inertial_spec_roundtrip() {
        let spec = InertialSpec {
            mass:    1.0_f64,
            com:     [0.1, 0.0, 0.0],
            inertia: Array2::<f64>::eye(3),
        };
        let spatial = from_inertial_spec(&spec);
        assert!((spatial.mass - 1.0).abs() < 1e-12);
    }
}

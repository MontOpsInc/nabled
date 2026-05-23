//! Joint types and limits.

use nabled_core::scalar::NabledReal;

use crate::ModelError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    Revolute,
    Prismatic,
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointAxis {
    X,
    Y,
    Z,
    Custom([f64; 3]),
}

impl JointAxis {
    /// Unit axis vector for revolute/prismatic motion.
    #[must_use]
    pub fn unit_vector<T: NabledReal>(&self) -> [T; 3] {
        match self {
            JointAxis::X => [T::one(), T::zero(), T::zero()],
            JointAxis::Y => [T::zero(), T::one(), T::zero()],
            JointAxis::Z => [T::zero(), T::zero(), T::one()],
            JointAxis::Custom(v) => {
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if norm <= 1e-12 {
                    return [T::zero(), T::zero(), T::one()];
                }
                [
                    T::from_f64(v[0] / norm).unwrap_or(T::zero()),
                    T::from_f64(v[1] / norm).unwrap_or(T::zero()),
                    T::from_f64(v[2] / norm).unwrap_or(T::zero()),
                ]
            }
        }
    }

    /// Parse `<axis xyz="..."/>` into a normalized axis.
    #[must_use]
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        let norm = (x * x + y * y + z * z).sqrt();
        if norm <= 1e-12 {
            return JointAxis::Z;
        }
        let nx = x / norm;
        let ny = y / norm;
        let nz = z / norm;
        if (nx - 1.0).abs() < 1e-6 && ny.abs() < 1e-6 && nz.abs() < 1e-6 {
            JointAxis::X
        } else if nx.abs() < 1e-6 && (ny - 1.0).abs() < 1e-6 && nz.abs() < 1e-6 {
            JointAxis::Y
        } else if nx.abs() < 1e-6 && ny.abs() < 1e-6 && (nz - 1.0).abs() < 1e-6 {
            JointAxis::Z
        } else {
            JointAxis::Custom([nx, ny, nz])
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointLimits<T> {
    pub lower:    T,
    pub upper:    T,
    pub velocity: T,
    pub effort:   T,
}

/// Validate joint limits ordering.
pub fn validate_limits<T: PartialOrd>(limits: &JointLimits<T>) -> Result<(), ModelError> {
    if limits.lower > limits.upper {
        return Err(ModelError::InvalidInput("lower limit exceeds upper limit".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn cardinal_axes_are_unit_length() {
        for axis in [JointAxis::X, JointAxis::Y, JointAxis::Z] {
            let v = axis.unit_vector::<f64>();
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert_relative_eq!(len, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn from_xyz_normalizes_custom_axis() {
        let axis = JointAxis::from_xyz(0.0, 2.0, 0.0);
        let v = axis.unit_vector::<f64>();
        assert_relative_eq!(v[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn limits_validate_ordering() {
        let ok = JointLimits { lower: -1.0, upper: 1.0, velocity: 2.0, effort: 10.0 };
        validate_limits(&ok).unwrap();
        let bad = JointLimits { lower: 1.0, upper: -1.0, velocity: 2.0, effort: 10.0 };
        assert!(validate_limits(&bad).is_err());
    }
}

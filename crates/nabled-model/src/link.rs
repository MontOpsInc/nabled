//! Link and inertial specifications.

use ndarray::Array2;

#[derive(Debug, Clone, PartialEq)]
pub struct LinkSpec {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InertialSpec<T> {
    pub mass: T,
    pub com: [T; 3],
    pub inertia: Array2<T>,
}

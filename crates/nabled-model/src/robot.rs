//! Robot model graph.

use crate::ModelError;
use crate::joint::{JointAxis, JointLimits, JointType};
use crate::link::{InertialSpec, LinkSpec};

#[derive(Debug, Clone, PartialEq)]
pub struct BodySpec<T> {
    pub link:       LinkSpec,
    pub joint_type: JointType,
    pub axis:       JointAxis,
    pub limits:     Option<JointLimits<T>>,
    pub inertial:   Option<InertialSpec<T>>,
    pub dh_a:       T,
    pub dh_alpha:   T,
    pub dh_d:       T,
    pub dh_theta:   T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobotModel<T> {
    bodies:  Vec<BodySpec<T>>,
    parents: Vec<Option<usize>>,
}

impl<T: Clone> Default for RobotModel<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Clone> RobotModel<T> {
    #[must_use]
    pub fn new() -> Self { Self { bodies: Vec::new(), parents: Vec::new() } }

    pub fn add_body(&mut self, parent: Option<usize>, body: BodySpec<T>) -> usize {
        let index = self.bodies.len();
        self.bodies.push(body);
        self.parents.push(parent);
        index
    }

    #[must_use]
    pub fn parent(&self, index: usize) -> Option<usize> {
        self.parents.get(index).copied().flatten()
    }

    #[must_use]
    pub fn joint(&self, index: usize) -> Option<&BodySpec<T>> { self.bodies.get(index) }

    #[must_use]
    pub fn dof(&self) -> usize {
        self.bodies.iter().filter(|b| !matches!(b.joint_type, JointType::Fixed)).count()
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.bodies.is_empty() {
            return Err(ModelError::EmptyModel);
        }
        for (i, parent) in self.parents.iter().enumerate() {
            if let Some(p) = parent
                && *p >= i
            {
                return Err(ModelError::InvalidInput(format!(
                    "parent index {p} must be less than child {i}"
                )));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn topological_order(&self) -> Vec<usize> { (0..self.bodies.len()).collect() }

    pub fn update_body(&mut self, index: usize, body: BodySpec<T>) -> Result<(), ModelError> {
        if index >= self.bodies.len() {
            return Err(ModelError::InvalidInput(format!("body index {index} out of range")));
        }
        self.bodies[index] = body;
        Ok(())
    }
}

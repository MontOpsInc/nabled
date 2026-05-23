//! Robot model graph.

use std::collections::{HashMap, VecDeque};

use nabled_linalg::geometry::Transform3;

use crate::ModelError;
use crate::joint::{JointLimits, JointType};
use crate::link::{InertialSpec, LinkSpec};

#[derive(Debug, Clone, PartialEq)]
pub struct BodySpec<T> {
    pub link: LinkSpec,
    pub parent_link: String,
    pub joint_type: JointType,
    pub axis: crate::joint::JointAxis,
    pub limits: Option<JointLimits<T>>,
    pub inertial: Option<InertialSpec<T>>,
    /// Static URDF `<origin>` transform from parent link to joint frame.
    pub joint_origin: Transform3<T>,
    pub dh_a: T,
    pub dh_alpha: T,
    pub dh_d: T,
    pub dh_theta: T,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobotModel<T> {
    bodies: Vec<BodySpec<T>>,
    parents: Vec<Option<usize>>,
    link_to_body: HashMap<String, usize>,
}

impl<T: Clone> Default for RobotModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> RobotModel<T> {
    #[must_use]
    pub fn new() -> Self {
        Self { bodies: Vec::new(), parents: Vec::new(), link_to_body: HashMap::new() }
    }

    pub fn add_body(&mut self, parent: Option<usize>, body: BodySpec<T>) -> usize {
        let index = self.bodies.len();
        let _ = self.link_to_body.insert(body.link.name.clone(), index);
        self.bodies.push(body);
        self.parents.push(parent);
        index
    }

    #[must_use]
    pub fn parent(&self, index: usize) -> Option<usize> {
        self.parents.get(index).copied().flatten()
    }

    #[must_use]
    pub fn joint(&self, index: usize) -> Option<&BodySpec<T>> {
        self.bodies.get(index)
    }

    #[must_use]
    pub fn body_index_for_link(&self, link_name: &str) -> Option<usize> {
        self.link_to_body.get(link_name).copied()
    }

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
                && *p >= self.bodies.len()
            {
                return Err(ModelError::InvalidInput(format!(
                    "parent index {p} out of range for body {i}"
                )));
            }
        }
        self.validate_acyclic()?;
        Ok(())
    }

    fn validate_acyclic(&self) -> Result<(), ModelError> {
        for start in 0..self.bodies.len() {
            let mut current = Some(start);
            let mut steps = 0usize;
            while let Some(idx) = current {
                steps += 1;
                if steps > self.bodies.len() {
                    return Err(ModelError::InvalidInput(
                        "cycle detected in robot tree".to_string(),
                    ));
                }
                current = self.parent(idx);
            }
        }
        Ok(())
    }

    /// BFS topological order from root bodies (no parent).
    #[must_use]
    pub fn topological_order(&self) -> Vec<usize> {
        let mut roots: Vec<usize> = self
            .parents
            .iter()
            .enumerate()
            .filter_map(|(i, p)| if p.is_none() { Some(i) } else { None })
            .collect();
        if roots.is_empty() && !self.bodies.is_empty() {
            roots.push(0);
        }
        let mut order = Vec::with_capacity(self.bodies.len());
        let mut queue: VecDeque<usize> = roots.into();
        let mut seen = vec![false; self.bodies.len()];
        while let Some(index) = queue.pop_front() {
            if seen[index] {
                continue;
            }
            seen[index] = true;
            order.push(index);
            for (child, parent) in self.parents.iter().enumerate() {
                if parent == &Some(index) && !seen[child] {
                    queue.push_back(child);
                }
            }
        }
        for (i, was_seen) in seen.iter().enumerate() {
            if !was_seen {
                order.push(i);
            }
        }
        order
    }

    /// Indices of actuated (non-fixed) bodies in topological order.
    #[must_use]
    pub fn actuated_indices(&self) -> Vec<usize> {
        self.topological_order()
            .into_iter()
            .filter(|&i| self.joint(i).is_some_and(|b| !matches!(b.joint_type, JointType::Fixed)))
            .collect()
    }

    /// Joint limits for actuated joint `joint_index` (0-based among actuated joints).
    #[must_use]
    pub fn limits_for_joint(&self, joint_index: usize) -> Option<&JointLimits<T>> {
        self.actuated_indices()
            .get(joint_index)
            .and_then(|&body_index| self.joint(body_index))
            .and_then(|body| body.limits.as_ref())
    }

    pub fn update_body(&mut self, index: usize, body: BodySpec<T>) -> Result<(), ModelError> {
        if index >= self.bodies.len() {
            return Err(ModelError::InvalidInput(format!("body index {index} out of range")));
        }
        let _ = self.link_to_body.insert(body.link.name.clone(), index);
        self.bodies[index] = body;
        Ok(())
    }
}

/// Extract a serial chain between named links (returns body indices in chain order).
pub fn extract_chain<T: Clone>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
) -> Result<Vec<usize>, ModelError> {
    let ee = model
        .body_index_for_link(ee_link)
        .ok_or_else(|| ModelError::InvalidInput(format!("unknown link {ee_link}")))?;

    let mut chain = Vec::new();
    let mut current = Some(ee);
    while let Some(idx) = current {
        let body = model
            .joint(idx)
            .ok_or_else(|| ModelError::InvalidInput(format!("missing body {idx}")))?;
        chain.push(idx);
        if body.parent_link == base_link {
            break;
        }
        current = model.parent(idx);
    }

    let reached_base =
        model.joint(*chain.last().unwrap()).is_some_and(|body| body.parent_link == base_link);
    if !reached_base {
        return Err(ModelError::InvalidInput(format!(
            "no kinematic path from {base_link} to {ee_link}"
        )));
    }

    chain.reverse();
    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::JointAxis;

    fn sample_body(name: &str, parent_link: &str) -> BodySpec<f64> {
        BodySpec {
            link: LinkSpec { name: name.to_string() },
            parent_link: parent_link.to_string(),
            joint_type: JointType::Revolute,
            axis: JointAxis::Z,
            limits: None,
            inertial: None,
            joint_origin: crate::origin::transform_from_xyz_rpy([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                .expect("valid origin"),
            dh_a: 1.0,
            dh_alpha: 0.0,
            dh_d: 0.0,
            dh_theta: 0.0,
        }
    }

    #[test]
    fn topological_order_is_bfs() {
        let mut model = RobotModel::new();
        let root = model.add_body(None, sample_body("link1", "base"));
        let _child = model.add_body(Some(root), sample_body("link2", "link1"));
        assert_eq!(model.topological_order(), vec![0, 1]);
    }

    #[test]
    fn extract_chain_finds_serial_path() {
        let mut model = RobotModel::new();
        let j1 = model.add_body(None, sample_body("link1", "base"));
        let _j2 = model.add_body(Some(j1), sample_body("link2", "link1"));
        let path = extract_chain(&model, "base", "link2").unwrap();
        assert_eq!(path, vec![0, 1]);
    }

    #[test]
    fn rejects_disconnected_extract_chain() {
        let mut model = RobotModel::new();
        let _ = model.add_body(None, sample_body("a", "base"));
        let _ = model.add_body(None, sample_body("b", "other_base"));
        assert!(extract_chain(&model, "base", "b").is_err());
    }
}

//! Minimal URDF parser (serial revolute/prismatic chains).

use nabled_core::scalar::NabledReal;
use ndarray::Array2;

use crate::ModelError;
use crate::joint::{JointAxis, JointType};
use crate::link::{InertialSpec, LinkSpec};
use crate::robot::{BodySpec, RobotModel};

/// Parse a minimal URDF string into a robot model.
pub fn from_urdf_str<T: NabledReal + Default>(urdf: &str) -> Result<RobotModel<T>, ModelError> {
    let mut model = RobotModel::new();
    let mut current_parent = None;
    for line in urdf.lines() {
        let line = line.trim();
        if line.starts_with("<joint") {
            let name = extract_attr(line, "name").unwrap_or_else(|| "joint".to_string());
            let jtype = extract_attr(line, "type").unwrap_or_else(|| "revolute".to_string());
            let joint_type = match jtype.as_str() {
                "revolute" => JointType::Revolute,
                "prismatic" => JointType::Prismatic,
                "fixed" => JointType::Fixed,
                _ => {
                    return Err(ModelError::ParseError(format!("unsupported joint type {jtype}")));
                }
            };
            let body = BodySpec {
                link: LinkSpec { name: name.clone() },
                joint_type,
                axis: JointAxis::Z,
                limits: None,
                inertial: None,
                dh_a: T::default(),
                dh_alpha: T::default(),
                dh_d: T::default(),
                dh_theta: T::default(),
            };
            let index = model.add_body(current_parent, body);
            current_parent = Some(index);
        } else if line.contains("<origin")
            && let Some(body_index) = current_parent
            && let Some(body) = model.joint(body_index)
        {
            let mut updated = body.clone();
            if let Some(xyz) = extract_attr(line, "xyz") {
                let parts: Vec<_> = xyz.split_whitespace().collect();
                if parts.len() == 3 {
                    updated.dh_d = parse_scalar(parts[2])?;
                    updated.dh_a = parse_scalar(parts[0])?;
                }
            }
            if let Some(rpy) = extract_attr(line, "rpy") {
                let parts: Vec<_> = rpy.split_whitespace().collect();
                if !parts.is_empty() {
                    updated.dh_alpha = parse_scalar(parts[0])?;
                    updated.dh_theta = parse_scalar(parts[1])?;
                }
            }
            model.update_body(body_index, updated)?;
        } else if line.contains("<mass")
            && let Some(body_index) = current_parent
            && let Some(value) = extract_attr(line, "value")
        {
            let mass = parse_scalar(&value)?;
            let mut body = model.joint(body_index).unwrap().clone();
            body.inertial = Some(InertialSpec {
                mass,
                com: [T::default(), T::default(), T::default()],
                inertia: Array2::<T>::eye(3),
            });
            model.update_body(body_index, body)?;
        }
    }
    if model.dof() == 0 {
        return Err(ModelError::ParseError("no actuated joints found".to_string()));
    }
    Ok(model)
}

/// Parse URDF from file path.
pub fn from_urdf_file<T: NabledReal + Default>(path: &str) -> Result<RobotModel<T>, ModelError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| ModelError::ParseError(format!("failed to read {path}: {err}")))?;
    from_urdf_str(&content)
}

fn extract_attr(line: &str, key: &str) -> Option<String> {
    let pattern = format!("{key}=\"");
    let start = line.find(&pattern)? + pattern.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn parse_scalar<T: NabledReal>(value: &str) -> Result<T, ModelError> {
    value
        .parse::<f64>()
        .ok()
        .and_then(|v| T::from_f64(v))
        .ok_or_else(|| ModelError::ParseError(format!("invalid scalar {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dh::to_chain_spec;

    const MINIMAL_URDF: &str = r#"
<robot name="planar2r">
  <link name="base"/>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="link1"/>
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
"#;

    #[test]
    fn parse_minimal_urdf() {
        let model = from_urdf_str::<f64>(MINIMAL_URDF).unwrap();
        assert_eq!(model.dof(), 2);
        let chain = to_chain_spec(&model).unwrap();
        assert_eq!(chain.num_joints(), 2);
    }
}

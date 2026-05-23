//! URDF parser (serial and branched tree models).

#![allow(clippy::too_many_lines, clippy::unnested_or_patterns, clippy::needless_pass_by_value)]

use std::collections::{HashMap, HashSet, VecDeque};

use nabled_core::scalar::NabledReal;
use ndarray::Array2;
use quick_xml::Reader;
use quick_xml::events::Event;

use crate::ModelError;
use crate::joint::{JointAxis, JointLimits, JointType};
use crate::link::{InertialSpec, LinkSpec};
use crate::origin::transform_from_xyz_rpy;
use crate::robot::{BodySpec, RobotModel};

#[derive(Clone)]
struct PendingJoint {
    name: String,
    joint_type: JointType,
    parent_link: String,
    child_link: String,
    origin_xyz: [f64; 3],
    origin_rpy: [f64; 3],
    axis: JointAxis,
    limits: Option<JointLimits<f64>>,
}

#[derive(Clone)]
struct PendingInertial {
    mass: f64,
    com: [f64; 3],
    inertia: Array2<f64>,
}

struct LinkData {
    inertial: Option<PendingInertial>,
}

/// Parse URDF string into a robot model.
pub fn from_urdf_str<T: NabledReal + Default>(urdf: &str) -> Result<RobotModel<T>, ModelError> {
    let parsed = parse_urdf_tree(urdf)?;
    build_model::<T>(&parsed)
}

/// Parse URDF from file path.
pub fn from_urdf_file<T: NabledReal + Default>(path: &str) -> Result<RobotModel<T>, ModelError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| ModelError::ParseError(format!("failed to read {path}: {err}")))?;
    from_urdf_str(&content)
}

struct ParsedUrdf {
    joints: Vec<PendingJoint>,
    links: HashMap<String, LinkData>,
}

fn parse_urdf_tree(urdf: &str) -> Result<ParsedUrdf, ModelError> {
    let mut reader = Reader::from_str(urdf);
    reader.config_mut().trim_text(true);

    let mut joints = Vec::new();
    let mut links: HashMap<String, LinkData> = HashMap::new();

    let mut buf = Vec::new();
    let mut in_joint = false;
    let mut in_link = false;
    let mut in_inertial = false;
    let mut current_link_name = String::new();
    let mut current_joint = empty_joint();
    let mut pending_inertial = empty_inertial();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                match tag.as_str() {
                    "joint" => {
                        in_joint = true;
                        current_joint = empty_joint();
                        current_joint.name = extract_attr(&e, "name").unwrap_or_default();
                        if let Some(jtype) = extract_attr(&e, "type") {
                            current_joint.joint_type = parse_joint_type(&jtype)?;
                        }
                    }
                    "link" => {
                        in_link = true;
                        current_link_name = extract_attr(&e, "name").unwrap_or_default();
                        let _ = links
                            .entry(current_link_name.clone())
                            .or_insert(LinkData { inertial: None });
                    }
                    "origin" if in_joint || in_inertial => {
                        if let Some(xyz) = extract_attr(&e, "xyz") {
                            let v = parse_vec3(&xyz)?;
                            if in_joint {
                                current_joint.origin_xyz = v;
                            } else {
                                pending_inertial.com = v;
                            }
                        }
                        if let Some(rpy) = extract_attr(&e, "rpy")
                            && in_joint
                        {
                            current_joint.origin_rpy = parse_vec3(&rpy)?;
                        }
                    }
                    "parent" if in_joint => {
                        current_joint.parent_link = extract_attr(&e, "link").unwrap_or_default();
                    }
                    "child" if in_joint => {
                        current_joint.child_link = extract_attr(&e, "link").unwrap_or_default();
                    }
                    "axis" if in_joint => {
                        if let Some(xyz) = extract_attr(&e, "xyz") {
                            let v = parse_vec3(&xyz)?;
                            current_joint.axis = JointAxis::from_xyz(v[0], v[1], v[2]);
                        }
                    }
                    "limit" if in_joint => {
                        current_joint.limits = Some(parse_limit_element(&e));
                    }
                    "inertial" if in_link => {
                        in_inertial = true;
                        pending_inertial = empty_inertial();
                    }
                    "mass" if in_inertial => {
                        if let Some(value) = extract_attr(&e, "value") {
                            pending_inertial.mass = value.parse::<f64>().map_err(|_| {
                                ModelError::ParseError(format!("invalid mass {value}"))
                            })?;
                        }
                    }
                    "inertia" if in_inertial => {
                        pending_inertial.inertia = parse_inertia_element(&e)?;
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name();
                let tag = String::from_utf8_lossy(name.as_ref());
                match tag.as_ref() {
                    "joint" if in_joint => {
                        joints.push(current_joint.clone());
                        in_joint = false;
                    }
                    "inertial" if in_inertial => {
                        if let Some(link) = links.get_mut(&current_link_name) {
                            link.inertial = Some(pending_inertial.clone());
                        }
                        in_inertial = false;
                    }
                    "link" if in_link => {
                        in_link = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(ModelError::ParseError(format!("XML parse error: {err}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(ParsedUrdf { joints, links })
}

fn empty_joint() -> PendingJoint {
    PendingJoint {
        name: String::new(),
        joint_type: JointType::Revolute,
        parent_link: String::new(),
        child_link: String::new(),
        origin_xyz: [0.0; 3],
        origin_rpy: [0.0; 3],
        axis: JointAxis::Z,
        limits: None,
    }
}

fn empty_inertial() -> PendingInertial {
    PendingInertial { mass: 0.0, com: [0.0; 3], inertia: Array2::<f64>::zeros((3, 3)) }
}

fn parse_joint_type(value: &str) -> Result<JointType, ModelError> {
    match value {
        "revolute" | "continuous" => Ok(JointType::Revolute),
        "prismatic" => Ok(JointType::Prismatic),
        "fixed" => Ok(JointType::Fixed),
        other => Err(ModelError::ParseError(format!("unsupported joint type {other}"))),
    }
}

fn parse_limit_element(e: &quick_xml::events::BytesStart<'_>) -> JointLimits<f64> {
    JointLimits {
        lower: parse_attr_f64(e, "lower").unwrap_or(-std::f64::consts::PI),
        upper: parse_attr_f64(e, "upper").unwrap_or(std::f64::consts::PI),
        velocity: parse_attr_f64(e, "velocity").unwrap_or(1.0),
        effort: parse_attr_f64(e, "effort").unwrap_or(1.0),
    }
}

fn parse_inertia_element(e: &quick_xml::events::BytesStart<'_>) -> Result<Array2<f64>, ModelError> {
    let ixx = parse_attr_f64(e, "ixx").unwrap_or(0.0);
    let ixy = parse_attr_f64(e, "ixy").unwrap_or(0.0);
    let ixz = parse_attr_f64(e, "ixz").unwrap_or(0.0);
    let iyy = parse_attr_f64(e, "iyy").unwrap_or(0.0);
    let iyz = parse_attr_f64(e, "iyz").unwrap_or(0.0);
    let izz = parse_attr_f64(e, "izz").unwrap_or(0.0);
    Array2::from_shape_vec((3, 3), vec![ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz])
        .map_err(|err| ModelError::ParseError(format!("invalid inertia tensor: {err}")))
}

fn build_model<T: NabledReal + Default>(parsed: &ParsedUrdf) -> Result<RobotModel<T>, ModelError> {
    if parsed.joints.is_empty() {
        return Err(ModelError::ParseError("no joints found".to_string()));
    }

    let mut child_links = HashSet::new();
    for joint in &parsed.joints {
        if !child_links.insert(joint.child_link.clone()) {
            return Err(ModelError::ParseError(format!(
                "duplicate child link {}",
                joint.child_link
            )));
        }
    }

    let base_links: Vec<String> = parsed
        .joints
        .iter()
        .map(|joint| joint.parent_link.clone())
        .filter(|parent| !child_links.contains(parent))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if base_links.is_empty() {
        return Err(ModelError::ParseError("no root link found".to_string()));
    }

    let mut model = RobotModel::new();
    let mut link_body: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<String> = base_links.into();
    let mut processed = HashSet::new();

    while let Some(parent_link) = queue.pop_front() {
        let child_joints: Vec<&PendingJoint> =
            parsed.joints.iter().filter(|joint| joint.parent_link == parent_link).collect();
        for joint in child_joints {
            if !processed.insert(joint.name.clone()) {
                continue;
            }
            let parent_body = link_body.get(&joint.parent_link).copied();
            let inertial =
                parsed.links.get(&joint.child_link).and_then(|link| link.inertial.as_ref());
            let body = body_from_joint::<T>(joint, inertial)?;
            let index = model.add_body(parent_body, body);
            let _ = link_body.insert(joint.child_link.clone(), index);
            queue.push_back(joint.child_link.clone());
        }
    }

    if processed.len() != parsed.joints.len() {
        return Err(ModelError::ParseError(
            "joint graph contains a cycle or disconnected component".to_string(),
        ));
    }

    if model.dof() == 0 {
        return Err(ModelError::ParseError("no actuated joints found".to_string()));
    }
    model.validate()?;
    Ok(model)
}

fn body_from_joint<T: NabledReal + Default>(
    joint: &PendingJoint,
    inertial: Option<&PendingInertial>,
) -> Result<BodySpec<T>, ModelError> {
    let inertial_spec = match inertial {
        Some(spec) => Some(InertialSpec {
            mass: parse_scalar::<T>(spec.mass)?,
            com: [
                parse_scalar::<T>(spec.com[0])?,
                parse_scalar::<T>(spec.com[1])?,
                parse_scalar::<T>(spec.com[2])?,
            ],
            inertia: spec.inertia.mapv(|value| parse_scalar::<T>(value).unwrap_or(T::zero())),
        }),
        None => None,
    };

    let limits = match joint.limits.as_ref() {
        Some(limits) => Some(JointLimits {
            lower: parse_scalar::<T>(limits.lower)?,
            upper: parse_scalar::<T>(limits.upper)?,
            velocity: parse_scalar::<T>(limits.velocity)?,
            effort: parse_scalar::<T>(limits.effort)?,
        }),
        None => None,
    };

    let xyz = [
        parse_scalar::<T>(joint.origin_xyz[0])?,
        parse_scalar::<T>(joint.origin_xyz[1])?,
        parse_scalar::<T>(joint.origin_xyz[2])?,
    ];
    let rpy = [
        parse_scalar::<T>(joint.origin_rpy[0])?,
        parse_scalar::<T>(joint.origin_rpy[1])?,
        parse_scalar::<T>(joint.origin_rpy[2])?,
    ];
    Ok(BodySpec {
        link: LinkSpec { name: joint.child_link.clone() },
        parent_link: joint.parent_link.clone(),
        joint_type: joint.joint_type,
        axis: joint.axis,
        limits,
        inertial: inertial_spec,
        joint_origin: transform_from_xyz_rpy(xyz, rpy)?,
        dh_a: xyz[0],
        dh_alpha: rpy[0],
        dh_d: xyz[2],
        dh_theta: rpy[1],
    })
}

fn extract_attr(e: &quick_xml::events::BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|attr| attr.key.as_ref() == key.as_bytes())
        .and_then(|attr| String::from_utf8(attr.value.into_owned()).ok())
}

fn parse_attr_f64(e: &quick_xml::events::BytesStart<'_>, key: &str) -> Option<f64> {
    extract_attr(e, key).and_then(|value| value.parse::<f64>().ok())
}

fn parse_vec3(value: &str) -> Result<[f64; 3], ModelError> {
    let parts: Vec<_> = value.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(ModelError::ParseError(format!("expected 3 values, got {value}")));
    }
    Ok([
        parts[0]
            .parse::<f64>()
            .map_err(|_| ModelError::ParseError(format!("invalid scalar {}", parts[0])))?,
        parts[1]
            .parse::<f64>()
            .map_err(|_| ModelError::ParseError(format!("invalid scalar {}", parts[1])))?,
        parts[2]
            .parse::<f64>()
            .map_err(|_| ModelError::ParseError(format!("invalid scalar {}", parts[2])))?,
    ])
}

fn parse_scalar<T: NabledReal>(value: f64) -> Result<T, ModelError> {
    T::from_f64(value).ok_or_else(|| ModelError::ParseError(format!("invalid scalar {value}")))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::dh::{extract_chain_spec, to_chain_spec};
    use crate::joint::validate_limits;

    const PLANAR2R: &str = r#"
<robot name="planar2r">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="0.5 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1" effort="10"/>
  </joint>
</robot>
"#;

    #[test]
    fn parse_planar2r_urdf() {
        const EXPECTED_LOWER: f64 = -314.0 / 100.0;
        let model = from_urdf_str::<f64>(PLANAR2R).unwrap();
        assert_eq!(model.dof(), 2);
        assert_eq!(model.topological_order(), vec![0, 1]);

        let chain = to_chain_spec(&model).unwrap();
        assert_eq!(chain.num_joints(), 2);
        assert_relative_eq!(chain.a[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(chain.a[1], 1.0, epsilon = 1e-10);

        let limits = model.limits_for_joint(0).unwrap();
        validate_limits(limits).unwrap();
        assert_relative_eq!(limits.lower, EXPECTED_LOWER, epsilon = 1e-10);

        let body = model.joint(model.actuated_indices()[0]).unwrap();
        let inertial = body.inertial.as_ref().unwrap();
        assert_relative_eq!(inertial.mass, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inertial.com[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(inertial.inertia[[0, 1]], inertial.inertia[[1, 0]], epsilon = 1e-12);

        let extracted = extract_chain_spec(&model, "base", "link2").unwrap();
        assert_eq!(extracted.a, chain.a);
    }

    #[test]
    fn axis_parsing() {
        let axis = JointAxis::from_xyz(0.0, 0.0, 1.0);
        assert_eq!(axis, JointAxis::Z);
        let v = axis.unit_vector::<f64>();
        assert_relative_eq!(v[2], 1.0, epsilon = 1e-10);

        let custom = JointAxis::from_xyz(0.0, 2.0, 0.0);
        let cv = custom.unit_vector::<f64>();
        assert_relative_eq!(cv[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn limit_round_trip() {
        let urdf = r#"
<robot name="limits">
  <link name="base"/>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="l1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="2.5" velocity="3" effort="4"/>
  </joint>
</robot>
"#;
        let model = from_urdf_str::<f64>(urdf).unwrap();
        let limits = model.limits_for_joint(0).unwrap();
        assert_relative_eq!(limits.lower, -1.5, epsilon = 1e-12);
        assert_relative_eq!(limits.upper, 2.5, epsilon = 1e-12);
        assert_relative_eq!(limits.velocity, 3.0, epsilon = 1e-12);
        assert_relative_eq!(limits.effort, 4.0, epsilon = 1e-12);
    }
}

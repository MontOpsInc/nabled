//! JSON fixture loader for Physical AI integration tests.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::{ChainSpec, DhConvention, JointType as KinJointType};
use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::ModelError;
use crate::joint::JointType;
use crate::link::InertialSpec;
use crate::origin::joint_origin_from_dh_scalars;
use crate::robot::{BodySpec, RobotModel};

#[derive(Debug, Deserialize)]
pub struct Planar2rFixture {
    pub description:   String,
    pub link_lengths:  Vec<f64>,
    pub dh_convention: String,
    pub dh_params:     Vec<[f64; 4]>,
    pub gravity:       Option<[f64; 3]>,
    pub links:         Option<Vec<LinkFixture>>,
    pub cases:         Vec<Planar2rCase>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct LinkFixture {
    pub mass:    f64,
    pub com:     [f64; 3],
    pub inertia: [[f64; 3]; 3],
}

#[derive(Debug, Deserialize)]
pub struct Planar2rCase {
    pub name:                 String,
    pub q:                    Vec<f64>,
    pub qd:                   Option<Vec<f64>>,
    pub qdd:                  Option<Vec<f64>>,
    pub tau:                  Option<Vec<f64>>,
    pub tau_gravity:          Option<Vec<f64>>,
    pub ee_translation:       Option<Vec<f64>>,
    pub jacobian_translation: Option<Vec<Vec<f64>>>,
}

impl Planar2rFixture {
    /// Build a `RobotModel` from fixture DH and optional link inertials.
    pub fn to_robot_model<T: NabledReal + Default>(&self) -> Result<RobotModel<T>, ModelError> {
        let mut model = RobotModel::new();
        let mut parent = None;
        for (i, params) in self.dh_params.iter().enumerate() {
            let inertial =
                self.links.as_ref().and_then(|links| links.get(i)).map(|link| InertialSpec {
                    mass:    parse_scalar::<T>(link.mass).unwrap_or(T::one()),
                    com:     [
                        parse_scalar::<T>(link.com[0]).unwrap_or(T::zero()),
                        parse_scalar::<T>(link.com[1]).unwrap_or(T::zero()),
                        parse_scalar::<T>(link.com[2]).unwrap_or(T::zero()),
                    ],
                    inertia: Array2::from_shape_fn((3, 3), |(r, c)| {
                        parse_scalar::<T>(link.inertia[r][c]).unwrap_or(T::zero())
                    }),
                });
            let body = BodySpec {
                link: crate::link::LinkSpec { name: format!("link{i}") },
                parent_link: if i == 0 { "base".to_string() } else { format!("link{}", i - 1) },
                joint_type: JointType::Revolute,
                axis: crate::joint::JointAxis::Z,
                limits: None,
                inertial,
                joint_origin: joint_origin_from_dh_scalars(
                    parse_scalar::<T>(params[0])?,
                    parse_scalar::<T>(params[1])?,
                    parse_scalar::<T>(params[2])?,
                    parse_scalar::<T>(params[3])?,
                )?,
                dh_a: parse_scalar::<T>(params[0])?,
                dh_alpha: parse_scalar::<T>(params[1])?,
                dh_d: parse_scalar::<T>(params[2])?,
                dh_theta: parse_scalar::<T>(params[3])?,
            };
            let index = model.add_body(parent, body);
            parent = Some(index);
        }
        model.validate()?;
        Ok(model)
    }

    /// Build a kinematic `ChainSpec` from fixture DH parameters.
    pub fn to_chain_spec<T: NabledReal>(&self) -> Result<ChainSpec<T>, ModelError> {
        let convention = match self.dh_convention.as_str() {
            "standard" => DhConvention::Standard,
            "modified" => DhConvention::Modified,
            other => {
                return Err(ModelError::InvalidInput(format!("unknown DH convention {other}")));
            }
        };
        let n = self.dh_params.len();
        let joint_types = vec![KinJointType::Revolute; n];
        let a = Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[0]).unwrap()));
        let alpha =
            Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[1]).unwrap()));
        let d = Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[2]).unwrap()));
        let theta_offset =
            Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[3]).unwrap()));
        ChainSpec::from_dh(convention, joint_types, a, alpha, d, theta_offset)
            .map_err(|_| ModelError::DimensionMismatch)
    }

    /// Load fixture from JSON file path.
    pub fn from_file(path: &str) -> Result<Self, ModelError> {
        let content = std::fs::read_to_string(path)
            .map_err(|err| ModelError::ParseError(format!("failed to read {path}: {err}")))?;
        serde_json::from_str(&content)
            .map_err(|err| ModelError::ParseError(format!("invalid JSON: {err}")))
    }
}

fn parse_scalar<T: NabledReal>(value: f64) -> Result<T, ModelError> {
    T::from_f64(value).ok_or_else(|| ModelError::ParseError(format!("invalid scalar {value}")))
}

/// Load the canonical planar 2R JSON fixture used by integration tests.
pub fn load_planar2r_json() -> Result<Planar2rFixture, ModelError> {
    let path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../nabled/tests/fixtures/physical_ai/2r_planar.json");
    Planar2rFixture::from_file(path)
}

#[derive(Debug, Deserialize)]
pub struct SixDofDhFixture {
    pub description:   String,
    pub dh_convention: String,
    pub dh_params:     Vec<[f64; 4]>,
    pub cases:         Vec<SixDofCase>,
}

#[derive(Debug, Deserialize)]
pub struct SixDofCase {
    pub name:           String,
    pub q:              Vec<f64>,
    pub ee_translation: Vec<f64>,
    #[serde(default)]
    pub tolerance:      f64,
}

impl SixDofDhFixture {
    /// Build a kinematic `ChainSpec` from fixture DH parameters.
    pub fn to_chain_spec<T: NabledReal>(&self) -> Result<ChainSpec<T>, ModelError> {
        let convention = match self.dh_convention.as_str() {
            "standard" => DhConvention::Standard,
            "modified" => DhConvention::Modified,
            other => {
                return Err(ModelError::InvalidInput(format!("unknown DH convention {other}")));
            }
        };
        let n = self.dh_params.len();
        let joint_types = vec![KinJointType::Revolute; n];
        let a = Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[0]).unwrap()));
        let alpha =
            Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[1]).unwrap()));
        let d = Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[2]).unwrap()));
        let theta_offset =
            Array1::from_iter(self.dh_params.iter().map(|p| parse_scalar::<T>(p[3]).unwrap()));
        ChainSpec::from_dh(convention, joint_types, a, alpha, d, theta_offset)
            .map_err(|_| ModelError::DimensionMismatch)
    }

    /// Load fixture from JSON file path.
    pub fn from_file(path: &str) -> Result<Self, ModelError> {
        let content = std::fs::read_to_string(path)
            .map_err(|err| ModelError::ParseError(format!("failed to read {path}: {err}")))?;
        serde_json::from_str(&content)
            .map_err(|err| ModelError::ParseError(format!("invalid JSON: {err}")))
    }
}

/// Load the canonical 6-DOF DH JSON fixture used by integration tests.
pub fn load_six_dof_dh_json() -> Result<SixDofDhFixture, ModelError> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nabled/tests/fixtures/physical_ai/six_dof_dh.json"
    );
    SixDofDhFixture::from_file(path)
}

#[derive(Debug, Deserialize)]
pub struct YBranchFixture {
    pub description: String,
    pub cases:       Vec<YBranchCase>,
}

#[derive(Debug, Deserialize)]
pub struct YBranchCase {
    pub name:                 String,
    pub q:                    Vec<f64>,
    pub left_ee_translation:  Vec<f64>,
    pub right_ee_translation: Vec<f64>,
}

impl YBranchFixture {
    /// Load fixture from JSON file path.
    pub fn from_file(path: &str) -> Result<Self, ModelError> {
        let content = std::fs::read_to_string(path)
            .map_err(|err| ModelError::ParseError(format!("failed to read {path}: {err}")))?;
        serde_json::from_str(&content)
            .map_err(|err| ModelError::ParseError(format!("invalid JSON: {err}")))
    }
}

/// Load the Y-branch tree fixture used by integration test S22.
pub fn load_y_branch_json() -> Result<YBranchFixture, ModelError> {
    let path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../nabled/tests/fixtures/physical_ai/y_branch.json");
    YBranchFixture::from_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_planar2r_fixture_from_repo() {
        let fixture = load_planar2r_json().unwrap();
        assert_eq!(fixture.dh_params.len(), 2);
        let chain = fixture.to_chain_spec::<f64>().unwrap();
        assert_eq!(chain.num_joints(), 2);
    }

    #[test]
    fn load_six_dof_fixture_from_repo() {
        let fixture = load_six_dof_dh_json().unwrap();
        assert_eq!(fixture.dh_params.len(), 6);
        let chain = fixture.to_chain_spec::<f64>().unwrap();
        assert_eq!(chain.num_joints(), 6);
    }
}

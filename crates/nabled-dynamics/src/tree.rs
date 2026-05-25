//! Tree dynamics (branch-routed RNEA / mass matrix / forward dynamics).
//!
//! These entry points operate on a single branch of a kinematic tree
//! identified by `base_link` and `ee_link`. They internally:
//!
//! 1. Extract a serial sub-chain via [`nabled_model::dh::extract_chain_spec_for_dynamics`].
//! 2. Construct a serial sub-`RobotModel` containing only that branch.
//! 3. Slice the full-model `q`/`qd`/etc. down to the branch.
//! 4. Run the existing serial RNEA / CRBA / FD algorithms.
//! 5. Scatter the branch-local result back into full-model actuated ordering.
//!
//! Whole-tree coupled dynamics (cross-branch RNEA, full mass matrix across
//! parallel branches) is intentionally out of scope here; callers needing that
//! must compose multiple branch calls.

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu;
use nabled_model::dh::{DynamicsBranchSpec, extract_chain_spec_for_dynamics};
use nabled_model::robot::{BodySpec, RobotModel};
use ndarray::{Array1, Array2, ArrayView1};

use crate::DynamicsError;
use crate::config::DynamicsConfig;
use crate::crba::mass_matrix;
use crate::fd::forward_dynamics_with_config;
use crate::rnea::rnea_with_config;

/// Build a serial sub-`RobotModel` containing exactly the bodies referenced by
/// `branch.body_indices`, re-parented into a serial chain.
fn sub_model_for_branch<T: NabledReal>(
    model: &RobotModel<T>,
    branch: &DynamicsBranchSpec<T>,
) -> Result<RobotModel<T>, DynamicsError> {
    let mut sub = RobotModel::new();
    for (chain_index, &body_index) in branch.body_indices.iter().enumerate() {
        let original = model.joint(body_index).ok_or(DynamicsError::EmptyModel)?;
        let parent_index = if chain_index == 0 { None } else { Some(chain_index - 1) };
        let mut body: BodySpec<T> = original.clone();
        if chain_index > 0 {
            let parent_body_index = branch.body_indices[chain_index - 1];
            let parent_body = model.joint(parent_body_index).ok_or(DynamicsError::EmptyModel)?;
            body.parent_link.clone_from(&parent_body.link.name);
        }
        let _ = sub.add_body(parent_index, body);
    }
    Ok(sub)
}

/// Slice a full-model actuated-ordering vector down to the branch.
fn slice_branch_vector<T: NabledReal>(
    full: &ArrayView1<'_, T>,
    q_indices: &[usize],
) -> Result<Array1<T>, DynamicsError> {
    if q_indices.iter().any(|&i| i >= full.len()) {
        return Err(DynamicsError::DimensionMismatch);
    }
    Ok(Array1::from_iter(q_indices.iter().map(|&i| full[i])))
}

fn validate_full_lengths<T: NabledReal>(
    model: &RobotModel<T>,
    arrays: &[&ArrayView1<'_, T>],
) -> Result<(), DynamicsError> {
    let expected = model.dof();
    if arrays.iter().any(|a| a.len() != expected) {
        return Err(DynamicsError::DimensionMismatch);
    }
    Ok(())
}

/// Branch [`rnea`](crate::rnea::rnea) on a tree model.
///
/// Inputs `q`, `qd`, `qdd` are in full-model actuated ordering. The returned
/// `tau` is also in full-model actuated ordering with non-branch entries set
/// to zero so callers can sum contributions from multiple branches.
pub fn rnea_tree<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    qdd: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    validate_full_lengths(model, &[q, qd, qdd])?;
    let branch = extract_chain_spec_for_dynamics(model, base_link, ee_link)
        .map_err(|err| DynamicsError::InvalidInput(err.to_string()))?;
    let sub_model = sub_model_for_branch(model, &branch)?;
    let q_branch = slice_branch_vector(q, &branch.q_indices)?;
    let qd_branch = slice_branch_vector(qd, &branch.q_indices)?;
    let qdd_branch = slice_branch_vector(qdd, &branch.q_indices)?;
    let tau_branch = rnea_with_config(
        &sub_model,
        &branch.chain,
        &q_branch.view(),
        &qd_branch.view(),
        &qdd_branch.view(),
        config,
    )?;
    scatter_full(&tau_branch, &branch.q_indices, model.dof())
}

/// Branch [`mass_matrix`](crate::crba::mass_matrix) on a tree model.
///
/// Returns an `n_full × n_full` matrix containing the branch's contribution
/// at the indices selected by the branch; off-branch rows/columns are zero.
pub fn mass_matrix_tree<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array2<T>, DynamicsError> {
    validate_full_lengths(model, &[q])?;
    let branch = extract_chain_spec_for_dynamics(model, base_link, ee_link)
        .map_err(|err| DynamicsError::InvalidInput(err.to_string()))?;
    let sub_model = sub_model_for_branch(model, &branch)?;
    let q_branch = slice_branch_vector(q, &branch.q_indices)?;
    let m_branch = mass_matrix(&sub_model, &branch.chain, &q_branch.view(), config)?;
    scatter_full_matrix(&m_branch, &branch.q_indices, model.dof())
}

/// Branch [`forward_dynamics_with_config`] on a tree model.
///
/// Inputs `q`, `qd`, `tau` are in full-model actuated ordering. The returned
/// `qdd` is also full-ordering, with off-branch entries zero. The function
/// only consumes `tau` entries inside the branch; non-branch torques are
/// ignored and not validated against actuator limits.
pub fn forward_dynamics_tree<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    validate_full_lengths(model, &[q, qd, tau])?;
    let branch = extract_chain_spec_for_dynamics(model, base_link, ee_link)
        .map_err(|err| DynamicsError::InvalidInput(err.to_string()))?;
    let sub_model = sub_model_for_branch(model, &branch)?;
    let q_branch = slice_branch_vector(q, &branch.q_indices)?;
    let qd_branch = slice_branch_vector(qd, &branch.q_indices)?;
    let tau_branch = slice_branch_vector(tau, &branch.q_indices)?;
    let qdd_branch = forward_dynamics_with_config(
        &sub_model,
        &branch.chain,
        &q_branch.view(),
        &qd_branch.view(),
        &tau_branch.view(),
        config,
    )?;
    scatter_full(&qdd_branch, &branch.q_indices, model.dof())
}

/// In-place RNEA branch variant.
pub fn rnea_tree_into<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    qdd: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
    output: &mut Array1<T>,
) -> Result<(), DynamicsError> {
    let tau = rnea_tree(model, base_link, ee_link, q, qd, qdd, config)?;
    if output.len() != tau.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&tau);
    Ok(())
}

/// In-place forward dynamics branch variant.
pub fn forward_dynamics_tree_into<T: NabledReal + Default + lu::LuProviderScalar>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
    output: &mut Array1<T>,
) -> Result<(), DynamicsError> {
    let qdd = forward_dynamics_tree(model, base_link, ee_link, q, qd, tau, config)?;
    if output.len() != qdd.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&qdd);
    Ok(())
}

/// In-place mass matrix branch variant.
pub fn mass_matrix_tree_into<T: NabledReal + Default>(
    model: &RobotModel<T>,
    base_link: &str,
    ee_link: &str,
    q: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
    output: &mut Array2<T>,
) -> Result<(), DynamicsError> {
    let m = mass_matrix_tree(model, base_link, ee_link, q, config)?;
    if output.dim() != m.dim() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&m);
    Ok(())
}

fn scatter_full<T: NabledReal>(
    branch: &Array1<T>,
    q_indices: &[usize],
    n_full: usize,
) -> Result<Array1<T>, DynamicsError> {
    if branch.len() != q_indices.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    let mut full = Array1::<T>::zeros(n_full);
    for (j, &i) in q_indices.iter().enumerate() {
        if i >= n_full {
            return Err(DynamicsError::DimensionMismatch);
        }
        full[i] = branch[j];
    }
    Ok(full)
}

fn scatter_full_matrix<T: NabledReal>(
    branch: &Array2<T>,
    q_indices: &[usize],
    n_full: usize,
) -> Result<Array2<T>, DynamicsError> {
    let m = branch.dim().0;
    if m != q_indices.len() || branch.dim().1 != m {
        return Err(DynamicsError::DimensionMismatch);
    }
    let mut full = Array2::<T>::zeros((n_full, n_full));
    for (j, &row) in q_indices.iter().enumerate() {
        for (k, &col) in q_indices.iter().enumerate() {
            if row >= n_full || col >= n_full {
                return Err(DynamicsError::DimensionMismatch);
            }
            full[[row, col]] = branch[[j, k]];
        }
    }
    Ok(full)
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments
)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_model::fixture::load_planar2r_json;
    use nabled_model::joint::{JointAxis, JointType as ModelJointType};
    use nabled_model::link::{InertialSpec, LinkSpec};
    use nabled_model::origin::joint_origin_from_dh_scalars;
    use nabled_model::robot::DhParams;
    use ndarray::arr1;

    use super::*;
    use crate::rnea::rnea_with_config;

    /// Programmatic DH-equipped Y-branch model: `base` → `trunk`; `trunk` → `left_ee`
    /// (a=1, theta=0); `trunk` → `right_ee` (a=1, theta=pi/2). Joint origins derive
    /// from the DH params so FK on each branch agrees with serial DH FK.
    fn build_y_branch_model() -> RobotModel<f64> {
        let mut model = RobotModel::new();
        let make_body =
            |name: &str, parent_link: &str, a: f64, alpha: f64, d: f64, theta_offset: f64| {
                BodySpec {
                    link:         LinkSpec { name: name.to_string() },
                    parent_link:  parent_link.to_string(),
                    joint_type:   ModelJointType::Revolute,
                    axis:         JointAxis::Z,
                    limits:       None,
                    inertial:     Some(InertialSpec {
                        mass:    1.0,
                        com:     [0.5, 0.0, 0.0],
                        inertia: Array2::<f64>::eye(3) * 0.01,
                    }),
                    joint_origin: joint_origin_from_dh_scalars(a, alpha, d, theta_offset)
                        .expect("origin"),
                    dh_params:    Some(DhParams { a, alpha, d, theta_offset }),
                }
            };
        let trunk = model.add_body(None, make_body("trunk", "base", 1.0, 0.0, 0.0, 0.0));
        let _left = model.add_body(Some(trunk), make_body("left_ee", "trunk", 1.0, 0.0, 0.0, 0.0));
        let _right = model.add_body(
            Some(trunk),
            make_body("right_ee", "trunk", 1.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );
        model.validate().expect("y-branch valid");
        model
    }

    #[test]
    fn rnea_tree_matches_serial_planar2r() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let chain = fixture.to_chain_spec::<f64>().unwrap();
        let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
        let config = DynamicsConfig { gravity, ..DynamicsConfig::default() };
        let q = arr1(&[0.3_f64, 0.5]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd = arr1(&[0.4_f64, -0.3]);

        let tau_serial =
            rnea_with_config(&model, &chain, &q.view(), &qd.view(), &qdd.view(), &config).unwrap();
        let tau_branch =
            rnea_tree(&model, "base", "link1", &q.view(), &qd.view(), &qdd.view(), &config)
                .unwrap();
        assert_relative_eq!(tau_serial, tau_branch, epsilon = 1e-9);
    }

    #[test]
    fn mass_matrix_tree_matches_serial_planar2r() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let chain = fixture.to_chain_spec::<f64>().unwrap();
        let q = arr1(&[0.3_f64, 0.5]);
        let config = DynamicsConfig::default();
        let m_serial = mass_matrix(&model, &chain, &q.view(), &config).unwrap();
        let m_branch = mass_matrix_tree(&model, "base", "link1", &q.view(), &config).unwrap();
        assert_eq!(m_serial.dim(), m_branch.dim());
        for ((row, col), value) in m_serial.indexed_iter() {
            assert_relative_eq!(*value, m_branch[[row, col]], epsilon = 1e-9);
        }
    }

    #[test]
    fn forward_dynamics_tree_round_trip_planar2r() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd_target = arr1(&[0.5_f64, 0.25]);
        let tau =
            rnea_tree(&model, "base", "link1", &q.view(), &qd.view(), &qdd_target.view(), &config)
                .unwrap();
        let recovered = forward_dynamics_tree(
            &model,
            "base",
            "link1",
            &q.view(),
            &qd.view(),
            &tau.view(),
            &config,
        )
        .unwrap();
        assert_relative_eq!(recovered, qdd_target, epsilon = 1e-6);
    }

    #[test]
    fn rnea_tree_supports_y_branch_separately() {
        let model = build_y_branch_model();
        let config = DynamicsConfig { gravity: [0.0, -9.81, 0.0], ..DynamicsConfig::default() };
        let n = model.dof();
        let q = Array1::<f64>::from_iter((0..n).map(|i| 0.1 * (i + 1) as f64));
        let zeros = Array1::<f64>::zeros(n);

        let tau_left =
            rnea_tree(&model, "base", "left_ee", &q.view(), &zeros.view(), &zeros.view(), &config)
                .unwrap();
        let tau_right =
            rnea_tree(&model, "base", "right_ee", &q.view(), &zeros.view(), &zeros.view(), &config)
                .unwrap();
        assert_eq!(tau_left.len(), n);
        assert_eq!(tau_right.len(), n);
        // Left branch covers trunk + left actuators (indices 0 and 1); right
        // branch covers trunk + right actuators (indices 0 and 2). The trunk
        // entry should appear in both.
        assert!(tau_left[0].abs() > 0.0);
        assert!(tau_left[1].abs() > 0.0);
        assert_eq!(tau_left[2], 0.0);
        assert!(tau_right[0].abs() > 0.0);
        assert!(tau_right[2].abs() > 0.0);
        assert_eq!(tau_right[1], 0.0);
    }

    #[test]
    fn mass_matrix_tree_y_branch_zero_off_branch() {
        let model = build_y_branch_model();
        let config = DynamicsConfig::default();
        let n = model.dof();
        let q = Array1::<f64>::from_iter((0..n).map(|i| 0.1 * (i + 1) as f64));
        let m_left = mass_matrix_tree(&model, "base", "left_ee", &q.view(), &config).unwrap();
        assert_eq!(m_left.dim(), (n, n));
        assert!(m_left[[0, 0]] > 0.0);
        assert!(m_left[[1, 1]] > 0.0);
        // Right branch indices (2) are not touched by the left branch.
        assert_eq!(m_left[[2, 2]], 0.0);
        assert_eq!(m_left[[1, 2]], 0.0);
        assert_eq!(m_left[[2, 1]], 0.0);
    }

    #[test]
    fn forward_dynamics_tree_y_branch_round_trip() {
        let model = build_y_branch_model();
        let config = DynamicsConfig::default();
        let n = model.dof();
        let q = arr1(&[0.1_f64, 0.2, -0.15]);
        let qd = arr1(&[0.05_f64, -0.1, 0.07]);
        let qdd_target = arr1(&[0.3_f64, 0.2, 0.0]);
        let tau = rnea_tree(
            &model,
            "base",
            "left_ee",
            &q.view(),
            &qd.view(),
            &qdd_target.view(),
            &config,
        )
        .unwrap();
        let qdd = forward_dynamics_tree(
            &model,
            "base",
            "left_ee",
            &q.view(),
            &qd.view(),
            &tau.view(),
            &config,
        )
        .unwrap();
        // Only branch entries (0, 1) should match; right entry (2) is zero.
        assert_eq!(qdd.len(), n);
        assert_relative_eq!(qdd[0], qdd_target[0], epsilon = 1e-6);
        assert_relative_eq!(qdd[1], qdd_target[1], epsilon = 1e-6);
        assert_eq!(qdd[2], 0.0);
    }

    #[test]
    fn rnea_tree_rejects_dimension_mismatch() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.0_f64]);
        let zeros = arr1(&[0.0_f64]);
        assert!(matches!(
            rnea_tree(&model, "base", "link1", &q.view(), &zeros.view(), &zeros.view(), &config),
            Err(DynamicsError::DimensionMismatch)
        ));
    }

    #[test]
    fn rnea_tree_into_writes_output_buffer() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.3_f64, 0.5]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let qdd = arr1(&[0.4_f64, -0.3]);
        let mut output = arr1(&[0.0_f64, 0.0]);
        rnea_tree_into(
            &model,
            "base",
            "link1",
            &q.view(),
            &qd.view(),
            &qdd.view(),
            &config,
            &mut output,
        )
        .unwrap();
        let tau = rnea_tree(&model, "base", "link1", &q.view(), &qd.view(), &qdd.view(), &config)
            .unwrap();
        assert_relative_eq!(output, tau, epsilon = 1e-12);
    }

    #[test]
    fn forward_dynamics_tree_into_writes_output() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.2_f64, 0.3]);
        let qd = arr1(&[0.1_f64, -0.2]);
        let tau = arr1(&[0.5_f64, 0.25]);
        let mut output = arr1(&[0.0_f64, 0.0]);
        forward_dynamics_tree_into(
            &model,
            "base",
            "link1",
            &q.view(),
            &qd.view(),
            &tau.view(),
            &config,
            &mut output,
        )
        .unwrap();
        let qdd = forward_dynamics_tree(
            &model,
            "base",
            "link1",
            &q.view(),
            &qd.view(),
            &tau.view(),
            &config,
        )
        .unwrap();
        assert_relative_eq!(output, qdd, epsilon = 1e-12);
    }

    #[test]
    fn mass_matrix_tree_into_writes_output() {
        let fixture = load_planar2r_json().unwrap();
        let model = fixture.to_robot_model::<f64>().unwrap();
        let config = DynamicsConfig::default();
        let q = arr1(&[0.2_f64, 0.3]);
        let mut output = Array2::<f64>::zeros((2, 2));
        mass_matrix_tree_into(&model, "base", "link1", &q.view(), &config, &mut output).unwrap();
        let m = mass_matrix_tree(&model, "base", "link1", &q.view(), &config).unwrap();
        for ((row, col), value) in output.indexed_iter() {
            assert_relative_eq!(*value, m[[row, col]], epsilon = 1e-12);
        }
    }
}

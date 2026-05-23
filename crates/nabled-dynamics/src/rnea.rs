//! Recursive Newton-Euler algorithm (serial chains).

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_kinematics::fk::link_transforms_view;
use nabled_linalg::geometry::{Transform3, se3, so3};
use nabled_model::robot::RobotModel;
use ndarray::{Array1, Array2, ArrayView1};

use crate::DynamicsError;
use crate::config::DynamicsConfig;
use crate::spatial::{
    force_cross_product, joint_motion_subspace, motion_cross_product, spatial_gravity,
    spatial_inertia_6x6, spatial_inertia_apply,
};

fn adjoint_from_transform<T: NabledReal>(transform: &Transform3<T>) -> Array2<T> {
    let r = &transform.rotation.matrix;
    let p = &transform.translation;
    let px = so3::hat(&p.view());
    let mut ad = Array2::<T>::zeros((6, 6));
    for i in 0..3 {
        for j in 0..3 {
            ad[[i, j]] = r[[i, j]];
            ad[[i + 3, j]] = px[[i, j]];
            ad[[i + 3, j + 3]] = r[[i, j]];
        }
    }
    ad
}

fn default_inertial<T: NabledReal + Default>() -> nabled_model::link::InertialSpec<T> {
    nabled_model::link::InertialSpec {
        mass:    T::one(),
        com:     [T::default(), T::default(), T::default()],
        inertia: Array2::<T>::eye(3) * T::from_f64(0.01).unwrap_or(T::zero()),
    }
}

pub fn rnea<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    rnea_with_config(model, chain, q, qd, qdd, &DynamicsConfig::default())
}

pub fn rnea_with_config<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    if model.dof() != chain.num_joints() || q.len() != chain.num_joints() {
        return Err(DynamicsError::DimensionMismatch);
    }
    if qd.len() != q.len() || qdd.len() != q.len() {
        return Err(DynamicsError::DimensionMismatch);
    }

    let transforms = link_transforms_view(chain, &q.view())
        .map_err(|_| DynamicsError::InvalidInput("FK failed".to_string()))?;
    let n = chain.num_joints();
    let indices = model.actuated_indices();
    if indices.len() != n {
        return Err(DynamicsError::DimensionMismatch);
    }

    let mut v = vec![Array1::<T>::zeros(6); n + 1];
    let mut a = vec![Array1::<T>::zeros(6); n + 1];
    a[0] = spatial_gravity(&config.gravity);

    let mut subspaces = Vec::with_capacity(n);
    let mut inertias = Vec::with_capacity(n);
    let mut adjoints = Vec::with_capacity(n);

    for joint_idx in 0..n {
        let body_idx = indices[joint_idx];
        let body = model.joint(body_idx).ok_or(DynamicsError::EmptyModel)?;
        let spec = body.inertial.as_ref().cloned().unwrap_or_else(default_inertial);
        inertias.push(spatial_inertia_6x6(&spec));
        subspaces.push(joint_motion_subspace(body.joint_type, body.axis));
        let parent_tf = &transforms[joint_idx];
        let child_tf = transforms.get(joint_idx + 1).ok_or(DynamicsError::EmptyModel)?;
        let relative = se3::compose(&se3::inverse(parent_tf), child_tf)
            .map_err(|_| DynamicsError::InvalidInput("transform compose failed".to_string()))?;
        adjoints.push(adjoint_from_transform(&relative));
    }

    for i in 0..n {
        let s = &subspaces[i];
        let vj = s.mapv(|val| val * qd[i]);
        v[i + 1] = adjoints[i].dot(&v[i]) + &vj;
        let aj = s.mapv(|val| val * qdd[i]);
        a[i + 1] = adjoints[i].dot(&a[i]) + aj + motion_cross_product(&v[i + 1].view(), &vj.view());
    }

    let mut tau = Array1::<T>::zeros(n);
    let mut f_child = Array1::<T>::zeros(6);
    for i in (0..n).rev() {
        let iv = spatial_inertia_apply(&inertias[i], &v[i + 1].view());
        let ia = spatial_inertia_apply(&inertias[i], &a[i + 1].view());
        let local = ia + force_cross_product(&v[i + 1].view(), &iv.view());
        let f_i = if i + 1 < n {
            local + adjoints[i + 1].t().dot(&f_child)
        } else {
            local
        };
        tau[i] = subspaces[i].dot(&f_i);
        f_child = f_i;
    }

    Ok(tau)
}

pub fn rnea_view<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    qdd: &ArrayView1<'_, T>,
) -> Result<Array1<T>, DynamicsError> {
    rnea(model, chain, &q.to_owned(), &qd.to_owned(), qdd)
}

pub fn rnea_into<T: NabledReal + Default>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &Array1<T>,
    qd: &Array1<T>,
    qdd: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), DynamicsError> {
    let tau = rnea(model, chain, q, qd, qdd)?;
    if output.len() != tau.len() {
        return Err(DynamicsError::DimensionMismatch);
    }
    output.assign(&tau);
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nabled_kinematics::chain::{ChainSpec, DhConvention, JointType};
    use nabled_model::fixture::Planar2rFixture;
    use nabled_model::joint::{JointAxis, JointType as ModelJointType};
    use nabled_model::robot::BodySpec;
    use ndarray::arr1;

    use super::*;

    fn load_fixture() -> Planar2rFixture {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../nabled/tests/fixtures/physical_ai/2r_planar.json"
        );
        Planar2rFixture::from_file(path).expect("fixture")
    }

    #[test]
    fn pendulum_gravity_torque() {
        let mut model = RobotModel::<f64>::new();
        let body = BodySpec {
            link: nabled_model::link::LinkSpec { name: "link1".to_string() },
            parent_link: "base".to_string(),
            joint_type: ModelJointType::Revolute,
            axis: JointAxis::Z,
            limits: None,
            inertial: Some(nabled_model::link::InertialSpec {
                mass:    1.0,
                com:     [0.5, 0.0, 0.0],
                inertia: Array2::<f64>::zeros((3, 3)),
            }),
            dh_a: 0.0,
            dh_alpha: 0.0,
            dh_d: 0.0,
            dh_theta: 0.0,
        };
        model.add_body(None, body);
        let chain = ChainSpec::from_dh(
            DhConvention::Standard,
            vec![JointType::Revolute],
            arr1(&[0.0]),
            arr1(&[0.0]),
            arr1(&[0.0]),
            arr1(&[0.0]),
        )
        .unwrap();
        let q = arr1(&[0.0_f64]);
        let zeros = arr1(&[0.0]);
        let config = DynamicsConfig { gravity: [0.0, -9.81, 0.0] };
        let tau = rnea_with_config(&model, &chain, &q, &zeros, &zeros.view(), &config).unwrap();
        assert!(tau[0].abs() > 0.1, "expected gravity torque, got {}", tau[0]);
    }

    #[test]
    fn planar2r_matches_fixture_when_present() {
        let fixture = load_fixture();
        if fixture.cases.iter().all(|c| c.tau.is_none()) {
            return;
        }
        let model = fixture.to_robot_model().unwrap();
        let chain = fixture.to_chain_spec().unwrap();
        let gravity: [f64; 3] = fixture.gravity.unwrap_or([0.0, -9.81, 0.0]);
        let config = DynamicsConfig { gravity };
        for case in &fixture.cases {
            if let (Some(qd), Some(qdd), Some(tau_ref)) = (&case.qd, &case.qdd, &case.tau) {
                let q = arr1(&case.q);
                let qd = arr1(qd);
                let qdd = arr1(qdd);
                let tau =
                    rnea_with_config(&model, &chain, &q, &qd, &qdd.view(), &config).unwrap();
                for (computed, expected) in tau.iter().zip(tau_ref.iter()) {
                    assert_relative_eq!(computed, expected, epsilon = 1e-6);
                }
            }
        }
    }
}

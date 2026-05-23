//! Planar 2R rigid-body simulation with semi-implicit Euler integration.
//!
//! Run: `cargo run -p nabled --example physical_ai_planar2r_sim`
//!
//! Composes: JSON fixture + URDF ingest → FK → forward dynamics → integrate `(q, qd)`.
//!
//! MT-A2: `nabled-sim` was not extracted — the sim step loop is ~15 lines and is not
//! shared with the LQR, EKF, or trajectory-IK examples (each targets a different domain).

use nabled::dynamics::config::DynamicsConfig;
use nabled::dynamics::fd::forward_dynamics_with_config;
use nabled::kinematics::fk::end_effector_pose;
use nabled::model::dh::to_chain_spec;
use nabled::model::fixture::load_planar2r_json;
use nabled::model::urdf::from_urdf_file;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = load_planar2r_json()?;
    let model = fixture.to_robot_model::<f64>()?;
    let chain = fixture.to_chain_spec::<f64>()?;
    let config = DynamicsConfig { gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]) };

    let urdf_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/physical_ai/planar2r.urdf");
    let urdf_model = from_urdf_file::<f64>(urdf_path)?;
    let urdf_chain = to_chain_spec(&urdf_model)?;
    assert_eq!(urdf_chain.num_joints(), chain.num_joints());

    let mut q = Array1::from_vec(vec![0.2, 0.4]);
    let mut qd = Array1::zeros(2);
    let tau = Array1::zeros(2);
    let dt = 0.01;
    let steps = 100;

    println!("Planar 2R forward-dynamics simulation (dt={dt}s, {steps} steps, tau=0)");
    println!("t(s)   q0      q1      ee_x    ee_y");

    for step in 0..=steps {
        let t = f64::from(step) * dt;
        let pose = end_effector_pose(&chain, &q)?;
        if step % 20 == 0 {
            println!(
                "{t:.2}   {:.3}   {:.3}   {:.3}   {:.3}",
                q[0], q[1], pose.translation[0], pose.translation[1]
            );
        }
        if step == steps {
            break;
        }
        let qdd = forward_dynamics_with_config(&model, &chain, &q, &qd, &tau.view(), &config)?;
        qd += &(&qdd * dt);
        q += &(&qd * dt);
    }

    Ok(())
}

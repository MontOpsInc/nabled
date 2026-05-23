//! Planar 2R rigid-body simulation with semi-implicit Euler integration.
//!
//! Run: `cargo run -p nabled --example physical_ai_planar2r_sim`
//!
//! Composes: JSON fixture + URDF ingest → `nabled::sim` orchestrator → FK log.

use nabled::dynamics::config::DynamicsConfig;
use nabled::kinematics::fk::end_effector_pose;
use nabled::model::dh::to_chain_spec;
use nabled::model::fixture::load_planar2r_json;
use nabled::model::urdf::from_urdf_file;
use nabled::sim::context::RobotContext;
use nabled::sim::sim::{SimConfig, SimState, semi_implicit_step};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = load_planar2r_json()?;
    let model = fixture.to_robot_model::<f64>()?;
    let chain = fixture.to_chain_spec::<f64>()?;
    let ctx = RobotContext::new(model, chain.clone(), DynamicsConfig {
        gravity: fixture.gravity.unwrap_or([0.0, -9.81, 0.0]),
        ..DynamicsConfig::default()
    });
    ctx.validate()?;

    let urdf_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/physical_ai/planar2r.urdf");
    let urdf_model = from_urdf_file::<f64>(urdf_path)?;
    let urdf_chain = to_chain_spec(&urdf_model)?;
    assert_eq!(urdf_chain.num_joints(), chain.num_joints());

    let mut state = SimState::new(Array1::from_vec(vec![0.2, 0.4]), Array1::zeros(2));
    let tau = Array1::zeros(2);
    let config = SimConfig { dt: 0.01, log_ee_pose: false };
    let steps = 100;

    println!("Planar 2R forward-dynamics simulation (dt={}s, {steps} steps, tau=0)", config.dt);
    println!("t(s)   q0      q1      ee_x    ee_y");

    for step in 0..=steps {
        let t = f64::from(step) * config.dt;
        let pose = end_effector_pose(&chain, &state.q)?;
        if step % 20 == 0 {
            println!(
                "{t:.2}   {:.3}   {:.3}   {:.3}   {:.3}",
                state.q[0], state.q[1], pose.translation[0], pose.translation[1]
            );
        }
        if step == steps {
            break;
        }
        let result = semi_implicit_step(&ctx, &state, &tau.view(), &config)?;
        state = result.state;
    }

    Ok(())
}

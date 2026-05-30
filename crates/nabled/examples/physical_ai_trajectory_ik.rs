//! Batch damped least-squares IK over a time grid with joint limits.
//!
//! Run: `cargo run -p nabled --example physical_ai_trajectory_ik`
//!
//! Composes: 6-DOF fixture → `nabled::sim::manipulation::TrajectoryIk` → FK verify.

#![expect(clippy::approx_constant)]

use nabled::kinematics::chain::JointLimits;
use nabled::kinematics::ik::IkConfig;
use nabled::model::fixture::load_six_dof_dh_json;
use nabled::sim::manipulation::{TrajectoryIk, TrajectoryIkConfig};
use ndarray::arr1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = load_six_dof_dh_json()?;
    let chain = fixture.to_chain_spec::<f64>()?;

    let trajectory = TrajectoryIk {
        times:   (0..=10).map(|i| f64::from(i) * 0.1).collect(),
        q_start: arr1(&[0.0, -0.3, 0.5, 0.2, -0.1, 0.4]),
        q_end:   arr1(&[0.3, -0.2, 0.4, 0.1, 0.0, 0.5]),
    };
    let limits: Vec<JointLimits<f64>> =
        (0..chain.num_joints()).map(|_| JointLimits { lower: -3.14, upper: 3.14 }).collect();
    let config = TrajectoryIkConfig {
        ik_config: IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() },
        limits:    Some(limits),
    };

    let result = trajectory.solve(&chain, &config)?;

    println!("t(s)   converged   pose_err     q0");
    for step in &result.steps {
        println!(
            "{:.1}    {}         {:.4e}   {:.3}",
            step.t,
            if step.result.converged { "yes" } else { "no " },
            step.pose_error_norm,
            step.result.q[0]
        );
    }

    println!(
        "Grid: {} steps, {} converged, max pose error {:.4e}",
        result.steps.len(),
        result.converged_count,
        result.max_error
    );
    Ok(())
}

//! Batch damped least-squares IK over a time grid with joint limits.
//!
//! Run: `cargo run -p nabled --example physical_ai_trajectory_ik`
//!
//! Composes: 6-DOF fixture → interpolated SE(3) targets → `inverse_kinematics_dls_with_limits`
//! → FK pose-error verification.
//!
//! MT-A2: `nabled-sim` was not extracted — the IK time-grid loop is local to this example;
//! shared fixture loading with the 2R sim is boilerplate, not >30% reusable orchestration.

use nabled::kinematics::chain::JointLimits;
use nabled::kinematics::fk::fk_view;
use nabled::kinematics::ik::{IkConfig, inverse_kinematics_dls_with_limits, pose_error};
use nabled::model::fixture::load_six_dof_dh_json;
use ndarray::arr1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = load_six_dof_dh_json()?;
    let chain = fixture.to_chain_spec::<f64>()?;

    let q_start = arr1(&[0.0, -0.3, 0.5, 0.2, -0.1, 0.4]);
    let q_end = arr1(&[0.3, -0.2, 0.4, 0.1, 0.0, 0.5]);
    let limits: Vec<JointLimits<f64>> =
        (0..chain.num_joints()).map(|_| JointLimits { lower: -3.14, upper: 3.14 }).collect();

    let config = IkConfig { max_iterations: 200, tolerance: 1e-3, ..IkConfig::default() };

    let times: Vec<f64> = (0..=10).map(|i| f64::from(i) * 0.1).collect();
    let duration = *times.last().expect("non-empty grid");

    let mut q_seed = q_start.clone();
    let mut max_error = 0.0_f64;
    let mut converged_steps = 0_usize;

    println!("t(s)   converged   pose_err     q0");
    for &t in &times {
        let blend = t / duration;
        let q_target = &q_start + &((&q_end - &q_start) * blend);
        let target = fk_view(&chain, &q_target.view())?;

        let result =
            inverse_kinematics_dls_with_limits(&chain, &q_seed, &target, &config, Some(&limits))?;
        let achieved = fk_view(&chain, &result.q.view())?;
        let err = pose_error(&achieved, &target)?;
        let err_norm = err.iter().map(|v| v * v).sum::<f64>().sqrt();
        max_error = max_error.max(err_norm);
        if result.converged {
            converged_steps += 1;
        }

        println!(
            "{t:.1}    {}         {err_norm:.4e}   {:.3}",
            if result.converged { "yes" } else { "no " },
            result.q[0]
        );
        q_seed = result.q;
    }

    println!(
        "Grid: {} steps, {converged_steps} converged, max pose error {max_error:.4e}",
        times.len()
    );
    Ok(())
}

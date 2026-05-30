//! Discrete LQR regulation with a Luenberger observer on a double integrator.
//!
//! Run: `cargo run -p nabled --example physical_ai_lqr_observer`
//!
//! Composes: `nabled::sim::control_loop::ClosedLoopStep` → closed-loop simulation.

use nabled::linalg::eigen::nonsymmetric;
use nabled::sim::control_loop::{ClosedLoopPlant, ClosedLoopState, ClosedLoopStep};
use ndarray::{arr1, arr2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.05;
    let plant = ClosedLoopPlant {
        a: arr2(&[[1.0, dt], [0.0, 1.0]]),
        b: arr2(&[[0.0], [dt]]),
        c: arr2(&[[1.0, 0.0]]),
    };
    let controller = ClosedLoopStep::design(
        plant.clone(),
        &arr2(&[[10.0, 0.0], [0.0, 1.0]]),
        &arr2(&[[0.1]]),
        &[-0.5, -0.6],
    )?;

    let closed = &plant.a - &plant.b.dot(&controller.gains.k);
    let eig = nonsymmetric(&closed)?;
    let max_pole = eig
        .eigenvalues
        .iter()
        .map(|lambda| f64::hypot(lambda.re, lambda.im))
        .fold(0.0_f64, f64::max);
    println!("LQR closed-loop max pole magnitude: {max_pole:.4}");

    let mut state = ClosedLoopState { x: arr1(&[1.0, 0.5]), x_hat: arr1(&[0.0, 0.0]) };

    println!("step   x       x_hat    u");
    for step in 0..=40 {
        let u = controller.step(&mut state)?;
        if step % 10 == 0 {
            println!("{step:3}    {:.3}   {:.3}   {:.3}", state.x[0], state.x_hat[0], u);
        }
    }

    let observer_err: f64 = (&state.x - &state.x_hat).mapv(|v: f64| v * v).sum().sqrt();
    println!("Final observer error norm: {observer_err:.4e}");
    println!("Final plant state: [{:.4}, {:.4}]", state.x[0], state.x[1]);

    Ok(())
}

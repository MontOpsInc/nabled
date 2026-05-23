//! Discrete LQR regulation with a Luenberger observer on a double integrator.
//!
//! Run: `cargo run -p nabled --example physical_ai_lqr_observer`
//!
//! Composes: `discrete_lqr` + `luenberger_gain` → closed-loop simulation with partial
//! state measurement.
//!
//! MT-A2: `nabled-sim` was not extracted — linear control simulation shares no integrator
//! or dynamics helpers with the rigid-body, sensor, or IK examples.

use nabled::control::lqr::{LqrResult, discrete_lqr};
use nabled::control::observer::luenberger_gain;
use nabled::linalg::eigen::nonsymmetric;
use ndarray::{arr1, arr2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.05;
    let a = arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b = arr2(&[[0.0], [dt]]);
    let c = arr2(&[[1.0, 0.0]]);
    let q_cost = arr2(&[[10.0, 0.0], [0.0, 1.0]]);
    let r_cost = arr2(&[[0.1]]);

    let LqrResult { gain: k, .. } = discrete_lqr(&a, &b, &q_cost, &r_cost)?;
    let observer_gain = luenberger_gain(&a, &c, &[-0.5, -0.6])?;

    let closed = &a - &b.dot(&k);
    let eig = nonsymmetric(&closed)?;
    let max_pole = eig
        .eigenvalues
        .iter()
        .map(|lambda| f64::hypot(lambda.re, lambda.im))
        .fold(0.0_f64, f64::max);
    println!("LQR closed-loop max pole magnitude: {max_pole:.4}");

    let mut state = arr1(&[1.0, 0.5]);
    let mut state_hat = arr1(&[0.0, 0.0]);

    println!("step   x       x_hat    u");
    for step in 0..=40 {
        let y = c.dot(&state);
        let u = -k.dot(&state_hat)[[0]];
        if step % 10 == 0 {
            println!("{step:3}    {:.3}   {:.3}   {:.3}", state[0], state_hat[0], u);
        }
        let innovation = &y - &c.dot(&state_hat);
        state = a.dot(&state) + &(b.column(0).to_owned() * u);
        state_hat = a.dot(&state_hat)
            + &(b.column(0).to_owned() * u)
            + &(observer_gain.column(0).to_owned() * innovation[[0]]);
    }

    let observer_err: f64 = (&state - &state_hat).mapv(|v: f64| v * v).sum().sqrt();
    println!("Final observer error norm: {observer_err:.4e}");
    println!("Final plant state: [{:.4}, {:.4}]", state[0], state[1]);

    Ok(())
}

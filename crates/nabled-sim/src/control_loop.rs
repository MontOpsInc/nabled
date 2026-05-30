//! Closed-loop LQR + Luenberger observer step (no sensor crate dependency).

use nabled_control::lqr::{LqrResult, discrete_lqr};
use nabled_control::observer::luenberger_gain;
use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array1, Array2};

use crate::SimError;

/// LTI plant matrices for closed-loop regulation.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedLoopPlant<T> {
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
}

/// Precomputed LQR and observer gains.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedLoopGains<T> {
    pub k: Array2<T>,
    pub l: Array2<T>,
}

/// Plant and estimated state.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedLoopState<T> {
    pub x:     Array1<T>,
    pub x_hat: Array1<T>,
}

/// One closed-loop regulation step with partial state measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedLoopStep<T> {
    pub plant: ClosedLoopPlant<T>,
    pub gains: ClosedLoopGains<T>,
}

impl<T: NabledReal + LuProviderScalar> ClosedLoopStep<T> {
    /// Design LQR gain and Luenberger observer from cost matrices and observer poles.
    pub fn design(
        plant: ClosedLoopPlant<T>,
        q_cost: &Array2<T>,
        r_cost: &Array2<T>,
        observer_poles: &[T],
    ) -> Result<Self, SimError> {
        let LqrResult { gain: k, .. } = discrete_lqr(&plant.a, &plant.b, q_cost, r_cost)?;
        let l = luenberger_gain(&plant.a, &plant.c, observer_poles)?;
        Ok(Self { plant, gains: ClosedLoopGains { k, l } })
    }

    /// Advance plant and observer by one step; returns control input `u`.
    pub fn step(&self, state: &mut ClosedLoopState<T>) -> Result<T, SimError> {
        let y = self.plant.c.dot(&state.x);
        let u = -self.gains.k.dot(&state.x_hat)[0];
        let innovation = &y - &self.plant.c.dot(&state.x_hat);
        state.x = self.plant.a.dot(&state.x) + &(self.plant.b.column(0).to_owned() * u);
        state.x_hat = self.plant.a.dot(&state.x_hat)
            + &(self.plant.b.column(0).to_owned() * u)
            + &(self.gains.l.column(0).to_owned() * innovation[[0]]);
        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn closed_loop_reduces_observer_error() {
        let dt = 0.05_f64;
        let plant = ClosedLoopPlant {
            a: arr2(&[[1.0, dt], [0.0, 1.0]]),
            b: arr2(&[[0.0], [dt]]),
            c: arr2(&[[1.0, 0.0]]),
        };
        let controller =
            ClosedLoopStep::design(plant, &arr2(&[[10.0, 0.0], [0.0, 1.0]]), &arr2(&[[0.1]]), &[
                -0.5, -0.6,
            ])
            .expect("design");
        let mut state = ClosedLoopState { x: arr1(&[1.0, 0.5]), x_hat: arr1(&[0.0, 0.0]) };
        for _ in 0..80 {
            let _ = controller.step(&mut state).expect("step");
        }
        let err = (&state.x - &state.x_hat).mapv(|v: f64| v * v).sum().sqrt();
        assert!(err < 1e-2, "observer error {err}");
        assert_relative_eq!(state.x[0], 0.0, epsilon = 0.15);
    }
}

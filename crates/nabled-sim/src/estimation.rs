//! EKF pipeline and innovation monitoring (no control crate dependency).

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::LuProviderScalar;
use nabled_ml::stats::rolling::rolling_covariance;
use nabled_sensor::ekf::{EkConfig, EkModel, ekf_predict, ekf_update};
use nabled_sensor::kalman::KalmanState;
use ndarray::{Array1, Array2, ArrayView1};

use crate::SimError;

/// Extended Kalman filter orchestration wrapper.
#[derive(Clone)]
pub struct EstimationPipeline<T> {
    pub model:  EkModel<T>,
    pub config: EkConfig<T>,
    pub state:  KalmanState<T>,
}

impl<T: NabledReal + LuProviderScalar> EstimationPipeline<T> {
    #[must_use]
    pub fn new(model: EkModel<T>, config: EkConfig<T>, state: KalmanState<T>) -> Self {
        Self { model, config, state }
    }

    /// Run predict step in place.
    pub fn predict(&mut self) -> Result<(), SimError> {
        self.state = ekf_predict(&self.state, &self.model, &self.config)?;
        Ok(())
    }

    /// Run update step in place.
    pub fn update(&mut self, measurement: &ArrayView1<'_, T>) -> Result<(), SimError> {
        let predicted = self.state.clone();
        self.state = ekf_update(&predicted, measurement, &self.model, &self.config)?;
        Ok(())
    }

    /// Predict then update in one call.
    pub fn predict_update(&mut self, measurement: &ArrayView1<'_, T>) -> Result<(), SimError> {
        self.predict()?;
        self.update(measurement)
    }
}

/// Track innovation vectors and expose rolling covariance (S20 pattern).
#[derive(Debug, Clone, PartialEq)]
pub struct InnovationMonitor<T> {
    pub window:  usize,
    innovations: Array2<T>,
    count:       usize,
}

impl<T: NabledReal + Default> InnovationMonitor<T> {
    pub fn new(window: usize, innovation_dim: usize) -> Result<Self, SimError> {
        if window == 0 {
            return Err(SimError::InvalidInput("window must be positive".to_string()));
        }
        Ok(Self { window, innovations: Array2::zeros((0, innovation_dim)), count: 0 })
    }

    /// Append one innovation row.
    pub fn push(&mut self, innovation: &Array1<T>) {
        let mut row = self.innovations.clone();
        if row.nrows() == 0 {
            row = Array2::zeros((1, innovation.len()));
            row.row_mut(0).assign(innovation);
        } else if row.ncols() != innovation.len() {
            return;
        } else {
            let mut stacked = Array2::zeros((row.nrows() + 1, innovation.len()));
            stacked.slice_mut(ndarray::s![0..row.nrows(), ..]).assign(&row);
            stacked.row_mut(row.nrows()).assign(innovation);
            row = stacked;
        }
        self.innovations = row;
        self.count += 1;
    }

    /// Number of stored innovations.
    #[must_use]
    pub fn len(&self) -> usize { self.count }

    #[must_use]
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Rolling covariance over stored innovations (delegates to `nabled-ml::stats`).
    #[must_use]
    pub fn rolling_covariance(&self) -> Array2<T> {
        if self.innovations.nrows() == 0 {
            return Array2::zeros((0, 0));
        }
        rolling_covariance(&self.innovations.view(), self.window)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr1, arr2};

    use super::*;

    #[test]
    fn innovation_monitor_tracks_rolling_covariance() {
        let mut monitor = InnovationMonitor::new(3, 2).expect("monitor");
        for row in [[0.1, -0.2], [0.0, 0.1], [-0.1, 0.0], [0.2, 0.1]] {
            monitor.push(&arr1(&row));
        }
        let cov: Array2<f64> = monitor.rolling_covariance();
        assert_eq!(cov.nrows(), 4);
        assert!(cov[[3, 3]].is_finite());
        assert!(cov[[3, 0]].is_finite());
    }

    #[test]
    fn estimation_pipeline_predict_update() {
        let model = EkModel {
            predict_state:    |x: &ArrayView1<'_, f64>| arr1(&[x[0].cos()]),
            predict_jacobian: |x: &ArrayView1<'_, f64>| arr2(&[[-x[0].sin()]]),
            measure:          |x: &ArrayView1<'_, f64>| arr1(&[x[0]]),
            measure_jacobian: |_: &ArrayView1<'_, f64>| arr2(&[[1.0]]),
        };
        let config =
            EkConfig { process_noise: arr2(&[[0.01]]), measurement_noise: arr2(&[[0.05]]) };
        let mut pipeline = EstimationPipeline::new(model, config, KalmanState {
            mean:       arr1(&[0.2]),
            covariance: arr2(&[[1.0]]),
        });
        pipeline.predict_update(&arr1(&[0.9]).view()).expect("predict-update");
        assert!(pipeline.state.mean[0] > 0.2);
    }
}

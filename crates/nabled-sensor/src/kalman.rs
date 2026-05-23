//! Linear Kalman filter.

use nabled_linalg::lu::LuProviderScalar;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::SensorError;

/// Kalman filter state (mean and covariance).
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanState<T> {
    pub mean:       Array1<T>,
    pub covariance: Array2<T>,
}

/// Predict step: x = F x, P = F P F' + Q.
pub fn predict<T: LuProviderScalar>(
    state: &KalmanState<T>,
    f: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<KalmanState<T>, SensorError> {
    if state.mean.len() != f.nrows() || f.nrows() != f.ncols() {
        return Err(SensorError::DimensionMismatch);
    }
    let mean = f.dot(&state.mean);
    let covariance = f.dot(&state.covariance).dot(&f.t()) + q;
    Ok(KalmanState { mean, covariance })
}

/// Predict into existing state.
pub fn predict_into<T: LuProviderScalar>(
    state: &mut KalmanState<T>,
    f: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<(), SensorError> {
    *state = predict(state, f, q)?;
    Ok(())
}

/// Update step with measurement z.
pub fn update<T: LuProviderScalar>(
    state: &KalmanState<T>,
    z: &ArrayView1<'_, T>,
    h: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
) -> Result<KalmanState<T>, SensorError> {
    let innovation = z - &h.dot(&state.mean);
    let s = h.dot(&state.covariance).dot(&h.t()) + r;
    let s_inv = nabled_linalg::lu::inverse(&s).map_err(|_| SensorError::NumericalInstability)?;
    let k = state.covariance.dot(&h.t()).dot(&s_inv);
    let mean = &state.mean + k.dot(&innovation);
    let n = state.mean.len();
    let identity = Array2::<T>::eye(n);
    let covariance = (identity - k.dot(h)).dot(&state.covariance);
    Ok(KalmanState { mean, covariance })
}

/// Update into existing state.
pub fn update_into<T: LuProviderScalar>(
    state: &mut KalmanState<T>,
    z: &ArrayView1<'_, T>,
    h: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
) -> Result<(), SensorError> {
    *state = update(state, z, h, r)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn kalman_fuses_measurement() {
        let state =
            KalmanState { mean: ndarray::arr1(&[0.0_f64]), covariance: arr2(&[[1.0]]) };
        let f = arr2(&[[1.0]]);
        let q = arr2(&[[0.01]]);
        let predicted = predict(&state, &f.view(), &q.view()).unwrap();
        let h = arr2(&[[1.0]]);
        let r = arr2(&[[0.1]]);
        let z = ndarray::arr1(&[1.0]);
        let updated = update(&predicted, &z.view(), &h.view(), &r.view()).unwrap();
        assert!(updated.mean[0] > 0.5);
        assert!(updated.covariance[[0, 0]] < 1.0);
    }
}

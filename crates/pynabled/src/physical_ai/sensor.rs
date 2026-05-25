//! Sensor bindings (Kalman, EKF, camera, IMU).

use nabled::linalg::lu;
use nabled::sensor::camera::{PinholeIntrinsics, pinhole_project};
use nabled::sensor::imu::{strapdown_predict_view, strapdown_predict_view_into};
use nabled::sensor::kalman::{KalmanState, predict, update};
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Kalman filter state carrier.
#[pyclass(name = "KalmanState")]
#[derive(Clone)]
pub struct PyKalmanState {
    pub(crate) inner: KalmanState<f64>,
}

#[pymethods]
impl PyKalmanState {
    #[new]
    fn new(mean: &Bound<'_, PyAny>, covariance: &Bound<'_, PyAny>) -> PyResult<Self> {
        match (utils::real_array1(mean, "mean")?, utils::real_array2(covariance, "covariance")?) {
            (utils::RealReadonlyArray1::F64(m), utils::RealReadonlyArray2::F64(p)) => Ok(Self {
                inner: KalmanState {
                    mean:       m.as_array().to_owned(),
                    covariance: p.as_array().to_owned(),
                },
            }),
            _ => Err(utils::matching_real_dtype_error(&["mean", "covariance"])),
        }
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(utils::pyarray1_from_owned(py, self.inner.mean.clone()))
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(utils::pyarray2_from_owned(py, self.inner.covariance.clone()))
    }
}

/// Pinhole camera intrinsics.
#[pyclass(name = "PinholeIntrinsics")]
#[derive(Clone)]
pub struct PyPinholeIntrinsics {
    pub(crate) inner: PinholeIntrinsics<f64>,
}

#[pymethods]
impl PyPinholeIntrinsics {
    #[new]
    fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { inner: PinholeIntrinsics { fx, fy, cx, cy } }
    }
}

fn kalman_state_to_py(state: KalmanState<f64>) -> PyKalmanState { PyKalmanState { inner: state } }

/// Linear Kalman predict step.
#[pyfunction]
pub fn kalman_predict(
    state: &PyKalmanState,
    transition: &Bound<'_, PyAny>,
    process_cov: &Bound<'_, PyAny>,
) -> PyResult<PyKalmanState> {
    match (
        utils::real_array2(transition, "transition")?,
        utils::real_array2(process_cov, "process_cov")?,
    ) {
        (utils::RealReadonlyArray2::F64(f), utils::RealReadonlyArray2::F64(q)) => {
            let predicted =
                predict(&state.inner, &f.as_array(), &q.as_array()).map_err(to_py_err)?;
            Ok(kalman_state_to_py(predicted))
        }
        _ => Err(utils::matching_real_dtype_error(&["transition", "process_cov"])),
    }
}

/// Linear Kalman update step.
#[pyfunction]
pub fn kalman_update(
    state: &PyKalmanState,
    measurement: &Bound<'_, PyAny>,
    observation: &Bound<'_, PyAny>,
    measurement_cov: &Bound<'_, PyAny>,
) -> PyResult<PyKalmanState> {
    match (
        utils::real_array1(measurement, "measurement")?,
        utils::real_array2(observation, "observation")?,
        utils::real_array2(measurement_cov, "measurement_cov")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(z),
            utils::RealReadonlyArray2::F64(h),
            utils::RealReadonlyArray2::F64(r),
        ) => {
            let updated = update(&state.inner, &z.as_array(), &h.as_array(), &r.as_array())
                .map_err(to_py_err)?;
            Ok(kalman_state_to_py(updated))
        }
        _ => Err(utils::matching_real_dtype_error(&[
            "measurement",
            "observation",
            "measurement_cov",
        ])),
    }
}

fn call_vector_f64(function: &Bound<'_, PyAny>, x: &ArrayView1<'_, f64>) -> PyResult<Array1<f64>> {
    let arg = utils::pyarray1_from_owned(function.py(), x.to_owned());
    let result = function.call1((arg,))?;
    match utils::real_array1(&result, "callback result")? {
        utils::RealReadonlyArray1::F64(arr) => Ok(arr.as_array().to_owned()),
        _ => Err(utils::matching_real_dtype_error(&["callback result"])),
    }
}

fn call_matrix_f64(function: &Bound<'_, PyAny>, x: &ArrayView1<'_, f64>) -> PyResult<Array2<f64>> {
    let arg = utils::pyarray1_from_owned(function.py(), x.to_owned());
    let result = function.call1((arg,))?;
    match utils::real_array2(&result, "callback result")? {
        utils::RealReadonlyArray2::F64(arr) => Ok(arr.as_array().to_owned()),
        _ => Err(utils::matching_real_dtype_error(&["callback result"])),
    }
}

/// EKF predict step with Python model callbacks.
#[pyfunction]
#[pyo3(signature = (state, predict_state, predict_jacobian, process_noise))]
pub fn ekf_predict(
    state: &PyKalmanState,
    predict_state: &Bound<'_, PyAny>,
    predict_jacobian: &Bound<'_, PyAny>,
    process_noise: &Bound<'_, PyAny>,
) -> PyResult<PyKalmanState> {
    let mean_view = state.inner.mean.view();
    let f = call_matrix_f64(predict_jacobian, &mean_view)?;
    let predicted_mean = call_vector_f64(predict_state, &mean_view)?;
    let q = match utils::real_array2(process_noise, "process_noise")? {
        utils::RealReadonlyArray2::F64(arr) => arr.as_array().to_owned(),
        _ => return Err(utils::matching_real_dtype_error(&["process_noise"])),
    };
    let covariance = f.dot(&state.inner.covariance).dot(&f.t()) + q;
    Ok(kalman_state_to_py(KalmanState { mean: predicted_mean, covariance }))
}

/// EKF update step with Python model callbacks.
#[pyfunction]
#[pyo3(signature = (state, measurement, measure, measure_jacobian, measurement_noise))]
pub fn ekf_update(
    state: &PyKalmanState,
    measurement: &Bound<'_, PyAny>,
    measure: &Bound<'_, PyAny>,
    measure_jacobian: &Bound<'_, PyAny>,
    measurement_noise: &Bound<'_, PyAny>,
) -> PyResult<PyKalmanState> {
    let z = match utils::real_array1(measurement, "measurement")? {
        utils::RealReadonlyArray1::F64(arr) => arr.as_array().to_owned(),
        _ => return Err(utils::matching_real_dtype_error(&["measurement"])),
    };
    let r = match utils::real_array2(measurement_noise, "measurement_noise")? {
        utils::RealReadonlyArray2::F64(arr) => arr.as_array().to_owned(),
        _ => return Err(utils::matching_real_dtype_error(&["measurement_noise"])),
    };
    let mean_view = state.inner.mean.view();
    let h = call_matrix_f64(measure_jacobian, &mean_view)?;
    let predicted = call_vector_f64(measure, &mean_view)?;
    let innovation = &z - &predicted;
    let s = h.dot(&state.inner.covariance).dot(&h.t()) + r;
    let s_inv = lu::inverse(&s).map_err(to_py_err)?;
    let k = state.inner.covariance.dot(&h.t()).dot(&s_inv);
    let mean = &state.inner.mean + k.dot(&innovation);
    let n = state.inner.mean.len();
    let identity = Array2::<f64>::eye(n);
    let covariance = (identity - k.dot(&h)).dot(&state.inner.covariance);
    Ok(kalman_state_to_py(KalmanState { mean, covariance }))
}

/// Pinhole camera projection.
#[pyfunction]
pub fn pinhole_project_py<'py>(
    py: Python<'py>,
    point: &Bound<'py, PyAny>,
    intrinsics: &PyPinholeIntrinsics,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(point, "point")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let uv = pinhole_project(&arr.as_array(), &intrinsics.inner).map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, uv))
        }
        _ => Err(utils::matching_real_dtype_error(&["point"])),
    }
}

/// IMU strapdown quaternion integration.
#[pyfunction]
pub fn strapdown_predict_py<'py>(
    py: Python<'py>,
    quaternion: &Bound<'py, PyAny>,
    gyro: &Bound<'py, PyAny>,
    dt: f64,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array1(quaternion, "quaternion")?, utils::real_array1(gyro, "gyro")?) {
        (utils::RealReadonlyArray1::F64(q), utils::RealReadonlyArray1::F64(g)) => {
            let q1 = strapdown_predict_view(&q.as_array(), &g.as_array(), dt).map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, q1))
        }
        _ => Err(utils::matching_real_dtype_error(&["quaternion", "gyro"])),
    }
}

/// Kalman predict into existing state buffers.
#[pyfunction]
pub fn kalman_predict_into(
    state: &mut PyKalmanState,
    transition: &Bound<'_, PyAny>,
    process_cov: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::real_array2(transition, "transition")?,
        utils::real_array2(process_cov, "process_cov")?,
    ) {
        (utils::RealReadonlyArray2::F64(f), utils::RealReadonlyArray2::F64(q)) => {
            nabled::sensor::kalman::predict_into(&mut state.inner, &f.as_array(), &q.as_array())
                .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["transition", "process_cov"])),
    }
}

/// Kalman update into existing state buffers.
#[pyfunction]
pub fn kalman_update_into(
    state: &mut PyKalmanState,
    measurement: &Bound<'_, PyAny>,
    observation: &Bound<'_, PyAny>,
    measurement_cov: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::real_array1(measurement, "measurement")?,
        utils::real_array2(observation, "observation")?,
        utils::real_array2(measurement_cov, "measurement_cov")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(z),
            utils::RealReadonlyArray2::F64(h),
            utils::RealReadonlyArray2::F64(r),
        ) => nabled::sensor::kalman::update_into(
            &mut state.inner,
            &z.as_array(),
            &h.as_array(),
            &r.as_array(),
        )
        .map_err(to_py_err),
        _ => Err(utils::matching_real_dtype_error(&[
            "measurement",
            "observation",
            "measurement_cov",
        ])),
    }
}

/// IMU strapdown into caller-provided quaternion buffer.
#[pyfunction]
pub fn strapdown_predict_into(
    quaternion: &Bound<'_, PyAny>,
    gyro: &Bound<'_, PyAny>,
    dt: f64,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array1(quaternion, "quaternion")?, utils::real_array1(gyro, "gyro")?) {
        (utils::RealReadonlyArray1::F64(q), utils::RealReadonlyArray1::F64(g)) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            let mut out_view = out.as_array_mut();
            let mut owned = Array1::<f64>::zeros(out_view.len());
            strapdown_predict_view_into(&q.as_array(), &g.as_array(), dt, &mut owned)
                .map_err(to_py_err)?;
            out_view.assign(&owned);
            Ok(())
        }
        _ => Err(utils::matching_real_dtype_error(&["quaternion", "gyro", "output"])),
    }
}

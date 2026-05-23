"""Sensor fusion bindings."""

from __future__ import annotations

from pynabled._pynabled import KalmanState as KalmanState
from pynabled._pynabled import PinholeIntrinsics as PinholeIntrinsics
from pynabled._pynabled import ekf_predict as _ekf_predict
from pynabled._pynabled import ekf_update as _ekf_update
from pynabled._pynabled import kalman_predict as _kalman_predict
from pynabled._pynabled import kalman_predict_into as _kalman_predict_into
from pynabled._pynabled import kalman_update as _kalman_update
from pynabled._pynabled import kalman_update_into as _kalman_update_into
from pynabled._pynabled import pinhole_project_py as pinhole_project
from pynabled._pynabled import strapdown_predict_into as _strapdown_predict_into
from pynabled._pynabled import strapdown_predict_py as _strapdown_predict


def kalman_predict(state: KalmanState, transition, process_cov, *, out: KalmanState | None = None):
    if out is None:
        return _kalman_predict(state, transition, process_cov)
    _kalman_predict_into(out, transition, process_cov)
    return out


def kalman_update(
    state: KalmanState,
    measurement,
    observation,
    measurement_cov,
    *,
    out: KalmanState | None = None,
):
    if out is None:
        return _kalman_update(state, measurement, observation, measurement_cov)
    _kalman_update_into(out, measurement, observation, measurement_cov)
    return out


def ekf_predict(state, predict_state, predict_jacobian, process_noise):
    return _ekf_predict(state, predict_state, predict_jacobian, process_noise)


def ekf_update(state, measurement, measure, measure_jacobian, measurement_noise):
    return _ekf_update(state, measurement, measure, measure_jacobian, measurement_noise)


def strapdown_predict(quaternion, gyro, dt, *, out=None):
    if out is None:
        return _strapdown_predict(quaternion, gyro, dt)
    _strapdown_predict_into(quaternion, gyro, dt, out)
    return out

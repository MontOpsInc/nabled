//! Dynamics bindings (RNEA, mass matrix, forward dynamics).

use nabled::dynamics::config::DynamicsConfig;
use nabled::dynamics::crba::mass_matrix;
use nabled::dynamics::fd::forward_dynamics_with_config;
use nabled::dynamics::rnea::rnea_with_config;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::physical_ai::kinematics::PyChainSpec;
use crate::physical_ai::model::PyRobotModel;
use crate::utils;

/// Dynamics configuration (gravity vector).
#[pyclass(name = "DynamicsConfig")]
#[derive(Clone)]
pub struct PyDynamicsConfig {
    pub(crate) inner: DynamicsConfig<f64>,
}

#[pymethods]
impl PyDynamicsConfig {
    #[new]
    #[pyo3(signature = (*, gravity=(0.0, 0.0, -9.81)))]
    fn new(gravity: (f64, f64, f64)) -> Self {
        Self { inner: DynamicsConfig { gravity: [gravity.0, gravity.1, gravity.2] } }
    }
}

/// Recursive Newton-Euler inverse dynamics.
#[pyfunction]
#[pyo3(signature = (model, chain, q, qd, qdd, config=None))]
pub fn rnea<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
    qd: &Bound<'py, PyAny>,
    qdd: &Bound<'py, PyAny>,
    config: Option<&PyDynamicsConfig>,
) -> PyResult<Py<PyAny>> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match (
        utils::real_array1(q, "q")?,
        utils::real_array1(qd, "qd")?,
        utils::real_array1(qdd, "qdd")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(q_arr),
            utils::RealReadonlyArray1::F64(qd_arr),
            utils::RealReadonlyArray1::F64(qdd_arr),
        ) => {
            let tau = rnea_with_config(
                &model.inner,
                &chain.inner,
                &q_arr.as_array().to_owned(),
                &qd_arr.as_array().to_owned(),
                &qdd_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, tau))
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "qd", "qdd"])),
    }
}

/// Composite rigid-body mass matrix.
#[pyfunction]
pub fn mass_matrix_py<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let m = mass_matrix(
                &model.inner,
                &chain.inner,
                &arr.as_array().to_owned(),
                &DynamicsConfig::default(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, m))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Forward dynamics given applied torques.
#[pyfunction]
#[pyo3(signature = (model, chain, q, qd, tau, config=None))]
pub fn forward_dynamics<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
    qd: &Bound<'py, PyAny>,
    tau: &Bound<'py, PyAny>,
    config: Option<&PyDynamicsConfig>,
) -> PyResult<Py<PyAny>> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match (
        utils::real_array1(q, "q")?,
        utils::real_array1(qd, "qd")?,
        utils::real_array1(tau, "tau")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(q_arr),
            utils::RealReadonlyArray1::F64(qd_arr),
            utils::RealReadonlyArray1::F64(tau_arr),
        ) => {
            let qdd = forward_dynamics_with_config(
                &model.inner,
                &chain.inner,
                &q_arr.as_array().to_owned(),
                &qd_arr.as_array().to_owned(),
                &tau_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, qdd))
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "qd", "tau"])),
    }
}

/// RNEA into caller-provided output buffer.
#[pyfunction]
#[pyo3(signature = (model, chain, q, qd, qdd, output, config=None))]
pub fn rnea_into(
    model: &PyRobotModel,
    chain: &PyChainSpec,
    q: &Bound<'_, PyAny>,
    qd: &Bound<'_, PyAny>,
    qdd: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    config: Option<&PyDynamicsConfig>,
) -> PyResult<()> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match (
        utils::real_array1(q, "q")?,
        utils::real_array1(qd, "qd")?,
        utils::real_array1(qdd, "qdd")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(q_arr),
            utils::RealReadonlyArray1::F64(qd_arr),
            utils::RealReadonlyArray1::F64(qdd_arr),
        ) => {
            let tau = rnea_with_config(
                &model.inner,
                &chain.inner,
                &q_arr.as_array().to_owned(),
                &qd_arr.as_array().to_owned(),
                &qdd_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            if out.as_array_mut().len() != tau.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "output length must match joint count",
                ));
            }
            out.as_array_mut().assign(&tau);
            Ok(())
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "qd", "qdd", "output"])),
    }
}

/// Forward dynamics into caller-provided output buffer.
#[pyfunction]
#[pyo3(signature = (model, chain, q, qd, tau, output, config=None))]
pub fn forward_dynamics_into(
    model: &PyRobotModel,
    chain: &PyChainSpec,
    q: &Bound<'_, PyAny>,
    qd: &Bound<'_, PyAny>,
    tau: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    config: Option<&PyDynamicsConfig>,
) -> PyResult<()> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match (
        utils::real_array1(q, "q")?,
        utils::real_array1(qd, "qd")?,
        utils::real_array1(tau, "tau")?,
    ) {
        (
            utils::RealReadonlyArray1::F64(q_arr),
            utils::RealReadonlyArray1::F64(qd_arr),
            utils::RealReadonlyArray1::F64(tau_arr),
        ) => {
            let qdd = forward_dynamics_with_config(
                &model.inner,
                &chain.inner,
                &q_arr.as_array().to_owned(),
                &qd_arr.as_array().to_owned(),
                &tau_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            out.as_array_mut().assign(&qdd);
            Ok(())
        }
        _ => Err(utils::matching_real_dtype_error(&["q", "qd", "tau", "output"])),
    }
}

//! Robot model bindings (URDF ingestion, chain extraction, fixtures).

use nabled::kinematics::chain::ChainSpec;
use nabled::model::dh::to_chain_spec;
use nabled::model::fixture::{Planar2rFixture, SixDofDhFixture};
use nabled::model::robot::RobotModel;
use nabled::model::urdf::{from_urdf_file, from_urdf_str};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::physical_ai::kinematics::PyChainSpec;

/// Robot model carrier.
#[pyclass(name = "RobotModel")]
#[derive(Clone)]
pub struct PyRobotModel {
    pub(crate) inner: RobotModel<f64>,
}

#[pymethods]
impl PyRobotModel {
    #[getter]
    fn dof(&self) -> usize {
        self.inner.dof()
    }
}

/// Load a robot model from a URDF file path.
#[pyfunction]
pub fn from_urdf_file_py(path: &str) -> PyResult<PyRobotModel> {
    let inner = from_urdf_file::<f64>(path).map_err(to_py_err)?;
    Ok(PyRobotModel { inner })
}

/// Load a robot model from a URDF string.
#[pyfunction]
pub fn from_urdf_str_py(urdf: &str) -> PyResult<PyRobotModel> {
    let inner = from_urdf_str::<f64>(urdf).map_err(to_py_err)?;
    Ok(PyRobotModel { inner })
}

/// Convert a serial robot model to a `ChainSpec`.
#[pyfunction]
pub fn to_chain_spec_py(model: &PyRobotModel) -> PyResult<PyChainSpec> {
    let inner: ChainSpec<f64> = to_chain_spec(&model.inner).map_err(to_py_err)?;
    Ok(PyChainSpec { inner })
}

/// Planar 2R JSON fixture carrier.
#[pyclass(name = "Planar2rFixture")]
pub struct PyPlanar2rFixture {
    pub(crate) inner: Planar2rFixture,
}

#[pymethods]
impl PyPlanar2rFixture {
    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[getter]
    fn gravity(&self) -> Option<[f64; 3]> {
        self.inner.gravity
    }

    fn to_robot_model(&self) -> PyResult<PyRobotModel> {
        let inner = self.inner.to_robot_model::<f64>().map_err(to_py_err)?;
        Ok(PyRobotModel { inner })
    }

    fn to_chain_spec(&self) -> PyResult<PyChainSpec> {
        let inner = self.inner.to_chain_spec::<f64>().map_err(to_py_err)?;
        Ok(PyChainSpec { inner })
    }
}

/// Six-DOF DH JSON fixture carrier.
#[pyclass(name = "SixDofDhFixture")]
pub struct PySixDofDhFixture {
    pub(crate) inner: SixDofDhFixture,
}

#[pymethods]
impl PySixDofDhFixture {
    fn to_chain_spec(&self) -> PyResult<PyChainSpec> {
        let inner = self.inner.to_chain_spec::<f64>().map_err(to_py_err)?;
        Ok(PyChainSpec { inner })
    }
}

/// Load the planar 2R JSON fixture from a file path.
#[pyfunction]
pub fn load_planar2r_fixture(path: &str) -> PyResult<PyPlanar2rFixture> {
    let inner = Planar2rFixture::from_file(path).map_err(to_py_err)?;
    Ok(PyPlanar2rFixture { inner })
}

/// Load the six-DOF DH JSON fixture from a file path.
#[pyfunction]
pub fn load_six_dof_dh_fixture(path: &str) -> PyResult<PySixDofDhFixture> {
    let inner = SixDofDhFixture::from_file(path).map_err(to_py_err)?;
    Ok(PySixDofDhFixture { inner })
}

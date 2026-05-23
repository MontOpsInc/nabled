//! Kinematics bindings (FK, Jacobian, IK).

use nabled::kinematics::chain::{ChainSpec, DhConvention, JointType};
use nabled::kinematics::fk::{end_effector_pose, fk_view};
use nabled::kinematics::ik::{IkConfig, IkResult, inverse_kinematics_dls_with_limits, pose_error};
use nabled::kinematics::jacobian::{jacobian, jacobian_translation};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::physical_ai::geometry::PyTransform3;
use crate::utils;

/// Serial DH chain specification.
#[pyclass(name = "ChainSpec")]
#[derive(Clone)]
pub struct PyChainSpec {
    pub(crate) inner: ChainSpec<f64>,
}

#[pymethods]
impl PyChainSpec {
    #[staticmethod]
    #[pyo3(signature = (joint_types, a, alpha, d, theta_offset, *, convention="standard"))]
    fn from_dh(
        joint_types: Vec<String>,
        a: &Bound<'_, PyAny>,
        alpha: &Bound<'_, PyAny>,
        d: &Bound<'_, PyAny>,
        theta_offset: &Bound<'_, PyAny>,
        convention: &str,
    ) -> PyResult<Self> {
        let dh_convention = match convention {
            "standard" => DhConvention::Standard,
            "modified" => DhConvention::Modified,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "convention must be 'standard' or 'modified'",
                ));
            }
        };
        let joint_types = joint_types
            .into_iter()
            .map(|name| match name.as_str() {
                "revolute" => Ok(JointType::Revolute),
                "prismatic" => Ok(JointType::Prismatic),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "joint_types entries must be 'revolute' or 'prismatic'",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        match (
            utils::real_array1(a, "a")?,
            utils::real_array1(alpha, "alpha")?,
            utils::real_array1(d, "d")?,
            utils::real_array1(theta_offset, "theta_offset")?,
        ) {
            (
                utils::RealReadonlyArray1::F64(a_arr),
                utils::RealReadonlyArray1::F64(alpha_arr),
                utils::RealReadonlyArray1::F64(d_arr),
                utils::RealReadonlyArray1::F64(theta_arr),
            ) => {
                let inner = ChainSpec::from_dh(
                    dh_convention,
                    joint_types,
                    a_arr.as_array().to_owned(),
                    alpha_arr.as_array().to_owned(),
                    d_arr.as_array().to_owned(),
                    theta_arr.as_array().to_owned(),
                )
                .map_err(to_py_err)?;
                Ok(Self { inner })
            }
            _ => Err(utils::matching_real_dtype_error(&["a", "alpha", "d", "theta_offset"])),
        }
    }

    #[getter]
    fn num_joints(&self) -> usize {
        self.inner.num_joints()
    }
}

/// IK solver configuration.
#[pyclass(name = "IkConfig")]
#[derive(Clone)]
pub struct PyIkConfig {
    pub(crate) inner: IkConfig<f64>,
}

#[pymethods]
impl PyIkConfig {
    #[new]
    #[pyo3(signature = (*, max_iterations=500, tolerance=1e-4, damping=0.01, step_scale=1.0))]
    fn new(max_iterations: usize, tolerance: f64, damping: f64, step_scale: f64) -> Self {
        Self { inner: IkConfig { max_iterations, tolerance, damping, step_scale } }
    }
}

/// DLS IK result.
#[pyclass(name = "IkResult")]
pub struct PyIkResult {
    #[pyo3(get)]
    pub q: Py<PyAny>,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub final_error: f64,
    #[pyo3(get)]
    pub converged: bool,
}

fn ik_result_to_py(py: Python<'_>, result: IkResult<f64>) -> PyResult<PyIkResult> {
    Ok(PyIkResult {
        q: utils::pyarray1_from_owned(py, result.q),
        iterations: result.iterations,
        final_error: result.final_error,
        converged: result.converged,
    })
}

/// End-effector pose for a joint configuration.
#[pyfunction]
pub fn end_effector_pose_py(chain: &PyChainSpec, q: &Bound<'_, PyAny>) -> PyResult<PyTransform3> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let pose =
                end_effector_pose(&chain.inner, &arr.as_array().to_owned()).map_err(to_py_err)?;
            Ok(PyTransform3 { inner: pose })
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Forward kinematics returning rotation and translation arrays.
#[pyfunction]
pub fn fk<'py>(
    py: Python<'py>,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let pose = fk_view(&chain.inner, &arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, pose.rotation.matrix),
                utils::pyarray1_from_owned(py, pose.translation),
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Full 6×n geometric Jacobian.
#[pyfunction]
pub fn jacobian_py<'py>(
    py: Python<'py>,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let j = jacobian(&chain.inner, &arr.as_array().to_owned()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, j))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Translational 2×n (or 3×n) Jacobian block.
#[pyfunction]
pub fn jacobian_translation_py<'py>(
    py: Python<'py>,
    chain: &PyChainSpec,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let j = jacobian_translation(&chain.inner, &arr.as_array().to_owned())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, j))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Pose error twist between two transforms.
#[pyfunction]
pub fn pose_error_py<'py>(
    py: Python<'py>,
    achieved: &PyTransform3,
    target: &PyTransform3,
) -> PyResult<Py<PyAny>> {
    let err = pose_error(&achieved.inner, &target.inner).map_err(to_py_err)?;
    Ok(utils::pyarray1_from_owned(py, err))
}

/// Damped least-squares inverse kinematics.
#[pyfunction]
#[pyo3(signature = (chain, q_init, target, config=None))]
pub fn inverse_kinematics_dls_py<'py>(
    py: Python<'py>,
    chain: &PyChainSpec,
    q_init: &Bound<'py, PyAny>,
    target: &PyTransform3,
    config: Option<&PyIkConfig>,
) -> PyResult<PyIkResult> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match utils::real_array1(q_init, "q_init")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let result = inverse_kinematics_dls_with_limits(
                &chain.inner,
                &arr.as_array().to_owned(),
                &target.inner,
                &config,
                None,
            )
            .map_err(to_py_err)?;
            ik_result_to_py(py, result)
        }
        _ => Err(utils::matching_real_dtype_error(&["q_init"])),
    }
}

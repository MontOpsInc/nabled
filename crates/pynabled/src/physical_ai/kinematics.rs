//! Kinematics bindings (FK, Jacobian, IK).

use nabled::kinematics::chain::{ChainSpec, DhConvention, JointType};
use nabled::kinematics::fk::fk_view;
use nabled::kinematics::ik::{
    IkConfig, IkResult, IkWorkspace, inverse_kinematics_dls_into,
    inverse_kinematics_dls_with_limits, inverse_kinematics_tree_dls_with_limits, pose_error,
};
use nabled::kinematics::jacobian::{jacobian_translation_view, jacobian_view};
use nabled::kinematics::tree::{
    end_effector_pose_tree_view, jacobian_tree_view, link_transforms_tree_view,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::error::to_py_err;
use crate::physical_ai::geometry::PyTransform3;
use crate::physical_ai::model::PyRobotModel;
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
    fn num_joints(&self) -> usize { self.inner.num_joints() }
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

/// Reusable workspace for DLS IK hot paths.
#[pyclass(name = "IkWorkspace")]
pub struct PyIkWorkspace {
    pub(crate) inner: IkWorkspace<f64>,
}

#[pymethods]
impl PyIkWorkspace {
    #[new]
    fn new(num_joints: usize) -> Self { Self { inner: IkWorkspace::new(num_joints) } }
}

/// DLS IK result.
#[pyclass(name = "IkResult")]
pub struct PyIkResult {
    #[pyo3(get)]
    pub q:           Py<PyAny>,
    #[pyo3(get)]
    pub iterations:  usize,
    #[pyo3(get)]
    pub final_error: f64,
    #[pyo3(get)]
    pub converged:   bool,
}

fn ik_result_to_py(py: Python<'_>, result: IkResult<f64>) -> PyResult<PyIkResult> {
    Ok(PyIkResult {
        q:           utils::pyarray1_from_owned(py, result.q),
        iterations:  result.iterations,
        final_error: result.final_error,
        converged:   result.converged,
    })
}

/// End-effector pose for a joint configuration.
#[pyfunction]
pub fn end_effector_pose_py(chain: &PyChainSpec, q: &Bound<'_, PyAny>) -> PyResult<PyTransform3> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let pose = fk_view(&chain.inner, &arr.as_array()).map_err(to_py_err)?;
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
            let j = jacobian_view(&chain.inner, &arr.as_array()).map_err(to_py_err)?;
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
            let j = jacobian_translation_view(&chain.inner, &arr.as_array()).map_err(to_py_err)?;
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

/// Pose error into caller-provided output buffer.
#[pyfunction]
pub fn pose_error_into_py(
    achieved: &PyTransform3,
    target: &PyTransform3,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
    if out.as_array_mut().len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err("output must have length 6"));
    }
    let err = pose_error(&achieved.inner, &target.inner).map_err(to_py_err)?;
    out.as_array_mut().assign(&err);
    Ok(())
}

/// End-effector pose on a branched kinematic tree.
#[pyfunction]
pub fn end_effector_pose_tree_py(
    model: &PyRobotModel,
    base_link: &str,
    ee_link: &str,
    q: &Bound<'_, PyAny>,
) -> PyResult<PyTransform3> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let pose =
                end_effector_pose_tree_view(&model.inner, base_link, ee_link, &arr.as_array())
                    .map_err(to_py_err)?;
            Ok(PyTransform3 { inner: pose })
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Geometric Jacobian for a branched kinematic tree.
#[pyfunction]
pub fn jacobian_tree_py<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    base_link: &str,
    ee_link: &str,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let j = jacobian_tree_view(&model.inner, base_link, ee_link, &arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, j))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// World-frame link transforms for a branched kinematic tree.
#[pyfunction]
pub fn link_transforms_tree_py<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    q: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let transforms =
                link_transforms_tree_view(&model.inner, &arr.as_array()).map_err(to_py_err)?;
            let dict = PyDict::new(py);
            for (link, transform) in transforms {
                dict.set_item(link, PyTransform3 { inner: transform })?;
            }
            Ok(dict)
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
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

/// DLS IK into caller buffers with reusable workspace.
#[pyfunction]
#[pyo3(signature = (chain, q_init, target, output, config=None, workspace=None))]
pub fn inverse_kinematics_dls_into_py<'py>(
    py: Python<'py>,
    chain: &PyChainSpec,
    q_init: &Bound<'py, PyAny>,
    target: &PyTransform3,
    output: &Bound<'py, PyAny>,
    config: Option<&PyIkConfig>,
    workspace: Option<&mut PyIkWorkspace>,
) -> PyResult<PyIkResult> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match utils::real_array1(q_init, "q_init")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            if out.as_array_mut().len() != chain.inner.num_joints() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "output length must match chain joint count",
                ));
            }
            let mut owned_workspace = IkWorkspace::new(chain.inner.num_joints());
            let workspace_ref =
                if let Some(ws) = workspace { &mut ws.inner } else { &mut owned_workspace };
            let mut q_out = ndarray::Array1::<f64>::zeros(chain.inner.num_joints());
            let result = inverse_kinematics_dls_into(
                &chain.inner,
                &arr.as_array().to_owned(),
                &target.inner,
                &config,
                None,
                workspace_ref,
                &mut q_out,
            )
            .map_err(to_py_err)?;
            out.as_array_mut().assign(&q_out);
            ik_result_to_py(py, result)
        }
        _ => Err(utils::matching_real_dtype_error(&["q_init", "output"])),
    }
}

/// Tree DLS inverse kinematics on a branched kinematic model.
#[pyfunction]
#[pyo3(signature = (model, base_link, ee_link, q_init, target, config=None))]
pub fn inverse_kinematics_tree_dls_py<'py>(
    py: Python<'py>,
    model: &PyRobotModel,
    base_link: &str,
    ee_link: &str,
    q_init: &Bound<'py, PyAny>,
    target: &PyTransform3,
    config: Option<&PyIkConfig>,
) -> PyResult<PyIkResult> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    match utils::real_array1(q_init, "q_init")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let result = inverse_kinematics_tree_dls_with_limits(
                &model.inner,
                base_link,
                ee_link,
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

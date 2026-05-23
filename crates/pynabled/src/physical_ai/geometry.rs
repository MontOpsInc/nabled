//! Geometry bindings (quaternions, SO(3), SE(3)).

use nabled::linalg::geometry::{self, AxisAngle, Quat, Transform3, quat, se3, so3};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::physical_ai::common::{transform_from_arrays, transform_to_parts};
use crate::utils;

/// Rigid transform carrier (rotation matrix + translation).
#[pyclass(name = "Transform3")]
#[derive(Clone)]
pub struct PyTransform3 {
    pub(crate) inner: Transform3<f64>,
}

#[pymethods]
impl PyTransform3 {
    #[new]
    fn new(rotation: &Bound<'_, PyAny>, translation: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self { inner: transform_from_arrays(rotation, translation)? })
    }

    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(utils::pyarray2_from_owned(py, self.inner.rotation.matrix.clone()))
    }

    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(utils::pyarray1_from_owned(py, self.inner.translation.clone()))
    }

    fn to_homogeneous<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(utils::pyarray2_from_owned(py, se3::to_homogeneous(&self.inner)))
    }
}

/// Build a unit quaternion from axis-angle parameters.
#[pyfunction]
#[pyo3(signature = (axis, angle))]
pub fn quat_from_axis_angle<'py>(
    py: Python<'py>,
    axis: [f64; 3],
    angle: f64,
) -> PyResult<Py<PyAny>> {
    let q = quat::from_axis_angle(&AxisAngle { axis, angle });
    Ok(utils::pyarray1_from_owned(py, ndarray::arr1(&[q.w, q.x, q.y, q.z])))
}

/// Convert quaternion `[w, x, y, z]` to a 3×3 rotation matrix.
#[pyfunction]
pub fn quat_to_rotation_matrix<'py>(py: Python<'py>, q: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array1(q, "q")? {
        utils::RealReadonlyArray1::F64(arr) => {
            if arr.as_array().len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "q must have length 4 [w, x, y, z]",
                ));
            }
            let quat = Quat {
                w: arr.as_array()[0],
                x: arr.as_array()[1],
                y: arr.as_array()[2],
                z: arr.as_array()[3],
            };
            Ok(utils::pyarray2_from_owned(py, quat::to_rotation_matrix(&quat).matrix))
        }
        _ => Err(utils::matching_real_dtype_error(&["q"])),
    }
}

/// Compose two rigid transforms.
#[pyfunction]
pub fn se3_compose(left: &PyTransform3, right: &PyTransform3) -> PyResult<PyTransform3> {
    let composed = se3::compose(&left.inner, &right.inner).map_err(to_py_err)?;
    Ok(PyTransform3 { inner: composed })
}

/// SE(3) logarithm map.
#[pyfunction]
pub fn se3_log<'py>(py: Python<'py>, transform: &PyTransform3) -> PyResult<Py<PyAny>> {
    let twist = se3::log(&transform.inner).map_err(to_py_err)?;
    Ok(utils::pyarray1_from_owned(py, twist))
}

/// SE(3) exponential map from a 6-vector twist.
#[pyfunction]
pub fn se3_exp(twist: &Bound<'_, PyAny>) -> PyResult<PyTransform3> {
    match utils::real_array1(twist, "twist")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let transform = se3::exp(&arr.as_array()).map_err(to_py_err)?;
            Ok(PyTransform3 { inner: transform })
        }
        _ => Err(utils::matching_real_dtype_error(&["twist"])),
    }
}

/// Build a `Transform3` from rotation and translation arrays.
#[pyfunction]
pub fn transform3_from_parts(
    rotation: &Bound<'_, PyAny>,
    translation: &Bound<'_, PyAny>,
) -> PyResult<PyTransform3> {
    Ok(PyTransform3 { inner: transform_from_arrays(rotation, translation)? })
}

/// Export rotation and translation arrays from a transform.
#[pyfunction]
pub fn transform3_to_parts<'py>(
    py: Python<'py>,
    transform: &PyTransform3,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    transform_to_parts(py, &transform.inner)
}

/// SO(3) compose two rotation matrices.
#[pyfunction]
pub fn so3_compose<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(left, "left")?, utils::real_array2(right, "right")?) {
        (utils::RealReadonlyArray2::F64(l), utils::RealReadonlyArray2::F64(r)) => {
            let rot_l = geometry::Rotation3 { matrix: l.as_array().to_owned() };
            let rot_r = geometry::Rotation3 { matrix: r.as_array().to_owned() };
            let composed = so3::compose(&rot_l, &rot_r).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, composed.matrix))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

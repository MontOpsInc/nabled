//! Shared Physical AI binding helpers.

use nabled::linalg::geometry::{Rotation3, Transform3};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::utils;

pub(crate) fn transform_from_arrays(
    rotation: &Bound<'_, PyAny>,
    translation: &Bound<'_, PyAny>,
) -> PyResult<Transform3<f64>> {
    let rotation_arr = utils::real_array2(rotation, "rotation")?;
    let translation_arr = utils::real_array1(translation, "translation")?;
    match (rotation_arr, translation_arr) {
        (utils::RealReadonlyArray2::F64(r), utils::RealReadonlyArray1::F64(t)) => {
            if r.as_array().nrows() != 3 || r.as_array().ncols() != 3 || t.as_array().len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "rotation must be 3x3 and translation must have length 3",
                ));
            }
            Ok(Transform3 {
                rotation:    Rotation3 { matrix: r.as_array().to_owned() },
                translation: t.as_array().to_owned(),
            })
        }
        _ => Err(utils::matching_real_dtype_error(&["rotation", "translation"])),
    }
}

pub(crate) fn transform_to_parts<'py>(
    py: Python<'py>,
    transform: &Transform3<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    Ok((
        utils::pyarray2_from_owned(py, transform.rotation.matrix.clone()),
        utils::pyarray1_from_owned(py, transform.translation.clone()),
    ))
}

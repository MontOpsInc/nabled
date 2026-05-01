//! Schur decomposition bindings for Python.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

pub(crate) enum PySchurWorkspaceInner {
    F32(nabled_linalg::schur::SchurWorkspace<f32>),
    F64(nabled_linalg::schur::SchurWorkspace<f64>),
    C64(nabled_linalg::schur::SchurComplexWorkspace),
}

#[pyclass(module = "pynabled._pynabled", name = "_SchurWorkspace")]
pub(crate) struct PySchurWorkspace {
    pub(crate) inner: PySchurWorkspaceInner,
}

#[pymethods]
impl PySchurWorkspace {
    #[new]
    fn new(dtype: &str) -> PyResult<Self> {
        match dtype {
            "float32" => Ok(Self {
                inner: PySchurWorkspaceInner::F32(nabled_linalg::schur::SchurWorkspace::default()),
            }),
            "float64" => Ok(Self {
                inner: PySchurWorkspaceInner::F64(nabled_linalg::schur::SchurWorkspace::default()),
            }),
            "complex128" => Ok(Self {
                inner: PySchurWorkspaceInner::C64(
                    nabled_linalg::schur::SchurComplexWorkspace::default(),
                ),
            }),
            _ => Err(PyTypeError::new_err("dtype must be float32, float64, or complex128")),
        }
    }

    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (PySchurWorkspaceInner::F32(workspace), utils::NumericReadonlyArray2::F32(arr)) => {
                let shape = arr.as_array().dim();
                let mut q = ndarray::Array2::<f32>::zeros(shape);
                let mut t = ndarray::Array2::<f32>::zeros(shape);
                nabled_linalg::schur::compute_schur_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q,
                    &mut t,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok((utils::pyarray2_from_owned(py, t), utils::pyarray2_from_owned(py, q)))
            }
            (PySchurWorkspaceInner::F64(workspace), utils::NumericReadonlyArray2::F64(arr)) => {
                let shape = arr.as_array().dim();
                let mut q = ndarray::Array2::<f64>::zeros(shape);
                let mut t = ndarray::Array2::<f64>::zeros(shape);
                nabled_linalg::schur::compute_schur_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q,
                    &mut t,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok((utils::pyarray2_from_owned(py, t), utils::pyarray2_from_owned(py, q)))
            }
            (PySchurWorkspaceInner::C64(workspace), utils::NumericReadonlyArray2::C64(arr)) => {
                let shape = arr.as_array().dim();
                let mut q = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
                let mut t = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
                nabled_linalg::schur::compute_schur_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q,
                    &mut t,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok((utils::pyarray2_from_owned(py, t), utils::pyarray2_from_owned(py, q)))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn compute_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output_q: &Bound<'_, PyAny>,
        output_t: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (PySchurWorkspaceInner::F32(workspace), utils::NumericReadonlyArray2::F32(arr)) => {
                let mut q = utils::output_array2::<f32>(output_q, "output_q", "float32")?;
                let mut t = utils::output_array2::<f32>(output_t, "output_t", "float32")?;
                nabled_linalg::schur::compute_schur_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q.as_array_mut(),
                    &mut t.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (PySchurWorkspaceInner::F64(workspace), utils::NumericReadonlyArray2::F64(arr)) => {
                let mut q = utils::output_array2::<f64>(output_q, "output_q", "float64")?;
                let mut t = utils::output_array2::<f64>(output_t, "output_t", "float64")?;
                nabled_linalg::schur::compute_schur_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q.as_array_mut(),
                    &mut t.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (PySchurWorkspaceInner::C64(workspace), utils::NumericReadonlyArray2::C64(arr)) => {
                let mut q = utils::output_array2::<num_complex::Complex64>(
                    output_q,
                    "output_q",
                    "complex128",
                )?;
                let mut t = utils::output_array2::<num_complex::Complex64>(
                    output_t,
                    "output_t",
                    "complex128",
                )?;
                nabled_linalg::schur::compute_schur_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut q.as_array_mut(),
                    &mut t.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&[
                "matrix",
                "output_q",
                "output_t",
                "workspace",
            ])),
        }
    }
}

/// Compute Schur decomposition. Returns (T, Q).
#[pyfunction(name = "schur_compute")]
pub fn compute_schur<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::schur::compute_schur_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::schur::compute_schur_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::schur::compute_schur_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
    }
}

/// Compute Schur decomposition into caller-provided `output_q` and `output_t`.
#[pyfunction(name = "schur_compute_into")]
pub fn compute_schur_into(
    matrix: &Bound<'_, PyAny>,
    output_q: &Bound<'_, PyAny>,
    output_t: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut q = utils::output_array2::<f32>(output_q, "output_q", "float32")?;
            let mut t = utils::output_array2::<f32>(output_t, "output_t", "float32")?;
            nabled_linalg::schur::compute_schur_into_view(
                &arr.as_array(),
                &mut q.as_array_mut(),
                &mut t.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut q = utils::output_array2::<f64>(output_q, "output_q", "float64")?;
            let mut t = utils::output_array2::<f64>(output_t, "output_t", "float64")?;
            nabled_linalg::schur::compute_schur_into_view(
                &arr.as_array(),
                &mut q.as_array_mut(),
                &mut t.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut q =
                utils::output_array2::<num_complex::Complex64>(output_q, "output_q", "complex128")?;
            let mut t =
                utils::output_array2::<num_complex::Complex64>(output_t, "output_t", "complex128")?;
            nabled_linalg::schur::compute_schur_complex_into_view(
                &arr.as_array(),
                &mut q.as_array_mut(),
                &mut t.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

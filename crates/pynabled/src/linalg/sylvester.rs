//! Sylvester and Lyapunov solver bindings for Python.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

pub(crate) enum PySylvesterWorkspaceInner {
    F32(nabled_linalg::sylvester::SylvesterWorkspace<f32>),
    F64(nabled_linalg::sylvester::SylvesterWorkspace<f64>),
    C64(nabled_linalg::sylvester::SylvesterComplexWorkspace),
}

#[pyclass(module = "pynabled._pynabled", name = "_SylvesterWorkspace")]
pub(crate) struct PySylvesterWorkspace {
    pub(crate) inner: PySylvesterWorkspaceInner,
}

#[pymethods]
impl PySylvesterWorkspace {
    #[new]
    fn new(dtype: &str) -> PyResult<Self> {
        match dtype {
            "float32" => Ok(Self {
                inner: PySylvesterWorkspaceInner::F32(
                    nabled_linalg::sylvester::SylvesterWorkspace::default(),
                ),
            }),
            "float64" => Ok(Self {
                inner: PySylvesterWorkspaceInner::F64(
                    nabled_linalg::sylvester::SylvesterWorkspace::default(),
                ),
            }),
            "complex128" => Ok(Self {
                inner: PySylvesterWorkspaceInner::C64(
                    nabled_linalg::sylvester::SylvesterComplexWorkspace::default(),
                ),
            }),
            _ => Err(PyTypeError::new_err("dtype must be float32, float64, or complex128")),
        }
    }

    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        matrix_a: &Bound<'py, PyAny>,
        matrix_b: &Bound<'py, PyAny>,
        matrix_c: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match (
            &mut self.inner,
            utils::numeric_array2(matrix_a, "matrix_a")?,
            utils::numeric_array2(matrix_b, "matrix_b")?,
            utils::numeric_array2(matrix_c, "matrix_c")?,
        ) {
            (
                PySylvesterWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(a),
                utils::NumericReadonlyArray2::F32(b),
                utils::NumericReadonlyArray2::F32(c),
            ) => {
                let mut output =
                    ndarray::Array2::<f32>::zeros((a.as_array().nrows(), b.as_array().ncols()));
                nabled_linalg::sylvester::solve_sylvester_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PySylvesterWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(a),
                utils::NumericReadonlyArray2::F64(b),
                utils::NumericReadonlyArray2::F64(c),
            ) => {
                let mut output =
                    ndarray::Array2::<f64>::zeros((a.as_array().nrows(), b.as_array().ncols()));
                nabled_linalg::sylvester::solve_sylvester_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PySylvesterWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(a),
                utils::NumericReadonlyArray2::C64(b),
                utils::NumericReadonlyArray2::C64(c),
            ) => {
                let mut output = ndarray::Array2::<num_complex::Complex64>::zeros((
                    a.as_array().nrows(),
                    b.as_array().ncols(),
                ));
                nabled_linalg::sylvester::solve_sylvester_complex_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&[
                "matrix_a",
                "matrix_b",
                "matrix_c",
                "workspace",
            ])),
        }
    }

    fn solve_into(
        &mut self,
        matrix_a: &Bound<'_, PyAny>,
        matrix_b: &Bound<'_, PyAny>,
        matrix_c: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (
            &mut self.inner,
            utils::numeric_array2(matrix_a, "matrix_a")?,
            utils::numeric_array2(matrix_b, "matrix_b")?,
            utils::numeric_array2(matrix_c, "matrix_c")?,
        ) {
            (
                PySylvesterWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(a),
                utils::NumericReadonlyArray2::F32(b),
                utils::NumericReadonlyArray2::F32(c),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::sylvester::solve_sylvester_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PySylvesterWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(a),
                utils::NumericReadonlyArray2::F64(b),
                utils::NumericReadonlyArray2::F64(c),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::sylvester::solve_sylvester_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PySylvesterWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(a),
                utils::NumericReadonlyArray2::C64(b),
                utils::NumericReadonlyArray2::C64(c),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::sylvester::solve_sylvester_complex_view_with_workspace_into(
                    &a.as_array(),
                    &b.as_array(),
                    &c.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&[
                "matrix_a",
                "matrix_b",
                "matrix_c",
                "output",
                "workspace",
            ])),
        }
    }

    fn lyapunov<'py>(
        &mut self,
        py: Python<'py>,
        a: &Bound<'py, PyAny>,
        q: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(a, "a")?, utils::numeric_array2(q, "q")?) {
            (
                PySylvesterWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(a_arr),
                utils::NumericReadonlyArray2::F32(q_arr),
            ) => {
                let mut output = ndarray::Array2::<f32>::zeros(q_arr.as_array().dim());
                nabled_linalg::sylvester::solve_lyapunov_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PySylvesterWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(a_arr),
                utils::NumericReadonlyArray2::F64(q_arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(q_arr.as_array().dim());
                nabled_linalg::sylvester::solve_lyapunov_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PySylvesterWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(a_arr),
                utils::NumericReadonlyArray2::C64(q_arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(q_arr.as_array().dim());
                nabled_linalg::sylvester::solve_lyapunov_complex_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["a", "q", "workspace"])),
        }
    }

    fn lyapunov_into(
        &mut self,
        a: &Bound<'_, PyAny>,
        q: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(a, "a")?, utils::numeric_array2(q, "q")?) {
            (
                PySylvesterWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(a_arr),
                utils::NumericReadonlyArray2::F32(q_arr),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::sylvester::solve_lyapunov_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PySylvesterWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(a_arr),
                utils::NumericReadonlyArray2::F64(q_arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::sylvester::solve_lyapunov_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PySylvesterWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(a_arr),
                utils::NumericReadonlyArray2::C64(q_arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::sylvester::solve_lyapunov_complex_view_with_workspace_into(
                    &a_arr.as_array(),
                    &q_arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["a", "q", "output", "workspace"])),
        }
    }
}

/// Solve Sylvester equation AX + XB = C.
#[pyfunction(name = "sylvester_solve")]
pub fn solve_sylvester<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    matrix_c: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array2(matrix_b, "matrix_b")?,
        utils::numeric_array2(matrix_c, "matrix_c")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(a),
            utils::NumericReadonlyArray2::F32(b),
            utils::NumericReadonlyArray2::F32(c),
        ) => {
            let mut output =
                ndarray::Array2::<f32>::zeros((a.as_array().nrows(), b.as_array().ncols()));
            nabled_linalg::sylvester::solve_sylvester_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::F64(a),
            utils::NumericReadonlyArray2::F64(b),
            utils::NumericReadonlyArray2::F64(c),
        ) => {
            let mut output =
                ndarray::Array2::<f64>::zeros((a.as_array().nrows(), b.as_array().ncols()));
            nabled_linalg::sylvester::solve_sylvester_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::C64(a),
            utils::NumericReadonlyArray2::C64(b),
            utils::NumericReadonlyArray2::C64(c),
        ) => {
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros((
                a.as_array().nrows(),
                b.as_array().ncols(),
            ));
            nabled_linalg::sylvester::solve_sylvester_complex_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix_a", "matrix_b", "matrix_c"])),
    }
}

/// Solve Sylvester equation `A X + X B = C` into `output`.
#[pyfunction(name = "sylvester_solve_into")]
pub fn solve_sylvester_into(
    matrix_a: &Bound<'_, PyAny>,
    matrix_b: &Bound<'_, PyAny>,
    matrix_c: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array2(matrix_b, "matrix_b")?,
        utils::numeric_array2(matrix_c, "matrix_c")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(a),
            utils::NumericReadonlyArray2::F32(b),
            utils::NumericReadonlyArray2::F32(c),
        ) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::sylvester::solve_sylvester_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(a),
            utils::NumericReadonlyArray2::F64(b),
            utils::NumericReadonlyArray2::F64(c),
        ) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::sylvester::solve_sylvester_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(a),
            utils::NumericReadonlyArray2::C64(b),
            utils::NumericReadonlyArray2::C64(c),
        ) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::sylvester::solve_sylvester_complex_view_into(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix_a", "matrix_b", "matrix_c"])),
    }
}

/// Solve Lyapunov equation AX + XA^T = Q.
#[pyfunction(name = "lyapunov_solve")]
pub fn solve_lyapunov<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array2(q, "q")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray2::F32(q_arr)) => {
            let mut output = ndarray::Array2::<f32>::zeros(q_arr.as_array().dim());
            nabled_linalg::sylvester::solve_lyapunov_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray2::F64(q_arr)) => {
            let mut output = ndarray::Array2::<f64>::zeros(q_arr.as_array().dim());
            nabled_linalg::sylvester::solve_lyapunov_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray2::C64(q_arr)) => {
            let mut output =
                ndarray::Array2::<num_complex::Complex64>::zeros(q_arr.as_array().dim());
            nabled_linalg::sylvester::solve_lyapunov_complex_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "q"])),
    }
}

/// Solve Lyapunov equation `A X + X A^T = -Q` into `output`.
#[pyfunction(name = "lyapunov_solve_into")]
pub fn solve_lyapunov_into(
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array2(q, "q")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray2::F32(q_arr)) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::sylvester::solve_lyapunov_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray2::F64(q_arr)) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::sylvester::solve_lyapunov_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray2::C64(q_arr)) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::sylvester::solve_lyapunov_complex_view_into(
                &a_arr.as_array(),
                &q_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "q"])),
    }
}

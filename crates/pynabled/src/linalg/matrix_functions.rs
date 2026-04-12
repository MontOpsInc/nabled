//! Matrix function bindings for Python.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

const DEFAULT_MAX_TERMS: usize = 64;

const DEFAULT_TOLERANCE: f64 = 1e-14;

fn complex_svd_component_error() -> PyErr {
    PyTypeError::new_err(
        "u, vt must both be complex128 and singular_values must be float64 for complex SVD results",
    )
}

pub(crate) enum PyMatrixFunctionWorkspaceInner {
    F32(nabled_linalg::matrix_functions::MatrixFunctionWorkspace<f32>),
    F64(nabled_linalg::matrix_functions::MatrixFunctionWorkspace<f64>),
    C64(nabled_linalg::matrix_functions::MatrixFunctionComplexWorkspace),
}

#[pyclass(module = "pynabled._pynabled", name = "_MatrixFunctionWorkspace")]
pub(crate) struct PyMatrixFunctionWorkspace {
    pub(crate) inner: PyMatrixFunctionWorkspaceInner,
}

#[pymethods]
impl PyMatrixFunctionWorkspace {
    #[new]
    fn new(dtype: &str) -> PyResult<Self> {
        match dtype {
            "float32" => Ok(Self {
                inner: PyMatrixFunctionWorkspaceInner::F32(
                    nabled_linalg::matrix_functions::MatrixFunctionWorkspace::default(),
                ),
            }),
            "float64" => Ok(Self {
                inner: PyMatrixFunctionWorkspaceInner::F64(
                    nabled_linalg::matrix_functions::MatrixFunctionWorkspace::default(),
                ),
            }),
            "complex128" => Ok(Self {
                inner: PyMatrixFunctionWorkspaceInner::C64(
                    nabled_linalg::matrix_functions::MatrixFunctionComplexWorkspace::default(),
                ),
            }),
            _ => Err(PyTypeError::new_err("dtype must be float32, float64, or complex128")),
        }
    }

    #[pyo3(signature = (matrix, max_terms=None, tolerance=None))]
    fn exp<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
        max_terms: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_complex_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    #[pyo3(signature = (matrix, output, max_terms=None, tolerance=None))]
    fn exp_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        max_terms: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<()> {
        let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_exp_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_exp_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_exp_complex_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn exp_eigen<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_exp_eigen_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn exp_eigen_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_exp_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_exp_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_exp_eigen_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn log_taylor<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
        max_terms: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
        match (&mut self.inner, utils::real_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::RealReadonlyArray2::F32(arr),
            ) => {
                let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_taylor_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::RealReadonlyArray2::F64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_taylor_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (PyMatrixFunctionWorkspaceInner::C64(_), _) => Err(PyTypeError::new_err(
                "matrix_log_taylor workspace must use dtype float32 or float64",
            )),
            _ => Err(utils::matching_real_dtype_error(&["matrix", "workspace"])),
        }
    }

    #[pyo3(signature = (matrix, output, max_terms=None, tolerance=None))]
    fn log_taylor_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        max_terms: Option<usize>,
        tolerance: Option<f64>,
    ) -> PyResult<()> {
        let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
        match (&mut self.inner, utils::real_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::RealReadonlyArray2::F32(arr),
            ) => {
                let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_log_taylor_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::RealReadonlyArray2::F64(arr),
            ) => {
                let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_log_taylor_view_with_workspace_into(
                    &arr.as_array(),
                    terms,
                    tol,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (PyMatrixFunctionWorkspaceInner::C64(_), _) => Err(PyTypeError::new_err(
                "matrix_log_taylor workspace must use dtype float32 or float64",
            )),
            _ => Err(utils::matching_real_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn log_eigen<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn log_eigen_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_log_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_log_eigen_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn log_svd<'py>(&mut self, py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_svd_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_svd_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_log_svd_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn log_svd_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_log_svd_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_log_svd_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_log_svd_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn power<'py>(
        &mut self,
        py: Python<'py>,
        matrix: &Bound<'py, PyAny>,
        power: f64,
    ) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let power = utils::f64_to_f32(power, "power")?;
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_power_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_power_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_power_complex_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn power_into(
        &mut self,
        matrix: &Bound<'_, PyAny>,
        power: f64,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let power = utils::f64_to_f32(power, "power")?;
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_power_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_power_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_power_complex_view_with_workspace_into(
                    &arr.as_array(),
                    power,
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }

    fn sign<'py>(&mut self, py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut output = ndarray::Array2::<f32>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_sign_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut output = ndarray::Array2::<f64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_sign_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut output =
                    ndarray::Array2::<num_complex::Complex64>::zeros(arr.as_array().dim());
                nabled_linalg::matrix_functions::matrix_sign_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut output,
                    workspace,
                )
                .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, output))
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "workspace"])),
        }
    }

    fn sign_into(&mut self, matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
        match (&mut self.inner, utils::numeric_array2(matrix, "matrix")?) {
            (
                PyMatrixFunctionWorkspaceInner::F32(workspace),
                utils::NumericReadonlyArray2::F32(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
                nabled_linalg::matrix_functions::matrix_sign_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::F64(workspace),
                utils::NumericReadonlyArray2::F64(arr),
            ) => {
                let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
                nabled_linalg::matrix_functions::matrix_sign_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            (
                PyMatrixFunctionWorkspaceInner::C64(workspace),
                utils::NumericReadonlyArray2::C64(arr),
            ) => {
                let mut out_arr =
                    utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
                nabled_linalg::matrix_functions::matrix_sign_complex_view_with_workspace_into(
                    &arr.as_array(),
                    &mut out_arr.as_array_mut(),
                    workspace,
                )
                .map_err(to_py_err)
            }
            _ => Err(utils::matching_numeric_dtype_error(&["matrix", "output", "workspace"])),
        }
    }
}

/// Matrix exponential via Taylor series.
#[pyfunction(name = "matrix_exp")]
pub fn matrix_exp<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_exp_complex_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix exponential via Taylor series into `output`.
#[pyfunction(name = "matrix_exp_into", signature = (matrix, output, max_terms=None, tolerance=None))]
pub fn matrix_exp_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<()> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_exp_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_exp_complex_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix exponential via eigendecomposition.
#[pyfunction(name = "matrix_exp_eigen")]
pub fn matrix_exp_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::matrix_functions::matrix_exp_eigen_complex_view(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Matrix exponential via eigendecomposition into `output`.
#[pyfunction(name = "matrix_exp_eigen_into")]
pub fn matrix_exp_eigen_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_exp_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_exp_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_exp_eigen_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via Taylor series.
#[pyfunction(name = "matrix_log_taylor")]
pub fn matrix_log_taylor<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via Taylor series into `output`.
#[pyfunction(
    name = "matrix_log_taylor_into",
    signature = (matrix, output, max_terms=None, tolerance=None)
)]
pub fn matrix_log_taylor_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<()> {
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tol = utils::f64_to_f32(tolerance.unwrap_or(DEFAULT_TOLERANCE), "tolerance")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_taylor_view_into(
                &arr.as_array(),
                terms,
                tol,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via eigendecomposition.
#[pyfunction(name = "matrix_log_eigen")]
pub fn matrix_log_eigen<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via eigendecomposition into `output`.
#[pyfunction(name = "matrix_log_eigen_into")]
pub fn matrix_log_eigen_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_log_eigen_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via SVD.
#[pyfunction(name = "matrix_log_svd")]
pub fn matrix_log_svd<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_log_svd_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix log via precomputed SVD factors.
#[pyfunction(name = "matrix_log_svd_from_factors")]
pub fn matrix_log_svd_from_factors<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    singular_values: &Bound<'py, PyAny>,
    vt: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let result = nabled_linalg::matrix_functions::matrix_log_svd_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let result = nabled_linalg::matrix_functions::matrix_log_svd_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let result = nabled_linalg::matrix_functions::matrix_log_svd_complex_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Matrix log via SVD into `output`.
#[pyfunction(name = "matrix_log_svd_into")]
pub fn matrix_log_svd_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_svd_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_log_svd_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix log via precomputed SVD factors into `output`.
#[pyfunction(name = "matrix_log_svd_from_factors_into")]
pub fn matrix_log_svd_from_factors_into(
    u: &Bound<'_, PyAny>,
    singular_values: &Bound<'_, PyAny>,
    vt: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_log_svd_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_log_svd_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_log_svd_complex_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Matrix power A^p.
#[pyfunction(name = "matrix_power")]
pub fn matrix_power<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    power: f64,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let power = utils::f64_to_f32(power, "power")?;
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_power_complex_view_into(
                &arr.as_array(),
                power,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix power into `output`.
#[pyfunction(name = "matrix_power_into")]
pub fn matrix_power_into(
    matrix: &Bound<'_, PyAny>,
    power: f64,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let power = utils::f64_to_f32(power, "power")?;
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_power_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_power_complex_view_into(
                &arr.as_array(),
                power,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Matrix sign function.
#[pyfunction(name = "matrix_sign")]
pub fn matrix_sign<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f32>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<f64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_view_into(&arr.as_array(), &mut output)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let shape = arr.as_array().dim();
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros(shape);
            nabled_linalg::matrix_functions::matrix_sign_complex_view_into(
                &arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Matrix sign function into `output`.
#[pyfunction(name = "matrix_sign_into")]
pub fn matrix_sign_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix_functions::matrix_sign_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix_functions::matrix_sign_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix_functions::matrix_sign_complex_view_into(
                &arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

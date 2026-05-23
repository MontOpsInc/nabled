//! Iterative solver bindings for Python (dense CG, GMRES).

use nabled_ml::iterative::IterativeConfig;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

#[expect(clippy::cast_possible_truncation)]
fn tolerance_f32(tolerance: Option<f64>, default: f32) -> f32 {
    tolerance.map_or(default, |value| value as f32)
}

/// Conjugate gradient solve for symmetric positive definite Ax = b.
#[pyfunction(name = "conjugate_gradient")]
pub fn conjugate_gradient<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(matrix_a, "matrix_a")?, utils::real_array1(matrix_b, "matrix_b")?) {
        (utils::RealReadonlyArray2::F32(a), utils::RealReadonlyArray1::F32(b)) => {
            let default = IterativeConfig::<f32>::default_f32();
            let config = IterativeConfig {
                tolerance: tolerance_f32(tolerance, default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let result = nabled_ml::iterative::conjugate_gradient_view(
                &a.as_array(),
                &b.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a), utils::RealReadonlyArray1::F64(b)) => {
            let default = IterativeConfig::<f64>::default_f64();
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let result = nabled_ml::iterative::conjugate_gradient_view(
                &a.as_array(),
                &b.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// Conjugate gradient solve for symmetric positive definite `Ax = b` into `output`.
#[pyfunction(name = "conjugate_gradient_into")]
pub fn conjugate_gradient_into(
    matrix_a: &Bound<'_, PyAny>,
    matrix_b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<()> {
    match (utils::real_array2(matrix_a, "matrix_a")?, utils::real_array1(matrix_b, "matrix_b")?) {
        (utils::RealReadonlyArray2::F32(a), utils::RealReadonlyArray1::F32(b)) => {
            let default = IterativeConfig::<f32>::default_f32();
            let config = IterativeConfig {
                tolerance: tolerance_f32(tolerance, default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let mut output_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_ml::iterative::conjugate_gradient_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(a), utils::RealReadonlyArray1::F64(b)) => {
            let default = IterativeConfig::<f64>::default_f64();
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let mut output_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_ml::iterative::conjugate_gradient_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b", "output"])),
    }
}

/// GMRES solve for general Ax = b.
#[pyfunction(name = "gmres")]
pub fn gmres<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(matrix_a, "matrix_a")?, utils::real_array1(matrix_b, "matrix_b")?) {
        (utils::RealReadonlyArray2::F32(a), utils::RealReadonlyArray1::F32(b)) => {
            let default = IterativeConfig::<f32>::default_f32();
            let config = IterativeConfig {
                tolerance: tolerance_f32(tolerance, default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let result = nabled_ml::iterative::gmres_view(&a.as_array(), &b.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a), utils::RealReadonlyArray1::F64(b)) => {
            let default = IterativeConfig::<f64>::default_f64();
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let result = nabled_ml::iterative::gmres_view(&a.as_array(), &b.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// GMRES solve for general `Ax = b` into `output`.
#[pyfunction(name = "gmres_into")]
pub fn gmres_into(
    matrix_a: &Bound<'_, PyAny>,
    matrix_b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<()> {
    match (utils::real_array2(matrix_a, "matrix_a")?, utils::real_array1(matrix_b, "matrix_b")?) {
        (utils::RealReadonlyArray2::F32(a), utils::RealReadonlyArray1::F32(b)) => {
            let default = IterativeConfig::<f32>::default_f32();
            let config = IterativeConfig {
                tolerance: tolerance_f32(tolerance, default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let mut output_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_ml::iterative::gmres_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(a), utils::RealReadonlyArray1::F64(b)) => {
            let default = IterativeConfig::<f64>::default_f64();
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(default.tolerance),
                max_iterations: max_iterations.unwrap_or(default.max_iterations),
            };
            let mut output_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_ml::iterative::gmres_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b", "output"])),
    }
}

/// Conjugate gradient solve for complex Hermitian positive definite Ax = b.
#[pyfunction(name = "conjugate_gradient_complex")]
pub fn conjugate_gradient_complex<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array1(matrix_b, "matrix_b")?,
    ) {
        (utils::NumericReadonlyArray2::C64(a), utils::NumericReadonlyArray1::C64(b)) => {
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(1e-10),
                max_iterations: max_iterations.unwrap_or(1000),
            };
            let result = nabled_ml::iterative::conjugate_gradient_complex_view(
                &a.as_array(),
                &b.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// Conjugate gradient solve for complex Hermitian positive definite `Ax = b` into `output`.
#[pyfunction(name = "conjugate_gradient_complex_into")]
pub fn conjugate_gradient_complex_into(
    matrix_a: &Bound<'_, PyAny>,
    matrix_b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array1(matrix_b, "matrix_b")?,
    ) {
        (utils::NumericReadonlyArray2::C64(a), utils::NumericReadonlyArray1::C64(b)) => {
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(1e-10),
                max_iterations: max_iterations.unwrap_or(1000),
            };
            let mut output_arr = utils::output_array1::<Complex64>(output, "output", "complex128")?;
            nabled_ml::iterative::conjugate_gradient_complex_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix_a", "matrix_b", "output"])),
    }
}

/// GMRES solve for general complex Ax = b.
#[pyfunction(name = "gmres_complex")]
pub fn gmres_complex<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array1(matrix_b, "matrix_b")?,
    ) {
        (utils::NumericReadonlyArray2::C64(a), utils::NumericReadonlyArray1::C64(b)) => {
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(1e-10),
                max_iterations: max_iterations.unwrap_or(1000),
            };
            let result =
                nabled_ml::iterative::gmres_complex_view(&a.as_array(), &b.as_array(), &config)
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix_a", "matrix_b"])),
    }
}

/// GMRES solve for general complex `Ax = b` into `output`.
#[pyfunction(name = "gmres_complex_into")]
pub fn gmres_complex_into(
    matrix_a: &Bound<'_, PyAny>,
    matrix_b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array1(matrix_b, "matrix_b")?,
    ) {
        (utils::NumericReadonlyArray2::C64(a), utils::NumericReadonlyArray1::C64(b)) => {
            let config = IterativeConfig {
                tolerance: tolerance.unwrap_or(1e-10),
                max_iterations: max_iterations.unwrap_or(1000),
            };
            let mut output_arr = utils::output_array1::<Complex64>(output, "output", "complex128")?;
            nabled_ml::iterative::gmres_complex_view_into(
                &a.as_array(),
                &b.as_array(),
                &config,
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix_a", "matrix_b", "output"])),
    }
}

//! Error conversion from nabled errors to Python exceptions.

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Convert any nabled error to a Python `ValueError`.
pub fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyErr::new::<PyValueError, _>(err.to_string())
}

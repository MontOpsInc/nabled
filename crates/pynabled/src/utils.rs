//! Utilities for Python bindings.

use numpy::PyUntypedArrayMethods;
use pyo3::PyResult;

/// Validate NumPy layout compatibility for borrowed-array ingress.
///
/// `pynabled` no longer rejects non-C-contiguous dense NumPy arrays at the Python boundary.
/// Dense kernels should borrow strided views when the Rust API admits them, and wrappers that
/// still materialize owned arrays must do so because of API shape rather than a blanket layout
/// restriction at ingress.
pub fn require_contiguous<'py, A: PyUntypedArrayMethods<'py>>(_array: &A) -> PyResult<()> { Ok(()) }

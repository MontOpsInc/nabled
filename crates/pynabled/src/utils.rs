//! Utilities for Python bindings.

use numpy::PyUntypedArrayMethods;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Ensure the array is C-contiguous for zero-copy access.
///
/// Returns a clear error if the array is not C-contiguous, suggesting
/// `np.ascontiguousarray(a)`.
pub fn require_contiguous<'py, A: PyUntypedArrayMethods<'py>>(array: &A) -> PyResult<()> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "array must be C-contiguous; use np.ascontiguousarray(a) first",
        ));
    }
    Ok(())
}

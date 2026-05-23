//! Signal processing bindings (FFT, autocorrelation) — requires `signal` feature.

use nabled::linalg::signal::correlation::autocorrelation_full;
use nabled::linalg::signal::fft::{bin_to_hz, dominant_frequency, irfft, irfft_into, rfft};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Real FFT returning complex spectrum coefficients.
#[pyfunction]
pub fn rfft_py<'py>(py: Python<'py>, signal: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array1(signal, "signal")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let spectrum = rfft(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, spectrum.bins))
        }
        _ => Err(utils::matching_real_dtype_error(&["signal"])),
    }
}

/// Inverse real FFT.
#[pyfunction]
pub fn irfft_py<'py>(py: Python<'py>, spectrum: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array1(spectrum, "spectrum")? {
        utils::NumericReadonlyArray1::C64(arr) => {
            use nabled::linalg::signal::fft::RfftSpectrum;
            let spec = RfftSpectrum { bins: arr.as_array().to_owned() };
            let reconstructed = irfft(&spec).map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, reconstructed))
        }
        _ => Err(utils::matching_complex_dtype_error(&["spectrum"])),
    }
}

/// Dominant frequency bin index.
#[pyfunction]
pub fn dominant_frequency_py(signal: &Bound<'_, PyAny>) -> PyResult<usize> {
    match utils::real_array1(signal, "signal")? {
        utils::RealReadonlyArray1::F64(arr) => {
            dominant_frequency(&arr.as_array()).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["signal"])),
    }
}

/// Convert FFT bin index to frequency in Hz.
#[pyfunction]
pub fn bin_to_hz_py(bin: usize, n: usize, sample_rate: f64) -> f64 {
    bin_to_hz(bin, n, sample_rate)
}

/// Full autocorrelation sequence.
#[pyfunction]
pub fn autocorrelation_full_py<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(signal, "signal")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let acf = autocorrelation_full(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, acf))
        }
        _ => Err(utils::matching_real_dtype_error(&["signal"])),
    }
}

/// Real FFT into caller-provided complex output buffer.
#[pyfunction]
pub fn rfft_into_py(signal: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::real_array1(signal, "signal")? {
        utils::RealReadonlyArray1::F64(arr) => {
            let spectrum = rfft(&arr.as_array()).map_err(to_py_err)?;
            let mut out =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            if out.as_array_mut().len() != spectrum.bins.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "output length must be n/2 + 1 for real input length n",
                ));
            }
            out.as_array_mut().assign(&spectrum.bins);
            Ok(())
        }
        _ => Err(utils::matching_real_dtype_error(&["signal", "output"])),
    }
}

/// Inverse real FFT into caller-provided output buffer.
#[pyfunction]
pub fn irfft_into_py(spectrum: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array1(spectrum, "spectrum")? {
        utils::NumericReadonlyArray1::C64(arr) => {
            use nabled::linalg::signal::fft::RfftSpectrum;
            let spec = RfftSpectrum { bins: arr.as_array().to_owned() };
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            let mut buffer = out.as_array_mut().to_owned();
            irfft_into(&spec, &mut buffer).map_err(to_py_err)?;
            out.as_array_mut().assign(&buffer);
            Ok(())
        }
        _ => Err(utils::matching_complex_dtype_error(&["spectrum", "output"])),
    }
}

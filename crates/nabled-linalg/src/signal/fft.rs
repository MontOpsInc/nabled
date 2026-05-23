//! Real FFT helpers (requires `signal` feature).

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, ArrayBase, ArrayView1, DataMut, Ix1};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use super::{SignalError, validate_output_len};

/// Real FFT magnitude spectrum (first n/2+1 bins).
pub fn rfft<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<Array1<T>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut buffer: Vec<Complex<f64>> =
        signal.iter().map(|v| Complex::new(v.to_f64().unwrap_or(0.0), 0.0)).collect();
    fft.process(&mut buffer);
    let out_len = signal.len() / 2 + 1;
    let mut output = Array1::<T>::zeros(out_len);
    for (i, slot) in output.iter_mut().enumerate().take(out_len) {
        let mag = buffer[i].norm();
        *slot = T::from_f64(mag).unwrap_or(T::zero());
    }
    Ok(output)
}

/// Inverse real FFT from Hermitian spectrum (minimal stub using full complex IFFT).
pub fn irfft<T: NabledReal>(spectrum: &ArrayView1<'_, T>) -> Result<Array1<T>, SignalError> {
    if spectrum.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = (spectrum.len() - 1) * 2;
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut buffer = vec![Complex::new(0.0, 0.0); n];
    for (i, value) in spectrum.iter().enumerate() {
        buffer[i] = Complex::new(value.to_f64().unwrap_or(0.0), 0.0);
    }
    for i in (spectrum.len())..n {
        buffer[i] = buffer[n - i].conj();
    }
    ifft.process(&mut buffer);
    let scale = 1.0 / n as f64;
    Ok(Array1::from_iter(buffer.iter().map(|c| T::from_f64(c.re * scale).unwrap_or(T::zero()))))
}

/// Real FFT into caller buffer (magnitude bins).
pub fn rfft_into<T, S>(
    signal: &ArrayView1<'_, T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    let spectrum = rfft(signal)?;
    validate_output_len(output, spectrum.len(), "rfft")?;
    output.assign(&spectrum);
    Ok(())
}

/// Power spectrum (squared magnitudes).
#[must_use]
pub fn power_spectrum<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<Array1<T>, SignalError> {
    let mag = rfft(signal)?;
    Ok(mag.mapv(|v| v * v))
}

/// Dominant frequency bin index (excluding DC).
pub fn dominant_frequency<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<usize, SignalError> {
    let power = power_spectrum(signal)?;
    if power.len() <= 1 {
        return Ok(0);
    }
    power
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or(SignalError::EmptyInput)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfft_smoke() {
        let signal = ndarray::arr1(&[1.0_f64, 0.0, -1.0, 0.0]);
        let spectrum = rfft(&signal.view()).unwrap();
        assert_eq!(spectrum.len(), 3);
        assert!(spectrum[1] > 0.0);
    }
}

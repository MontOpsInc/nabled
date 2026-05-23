//! Autocorrelation and cross-correlation at a fixed lag.

use nabled_core::scalar::NabledReal;
#[cfg(feature = "signal")]
use ndarray::Array1;
use ndarray::ArrayView1;
#[cfg(feature = "signal")]
use num_complex::Complex;

#[cfg(feature = "signal")]
use super::fft::{RfftSpectrum, irfft, rfft};
use super::{SignalError, dot_segment};

fn validate_lag(len: usize, lag: usize) -> Result<(), SignalError> {
    if len == 0 {
        return Err(SignalError::EmptyInput);
    }
    if lag >= len {
        return Err(SignalError::InvalidInput(format!(
            "lag {lag} must be less than signal length {len}"
        )));
    }
    Ok(())
}

/// Autocorrelation at fixed lag (unnormalized dot product of overlapping segments).
pub fn autocorrelation_at_lag<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    lag: usize,
) -> Result<T, SignalError> {
    autocorrelation_at_lag_view(signal, lag)
}

/// Autocorrelation at fixed lag from a view.
pub fn autocorrelation_at_lag_view<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    lag: usize,
) -> Result<T, SignalError> {
    validate_lag(signal.len(), lag)?;
    let n = signal.len() - lag;
    Ok(dot_segment(&signal.slice(ndarray::s![..n]), &signal.slice(ndarray::s![lag..lag + n])))
}

/// Cross-correlation at fixed lag between `a` and `b`.
pub fn cross_correlation_at_lag<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
    lag: usize,
) -> Result<T, SignalError> {
    cross_correlation_at_lag_view(a, b, lag)
}

/// Cross-correlation at fixed lag from views.
pub fn cross_correlation_at_lag_view<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
    lag: usize,
) -> Result<T, SignalError> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(SignalError::InvalidInput(
            "signals must have equal length for cross-correlation".to_string(),
        ));
    }
    validate_lag(a.len(), lag)?;
    let n = a.len() - lag;
    Ok(dot_segment(&a.slice(ndarray::s![..n]), &b.slice(ndarray::s![lag..lag + n])))
}

/// Batch autocorrelation at lag for matrix columns (rows = time).
pub fn autocorrelation_at_lag_into<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    lag: usize,
    output: &mut T,
) -> Result<(), SignalError> {
    *output = autocorrelation_at_lag_view(signal, lag)?;
    Ok(())
}

/// Batch cross-correlation at lag into scalar output.
pub fn cross_correlation_at_lag_into<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
    lag: usize,
    output: &mut T,
) -> Result<(), SignalError> {
    *output = cross_correlation_at_lag_view(a, b, lag)?;
    Ok(())
}

/// Full linear autocorrelation for lags `0..n` via FFT convolution (requires `signal` feature).
#[cfg(feature = "signal")]
pub fn autocorrelation_full<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
) -> Result<Array1<T>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = signal.len();
    let padded_len = n * 2;
    let mut padded = Array1::<T>::zeros(padded_len);
    padded.slice_mut(ndarray::s![..n]).assign(signal);
    let spectrum = rfft(&padded.view())?;
    let power = spectrum.power();
    let power_spectrum = RfftSpectrum { bins: power.mapv(|value| Complex::new(value, T::zero())) };
    let circular = irfft(&power_spectrum)?;
    Ok(Array1::from_iter(circular.iter().take(n).copied()))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn autocorr_lag_zero_is_energy() {
        let signal = ndarray::arr1(&[1.0_f64, 2.0, 3.0]);
        let r0 = autocorrelation_at_lag(&signal.view(), 0).unwrap();
        assert_relative_eq!(r0, 14.0, epsilon = 1e-12);
    }

    #[cfg(feature = "signal")]
    #[test]
    fn autocorrelation_full_periodic_peak() {
        let period: u16 = 8;
        let n: u16 = 32;
        let signal: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| (2.0 * std::f64::consts::PI * f64::from(i) / f64::from(period)).sin()),
        );
        let acf = autocorrelation_full(&signal.view()).unwrap();
        assert!(acf[0] >= acf[usize::from(period)]);
        let (peak_lag, _) = acf
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(peak_lag, usize::from(period));
    }
}

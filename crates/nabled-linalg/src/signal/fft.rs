//! Real FFT helpers (requires `signal` feature).

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, ArrayBase, ArrayView1, DataMut, Ix1};
use num_complex::Complex;
use num_traits::FromPrimitive;
use realfft::RealFftPlanner;

use super::window::{WindowKind, apply_window};
use super::{SignalError, validate_output_len};

/// Half-complex spectrum from a real-valued forward FFT (`n/2 + 1` bins).
#[derive(Debug, Clone, PartialEq)]
pub struct RfftSpectrum<T: NabledReal> {
    /// Complex frequency bins (DC through Nyquist when `n` is even).
    pub bins: Array1<Complex<T>>,
    /// Original time-domain length used for the forward transform.
    ///
    /// When zero, inverse transforms assume an even length of `2 * (bins.len() - 1)`.
    pub time_len: usize,
}

impl<T: NabledReal> RfftSpectrum<T> {
    /// Number of complex bins.
    #[must_use]
    pub fn len(&self) -> usize { self.bins.len() }

    /// True when the spectrum contains no bins.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.bins.is_empty() }

    /// Per-bin power `|z|²`.
    #[must_use]
    pub fn power(&self) -> Array1<T> { self.bins.mapv(|c| c.re * c.re + c.im * c.im) }

    /// Per-bin magnitude `|z|`.
    #[must_use]
    pub fn magnitude(&self) -> Array1<T> {
        self.bins.mapv(|c| {
            let mag_sq = c.re * c.re + c.im * c.im;
            T::from_f64(mag_sq.to_f64().unwrap_or(0.0).sqrt()).unwrap_or(T::zero())
        })
    }
}

fn map_realfft_error(err: &realfft::FftError) -> SignalError {
    SignalError::InvalidInput(err.to_string())
}

fn rfft_f64(signal: &[f64]) -> Result<(Vec<Complex<f64>>, usize), SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = signal.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n);
    let mut indata = r2c.make_input_vec();
    indata.copy_from_slice(signal);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut indata, &mut spectrum).map_err(|err| map_realfft_error(&err))?;
    Ok((spectrum, n))
}

fn irfft_f64(spectrum: &[Complex<f64>], n: usize) -> Result<Vec<f64>, SignalError> {
    if spectrum.is_empty() || n == 0 {
        return Err(SignalError::EmptyInput);
    }
    if spectrum.len() != n / 2 + 1 {
        return Err(SignalError::InvalidInput(format!(
            "irfft spectrum length must be {expected}, got {actual}",
            expected = n / 2 + 1,
            actual = spectrum.len()
        )));
    }
    let mut planner = RealFftPlanner::<f64>::new();
    let c2r = planner.plan_fft_inverse(n);
    let mut spec = spectrum.to_vec();
    let mut outdata = c2r.make_output_vec();
    c2r.process(&mut spec, &mut outdata).map_err(|err| map_realfft_error(&err))?;
    let scale = 1.0 / f64::from_usize(n).unwrap_or(f64::from(u32::MAX));
    Ok(outdata.iter().map(|sample| sample * scale).collect())
}

fn complex_from_f64<T: NabledReal>(value: Complex<f64>) -> Complex<T> {
    Complex::new(
        T::from_f64(value.re).unwrap_or(T::zero()),
        T::from_f64(value.im).unwrap_or(T::zero()),
    )
}

fn complex_to_f64<T: NabledReal>(value: Complex<T>) -> Complex<f64> {
    Complex::new(value.re.to_f64().unwrap_or(0.0), value.im.to_f64().unwrap_or(0.0))
}

/// Forward real FFT returning the half-complex spectrum (`n/2 + 1` bins).
///
/// Forward transforms are unnormalized; apply [`irfft`] (which scales by `1/n`) for round-trip
/// reconstruction.
pub fn rfft<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<RfftSpectrum<T>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let signal_f64: Vec<f64> = signal.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let (spectrum, _) = rfft_f64(&signal_f64)?;
    Ok(RfftSpectrum {
        bins: Array1::from_iter(spectrum.into_iter().map(complex_from_f64::<T>)),
        time_len: signal.len(),
    })
}

/// Magnitude spectrum (`|X[k]|`) for each half-complex bin.
pub fn rfft_magnitude<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<Array1<T>, SignalError> {
    Ok(rfft(signal)?.magnitude())
}

/// Inverse real FFT from a half-complex spectrum, scaling the result by `1/n`.
pub fn irfft<T: NabledReal>(spectrum: &RfftSpectrum<T>) -> Result<Array1<T>, SignalError> {
    if spectrum.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = if spectrum.time_len > 0 {
        spectrum.time_len
    } else {
        (spectrum.bins.len().saturating_sub(1)) * 2
    };
    let spectrum_f64: Vec<Complex<f64>> =
        spectrum.bins.iter().copied().map(complex_to_f64).collect();
    let output = irfft_f64(&spectrum_f64, n)?;
    Ok(Array1::from_iter(output.into_iter().map(|v| T::from_f64(v).unwrap_or(T::zero()))))
}

/// Forward real FFT into caller-owned spectrum storage.
pub fn rfft_into<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    output: &mut RfftSpectrum<T>,
) -> Result<(), SignalError> {
    let spectrum = rfft(signal)?;
    validate_output_len(&output.bins, spectrum.len(), "rfft")?;
    output.bins.assign(&spectrum.bins);
    Ok(())
}

/// Inverse real FFT into caller-owned time-domain buffer.
pub fn irfft_into<T, S>(
    spectrum: &RfftSpectrum<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    let time = irfft(spectrum)?;
    validate_output_len(output, time.len(), "irfft")?;
    output.assign(&time);
    Ok(())
}

/// Windowed forward real FFT composing [`super::window`] helpers.
pub fn windowed_rfft<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    window: WindowKind,
    normalize: bool,
) -> Result<RfftSpectrum<T>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let windowed = apply_window(signal, window, normalize)?;
    rfft(&windowed.view())
}

/// Map a half-complex bin index to frequency in hertz.
#[must_use]
pub fn bin_to_hz<T: NabledReal>(bin: usize, n: usize, sample_rate: T) -> T {
    if n == 0 {
        return T::zero();
    }
    T::from_usize(bin).unwrap_or(T::zero()) * sample_rate / T::from_usize(n).unwrap_or(T::one())
}

/// Power spectrum (squared magnitudes of complex bins).
pub fn power_spectrum<T: NabledReal>(signal: &ArrayView1<'_, T>) -> Result<Array1<T>, SignalError> {
    Ok(rfft(signal)?.power())
}

/// Dominant frequency bin index (excluding DC) using complex-bin power.
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
    use approx::assert_relative_eq;
    use rand::RngExt;

    use super::*;

    #[test]
    fn rfft_sine_peak_at_expected_bin() {
        let n: u16 = 64;
        let bin: u16 = 5;
        let sample_rate = 1_000.0_f64;
        let freq = bin_to_hz(usize::from(bin), usize::from(n), sample_rate);
        let n_f = f64::from(n);
        let signal: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            let t = f64::from(i) / sample_rate;
            (2.0 * std::f64::consts::PI * freq * t).sin()
        }));
        let peak = dominant_frequency(&signal.view()).unwrap();
        assert_eq!(peak, usize::from(bin));
        assert_relative_eq!(
            bin_to_hz(peak, usize::from(n), sample_rate),
            freq,
            epsilon = sample_rate / n_f
        );
    }

    #[test]
    fn irfft_rfft_round_trip_even_length() {
        let mut rng = rand::rng();
        let n = 128;
        let signal: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.random_range(-1.0..1.0)));
        let spectrum = rfft(&signal.view()).unwrap();
        let reconstructed = irfft(&spectrum).unwrap();
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }

    #[test]
    fn rfft_magnitude_matches_complex_bins() {
        let signal = ndarray::arr1(&[1.0_f64, 0.0, -1.0, 0.0]);
        let spectrum = rfft(&signal.view()).unwrap();
        let magnitude = rfft_magnitude(&signal.view()).unwrap();
        assert_eq!(magnitude.len(), spectrum.len());
        for (mag, bin) in magnitude.iter().zip(spectrum.bins.iter()) {
            let expected = (bin.re * bin.re + bin.im * bin.im).sqrt();
            assert_relative_eq!(mag, &expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn windowed_rfft_smoke() {
        let signal = ndarray::arr1(&[1.0_f64, 0.5, -0.5, -1.0]);
        let spectrum = windowed_rfft(&signal.view(), WindowKind::Hann, true).unwrap();
        assert_eq!(spectrum.len(), 3);
    }

    #[test]
    fn rfft_empty_input_errors() {
        let empty = ndarray::arr1::<f64>(&[]);
        assert_eq!(rfft(&empty.view()), Err(SignalError::EmptyInput));
        assert_eq!(windowed_rfft(&empty.view(), WindowKind::Hann, false), Err(SignalError::EmptyInput));
    }

    #[test]
    fn irfft_empty_spectrum_errors() {
        let empty = RfftSpectrum::<f64> { bins: ndarray::arr1(&[]), time_len: 0 };
        assert_eq!(irfft(&empty), Err(SignalError::EmptyInput));
    }

    #[test]
    fn irfft_rfft_round_trip_odd_length() {
        let mut rng = rand::rng();
        let n = 127;
        let signal: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.random_range(-1.0..1.0)));
        let spectrum = rfft(&signal.view()).unwrap();
        assert_eq!(spectrum.len(), n / 2 + 1);
        let reconstructed = irfft(&spectrum).unwrap();
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }

    #[test]
    fn irfft_rfft_round_trip_fixed_odd_vector() {
        let signal = ndarray::arr1(&[0.2_f64, -0.5, 1.0, 0.0, -1.0, 0.3, 0.7]);
        let spectrum = rfft(&signal.view()).unwrap();
        let reconstructed = irfft(&spectrum).unwrap();
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(orig, recon, epsilon = 1e-10);
        }
    }
}

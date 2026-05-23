//! Window functions for spectral analysis.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, ArrayBase, ArrayView1, DataMut, Ix1};

use super::{SignalError, validate_output_len};

fn scalar_pi<T: NabledReal>() -> T {
    T::from_f64(std::f64::consts::PI).unwrap_or(T::zero())
}

fn scalar_two<T: NabledReal>() -> T {
    T::from_f64(2.0).unwrap_or(T::one() + T::one())
}

/// Built-in window families for spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// Hann (raised cosine) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}

/// Generate a window of the given kind into `output`.
pub fn window_into<T, S>(
    kind: WindowKind,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    match kind {
        WindowKind::Hann => hann_into(output.len(), output),
        WindowKind::Hamming => hamming_into(output.len(), output),
        WindowKind::Blackman => blackman_into(output.len(), output),
    }
}

/// Element-wise multiply `signal` by a window, optionally normalizing the window to unit coherent gain.
pub fn apply_window<T: NabledReal>(
    signal: &ArrayView1<'_, T>,
    kind: WindowKind,
    normalize: bool,
) -> Result<Array1<T>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::EmptyInput);
    }
    let n = signal.len();
    let mut window = Array1::<T>::zeros(n);
    window_into(kind, &mut window)?;
    if normalize {
        let sum = window.iter().copied().fold(T::zero(), |acc, v| acc + v);
        if sum > T::epsilon() {
            let scale = T::from_usize(n).unwrap_or(T::one()) / sum;
            window.mapv_inplace(|value| value * scale);
        }
    }
    Ok(signal * window)
}

/// Hann window of length `n`.
#[must_use]
pub fn hann<T: NabledReal>(n: usize) -> Array1<T> {
    let mut window = Array1::<T>::zeros(n);
    drop(hann_into(n, &mut window));
    window
}

/// Hann window into caller buffer.
pub fn hann_into<T, S>(n: usize, output: &mut ArrayBase<S, Ix1>) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    validate_output_len(output, n, "hann")?;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        output[0] = T::one();
        return Ok(());
    }
    let denom = T::from_usize(n - 1).unwrap_or(T::one());
    for i in 0..n {
        let phase =
            scalar_two::<T>() * scalar_pi::<T>() * T::from_usize(i).unwrap_or(T::zero()) / denom;
        output[i] = T::from_f64(0.5).unwrap_or(T::zero()) * (T::one() - phase.cos());
    }
    Ok(())
}

/// Hamming window of length `n`.
#[must_use]
pub fn hamming<T: NabledReal>(n: usize) -> Array1<T> {
    let mut window = Array1::<T>::zeros(n);
    drop(hamming_into(n, &mut window));
    window
}

/// Hamming window into caller buffer.
pub fn hamming_into<T, S>(n: usize, output: &mut ArrayBase<S, Ix1>) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    validate_output_len(output, n, "hamming")?;
    if n == 0 {
        return Ok(());
    }
    let a0 = T::from_f64(0.54).unwrap_or(T::zero());
    let a1 = T::from_f64(0.46).unwrap_or(T::zero());
    let denom = T::from_usize(n.saturating_sub(1).max(1)).unwrap_or(T::one());
    for i in 0..n {
        let phase =
            scalar_two::<T>() * scalar_pi::<T>() * T::from_usize(i).unwrap_or(T::zero()) / denom;
        output[i] = a0 - a1 * phase.cos();
    }
    Ok(())
}

/// Blackman window of length `n`.
#[must_use]
pub fn blackman<T: NabledReal>(n: usize) -> Array1<T> {
    let mut window = Array1::<T>::zeros(n);
    drop(blackman_into(n, &mut window));
    window
}

/// Blackman window into caller buffer.
pub fn blackman_into<T, S>(n: usize, output: &mut ArrayBase<S, Ix1>) -> Result<(), SignalError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    validate_output_len(output, n, "blackman")?;
    if n == 0 {
        return Ok(());
    }
    let a0 = T::from_f64(0.42).unwrap_or(T::zero());
    let a1 = T::from_f64(0.5).unwrap_or(T::zero());
    let a2 = T::from_f64(0.08).unwrap_or(T::zero());
    let denom = T::from_usize(n.saturating_sub(1).max(1)).unwrap_or(T::one());
    for i in 0..n {
        let phase =
            scalar_two::<T>() * scalar_pi::<T>() * T::from_usize(i).unwrap_or(T::zero()) / denom;
        output[i] = a0 - a1 * phase.cos() + a2 * (scalar_two::<T>() * phase).cos();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn hann_endpoints_are_zero_for_n_gt_1() {
        let w = hann::<f64>(8);
        assert_relative_eq!(w[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(w[7], 0.0, epsilon = 1e-12);
        assert!(w[4] > 0.9);
    }
}

//! Time- and frequency-domain signal processing.

#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use nabled_core::scalar::NabledReal;
use ndarray::{ArrayBase, ArrayView1, DataMut, Ix1};
use thiserror::Error;

pub mod correlation;
pub mod window;

#[cfg(feature = "signal")]
pub mod fft;

/// Errors for signal processing operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SignalError {
    #[error("input cannot be empty")]
    EmptyInput,
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("numerical instability detected")]
    NumericalInstability,
}

pub(crate) fn validate_output_len<T, S>(
    output: &ArrayBase<S, Ix1>,
    expected: usize,
    name: &str,
) -> Result<(), SignalError>
where
    S: DataMut<Elem = T>,
{
    if output.len() != expected {
        return Err(SignalError::InvalidInput(format!("{name} output length must be {expected}")));
    }
    Ok(())
}

pub(crate) fn dot_segment<T: NabledReal>(a: &ArrayView1<'_, T>, b: &ArrayView1<'_, T>) -> T {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).fold(T::zero(), |acc, v| acc + v)
}

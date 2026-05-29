//! Statistical utilities over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, DataMut, Ix1, Ix2};
use num_complex::Complex64;

/// Error type for matrix statistics.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix needs at least two rows.
    InsufficientSamples,
    /// Input or output shapes are incompatible.
    InvalidInput(String),
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            StatsError::InsufficientSamples => {
                write!(f, "At least two observations are required")
            }
            StatsError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            StatsError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for StatsError {}

fn usize_to_scalar<T: NabledReal>(value: usize) -> T {
    T::from_usize(value).unwrap_or(T::max_value())
}

fn complex_is_finite(value: Complex64) -> bool { value.re.is_finite() && value.im.is_finite() }

fn validate_vector_output_len<T, S>(
    output: &ArrayBase<S, Ix1>,
    expected_len: usize,
    name: &str,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = T>,
{
    if output.len() != expected_len {
        return Err(StatsError::InvalidInput(format!(
            "{name} output length must match expected length {expected_len}",
        )));
    }
    Ok(())
}

fn validate_matrix_output_shape<T, S>(
    output: &ArrayBase<S, Ix2>,
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = T>,
{
    if output.nrows() != expected_rows || output.ncols() != expected_cols {
        return Err(StatsError::InvalidInput(format!(
            "{name} output shape must be ({expected_rows}, {expected_cols})",
        )));
    }
    Ok(())
}

fn column_means_into_impl<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    validate_vector_output_len(output, matrix.ncols(), "column_means")?;

    if matrix.nrows() == 0 {
        output.fill(T::zero());
        return Ok(());
    }

    let denom = usize_to_scalar::<T>(matrix.nrows());
    for col in 0..matrix.ncols() {
        let mut sum = T::zero();
        for row in 0..matrix.nrows() {
            sum += matrix[[row, col]];
        }
        output[col] = sum / denom;
    }

    Ok(())
}

fn column_means_impl<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array1<T> {
    matrix.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(matrix.ncols()))
}

/// Compute column means.
#[must_use]
pub fn column_means<T: NabledReal>(matrix: &Array2<T>) -> Array1<T> {
    column_means_impl(&matrix.view())
}

/// Compute column means from a matrix view.
#[must_use]
pub fn column_means_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array1<T> {
    column_means_impl(matrix)
}

/// Compute column means into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the column count.
pub fn column_means_into<T, S>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    column_means_into_impl(&matrix.view(), output)
}

/// Compute column means from a matrix view into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the column count.
pub fn column_means_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    column_means_into_impl(matrix, output)
}

fn center_columns_impl<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array2<T> {
    let means = column_means_impl(matrix);
    let mut centered = Array2::<T>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - means[col];
        }
    }
    centered
}

/// Center columns by subtracting their means.
#[must_use]
pub fn center_columns<T: NabledReal>(matrix: &Array2<T>) -> Array2<T> {
    center_columns_impl(&matrix.view())
}

/// Center columns by subtracting their means from a matrix view.
#[must_use]
pub fn center_columns_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array2<T> {
    center_columns_impl(matrix)
}

fn center_columns_into_impl<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    validate_matrix_output_shape(output, matrix.nrows(), matrix.ncols(), "center_columns")?;

    let mut means = Array1::<T>::zeros(matrix.ncols());
    column_means_into_impl(matrix, &mut means)?;

    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            output[[row, col]] = matrix[[row, col]] - means[col];
        }
    }

    Ok(())
}

/// Center columns by subtracting their means into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the input shape.
pub fn center_columns_into<T, S>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    center_columns_into_impl(&matrix.view(), output)
}

/// Center columns by subtracting their means from a matrix view into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the input shape.
pub fn center_columns_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    center_columns_into_impl(matrix, output)
}

fn covariance_matrix_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let centered = center_columns_impl(matrix);
    let covariance: Array2<T> =
        centered.t().dot(&centered) / usize_to_scalar::<T>(matrix.nrows() - 1);

    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(covariance)
}

/// Compute sample covariance matrix.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, StatsError> {
    covariance_matrix_impl(&matrix.view())
}

/// Compute sample covariance matrix from a matrix view.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    covariance_matrix_impl(matrix)
}

fn covariance_matrix_into_impl<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }
    validate_matrix_output_shape(output, matrix.ncols(), matrix.ncols(), "covariance_matrix")?;

    let mut means = Array1::<T>::zeros(matrix.ncols());
    column_means_into_impl(matrix, &mut means)?;
    let denom = usize_to_scalar::<T>(matrix.nrows() - 1);

    for i in 0..matrix.ncols() {
        for j in i..matrix.ncols() {
            let mut sum = T::zero();
            for row in 0..matrix.nrows() {
                let left = matrix[[row, i]] - means[i];
                let right = matrix[[row, j]] - means[j];
                sum += left * right;
            }
            let value = sum / denom;
            output[[i, j]] = value;
            if i != j {
                output[[j, i]] = value;
            }
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(())
}

/// Compute sample covariance matrix into caller-provided output.
///
/// # Errors
/// Returns an error for empty input, fewer than two samples, or incompatible output shape.
pub fn covariance_matrix_into<T, S>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    covariance_matrix_into_impl(&matrix.view(), output)
}

/// Compute sample covariance matrix from a matrix view into caller-provided output.
///
/// # Errors
/// Returns an error for empty input, fewer than two samples, or incompatible output shape.
pub fn covariance_matrix_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    covariance_matrix_into_impl(matrix, output)
}

fn correlation_matrix_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    let covariance = covariance_matrix_impl(matrix)?;
    let n = covariance.nrows();
    let mut correlation = Array2::<T>::zeros((n, n));

    for i in 0..n {
        let sigma_i = covariance[[i, i]].sqrt();
        for j in 0..n {
            let sigma_j = covariance[[j, j]].sqrt();
            let denom = (sigma_i * sigma_j).max(T::epsilon());
            correlation[[i, j]] = covariance[[i, j]] / denom;
        }
    }

    Ok(correlation)
}

/// Compute correlation matrix.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, StatsError> {
    correlation_matrix_impl(&matrix.view())
}

/// Compute correlation matrix from a matrix view.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    correlation_matrix_impl(matrix)
}

fn correlation_matrix_into_impl<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    covariance_matrix_into_impl(matrix, output)?;
    let mut sigmas = Array1::<T>::zeros(output.nrows());
    for i in 0..output.nrows() {
        sigmas[i] = output[[i, i]].sqrt();
    }

    for i in 0..output.nrows() {
        let sigma_i = sigmas[i];
        for j in 0..output.ncols() {
            let sigma_j = sigmas[j];
            let denom = (sigma_i * sigma_j).max(T::epsilon());
            output[[i, j]] /= denom;
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(())
}

/// Compute correlation matrix into caller-provided output.
///
/// # Errors
/// Returns an error if covariance computation fails or `output` shape is incompatible.
pub fn correlation_matrix_into<T, S>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    correlation_matrix_into_impl(&matrix.view(), output)
}

/// Compute correlation matrix from a matrix view into caller-provided output.
///
/// # Errors
/// Returns an error if covariance computation fails or `output` shape is incompatible.
pub fn correlation_matrix_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    correlation_matrix_into_impl(matrix, output)
}

fn column_means_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Array1<Complex64> {
    if matrix.nrows() == 0 {
        return Array1::zeros(matrix.ncols());
    }

    let mut means = Array1::<Complex64>::zeros(matrix.ncols());
    for col in 0..matrix.ncols() {
        let mut sum = Complex64::new(0.0, 0.0);
        for row in 0..matrix.nrows() {
            sum += matrix[[row, col]];
        }
        means[col] = sum / usize_to_scalar::<f64>(matrix.nrows());
    }
    means
}

/// Compute complex column means.
#[must_use]
pub fn column_means_complex(matrix: &Array2<Complex64>) -> Array1<Complex64> {
    column_means_complex_impl(&matrix.view())
}

/// Compute complex column means from a matrix view.
#[must_use]
pub fn column_means_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Array1<Complex64> {
    column_means_complex_impl(matrix)
}

fn column_means_complex_into_impl<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    validate_vector_output_len(output, matrix.ncols(), "column_means_complex")?;

    if matrix.nrows() == 0 {
        output.fill(Complex64::new(0.0, 0.0));
        return Ok(());
    }

    let denom = usize_to_scalar::<f64>(matrix.nrows());
    for col in 0..matrix.ncols() {
        let mut sum = Complex64::new(0.0, 0.0);
        for row in 0..matrix.nrows() {
            sum += matrix[[row, col]];
        }
        output[col] = sum / denom;
    }

    Ok(())
}

/// Compute complex column means into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the column count.
pub fn column_means_complex_into<S>(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    column_means_complex_into_impl(&matrix.view(), output)
}

/// Compute complex column means from a matrix view into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the column count.
pub fn column_means_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    column_means_complex_into_impl(matrix, output)
}

fn center_columns_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Array2<Complex64> {
    let means = column_means_complex_impl(matrix);
    let mut centered = Array2::<Complex64>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - means[col];
        }
    }
    centered
}

/// Center complex columns by subtracting their means.
#[must_use]
pub fn center_columns_complex(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    center_columns_complex_impl(&matrix.view())
}

/// Center complex columns by subtracting their means from a matrix view.
#[must_use]
pub fn center_columns_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Array2<Complex64> {
    center_columns_complex_impl(matrix)
}

fn center_columns_complex_into_impl<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    validate_matrix_output_shape(output, matrix.nrows(), matrix.ncols(), "center_columns_complex")?;

    let mut means = Array1::<Complex64>::zeros(matrix.ncols());
    column_means_complex_into_impl(matrix, &mut means)?;

    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            output[[row, col]] = matrix[[row, col]] - means[col];
        }
    }

    Ok(())
}

/// Center complex columns by subtracting their means into caller-provided output.
///
/// # Errors
/// Returns an error if `output` does not match the input shape.
pub fn center_columns_complex_into<S>(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    center_columns_complex_into_impl(&matrix.view(), output)
}

/// Center complex columns by subtracting their means from a matrix view into caller-provided
/// output.
///
/// # Errors
/// Returns an error if `output` does not match the input shape.
pub fn center_columns_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    center_columns_complex_into_impl(matrix, output)
}

fn covariance_matrix_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let centered = center_columns_complex_impl(matrix);
    let conjugate_transpose = centered.t().mapv(|value| value.conj());
    let covariance: Array2<Complex64> =
        conjugate_transpose.dot(&centered) / usize_to_scalar::<f64>(matrix.nrows() - 1);

    if covariance.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(covariance)
}

/// Compute sample covariance matrix for complex observations.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    covariance_matrix_complex_impl(&matrix.view())
}

/// Compute sample covariance matrix for complex observations from a matrix view.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    covariance_matrix_complex_impl(matrix)
}

fn covariance_matrix_complex_into_impl<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }
    validate_matrix_output_shape(
        output,
        matrix.ncols(),
        matrix.ncols(),
        "covariance_matrix_complex",
    )?;

    let mut means = Array1::<Complex64>::zeros(matrix.ncols());
    column_means_complex_into_impl(matrix, &mut means)?;
    let denom = usize_to_scalar::<f64>(matrix.nrows() - 1);

    for i in 0..matrix.ncols() {
        for j in i..matrix.ncols() {
            let mut sum = Complex64::new(0.0, 0.0);
            for row in 0..matrix.nrows() {
                let left = matrix[[row, i]] - means[i];
                let right = matrix[[row, j]] - means[j];
                sum += left.conj() * right;
            }
            let value = sum / denom;
            output[[i, j]] = value;
            if i != j {
                output[[j, i]] = value.conj();
            }
        }
    }

    if output.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(())
}

/// Compute sample covariance matrix for complex observations into caller-provided output.
///
/// # Errors
/// Returns an error for empty input, fewer than two samples, or incompatible output shape.
pub fn covariance_matrix_complex_into<S>(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    covariance_matrix_complex_into_impl(&matrix.view(), output)
}

/// Compute sample covariance matrix for complex observations from a matrix view into
/// caller-provided output.
///
/// # Errors
/// Returns an error for empty input, fewer than two samples, or incompatible output shape.
pub fn covariance_matrix_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    covariance_matrix_complex_into_impl(matrix, output)
}

fn correlation_matrix_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    let covariance = covariance_matrix_complex_impl(matrix)?;
    let n = covariance.nrows();
    let mut correlation = Array2::<Complex64>::zeros((n, n));

    for i in 0..n {
        let sigma_i = covariance[[i, i]].re.max(0.0).sqrt();
        for j in 0..n {
            let sigma_j = covariance[[j, j]].re.max(0.0).sqrt();
            let denom = (sigma_i * sigma_j).max(f64::EPSILON);
            correlation[[i, j]] = covariance[[i, j]] / denom;
        }
    }

    if correlation.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(correlation)
}

/// Compute correlation matrix for complex observations.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    correlation_matrix_complex_impl(&matrix.view())
}

/// Compute correlation matrix for complex observations from a matrix view.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    correlation_matrix_complex_impl(matrix)
}

fn correlation_matrix_complex_into_impl<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    covariance_matrix_complex_into_impl(matrix, output)?;
    let mut sigmas = Array1::<f64>::zeros(output.nrows());
    for i in 0..output.nrows() {
        sigmas[i] = output[[i, i]].re.max(0.0).sqrt();
    }

    for i in 0..output.nrows() {
        let sigma_i = sigmas[i];
        for j in 0..output.ncols() {
            let sigma_j = sigmas[j];
            let denom = (sigma_i * sigma_j).max(f64::EPSILON);
            output[[i, j]] /= denom;
        }
    }

    if output.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(())
}

/// Compute correlation matrix for complex observations into caller-provided output.
///
/// # Errors
/// Returns an error if covariance computation fails or `output` shape is incompatible.
pub fn correlation_matrix_complex_into<S>(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    correlation_matrix_complex_into_impl(&matrix.view(), output)
}

/// Compute correlation matrix for complex observations from a matrix view into caller-provided
/// output.
///
/// # Errors
/// Returns an error if covariance computation fails or `output` shape is incompatible.
pub fn correlation_matrix_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), StatsError>
where
    S: DataMut<Elem = Complex64>,
{
    correlation_matrix_complex_into_impl(matrix, output)
}

// --- Physical AI streaming / rolling extensions ---

pub mod online {
    #![allow(clippy::missing_errors_doc)]
    use nabled_core::scalar::NabledReal;

    /// Online mean accumulator (Welford-style count/mean).
    #[derive(Debug, Clone, PartialEq)]
    pub struct OnlineMean<T> {
        count: usize,
        mean:  T,
    }

    impl<T: NabledReal> Default for OnlineMean<T> {
        fn default() -> Self { Self { count: 0, mean: T::zero() } }
    }

    impl<T: NabledReal> OnlineMean<T> {
        pub fn push(&mut self, value: T) {
            self.count += 1;
            let n = T::from_usize(self.count).unwrap_or(T::one());
            self.mean += (value - self.mean) / n;
        }

        #[must_use]
        pub fn mean(&self) -> T { self.mean }

        pub fn reset(&mut self) {
            self.count = 0;
            self.mean = T::zero();
        }
    }

    /// Online variance accumulator (Welford).
    #[derive(Debug, Clone, PartialEq)]
    pub struct OnlineVariance<T> {
        count: usize,
        mean:  T,
        m2:    T,
    }

    impl<T: NabledReal> Default for OnlineVariance<T> {
        fn default() -> Self { Self { count: 0, mean: T::zero(), m2: T::zero() } }
    }

    impl<T: NabledReal> OnlineVariance<T> {
        pub fn push(&mut self, value: T) {
            self.count += 1;
            let n = T::from_usize(self.count).unwrap_or(T::one());
            let delta = value - self.mean;
            self.mean += delta / n;
            let delta2 = value - self.mean;
            self.m2 += delta * delta2;
        }

        #[must_use]
        pub fn mean(&self) -> T { self.mean }

        #[must_use]
        pub fn variance(&self) -> T {
            if self.count < 2 {
                return T::zero();
            }
            self.m2 / T::from_usize(self.count - 1).unwrap_or(T::one())
        }

        pub fn reset(&mut self) {
            self.count = 0;
            self.mean = T::zero();
            self.m2 = T::zero();
        }
    }
}

pub mod ewma {
    #![allow(clippy::missing_errors_doc)]
    use nabled_core::scalar::NabledReal;
    use ndarray::{Array1, ArrayBase, ArrayView1, DataMut, Ix1};

    use super::StatsError;

    /// Incremental EWMA state.
    #[derive(Debug, Clone, PartialEq)]
    pub struct EwmaState<T> {
        alpha: T,
        value: Option<T>,
    }

    impl<T: NabledReal> EwmaState<T> {
        pub fn new(alpha: T) -> Self { Self { alpha, value: None } }

        pub fn push(&mut self, sample: T) -> T {
            if let Some(prev) = self.value {
                let next = self.alpha * sample + (T::one() - self.alpha) * prev;
                self.value = Some(next);
                next
            } else {
                self.value = Some(sample);
                sample
            }
        }
    }

    #[must_use]
    pub fn ewma<T: NabledReal>(signal: &Array1<T>, alpha: T) -> Array1<T> {
        ewma_view(&signal.view(), alpha)
    }

    #[must_use]
    pub fn ewma_view<T: NabledReal>(signal: &ArrayView1<'_, T>, alpha: T) -> Array1<T> {
        let mut out = Array1::<T>::zeros(signal.len());
        drop(ewma_into(signal, alpha, &mut out));
        out
    }

    pub fn ewma_into<T, S>(
        signal: &ArrayView1<'_, T>,
        alpha: T,
        output: &mut ArrayBase<S, Ix1>,
    ) -> Result<(), StatsError>
    where
        T: NabledReal,
        S: DataMut<Elem = T>,
    {
        if output.len() != signal.len() {
            return Err(StatsError::InvalidInput("ewma output length mismatch".to_string()));
        }
        let mut state = EwmaState::new(alpha);
        for (i, &sample) in signal.iter().enumerate() {
            output[i] = state.push(sample);
        }
        Ok(())
    }
}

pub mod rolling {
    #![allow(clippy::missing_errors_doc)]
    use nabled_core::scalar::NabledReal;
    use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, DataMut, Ix1, Ix2};

    use super::StatsError;

    #[must_use]
    pub fn rolling_mean<T: NabledReal>(signal: &ArrayView1<'_, T>, window: usize) -> Array1<T> {
        let mut out = Array1::<T>::zeros(signal.len());
        drop(rolling_mean_into(signal, window, &mut out));
        out
    }

    pub fn rolling_mean_view<T: NabledReal>(
        signal: &ArrayView1<'_, T>,
        window: usize,
    ) -> Array1<T> {
        rolling_mean(signal, window)
    }

    pub fn rolling_mean_into<T, S>(
        signal: &ArrayView1<'_, T>,
        window: usize,
        output: &mut ArrayBase<S, Ix1>,
    ) -> Result<(), StatsError>
    where
        T: NabledReal,
        S: DataMut<Elem = T>,
    {
        if window == 0 {
            return Err(StatsError::InvalidInput("window must be positive".to_string()));
        }
        if output.len() != signal.len() {
            return Err(StatsError::InvalidInput(
                "rolling_mean output length mismatch".to_string(),
            ));
        }
        for i in 0..signal.len() {
            let start = i.saturating_sub(window - 1);
            let slice = signal.slice(ndarray::s![start..=i]);
            let sum = slice.iter().fold(T::zero(), |acc, v| acc + *v);
            let count = T::from_usize(slice.len()).unwrap_or(T::one());
            output[i] = sum / count;
        }
        Ok(())
    }

    #[must_use]
    pub fn rolling_variance<T: NabledReal>(signal: &ArrayView1<'_, T>, window: usize) -> Array1<T> {
        let mut out = Array1::<T>::zeros(signal.len());
        drop(rolling_variance_into(signal, window, &mut out));
        out
    }

    pub fn rolling_variance_into<T, S>(
        signal: &ArrayView1<'_, T>,
        window: usize,
        output: &mut ArrayBase<S, Ix1>,
    ) -> Result<(), StatsError>
    where
        T: NabledReal,
        S: DataMut<Elem = T>,
    {
        if window == 0 {
            return Err(StatsError::InvalidInput("window must be positive".to_string()));
        }
        if output.len() != signal.len() {
            return Err(StatsError::InvalidInput(
                "rolling_variance output length mismatch".to_string(),
            ));
        }
        for i in 0..signal.len() {
            let start = i.saturating_sub(window - 1);
            let slice = signal.slice(ndarray::s![start..=i]);
            let mean = slice.iter().fold(T::zero(), |acc, v| acc + *v)
                / T::from_usize(slice.len()).unwrap_or(T::one());
            let var = slice
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .fold(T::zero(), |acc, v| acc + v)
                / T::from_usize(slice.len()).unwrap_or(T::one());
            output[i] = var;
        }
        Ok(())
    }

    #[must_use]
    pub fn rolling_covariance<T: NabledReal>(
        matrix: &ArrayView2<'_, T>,
        window: usize,
    ) -> Array2<T> {
        let mut out = Array2::<T>::zeros((matrix.nrows(), matrix.ncols() * matrix.ncols()));
        drop(rolling_covariance_into(matrix, window, &mut out));
        out
    }

    pub fn rolling_covariance_view<T: NabledReal>(
        matrix: &ArrayView2<'_, T>,
        window: usize,
    ) -> Array2<T> {
        rolling_covariance(matrix, window)
    }

    pub fn rolling_covariance_into<T, S>(
        matrix: &ArrayView2<'_, T>,
        window: usize,
        output: &mut ArrayBase<S, Ix2>,
    ) -> Result<(), StatsError>
    where
        T: NabledReal,
        S: DataMut<Elem = T>,
    {
        let cols = matrix.ncols();
        if output.nrows() != matrix.nrows() || output.ncols() != cols * cols {
            return Err(StatsError::InvalidInput(
                "rolling_covariance output shape mismatch".to_string(),
            ));
        }
        if window == 0 {
            return Err(StatsError::InvalidInput("window must be positive".to_string()));
        }
        for row in 0..matrix.nrows() {
            let start = row.saturating_sub(window - 1);
            let block = matrix.slice(ndarray::s![start..=row, ..]);
            let cov = if block.nrows() >= 2 {
                super::covariance_matrix_view(&block)?
            } else {
                Array2::<T>::zeros((cols, cols))
            };
            for i in 0..cols {
                for j in 0..cols {
                    output[[row, i * cols + j]] = cov[[i, j]];
                }
            }
        }
        Ok(())
    }
}

pub mod lag {
    #![allow(clippy::missing_errors_doc)]
    use nabled_core::scalar::NabledReal;
    use ndarray::{Array2, ArrayView2};

    use super::StatsError;

    /// Zero-copy lag view: rows `[0..n-lag)` of `matrix` aligned with rows `[lag..n)`.
    pub fn lag_view<'a, T>(
        matrix: &'a ArrayView2<'_, T>,
        lag: usize,
    ) -> Result<ArrayView2<'a, T>, StatsError> {
        if lag >= matrix.nrows() {
            return Err(StatsError::InvalidInput(format!(
                "lag {lag} must be less than row count {}",
                matrix.nrows()
            )));
        }
        Ok(matrix.slice(ndarray::s![..matrix.nrows() - lag, ..]))
    }

    /// Shift columns down by `lag` rows, filling top rows with zero.
    pub fn shift_columns_into<T: NabledReal>(
        matrix: &ArrayView2<'_, T>,
        lag: usize,
        output: &mut Array2<T>,
    ) -> Result<(), StatsError> {
        if output.dim() != matrix.dim() {
            return Err(StatsError::InvalidInput(
                "shift_columns_into output shape mismatch".to_string(),
            ));
        }
        output.fill(T::zero());
        if lag >= matrix.nrows() {
            return Ok(());
        }
        let rows = matrix.nrows() - lag;
        output.slice_mut(ndarray::s![lag.., ..]).assign(&matrix.slice(ndarray::s![..rows, ..]));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn covariance_and_correlation_are_well_formed() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let covariance = covariance_matrix(&matrix).unwrap();
        let correlation = correlation_matrix(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }

    #[test]
    fn stats_rejects_empty_and_insufficient_inputs() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(covariance_matrix(&empty), Err(StatsError::EmptyMatrix)));

        let one_row = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        assert!(matches!(covariance_matrix(&one_row), Err(StatsError::InsufficientSamples)));
    }

    #[test]
    fn center_columns_zeroes_means() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let centered = center_columns(&matrix);
        let means = column_means(&centered);
        assert!(means.iter().all(|value| num_traits::Float::abs(*value) < 1e-12));
    }

    #[test]
    fn column_means_handles_empty_input() {
        let matrix = Array2::<f64>::zeros((0, 3));
        let means = column_means(&matrix);
        assert_eq!(means.len(), 3);
        assert!(means.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn covariance_reports_numerical_instability() {
        let matrix = Array2::from_shape_vec((2, 2), vec![f64::MAX, 0.0, -f64::MAX, 0.0]).unwrap();
        let result = covariance_matrix(&matrix);
        assert!(matches!(result, Err(StatsError::NumericalInstability)));
    }

    #[test]
    fn correlation_handles_zero_variance_column() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 10.0, 1.0, 20.0, 1.0, 30.0]).unwrap();
        let correlation = correlation_matrix(&matrix).unwrap();
        assert!(correlation[[0, 0]].is_finite());
        assert!(correlation[[0, 1]].is_finite());
        assert!(correlation[[1, 0]].is_finite());
        assert!(correlation[[1, 1]].is_finite());
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let means_owned = column_means(&matrix);
        let means_view = column_means_view(&matrix.view());
        let centered_owned = center_columns(&matrix);
        let centered_view = center_columns_view(&matrix.view());
        let covariance_owned = covariance_matrix(&matrix).unwrap();
        let covariance_view = covariance_matrix_view(&matrix.view()).unwrap();
        let correlation_owned = correlation_matrix(&matrix).unwrap();
        let correlation_view = correlation_matrix_view(&matrix.view()).unwrap();

        for i in 0..means_owned.len() {
            assert!((means_owned[i] - means_view[i]).abs() < 1e-12);
        }
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((centered_owned[[i, j]] - centered_view[[i, j]]).abs() < 1e-12);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((covariance_owned[[i, j]] - covariance_view[[i, j]]).abs() < 1e-12);
                assert!((correlation_owned[[i, j]] - correlation_view[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn complex_covariance_and_correlation_are_well_formed() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(3.0, 0.2),
            Complex64::new(1.0, -0.3),
            Complex64::new(4.0, 0.7),
            Complex64::new(0.0, 0.0),
        ])
        .unwrap();

        let covariance = covariance_matrix_complex(&matrix).unwrap();
        let correlation = correlation_matrix_complex(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }

    #[test]
    fn complex_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(3.0, -2.0),
            Complex64::new(4.0, 1.0),
        ])
        .unwrap();

        let means_owned = column_means_complex(&matrix);
        let means_view = column_means_complex_view(&matrix.view());
        let centered_owned = center_columns_complex(&matrix);
        let centered_view = center_columns_complex_view(&matrix.view());
        let covariance_owned = covariance_matrix_complex(&matrix).unwrap();
        let covariance_view = covariance_matrix_complex_view(&matrix.view()).unwrap();
        let correlation_owned = correlation_matrix_complex(&matrix).unwrap();
        let correlation_view = correlation_matrix_complex_view(&matrix.view()).unwrap();

        for i in 0..means_owned.len() {
            assert!((means_owned[i] - means_view[i]).norm() < 1e-12);
        }
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((centered_owned[[i, j]] - centered_view[[i, j]]).norm() < 1e-12);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((covariance_owned[[i, j]] - covariance_view[[i, j]]).norm() < 1e-12);
                assert!((correlation_owned[[i, j]] - correlation_view[[i, j]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn stats_view_into_reuses_outputs() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();

        let mut means = Array1::<f64>::zeros(2);
        let mut centered = Array2::<f64>::zeros((4, 2));
        let mut covariance = Array2::<f64>::zeros((2, 2));
        let mut correlation = Array2::<f64>::zeros((2, 2));

        column_means_view_into(&matrix.view(), &mut means).unwrap();
        center_columns_view_into(&matrix.view(), &mut centered).unwrap();
        covariance_matrix_view_into(&matrix.view(), &mut covariance).unwrap();
        correlation_matrix_view_into(&matrix.view(), &mut correlation).unwrap();

        assert_eq!(means, column_means(&matrix));
        assert_eq!(centered, center_columns(&matrix));
        assert_eq!(covariance, covariance_matrix(&matrix).unwrap());
        assert_eq!(correlation, correlation_matrix(&matrix).unwrap());
    }

    #[test]
    fn complex_stats_view_into_reuses_outputs() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(3.0, -2.0),
            Complex64::new(4.0, 1.0),
        ])
        .unwrap();

        let mut means = Array1::<Complex64>::zeros(2);
        let mut centered = Array2::<Complex64>::zeros((3, 2));
        let mut covariance = Array2::<Complex64>::zeros((2, 2));
        let mut correlation = Array2::<Complex64>::zeros((2, 2));

        column_means_complex_view_into(&matrix.view(), &mut means).unwrap();
        center_columns_complex_view_into(&matrix.view(), &mut centered).unwrap();
        covariance_matrix_complex_view_into(&matrix.view(), &mut covariance).unwrap();
        correlation_matrix_complex_view_into(&matrix.view(), &mut correlation).unwrap();

        assert_eq!(means, column_means_complex(&matrix));
        assert_eq!(centered, center_columns_complex(&matrix));
        assert_eq!(covariance, covariance_matrix_complex(&matrix).unwrap());
        assert_eq!(correlation, correlation_matrix_complex(&matrix).unwrap());
    }

    #[test]
    fn stats_owned_into_paths_cover_empty_valid_and_error_cases() {
        assert_eq!(
            StatsError::InvalidInput("bad shape".to_string()).to_string(),
            "Invalid input: bad shape"
        );

        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let mut means = Array1::<f64>::zeros(2);
        let mut centered = Array2::<f64>::zeros((4, 2));
        let mut covariance = Array2::<f64>::zeros((2, 2));
        let mut correlation = Array2::<f64>::zeros((2, 2));

        column_means_into(&matrix, &mut means).unwrap();
        center_columns_into(&matrix, &mut centered).unwrap();
        covariance_matrix_into(&matrix, &mut covariance).unwrap();
        correlation_matrix_into(&matrix, &mut correlation).unwrap();

        assert_eq!(means, column_means(&matrix));
        assert_eq!(centered, center_columns(&matrix));
        assert_eq!(covariance, covariance_matrix(&matrix).unwrap());
        assert_eq!(correlation, correlation_matrix(&matrix).unwrap());

        let empty_columns = Array2::<f64>::zeros((0, 3));
        let mut empty_means = Array1::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        column_means_into(&empty_columns, &mut empty_means).unwrap();
        assert!(empty_means.iter().all(|value| *value == 0.0));

        let mut bad_means = Array1::<f64>::zeros(3);
        assert!(matches!(
            column_means_into(&matrix, &mut bad_means),
            Err(StatsError::InvalidInput(_))
        ));

        let mut bad_centered = Array2::<f64>::zeros((4, 3));
        assert!(matches!(
            center_columns_into(&matrix, &mut bad_centered),
            Err(StatsError::InvalidInput(_))
        ));

        let empty = Array2::<f64>::zeros((0, 0));
        let mut empty_covariance = Array2::<f64>::zeros((0, 0));
        assert!(matches!(
            covariance_matrix_into(&empty, &mut empty_covariance),
            Err(StatsError::EmptyMatrix)
        ));

        let one_row = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let mut one_row_covariance = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            covariance_matrix_into(&one_row, &mut one_row_covariance),
            Err(StatsError::InsufficientSamples)
        ));

        let mut bad_covariance = Array2::<f64>::zeros((3, 3));
        assert!(matches!(
            covariance_matrix_into(&matrix, &mut bad_covariance),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            correlation_matrix_into(&matrix, &mut bad_covariance),
            Err(StatsError::InvalidInput(_))
        ));

        let unstable = Array2::from_shape_vec((2, 2), vec![f64::MAX, 0.0, -f64::MAX, 0.0]).unwrap();
        let mut unstable_covariance = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            covariance_matrix_into(&unstable, &mut unstable_covariance),
            Err(StatsError::NumericalInstability)
        ));
    }

    #[test]
    fn complex_stats_owned_into_paths_cover_empty_valid_and_error_cases() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(3.0, -2.0),
            Complex64::new(4.0, 1.0),
        ])
        .unwrap();
        let mut means = Array1::<Complex64>::zeros(2);
        let mut centered = Array2::<Complex64>::zeros((3, 2));
        let mut covariance = Array2::<Complex64>::zeros((2, 2));
        let mut correlation = Array2::<Complex64>::zeros((2, 2));

        column_means_complex_into(&matrix, &mut means).unwrap();
        center_columns_complex_into(&matrix, &mut centered).unwrap();
        covariance_matrix_complex_into(&matrix, &mut covariance).unwrap();
        correlation_matrix_complex_into(&matrix, &mut correlation).unwrap();

        assert_eq!(means, column_means_complex(&matrix));
        assert_eq!(centered, center_columns_complex(&matrix));
        assert_eq!(covariance, covariance_matrix_complex(&matrix).unwrap());
        assert_eq!(correlation, correlation_matrix_complex(&matrix).unwrap());

        let empty_columns = Array2::<Complex64>::zeros((0, 3));
        let mut empty_means = Array1::<Complex64>::from_vec(vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, 1.0),
        ]);
        column_means_complex_into(&empty_columns, &mut empty_means).unwrap();
        assert!(empty_means.iter().all(|value| *value == Complex64::new(0.0, 0.0)));

        let mut bad_means = Array1::<Complex64>::zeros(3);
        assert!(matches!(
            column_means_complex_into(&matrix, &mut bad_means),
            Err(StatsError::InvalidInput(_))
        ));

        let mut bad_centered = Array2::<Complex64>::zeros((3, 3));
        assert!(matches!(
            center_columns_complex_into(&matrix, &mut bad_centered),
            Err(StatsError::InvalidInput(_))
        ));

        let empty = Array2::<Complex64>::zeros((0, 0));
        let mut empty_covariance = Array2::<Complex64>::zeros((0, 0));
        assert!(matches!(
            covariance_matrix_complex_into(&empty, &mut empty_covariance),
            Err(StatsError::EmptyMatrix)
        ));

        let one_row = Array2::from_shape_vec((1, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ])
        .unwrap();
        let mut one_row_covariance = Array2::<Complex64>::zeros((2, 2));
        assert!(matches!(
            covariance_matrix_complex_into(&one_row, &mut one_row_covariance),
            Err(StatsError::InsufficientSamples)
        ));

        let mut bad_covariance = Array2::<Complex64>::zeros((3, 3));
        assert!(matches!(
            covariance_matrix_complex_into(&matrix, &mut bad_covariance),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            correlation_matrix_complex_into(&matrix, &mut bad_covariance),
            Err(StatsError::InvalidInput(_))
        ));

        let unstable = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(f64::MAX, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-f64::MAX, 0.0),
            Complex64::new(0.0, 0.0),
        ])
        .unwrap();
        let mut unstable_covariance = Array2::<Complex64>::zeros((2, 2));
        assert!(matches!(
            covariance_matrix_complex_into(&unstable, &mut unstable_covariance),
            Err(StatsError::NumericalInstability)
        ));
    }

    #[test]
    fn stats_view_into_rejects_wrong_output_shapes() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let mut bad_means = Array1::<f64>::zeros(3);
        let mut bad_centered = Array2::<f64>::zeros((4, 3));
        let mut bad_covariance = Array2::<f64>::zeros((3, 3));

        assert!(matches!(
            column_means_view_into(&matrix.view(), &mut bad_means),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            center_columns_view_into(&matrix.view(), &mut bad_centered),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            covariance_matrix_view_into(&matrix.view(), &mut bad_covariance),
            Err(StatsError::InvalidInput(_))
        ));
    }

    fn approx_eq(a: f64, b: f64) -> bool { num_traits::Float::abs(a - b) < 1e-10 }

    #[test]
    fn online_mean_and_variance_track_batch_statistics() {
        let mut mean = online::OnlineMean::<f64>::default();
        let mut variance = online::OnlineVariance::<f64>::default();
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            mean.push(value);
            variance.push(value);
        }
        assert!(approx_eq(mean.mean(), 3.0));
        assert!(approx_eq(variance.mean(), 3.0));
        assert!(approx_eq(variance.variance(), 2.5));
    }

    #[test]
    fn online_accumulators_reset_to_initial_state() {
        let mut mean = online::OnlineMean::<f64>::default();
        let mut variance = online::OnlineVariance::<f64>::default();
        mean.push(4.0);
        variance.push(4.0);
        mean.reset();
        variance.reset();
        assert!(approx_eq(mean.mean(), 0.0));
        assert!(approx_eq(variance.mean(), 0.0));
        assert!(approx_eq(variance.variance(), 0.0));
    }

    #[test]
    fn online_variance_is_zero_with_fewer_than_two_samples() {
        let mut variance = online::OnlineVariance::<f64>::default();
        variance.push(2.0);
        assert!(approx_eq(variance.variance(), 0.0));
    }

    #[test]
    fn ewma_state_and_vector_apis_match_expected_smoothing() {
        let signal = Array1::from_vec(vec![10.0_f64, 0.0]);
        let alpha = 0.5;
        let mut state = ewma::EwmaState::new(alpha);
        assert!(approx_eq(state.push(10.0), 10.0));
        assert!(approx_eq(state.push(0.0), 5.0));

        let owned = ewma::ewma(&signal, alpha);
        let viewed = ewma::ewma_view(&signal.view(), alpha);
        assert!(approx_eq(owned[0], 10.0));
        assert!(approx_eq(owned[1], 5.0));
        assert_eq!(owned, viewed);

        let mut output = Array1::<f64>::zeros(2);
        ewma::ewma_into(&signal.view(), alpha, &mut output).unwrap();
        assert_eq!(output, owned);

        let mut bad_output = Array1::<f64>::zeros(3);
        assert!(matches!(
            ewma::ewma_into(&signal.view(), alpha, &mut bad_output),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn rolling_mean_and_variance_match_manual_windows() {
        let signal = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
        let window = 3;
        let mean = rolling::rolling_mean(&signal.view(), window);
        let mean_view = rolling::rolling_mean_view(&signal.view(), window);
        assert_eq!(mean, mean_view);
        assert!(approx_eq(mean[0], 1.0));
        assert!(approx_eq(mean[2], 2.0));
        assert!(approx_eq(mean[4], 4.0));

        let variance = rolling::rolling_variance(&signal.view(), window);
        assert!(approx_eq(variance[2], 2.0 / 3.0));

        let mut mean_out = Array1::<f64>::zeros(signal.len());
        rolling::rolling_mean_into(&signal.view(), window, &mut mean_out).unwrap();
        assert_eq!(mean_out, mean);

        let mut variance_out = Array1::<f64>::zeros(signal.len());
        rolling::rolling_variance_into(&signal.view(), window, &mut variance_out).unwrap();
        assert_eq!(variance_out, variance);

        assert!(matches!(
            rolling::rolling_mean_into(&signal.view(), 0, &mut mean_out),
            Err(StatsError::InvalidInput(_))
        ));
        let mut bad_mean_out = Array1::<f64>::zeros(signal.len() + 1);
        assert!(matches!(
            rolling::rolling_mean_into(&signal.view(), window, &mut bad_mean_out),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            rolling::rolling_variance_into(&signal.view(), 0, &mut variance_out),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn rolling_covariance_into_handles_valid_and_invalid_inputs() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let window = 2;
        let cov = rolling::rolling_covariance(&matrix.view(), window);
        let cov_view = rolling::rolling_covariance_view(&matrix.view(), window);
        assert_eq!(cov, cov_view);
        assert!(approx_eq(cov[[0, 0]], 0.0));

        let mut output = Array2::<f64>::zeros((3, 4));
        rolling::rolling_covariance_into(&matrix.view(), window, &mut output).unwrap();
        assert_eq!(output, cov);

        let mut bad_shape = Array2::<f64>::zeros((3, 3));
        assert!(matches!(
            rolling::rolling_covariance_into(&matrix.view(), window, &mut bad_shape),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            rolling::rolling_covariance_into(&matrix.view(), 0, &mut output),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn lag_view_and_shift_columns_into_cover_alignment_paths() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = matrix.view();
        let lagged = lag::lag_view(&view, 1).unwrap();
        assert_eq!(lagged.dim(), (2, 2));
        assert!(approx_eq(lagged[[0, 0]], 1.0));
        assert!(approx_eq(lagged[[1, 1]], 4.0));

        let mut shifted = Array2::<f64>::zeros((3, 2));
        lag::shift_columns_into(&view, 1, &mut shifted).unwrap();
        assert!(approx_eq(shifted[[0, 0]], 0.0));
        assert!(approx_eq(shifted[[1, 0]], 1.0));
        assert!(approx_eq(shifted[[2, 1]], 4.0));

        assert!(matches!(lag::lag_view(&view, matrix.nrows()), Err(StatsError::InvalidInput(_))));

        let mut bad_output = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            lag::shift_columns_into(&view, 1, &mut bad_output),
            Err(StatsError::InvalidInput(_))
        ));

        let mut zeroed = Array2::<f64>::ones((3, 2));
        lag::shift_columns_into(&view, matrix.nrows(), &mut zeroed).unwrap();
        assert!(zeroed.iter().all(|value| approx_eq(*value, 0.0)));
    }
}

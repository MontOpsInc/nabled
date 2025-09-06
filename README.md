# Rust Linear Algebra Library

Advanced linear algebra functions built on top of `nalgebra` and `ndarray` crates in Rust. This library provides enhanced implementations of common linear algebra operations with focus on performance and numerical stability.

## Features

- **Singular Value Decomposition (SVD)** implementations using both `nalgebra` and `ndarray`
- **Matrix Functions** - Matrix exponential, logarithm, and power operations
- **Jacobian Computation** - Numerical differentiation and gradient computation (real and complex)
- **Truncated SVD** for dimensionality reduction
- **Matrix reconstruction** from SVD components
- **Condition number** and **matrix rank** computation
- **Comprehensive test suite** with integration tests
- **Performance benchmarks** comparing different implementations
- **Utility functions** for matrix operations and conversions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-linalg = "0.2.0"
```

## Quick Start

### Basic SVD Usage

```rust
use rust_linalg::svd::{nalgebra_svd, ndarray_svd};
use nalgebra::DMatrix;
use ndarray::Array2;

// Using nalgebra
let matrix = DMatrix::from_row_slice(3, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
]);

let svd = nalgebra_svd::compute_svd(&matrix)?;
println!("Singular values: {:?}", svd.singular_values);
println!("Condition number: {}", nalgebra_svd::condition_number(&svd));

// Using ndarray
let matrix = Array2::from_shape_vec((3, 3), vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
])?;

let svd = ndarray_svd::compute_svd(&matrix)?;
println!("Matrix rank: {}", ndarray_svd::matrix_rank(&svd, None));
```

### Truncated SVD

```rust
// Keep only the 2 largest singular values
let truncated_svd = nalgebra_svd::compute_truncated_svd(&matrix, 2)?;
println!("Truncated singular values: {:?}", truncated_svd.singular_values);
```

### Matrix Reconstruction

```rust
let svd = nalgebra_svd::compute_svd(&matrix)?;
let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
// reconstructed should be approximately equal to the original matrix
```

### Matrix Functions

```rust
use rust_linalg::matrix_functions::nalgebra_matrix_functions;
use nalgebra::DMatrix;

// Matrix exponential
let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
let exp_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&matrix)?;
println!("exp(A): {}", exp_matrix);

// Matrix logarithm
let log_matrix = nalgebra_matrix_functions::matrix_log_eigen(&matrix)?;
println!("log(A): {}", log_matrix);

// Matrix power (e.g., square root)
let sqrt_matrix = nalgebra_matrix_functions::matrix_power(&matrix, 0.5)?;
println!("A^0.5: {}", sqrt_matrix);
```

### Jacobian Computation

```rust
use rust_linalg::jacobian::nalgebra_jacobian;
use nalgebra::DVector;

// Jacobian of vector-valued function
let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
    Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
};

let x = DVector::from_vec(vec![2.0, 3.0]);
let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
println!("Jacobian: {}", jacobian);
```

### Gradient Computation

```rust
use rust_linalg::jacobian::nalgebra_jacobian;
use nalgebra::DVector;

// Gradient of scalar function
let f = |x: &DVector<f64>| -> Result<f64, String> {
    Ok(x[0] * x[0] + x[1] * x[1])
};

let x = DVector::from_vec(vec![3.0, 4.0]);
let gradient = nalgebra_jacobian::numerical_gradient(&f, &x, &Default::default())?;
println!("Gradient: {}", gradient);
```

### Complex Derivatives

```rust
use rust_linalg::jacobian::complex_jacobian;
use nalgebra::DVector;
use num_complex::Complex;

// Complex function f(z) = z²
let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
    let mut result = DVector::zeros(x.len());
    for i in 0..x.len() {
        result[i] = x[i] * x[i];
    }
    Ok(result)
};

let x = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
let jacobian = complex_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
println!("Complex Jacobian: {}", jacobian);
```

### Hessian Matrix

```rust
use rust_linalg::jacobian::nalgebra_jacobian;
use nalgebra::DVector;

// Hessian of scalar function
let f = |x: &DVector<f64>| -> Result<f64, String> {
    Ok(x[0] * x[0] * x[0] + x[1] * x[1] * x[1])
};

let x = DVector::from_vec(vec![2.0, 3.0]);
let hessian = nalgebra_jacobian::numerical_hessian(&f, &x, &Default::default())?;
println!("Hessian: {}", hessian);
```

## API Reference

### Nalgebra SVD

- `compute_svd(matrix)` - Compute full SVD
- `compute_truncated_svd(matrix, k)` - Compute truncated SVD keeping k largest singular values
- `reconstruct_matrix(svd)` - Reconstruct original matrix from SVD components
- `condition_number(svd)` - Compute condition number from singular values
- `matrix_rank(svd, tolerance)` - Compute matrix rank from singular values

### Ndarray SVD

- `compute_svd(matrix)` - Compute full SVD
- `compute_truncated_svd(matrix, k)` - Compute truncated SVD keeping k largest singular values
- `reconstruct_matrix(svd)` - Reconstruct original matrix from SVD components
- `condition_number(svd)` - Compute condition number from singular values
- `matrix_rank(svd, tolerance)` - Compute matrix rank from singular values

### Matrix Functions

#### Nalgebra Matrix Functions
- `matrix_exp(matrix, max_iterations, tolerance)` - Compute matrix exponential using Taylor series
- `matrix_exp_eigen(matrix)` - Compute matrix exponential using eigenvalue decomposition
- `matrix_log_taylor(matrix, max_iterations, tolerance)` - Compute matrix logarithm using Taylor series
- `matrix_log_eigen(matrix)` - Compute matrix logarithm using eigenvalue decomposition
- `matrix_log_svd(matrix)` - Compute matrix logarithm using SVD decomposition
- `matrix_power(matrix, power)` - Compute matrix power using eigenvalue decomposition

#### Ndarray Matrix Functions
- `matrix_exp(matrix, max_iterations, tolerance)` - Compute matrix exponential
- `matrix_exp_eigen(matrix)` - Compute matrix exponential using eigenvalue decomposition
- `matrix_log_taylor(matrix, max_iterations, tolerance)` - Compute matrix logarithm using Taylor series
- `matrix_log_eigen(matrix)` - Compute matrix logarithm using eigenvalue decomposition
- `matrix_log_svd(matrix)` - Compute matrix logarithm using SVD decomposition
- `matrix_power(matrix, power)` - Compute matrix power

### Jacobian Computation

#### Nalgebra Jacobian Functions
- `numerical_jacobian(f, x, config)` - Compute Jacobian using forward finite differences
- `numerical_jacobian_central(f, x, config)` - Compute Jacobian using central finite differences (more accurate)
- `numerical_gradient(f, x, config)` - Compute gradient for scalar functions
- `numerical_hessian(f, x, config)` - Compute Hessian matrix (second-order derivatives)

#### Ndarray Jacobian Functions
- `numerical_jacobian(f, x, config)` - Compute Jacobian using forward finite differences
- `numerical_jacobian_central(f, x, config)` - Compute Jacobian using central finite differences
- `numerical_gradient(f, x, config)` - Compute gradient for scalar functions
- `numerical_hessian(f, x, config)` - Compute Hessian matrix (second-order derivatives)

## Examples

Run the examples:

```bash
# SVD example
cargo run --example svd_example

# Matrix functions example
cargo run --example matrix_functions_example

# Jacobian computation example
cargo run --example jacobian_example
```

## Testing

Run the test suite:

```bash
cargo test
```

Run integration tests:

```bash
cargo test --test integration_tests
```

## Benchmarks

Run performance benchmarks:

```bash
cargo bench
```

The benchmarks compare:
- Nalgebra vs Ndarray SVD performance
- Full SVD vs Truncated SVD performance
- Performance across different matrix sizes

## Error Handling

The library provides comprehensive error handling through custom error enums:

### SVD Errors
```rust
pub enum SVDError {
    NotSquare,
    EmptyMatrix,
    ConvergenceFailed,
    InvalidInput(String),
}
```

### Matrix Function Errors
```rust
pub enum MatrixFunctionError {
    NotSquare,
    EmptyMatrix,
    SingularMatrix,
    ConvergenceFailed,
    InvalidInput(String),
    NegativeEigenvalues,
}
```

### Jacobian Errors
```rust
pub enum JacobianError {
    FunctionError(String),
    InvalidDimensions(String),
    InvalidStepSize,
    ConvergenceFailed,
    EmptyInput,
    DimensionMismatch,
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] QR decomposition implementations
- [ ] LU decomposition implementations
- [ ] Eigenvalue decomposition
- [ ] Matrix factorization algorithms
- [ ] GPU acceleration support
- [ ] More advanced numerical algorithms

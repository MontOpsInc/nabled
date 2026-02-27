# Rust Linear Algebra Library - Comprehensive Documentation

This document provides detailed documentation for all modules in the rust-linalg library, including function signatures, parameters, return values, and examples.

## Table of Contents

1. [SVD (Singular Value Decomposition)](#svd-singular-value-decomposition)
2. [Matrix Functions](#matrix-functions)
3. [Jacobian Computation](#jacobian-computation)
4. [QR Decomposition](#qr-decomposition)
5. [Cholesky Decomposition](#cholesky-decomposition)
6. [LU Decomposition](#lu-decomposition)
7. [Eigenvalue Decomposition](#eigenvalue-decomposition)
8. [Statistics](#statistics)
9. [PCA](#pca)
10. [Linear Regression](#linear-regression)
11. [Polar Decomposition](#polar-decomposition)
12. [Schur Decomposition](#schur-decomposition)
13. [Sylvester Equation](#sylvester-equation)
14. [Orthogonalization](#orthogonalization)
15. [Iterative](#iterative)
16. [Triangular](#triangular)
17. [Utility Functions](#utility-functions)

---

## SVD (Singular Value Decomposition)

The SVD module provides implementations for both `nalgebra` and `ndarray` that compute the singular value decomposition of matrices.

### Nalgebra SVD Functions

#### `compute_svd(matrix: &DMatrix<T>) -> Result<NalgebraSVD<T>, SVDError>`

Computes the full singular value decomposition of a matrix.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix to decompose (must be square or rectangular)

**Returns:**
- `Result<NalgebraSVD<T>, SVDError>` - SVD result containing U, S, and V^T matrices

**Example:**
```rust
use nabled::svd::nalgebra_svd;
use nalgebra::DMatrix;

let matrix = DMatrix::from_row_slice(3, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
]);

let svd = nalgebra_svd::compute_svd(&matrix)?;
println!("Singular values: {:?}", svd.singular_values);
```

#### `compute_truncated_svd(matrix: &DMatrix<T>, k: usize) -> Result<NalgebraSVD<T>, SVDError>`

Computes truncated SVD keeping only the k largest singular values.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix to decompose
- `k: usize` - Number of largest singular values to keep

**Returns:**
- `Result<NalgebraSVD<T>, SVDError>` - Truncated SVD result

**Example:**
```rust
let truncated_svd = nalgebra_svd::compute_truncated_svd(&matrix, 2)?;
println!("Truncated singular values: {:?}", truncated_svd.singular_values);
```

#### `reconstruct_matrix(svd: &NalgebraSVD<T>) -> DMatrix<T>`

Reconstructs the original matrix from SVD components.

**Parameters:**
- `svd: &NalgebraSVD<T>` - SVD result containing U, S, and V^T

**Returns:**
- `DMatrix<T>` - Reconstructed matrix

**Example:**
```rust
let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
// reconstructed should be approximately equal to the original matrix
```

#### `condition_number(svd: &NalgebraSVD<T>) -> T`

Computes the condition number from singular values.

**Parameters:**
- `svd: &NalgebraSVD<T>` - SVD result

**Returns:**
- `T` - Condition number (ratio of largest to smallest singular value)

**Example:**
```rust
let cond_num = nalgebra_svd::condition_number(&svd);
println!("Condition number: {}", cond_num);
```

#### `matrix_rank(svd: &NalgebraSVD<T>, tolerance: Option<T>) -> usize`

Computes the matrix rank from singular values.

**Parameters:**
- `svd: &NalgebraSVD<T>` - SVD result
- `tolerance: Option<T>` - Optional tolerance for rank determination

**Returns:**
- `usize` - Matrix rank

**Example:**
```rust
let rank = nalgebra_svd::matrix_rank(&svd, None);
println!("Matrix rank: {}", rank);
```

### Ndarray SVD Functions

All ndarray functions have the same signatures and behavior as nalgebra functions, but work with `Array2<T>` instead of `DMatrix<T>`.

#### `compute_svd(matrix: &Array2<T>) -> Result<NdarraySVD<T>, SVDError>`
#### `compute_truncated_svd(matrix: &Array2<T>, k: usize) -> Result<NdarraySVD<T>, SVDError>`
#### `reconstruct_matrix(svd: &NdarraySVD<T>) -> Array2<T>`
#### `condition_number(svd: &NdarraySVD<T>) -> T`
#### `matrix_rank(svd: &NdarraySVD<T>, tolerance: Option<T>) -> usize`

---

## Matrix Functions

The matrix functions module provides matrix exponential, logarithm, and power operations.

### Nalgebra Matrix Functions

#### `matrix_exp(matrix: &DMatrix<T>, max_iterations: usize, tolerance: T) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix exponential using Taylor series expansion.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)
- `max_iterations: usize` - Maximum number of Taylor series terms
- `tolerance: T` - Convergence tolerance

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix exponential

**Example:**
```rust
use nabled::matrix_functions::nalgebra_matrix_functions;
use nalgebra::DMatrix;

let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
let exp_matrix = nalgebra_matrix_functions::matrix_exp(&matrix, 50, 1e-10)?;
println!("exp(A): {}", exp_matrix);
```

#### `matrix_exp_eigen(matrix: &DMatrix<T>) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix exponential using eigenvalue decomposition (more efficient for symmetric matrices).

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix exponential

**Example:**
```rust
let exp_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&matrix)?;
println!("exp(A): {}", exp_matrix);
```

#### `matrix_log_taylor(matrix: &DMatrix<T>, max_iterations: usize, tolerance: T) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix logarithm using Taylor series expansion.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)
- `max_iterations: usize` - Maximum number of Taylor series terms
- `tolerance: T` - Convergence tolerance

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix logarithm

**Example:**
```rust
let log_matrix = nalgebra_matrix_functions::matrix_log_taylor(&matrix, 50, 1e-10)?;
println!("log(A): {}", log_matrix);
```

#### `matrix_log_eigen(matrix: &DMatrix<T>) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix logarithm using eigenvalue decomposition.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix logarithm

**Example:**
```rust
let log_matrix = nalgebra_matrix_functions::matrix_log_eigen(&matrix)?;
println!("log(A): {}", log_matrix);
```

#### `matrix_log_svd(matrix: &DMatrix<T>) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix logarithm using SVD decomposition.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix logarithm

**Example:**
```rust
let log_matrix = nalgebra_matrix_functions::matrix_log_svd(&matrix)?;
println!("log(A): {}", log_matrix);
```

#### `matrix_power(matrix: &DMatrix<T>, power: T) -> Result<DMatrix<T>, MatrixFunctionError>`

Computes matrix power using eigenvalue decomposition.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix (must be square)
- `power: T` - Power to raise matrix to

**Returns:**
- `Result<DMatrix<T>, MatrixFunctionError>` - Matrix power

**Example:**
```rust
let sqrt_matrix = nalgebra_matrix_functions::matrix_power(&matrix, 0.5)?;
println!("A^0.5: {}", sqrt_matrix);
```

### Ndarray Matrix Functions

All ndarray functions have the same signatures and behavior as nalgebra functions, but work with `Array2<T>` instead of `DMatrix<T>`.

---

## Jacobian Computation

The Jacobian module provides numerical differentiation and gradient computation for both vector-valued and scalar functions.

### Configuration

#### `JacobianConfig<T>`

Configuration structure for Jacobian computation.

**Fields:**
- `step_size: T` - Step size for finite differences (default: 1e-6)
- `tolerance: T` - Relative tolerance for convergence (default: 1e-8)
- `max_iterations: usize` - Maximum number of iterations (default: 100)

**Example:**
```rust
use nabled::jacobian::JacobianConfig;

let config = JacobianConfig {
    step_size: 1e-8,
    tolerance: 1e-10,
    max_iterations: 200,
};
```

### Nalgebra Jacobian Functions

#### `numerical_jacobian(f: &F, x: &DVector<T>, config: &JacobianConfig<T>) -> Result<DMatrix<T>, JacobianError>`

Computes numerical Jacobian using forward finite differences.

**Parameters:**
- `f: &F` - Function that takes `&DVector<T>` and returns `Result<DVector<T>, String>`
- `x: &DVector<T>` - Point at which to compute the Jacobian
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DMatrix<T>, JacobianError>` - Jacobian matrix (m×n where m is output dimension, n is input dimension)

**Example:**
```rust
use nabled::jacobian::nalgebra_jacobian;
use nalgebra::DVector;

let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
    Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
};

let x = DVector::from_vec(vec![2.0, 3.0]);
let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
// Jacobian should be approximately [[4.0, 0.0], [0.0, 6.0]]
```

#### `numerical_jacobian_central(f: &F, x: &DVector<T>, config: &JacobianConfig<T>) -> Result<DMatrix<T>, JacobianError>`

Computes numerical Jacobian using central finite differences (more accurate).

**Parameters:**
- `f: &F` - Function that takes `&DVector<T>` and returns `Result<DVector<T>, String>`
- `x: &DVector<T>` - Point at which to compute the Jacobian
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DMatrix<T>, JacobianError>` - Jacobian matrix

**Example:**
```rust
let jacobian = nalgebra_jacobian::numerical_jacobian_central(&f, &x, &Default::default())?;
// More accurate than forward differences
```

#### `numerical_gradient(f: &F, x: &DVector<T>, config: &JacobianConfig<T>) -> Result<DVector<T>, JacobianError>`

Computes gradient for scalar functions.

**Parameters:**
- `f: &F` - Function that takes `&DVector<T>` and returns `Result<T, String>`
- `x: &DVector<T>` - Point at which to compute the gradient
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DVector<T>, JacobianError>` - Gradient vector

**Example:**
```rust
let f = |x: &DVector<f64>| -> Result<f64, String> {
    Ok(x[0] * x[0] + x[1] * x[1])
};

let x = DVector::from_vec(vec![3.0, 4.0]);
let gradient = nalgebra_jacobian::numerical_gradient(&f, &x, &Default::default())?;
// Gradient should be approximately [6.0, 8.0]
```

#### `numerical_hessian(f: &F, x: &DVector<T>, config: &JacobianConfig<T>) -> Result<DMatrix<T>, JacobianError>`

Computes Hessian matrix (second-order partial derivatives).

**Parameters:**
- `f: &F` - Function that takes `&DVector<T>` and returns `Result<T, String>`
- `x: &DVector<T>` - Point at which to compute the Hessian
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DMatrix<T>, JacobianError>` - Hessian matrix (n×n where n is input dimension)

**Example:**
```rust
let f = |x: &DVector<f64>| -> Result<f64, String> {
    Ok(x[0] * x[0] * x[0] + x[1] * x[1] * x[1])
};

let x = DVector::from_vec(vec![2.0, 3.0]);
let hessian = nalgebra_jacobian::numerical_hessian(&f, &x, &Default::default())?;
// Hessian should be approximately [[12.0, 0.0], [0.0, 18.0]]
```

### Ndarray Jacobian Functions

All ndarray functions have the same signatures and behavior as nalgebra functions, but work with `Array1<T>` and `Array2<T>` instead of `DVector<T>` and `DMatrix<T>`.

#### `numerical_jacobian(f: &F, x: &Array1<T>, config: &JacobianConfig<T>) -> Result<Array2<T>, JacobianError>`
#### `numerical_jacobian_central(f: &F, x: &Array1<T>, config: &JacobianConfig<T>) -> Result<Array2<T>, JacobianError>`
#### `numerical_gradient(f: &F, x: &Array1<T>, config: &JacobianConfig<T>) -> Result<Array1<T>, JacobianError>`
#### `numerical_hessian(f: &F, x: &Array1<T>, config: &JacobianConfig<T>) -> Result<Array2<T>, JacobianError>`

### Complex Derivatives

The Jacobian module also provides complex derivative computation using the complex step method, which can provide higher accuracy for certain types of functions.

#### `complex_jacobian::numerical_jacobian(f: &F, x: &DVector<Complex<T>>, config: &JacobianConfig<T>) -> Result<DMatrix<Complex<T>>, JacobianError>`

Computes numerical Jacobian for complex-valued functions using the complex step method.

**Parameters:**
- `f: &F` - Function that takes `&DVector<Complex<T>>` and returns `Result<DVector<Complex<T>>, String>`
- `x: &DVector<Complex<T>>` - Complex point at which to compute the Jacobian
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DMatrix<Complex<T>>, JacobianError>` - Complex Jacobian matrix

**Example:**
```rust
use nabled::jacobian::complex_jacobian;
use nalgebra::DVector;
use num_complex::Complex;

let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
    let mut result = DVector::zeros(x.len());
    for i in 0..x.len() {
        result[i] = x[i] * x[i]; // f(z) = z²
    }
    Ok(result)
};

let x = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
let jacobian = complex_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
// Jacobian should be approximately [[2.0, 0.0], [0.0, 4.0]]
```

#### `complex_jacobian::numerical_gradient(f: &F, x: &DVector<Complex<T>>, config: &JacobianConfig<T>) -> Result<DVector<Complex<T>>, JacobianError>`

Computes gradient for complex scalar functions.

**Parameters:**
- `f: &F` - Function that takes `&DVector<Complex<T>>` and returns `Result<Complex<T>, String>`
- `x: &DVector<Complex<T>>` - Complex point at which to compute the gradient
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DVector<Complex<T>>, JacobianError>` - Complex gradient vector

#### `complex_jacobian::numerical_hessian(f: &F, x: &DVector<Complex<T>>, config: &JacobianConfig<T>) -> Result<DMatrix<Complex<T>>, JacobianError>`

Computes Hessian matrix for complex scalar functions.

**Parameters:**
- `f: &F` - Function that takes `&DVector<Complex<T>>` and returns `Result<Complex<T>, String>`
- `x: &DVector<Complex<T>>` - Complex point at which to compute the Hessian
- `config: &JacobianConfig<T>` - Configuration parameters

**Returns:**
- `Result<DMatrix<Complex<T>>, JacobianError>` - Complex Hessian matrix

**Note:** The complex step method for second derivatives may have limitations and may not provide accurate results for all functions.

### Error Types

#### `JacobianError`

Error types for Jacobian computation:

- `FunctionError(String)` - Function returned an error during evaluation
- `InvalidDimensions(String)` - Invalid input dimensions
- `InvalidStepSize` - Step size too small or invalid
- `ConvergenceFailed` - Convergence failed for iterative methods
- `EmptyInput` - Empty input or output
- `DimensionMismatch` - Dimension mismatch between input and output

---

## Utility Functions

The utility module provides helper functions for matrix operations and conversions.

### Matrix Conversion Functions

#### `nalgebra_to_ndarray(matrix: &DMatrix<T>) -> Array2<T>`

Converts nalgebra matrix to ndarray matrix.

**Parameters:**
- `matrix: &DMatrix<T>` - Input nalgebra matrix

**Returns:**
- `Array2<T>` - Converted ndarray matrix

#### `ndarray_to_nalgebra(array: &Array2<T>) -> DMatrix<T>`

Converts ndarray matrix to nalgebra matrix.

**Parameters:**
- `array: &Array2<T>` - Input ndarray matrix

**Returns:**
- `DMatrix<T>` - Converted nalgebra matrix

### Norm Functions

#### `frobenius_norm(matrix: &Array2<T>) -> T`

Computes the Frobenius norm of a matrix.

**Parameters:**
- `matrix: &Array2<T>` - Input matrix

**Returns:**
- `T` - Frobenius norm (square root of sum of squared elements)

#### `spectral_norm(matrix: &Array2<T>) -> Result<T, SVDError>`

Computes the spectral norm (largest singular value) of a matrix.

**Parameters:**
- `matrix: &Array2<T>` - Input matrix

**Returns:**
- `Result<T, SVDError>` - Spectral norm

### Random Matrix Generation

#### `random_matrix(rows: usize, cols: usize) -> Array2<f64>`

Generates a random matrix (currently returns zeros as placeholder).

**Parameters:**
- `rows: usize` - Number of rows
- `cols: usize` - Number of columns

**Returns:**
- `Array2<f64>` - Random matrix

---

## Error Handling

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

### QR Errors

```rust
pub enum QRError {
    EmptyMatrix,
    SingularMatrix,
    ConvergenceFailed,
    InvalidDimensions(String),
    NumericalInstability,
    InvalidInput(String),
}
```

### Cholesky Errors

```rust
pub enum CholeskyError {
    EmptyMatrix,
    NotSquare,
    NotPositiveDefinite,
    NumericalInstability,
    InvalidInput(String),
}
```

### LU Errors

```rust
pub enum LUError {
    EmptyMatrix,
    NotSquare,
    SingularMatrix,
    NumericalInstability,
    InvalidInput(String),
}
```

### Eigen Errors

```rust
pub enum EigenError {
    EmptyMatrix,
    NotSquare,
    NonSymmetric,
    NotPositiveDefinite,
    DimensionMismatch,
    ConvergenceFailed,
    NumericalInstability,
    InvalidInput(String),
}
```

### Regression Errors

```rust
pub enum RegressionError {
    EmptyInput,
    DimensionMismatch(String),
    SingularMatrix,
    QRError(String),
}
```

### PCA Errors

```rust
pub enum PCAError {
    EmptyMatrix,
    InsufficientSamples,
    InvalidComponents,
    Computation(String),
}
```

### Polar Errors

```rust
pub enum PolarError {
    EmptyMatrix,
    NotSquare,
    SVDError(SVDError),
    InvalidInput(String),
}
```

### Schur Errors

```rust
pub enum SchurError {
    EmptyMatrix,
    NotSquare,
    ConvergenceFailed,
    InvalidInput(String),
}
```

### Sylvester Errors

```rust
pub enum SylvesterError {
    EmptyMatrix,
    DimensionMismatch,
    SchurError(SchurError),
    InvalidInput(String),
}
```

### Orthogonalization Errors

```rust
pub enum OrthogonalizationError {
    EmptyMatrix,
    InvalidInput(String),
}
```

### Stats Errors

```rust
pub enum StatsError {
    EmptyMatrix,
    InsufficientSamples,
    InvalidInput(String),
}
```

### Triangular Errors

```rust
pub enum TriangularError {
    EmptyMatrix,
    NotSquare,
    SingularMatrix,
    InvalidInput(String),
}
```

### Iterative Errors

```rust
pub enum IterativeError {
    NotConverged,
    InvalidInput(String),
}
```

---

## Performance Considerations

### SVD
- Nalgebra is generally faster for small to medium matrices
- Both implementations scale similarly for large matrices
- Truncated SVD is faster than full SVD when k << min(m,n)

### Matrix Functions
- Eigenvalue decomposition methods are faster for symmetric matrices
- Taylor series methods are more general but slower
- SVD-based methods are most robust for ill-conditioned matrices

### Jacobian Computation
- Central differences are more accurate but require 2x function evaluations
- Smaller step sizes improve accuracy but may cause numerical issues
- Hessian computation is O(n²) in function evaluations

---

## QR Decomposition

The QR decomposition module provides implementations for both `nalgebra` and `ndarray` that compute the QR decomposition of matrices using Householder reflections.

### Nalgebra QR Functions

#### `compute_qr(matrix: &DMatrix<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes the full QR decomposition of a matrix using nalgebra's built-in QR algorithm.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix to decompose (can be square or rectangular)
- `config: &QRConfig<T>` - Configuration for the decomposition

**Returns:**
- `Result<QRResult<T>, QRError>` - QR result containing Q, R, and rank information

**Example:**
```rust
use nabled::qr::nalgebra_qr;
use nalgebra::DMatrix;

let matrix = DMatrix::from_row_slice(3, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 10.0
]);

let qr = nalgebra_qr::compute_qr(&matrix, &Default::default())?;
println!("Q: {}", qr.q);
println!("R: {}", qr.r);
println!("Rank: {}", qr.rank);
```

#### `compute_reduced_qr(matrix: &DMatrix<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes the reduced QR decomposition (economy size) for rectangular matrices.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix to decompose
- `config: &QRConfig<T>` - Configuration for the decomposition

**Returns:**
- `Result<QRResult<T>, QRError>` - Reduced QR result with Q as m×min(m,n) and R as min(m,n)×n

**Example:**
```rust
use nabled::qr::nalgebra_qr;
use nalgebra::DMatrix;

let matrix = DMatrix::from_row_slice(4, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0
]);

let qr = nalgebra_qr::compute_reduced_qr(&matrix, &Default::default())?;
// Q is 4×3, R is 3×3
```

#### `compute_qr_with_pivoting(matrix: &DMatrix<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes QR decomposition with column pivoting for numerical stability.

**Parameters:**
- `matrix: &DMatrix<T>` - Input matrix to decompose
- `config: &QRConfig<T>` - Configuration for the decomposition

**Returns:**
- `Result<QRResult<T>, QRError>` - QR result with pivoting information

#### `solve_least_squares(matrix: &DMatrix<T>, rhs: &DVector<T>, config: &QRConfig<T>) -> Result<DVector<T>, QRError>`

Solves least squares problem min ||Ax - b||₂ using QR decomposition.

**Parameters:**
- `matrix: &DMatrix<T>` - Coefficient matrix A
- `rhs: &DVector<T>` - Right-hand side vector b
- `config: &QRConfig<T>` - Configuration for the decomposition

**Returns:**
- `Result<DVector<T>, QRError>` - Solution vector x

**Example:**
```rust
use nabled::qr::nalgebra_qr;
use nalgebra::{DMatrix, DVector};

let a = DMatrix::from_row_slice(4, 2, &[
    1.0, 1.0,
    1.0, 2.0,
    1.0, 3.0,
    1.0, 4.0
]);
let b = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

let x = nalgebra_qr::solve_least_squares(&a, &b, &Default::default())?;
println!("Solution: {}", x);
```

### Edge Cases and Robustness

The QR decomposition implementation includes comprehensive edge case handling for numerical robustness:

#### **Empty and Zero Matrices**
```rust
// Empty matrix - returns QRError::EmptyMatrix
let empty = DMatrix::<f64>::zeros(0, 0);
let result = nalgebra_qr::compute_qr(&empty, &config);

// Zero matrix - handled gracefully with rank 0
let zero_matrix = DMatrix::zeros(3, 3);
let qr = nalgebra_qr::compute_qr(&zero_matrix, &config)?;
assert_eq!(qr.rank, 0);
```

#### **Single Element Matrices**
```rust
// Single element matrix - optimized path
let single = DMatrix::from_element(1, 1, 5.0);
let qr = nalgebra_qr::compute_qr(&single, &config)?;
// Q = [[1]], R = [[5]], rank = 1
```

#### **Numerical Stability Checks**
```rust
// NaN/Infinity detection - returns QRError::NumericalInstability
let mut nan_matrix = DMatrix::from_element(2, 2, 1.0);
nan_matrix[(0, 0)] = f64::NAN;
let result = nalgebra_qr::compute_qr(&nan_matrix, &config);

// Very small rank tolerance - returns QRError::InvalidInput
let bad_config = QRConfig { 
    rank_tolerance: 1e-20,
    max_iterations: 100,
    use_pivoting: false
};
```

#### **Rank-Deficient Matrices**
```rust
// Rank-deficient matrix - detected and handled
let rank_deficient = DMatrix::from_row_slice(2, 2, &[
    1.0, 2.0,
    2.0, 4.0  // Second row is 2 * first row
]);
let qr = nalgebra_qr::compute_qr(&rank_deficient, &config)?;
assert_eq!(qr.rank, 1); // Correctly detects rank deficiency
```

#### **Least Squares Edge Cases**
```rust
// Dimension mismatch - returns QRError::InvalidDimensions
let matrix = DMatrix::from_element(2, 2, 1.0);
let vector = DVector::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
let result = nalgebra_qr::solve_least_squares(&matrix, &vector, &config);

// Underdetermined system - returns QRError::InvalidDimensions
let matrix = DMatrix::from_element(2, 3, 1.0); // 2 equations, 3 unknowns
let vector = DVector::from_vec(vec![1.0, 2.0]);
let result = nalgebra_qr::solve_least_squares(&matrix, &vector, &config);

// Singular matrix - returns QRError::SingularMatrix
let singular = DMatrix::from_row_slice(2, 2, &[
    1.0, 2.0,
    2.0, 4.0
]);
let vector = DVector::from_vec(vec![1.0, 2.0]);
let result = nalgebra_qr::solve_least_squares(&singular, &vector, &config);
```

#### `reconstruct_matrix(qr: &QRResult<T>) -> DMatrix<T>`

Reconstructs the original matrix from QR decomposition components.

**Parameters:**
- `qr: &QRResult<T>` - QR decomposition result

**Returns:**
- `DMatrix<T>` - Reconstructed matrix A = QR

#### `condition_number(qr: &QRResult<T>) -> T`

Computes the condition number from QR decomposition.

**Parameters:**
- `qr: &QRResult<T>` - QR decomposition result

**Returns:**
- `T` - Condition number (ratio of largest to smallest diagonal element of R)

### Ndarray QR Functions

#### `compute_qr(matrix: &Array2<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes QR decomposition for ndarray matrices by converting to nalgebra.

**Parameters:**
- `matrix: &Array2<T>` - Input ndarray matrix
- `config: &QRConfig<T>` - Configuration for the decomposition

**Returns:**
- `Result<QRResult<T>, QRError>` - QR result (nalgebra matrices)

#### `compute_reduced_qr(matrix: &Array2<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes reduced QR decomposition for ndarray matrices.

#### `compute_qr_with_pivoting(matrix: &Array2<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>`

Computes QR decomposition with pivoting for ndarray matrices.

#### `solve_least_squares(matrix: &Array2<T>, rhs: &Array1<T>, config: &QRConfig<T>) -> Result<Array1<T>, QRError>`

Solves least squares problem for ndarray matrices.

### QR Configuration

#### `QRConfig<T>`

Configuration structure for QR decomposition parameters.

**Fields:**
- `rank_tolerance: T` - Tolerance for rank determination (default: 1e-12)
- `max_iterations: usize` - Maximum iterations for iterative methods (default: 100)
- `use_pivoting: bool` - Enable column pivoting (default: false)

### QR Result

#### `QRResult<T>`

Result structure containing QR decomposition components.

**Fields:**
- `q: DMatrix<T>` - Orthogonal matrix Q
- `r: DMatrix<T>` - Upper triangular matrix R
- `p: Option<DMatrix<T>>` - Column permutation matrix (if pivoting was used)
- `rank: usize` - Matrix rank

### QR Error Handling

#### `QRError`

Error types for QR decomposition operations.

**Variants:**
- `EmptyMatrix` - Matrix is empty
- `SingularMatrix` - Matrix is singular or rank-deficient
- `ConvergenceFailed` - Convergence failed for iterative methods
- `InvalidDimensions(String)` - Invalid input dimensions
- `NumericalInstability` - Numerical instability detected
- `InvalidInput(String)` - Invalid input parameters

### Performance Considerations

- QR decomposition is O(mn²) for m×n matrices
- Reduced QR is more memory efficient for rectangular matrices
- Column pivoting improves numerical stability but adds overhead
- Least squares solving is O(mn²) using QR decomposition
- Condition number computation is O(min(m,n))

---

## Cholesky Decomposition

Cholesky decomposition for symmetric positive-definite matrices. Used for solving linear systems and computing matrix inverses.

### Functions

#### `compute_cholesky(matrix)` (nalgebra_cholesky / ndarray_cholesky)

Computes the Cholesky decomposition A = LL^T where L is lower triangular.

**Returns:** `NalgebraCholeskyResult` or `NdarrayCholeskyResult` with `.l` (lower triangular factor)

#### `solve(matrix, rhs)`

Solves Ax = b using the Cholesky decomposition.

#### `inverse(matrix)`

Computes the matrix inverse using Cholesky.

---

## LU Decomposition

LU decomposition for solving linear systems Ax = b and computing matrix inverse.

### Functions

#### `compute_lu(matrix)` (nalgebra_lu / ndarray_lu)

Computes the LU decomposition with partial pivoting.

**Returns:** `NalgebraLUResult` or `NdarrayLUResult` with `.l`, `.u`, `.p`

#### `solve(matrix, rhs)` / `inverse(matrix)` / `log_det(matrix)`

Solve linear systems, compute inverse, or log determinant from LU result.

---

## Eigenvalue Decomposition

Symmetric eigenvalue decomposition for real symmetric matrices.

### Functions

#### `compute_symmetric_eigen(matrix)` (nalgebra_eigen / ndarray_eigen)

Computes eigenvalues and eigenvectors of a symmetric matrix.

**Returns:** `NalgebraEigenResult` or `NdarrayEigenResult` with `.eigenvalues`, `.eigenvectors`

#### `compute_generalized_eigen(a, b)`

Solves the generalized eigenvalue problem Av = λBv for symmetric A and positive-definite B.

---

## Statistics

Covariance and correlation for data matrices.

### Functions

#### `column_means(matrix)` / `center_columns(matrix)`

Compute column means and center columns (subtract mean).

#### `covariance_matrix(matrix)` / `correlation_matrix(matrix)`

Compute sample covariance matrix (Bessel correction n-1) and correlation matrix.

**Module:** `nabled::stats::nalgebra_stats` / `ndarray_stats`

---

## PCA

Principal Component Analysis via SVD of centered data.

### Functions

#### `compute_pca(matrix, n_components)`

Computes PCA: centers data, performs SVD, returns components, scores, explained variance.

**Returns:** `NalgebraPCAResult` / `NdarrayPCAResult` with `.components`, `.scores`, `.explained_variance`, `.explained_variance_ratio`

#### `transform(result, data)` / `inverse_transform(result, scores)`

Project data onto principal components or reconstruct from scores.

---

## Linear Regression

Ordinary least squares linear regression via QR decomposition.

### Functions

#### `linear_regression(x, y, intercept)`

Computes OLS coefficients. Set `intercept` true to include intercept term.

**Returns:** `NalgebraRegressionResult` / `NdarrayRegressionResult` with `.coefficients`, `.fitted_values`, `.residuals`, `.r_squared`

---

## Polar Decomposition

Polar decomposition A = UP where U is unitary and P is positive-semidefinite. Uses SVD.

### Functions

#### `compute_polar(matrix)` (nalgebra_polar / ndarray_polar)

**Returns:** `NalgebraPolarResult` / `NdarrayPolarResult` with `.u` (unitary), `.p` (positive-semidefinite)

---

## Schur Decomposition

Schur decomposition: A = QUQ^T where Q is orthogonal and U is upper quasi-triangular.

### Functions

#### `compute_schur(matrix)` (nalgebra_schur / ndarray_schur)

**Returns:** `NalgebraSchurResult` / `NdarraySchurResult` with `.q`, `.t` (quasi-triangular)

---

## Sylvester Equation

Solves the Sylvester equation AX + XB = C.

### Functions

#### `solve_sylvester(a, b, c)` (nalgebra_sylvester / ndarray_sylvester)

Uses Schur decomposition. Matrices A, B, C; returns solution X.

**Requires:** A and B square; dimensions must be compatible.

---

## Orthogonalization

Gram-Schmidt and related orthogonalization procedures.

### Functions

Provides orthogonalization of a set of vectors (e.g., Gram-Schmidt process). See `nabled::orthogonalization` for available functions.

---

## Iterative

Iterative solver configuration and infrastructure.

### `IterativeConfig`

Configuration for iterative methods (max iterations, tolerance, etc.). See `nabled::iterative`.

---

## Triangular

Triangular matrix solve operations.

### Functions

Solves lower/upper triangular systems. See `nabled::triangular` module.

---

## Best Practices

1. **Use appropriate step sizes** for Jacobian computation (typically 1e-6 to 1e-8)
2. **Prefer central differences** for higher accuracy when function evaluation is cheap
3. **Use eigenvalue decomposition** for matrix functions when matrices are symmetric
4. **Check condition numbers** before using SVD for matrix functions
5. **Use QR decomposition** for least squares problems and overdetermined systems
6. **Prefer reduced QR** for rectangular matrices to save memory
7. **Enable column pivoting** for numerically challenging matrices
8. **Handle errors appropriately** - all functions return Results for robust error handling
9. **Consider matrix size** when choosing between nalgebra and ndarray implementations

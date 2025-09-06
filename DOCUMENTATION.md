# Rust Linear Algebra Library - Comprehensive Documentation

This document provides detailed documentation for all modules in the rust-linalg library, including function signatures, parameters, return values, and examples.

## Table of Contents

1. [SVD (Singular Value Decomposition)](#svd-singular-value-decomposition)
2. [Matrix Functions](#matrix-functions)
3. [Jacobian Computation](#jacobian-computation)
4. [Utility Functions](#utility-functions)

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
use rust_linalg::svd::nalgebra_svd;
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
use rust_linalg::matrix_functions::nalgebra_matrix_functions;
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
use rust_linalg::jacobian::JacobianConfig;

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
use rust_linalg::jacobian::nalgebra_jacobian;
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

## Best Practices

1. **Use appropriate step sizes** for Jacobian computation (typically 1e-6 to 1e-8)
2. **Prefer central differences** for higher accuracy when function evaluation is cheap
3. **Use eigenvalue decomposition** for matrix functions when matrices are symmetric
4. **Check condition numbers** before using SVD for matrix functions
5. **Handle errors appropriately** - all functions return Results for robust error handling
6. **Consider matrix size** when choosing between nalgebra and ndarray implementations

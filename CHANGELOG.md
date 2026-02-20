# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- Arrow integration (extracted to separate repo). This repo is a linear algebra library written in Rust.

## [0.3.0] - 2025-02-19

### Added
- **LU Decomposition** - Solve Ax = b and compute matrix inverse
  - Nalgebra and ndarray implementations (ndarray via nalgebra conversion)
  - `compute_lu`, `solve`, `inverse` functions
- **Cholesky Decomposition** - For symmetric positive-definite matrices
  - `compute_cholesky`, `solve`, `inverse`
- **Eigenvalue Decomposition** - Symmetric matrices only
  - `compute_symmetric_eigen` returning eigenvalues and eigenvectors
- **Statistics** - Covariance and correlation
  - `column_means`, `center_columns`, `covariance_matrix`, `correlation_matrix`
  - Bessel correction (n-1) for sample covariance
- **PCA** - Principal Component Analysis
  - `compute_pca`, `transform`, `inverse_transform`
  - Explained variance and explained variance ratio
- **Linear Regression** - Ordinary least squares via QR
  - `linear_regression` with optional intercept, R-squared, residuals
- **Polar Decomposition** - A = UP via SVD
- **Schur Decomposition** - Upper quasi-triangular form
- **Sylvester Equation Solver** - AX + XB = C
- **Orthogonalization** - Gram-Schmidt and related functions
- **Iterative solvers** - IterativeConfig infrastructure
- **Triangular** - Triangular solve operations
- **Dual license** - MIT OR Apache-2.0
- Examples: lu_example, cholesky_example, pca_example, regression_example

### Changed
- Removed ndarray-linalg dependency (avoids LAPACK)
- Bumped version to 0.3.0

### Added (previous)
- **QR Decomposition** - Full and reduced QR decomposition with least squares solving
  - Full QR decomposition using nalgebra's built-in QR algorithm
  - Reduced QR decomposition (economy size) for rectangular matrices
  - QR decomposition with column pivoting for numerical stability
  - Least squares problem solving using QR decomposition
  - Matrix reconstruction from QR components
  - Condition number computation from QR decomposition
  - Support for both nalgebra and ndarray matrices
  - Comprehensive error handling with QRError enum
  - Configurable rank tolerance and parameters via QRConfig
  - **Enhanced Edge Case Handling** - Robust handling of numerical edge cases
    - Empty matrix detection and proper error reporting
    - Zero matrix handling with rank 0 detection
    - Single element matrix optimization
    - NaN and infinity value detection with numerical instability errors
    - Rank-deficient matrix detection and graceful handling
    - Dimension mismatch validation for least squares problems
    - Underdetermined system detection
    - Singular matrix detection in least squares solving
    - Numerical overflow/underflow protection
    - Very small rank tolerance validation
  - Extensive test suite with 10 new tests covering all functionality and edge cases
  - Working example demonstrating all QR features
  - Integration with existing library structure
- **Jacobian Computation** - Numerical differentiation and gradient computation
  - Numerical Jacobian computation using finite differences for both nalgebra and ndarray
  - Forward and central difference methods for higher accuracy
  - Gradient computation for scalar functions
  - Hessian matrix computation (second-order partial derivatives)
  - **Complex Derivatives** - Support for complex-valued functions using complex step method
    - Complex Jacobian computation for vector-valued complex functions
    - Complex gradient computation for scalar complex functions
    - Complex Hessian computation (with limitations noted)
    - Higher accuracy for certain types of functions compared to finite differences
  - Configurable step size and tolerance via JacobianConfig
  - Comprehensive error handling with JacobianError enum
  - Extensive test suite with 12 new tests covering all functionality (including complex derivatives)
  - Working examples demonstrating all Jacobian features (real and complex)
  - Support for both vector-valued and scalar functions (real and complex)

## [0.2.0] - 2024-12-19

### Added
- **Matrix Functions** - Matrix exponential, logarithm, and power operations
  - Matrix exponential using Taylor series and eigenvalue decomposition
  - Matrix logarithm using Taylor series, eigenvalue decomposition, and SVD
  - Matrix power computation using eigenvalue decomposition
  - Support for both nalgebra and ndarray implementations
  - Comprehensive error handling for edge cases (non-square, singular, negative eigenvalues)
  - Extensive test suite with 7 new tests covering all matrix functions
  - Working example demonstrating all matrix function capabilities
- **Enhanced Documentation**
  - Updated library documentation with matrix function examples
  - Added comprehensive usage examples for matrix exponential, logarithm, and power
  - Improved API documentation with inline code examples

## [0.1.0] - 2024-12-19

### Added
- Initial release of rust-linalg library
- **SVD (Singular Value Decomposition)** implementations for both nalgebra and ndarray
  - Full SVD computation
  - Truncated SVD for dimensionality reduction
  - Matrix reconstruction from SVD components
  - Condition number calculation
  - Matrix rank computation
- **Utility functions**
  - Matrix conversion utilities between nalgebra and ndarray
  - Frobenius norm computation
  - Spectral norm computation
  - Matrix comparison utilities
- **Comprehensive testing**
  - Unit tests for all implementations (20 tests total: 5 SVD + 7 matrix functions + 8 Jacobian)
  - Integration tests covering edge cases (12 tests)
  - Test coverage for error handling
  - Matrix reconstruction accuracy verification
- **Performance benchmarking**
  - Comparative benchmarks between nalgebra and ndarray
  - Performance analysis across different matrix sizes (10x10 to 200x200)
  - Full SVD vs truncated SVD performance comparison
- **Documentation and examples**
  - Comprehensive README with usage examples
  - Working example demonstrating library features
  - API documentation with clear function descriptions
  - Inline code examples in library documentation

### Technical Details
- **Dependencies**: nalgebra 0.32, ndarray 0.15, ndarray-linalg 0.16
- **Error Handling**: Robust error handling with custom `SVDError`, `MatrixFunctionError`, and `JacobianError` enums
- **Generic Programming**: Flexible type system supporting different numeric types
- **Memory Safety**: All operations are memory-safe with proper ownership handling
- **Cross-compatibility**: Seamless conversion between nalgebra and ndarray ecosystems

### Performance Benchmarks
- **Small matrices (10x10)**: Nalgebra ~8.6µs, Ndarray ~9.9µs
- **Medium matrices (50x50)**: Nalgebra ~339µs, Ndarray ~353µs  
- **Large matrices (100x100)**: Nalgebra ~2.24ms, Ndarray ~2.37ms
- **Very large matrices (200x200)**: Nalgebra ~15.4ms, Ndarray ~16.0ms

[Unreleased]: https://github.com/NiklausParcell/rust-linearalgebra-better/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NiklausParcell/rust-linearalgebra-better/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NiklausParcell/rust-linearalgebra-better/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NiklausParcell/rust-linearalgebra-better/releases/tag/v0.1.0

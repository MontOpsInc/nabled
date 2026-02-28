# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Workspace-oriented structure with focused crates:
  - `nabled-core`
  - `nabled-linalg`
  - `nabled-ml`
- ndarray-native implementations across decomposition and matrix-function modules.
- Shared internal numerical helpers for LU/QR/Jacobi eigen routines and validation.

### Changed
- Standardized public APIs on `ndarray` data structures.
- Updated tests, examples, and benchmarks to the ndarray-first surface.
- Updated benchmark reporting groups to match current ndarray modules.

### Removed
- Legacy backend/interop pathways that depended on cross-library matrix conversion.
- Deprecated utility and backend modules superseded by ndarray-native kernels.

## [0.3.0] - 2025-02-19

### Added
- LU decomposition and solve/inverse APIs.
- Cholesky decomposition for SPD matrices.
- Symmetric/generalized eigen decomposition APIs.
- Statistics, PCA, regression, polar, schur, sylvester, orthogonalization, iterative, and triangular modules.
- Expanded examples and test coverage.

### Changed
- Broader decomposition and solver coverage across the library surface.

## [0.2.0] - 2024-12-19

### Added
- Matrix function APIs (exponential, logarithm, power).
- Expanded documentation and usage examples.

## [0.1.0] - 2024-12-19

### Added
- Initial release with SVD, supporting matrix operations, tests, and benchmarking.

[Unreleased]: https://github.com/MontOpsInc/nabled/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MontOpsInc/nabled/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MontOpsInc/nabled/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MontOpsInc/nabled/releases/tag/v0.1.0

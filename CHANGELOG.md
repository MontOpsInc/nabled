# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- QR decomposition implementations
- LU decomposition implementations
- Eigenvalue decomposition
- GPU acceleration support

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
  - Unit tests for both implementations (5 tests)
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
- **Error Handling**: Robust error handling with custom `SVDError` enum
- **Generic Programming**: Flexible type system supporting different numeric types
- **Memory Safety**: All operations are memory-safe with proper ownership handling
- **Cross-compatibility**: Seamless conversion between nalgebra and ndarray ecosystems

### Performance Benchmarks
- **Small matrices (10x10)**: Nalgebra ~8.6µs, Ndarray ~9.9µs
- **Medium matrices (50x50)**: Nalgebra ~339µs, Ndarray ~353µs  
- **Large matrices (100x100)**: Nalgebra ~2.24ms, Ndarray ~2.37ms
- **Very large matrices (200x200)**: Nalgebra ~15.4ms, Ndarray ~16.0ms

[Unreleased]: https://github.com/NiklausParcell/rust-linearalgebra-better/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NiklausParcell/rust-linearalgebra-better/releases/tag/v0.1.0

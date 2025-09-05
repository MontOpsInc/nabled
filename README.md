# Rust Linear Algebra Library

Advanced linear algebra functions built on top of `nalgebra` and `ndarray` crates in Rust. This library provides enhanced implementations of common linear algebra operations with focus on performance and numerical stability.

## Features

- **Singular Value Decomposition (SVD)** implementations using both `nalgebra` and `ndarray`
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
rust-linalg = "0.1.0"
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

## Examples

Run the example:

```bash
cargo run --example svd_example
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

The library provides comprehensive error handling through the `SVDError` enum:

```rust
pub enum SVDError {
    NotSquare,
    EmptyMatrix,
    ConvergenceFailed,
    InvalidInput(String),
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

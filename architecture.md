# Architecture

This is a linear algebra library written in Rust. It provides dual backends (nalgebra and ndarray) and operates solely on matrix types from those crates. Data formats (Arrow, Lance, etc.) can sit on top via separate integrations in other repositories.

## Module Dependency Diagram

```mermaid
flowchart TB
    subgraph External [External Crates]
        nalgebra[nalgebra]
        ndarray[ndarray]
    end

    subgraph Core [Core Decompositions]
        svd[SVD]
        qr[QR]
        lu[LU]
        cholesky[Cholesky]
        eigen[Eigen]
        schur[Schur]
    end

    subgraph HigherLevel [Higher-Level Modules]
        pca[PCA]
        regression[Regression]
        polar[Polar]
        sylvester[Sylvester]
    end

    subgraph Utils [Utils and Support]
        stats[Stats]
        utils[Utils]
        matrix_functions[Matrix Functions]
        jacobian[Jacobian]
        orthogonalization[Orthogonalization]
        triangular[Triangular]
        iterative[Iterative]
    end

    pca --> stats
    pca --> svd
    regression --> qr
    polar --> svd
    sylvester --> schur
```

## Data Flow

Matrices flow from nalgebra (`DMatrix`, `DVector`) or ndarray (`Array2`, `Array1`) into decomposition modules. Results are returned as nalgebra or ndarray types. The library does not depend on any data format; conversions happen in calling code or in separate integration crates.

## File Reference

| Module | Source File |
|--------|-------------|
| cholesky | `src/cholesky.rs` |
| eigen | `src/eigen.rs` |
| iterative | `src/iterative.rs` |
| jacobian | `src/jacobian.rs` |
| lu | `src/lu.rs` |
| matrix_functions | `src/matrix_functions.rs` |
| orthogonalization | `src/orthogonalization.rs` |
| pca | `src/pca.rs` |
| polar | `src/polar.rs` |
| qr | `src/qr.rs` |
| regression | `src/regression.rs` |
| schur | `src/schur.rs` |
| stats | `src/stats.rs` |
| svd | `src/svd.rs` |
| sylvester | `src/sylvester.rs` |
| triangular | `src/triangular.rs` |
| utils | `src/utils.rs` |

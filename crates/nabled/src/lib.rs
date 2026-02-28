//! # nabled
//!
//! Workspace facade crate for ndarray-first numerical modules.

pub use nabled_core::prelude;
pub use nabled_linalg::{
    CholeskyError, EigenError, LUError, LogDetResult, MatrixFunctionError, NdarrayCholeskyResult,
    NdarrayEigenResult, NdarrayGeneralizedEigenResult, NdarrayLUResult, NdarrayPolarResult,
    NdarraySVD, NdarraySchurResult, OrthogonalizationError, PolarError, PseudoInverseConfig,
    QRConfig, QRError, QRResult, SVDError, SchurError, SylvesterError, TriangularError, cholesky,
    eigen, lu, matrix_functions, orthogonalization, polar, qr, schur, svd, sylvester, triangular,
};
pub use nabled_ml::{
    IterativeConfig, IterativeError, JacobianConfig, JacobianError, NdarrayPCAResult,
    NdarrayRegressionResult, PCAError, RegressionError, StatsError, iterative, jacobian, pca,
    regression, stats,
};

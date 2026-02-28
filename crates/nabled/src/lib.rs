//! # nabled
//!
//! Workspace facade crate for ndarray-first numerical modules.

pub use nabled_core::prelude;
pub use nabled_linalg::{
    CholeskyError, CsrMatrix, EigenError, LUError, LogDetResult, MatrixFunctionError,
    MatrixFunctionWorkspace, NdarrayCholeskyResult, NdarrayEigenResult,
    NdarrayGeneralizedEigenResult, NdarrayLUResult, NdarrayPolarResult, NdarraySVD,
    NdarraySchurResult, OrthogonalizationError, PairwiseCosineWorkspace, PolarError,
    PseudoInverseConfig, QRConfig, QRError, QRResult, SVDError, SchurError, SchurWorkspace,
    SparseError, SylvesterError, SylvesterWorkspace, TriangularError, VectorError, cholesky, eigen,
    lu, matrix_functions, orthogonalization, polar, qr, schur, sparse, svd, sylvester, triangular,
    vector,
};
pub use nabled_ml::{
    AdamConfig, IterativeConfig, IterativeError, JacobianConfig, JacobianError, LineSearchConfig,
    NdarrayPCAResult, NdarrayRegressionResult, OptimizationError, PCAError, RegressionError,
    SGDConfig, StatsError, iterative, jacobian, optimization, pca, regression, stats,
};

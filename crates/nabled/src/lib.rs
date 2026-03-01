//! # nabled
//!
//! Workspace facade crate for ndarray-first numerical modules.

pub use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
pub use nabled_core::prelude;
pub use nabled_linalg::{
    AcceleratorError, BackendKind, CholeskyError, CooMatrix, CpuBackend, CscMatrix, CsrMatrix,
    CudaBackend, DistributedBackend, EigenError, JacobiPreconditioner, LUError, LogDetResult,
    MatrixError, MatrixFunctionError, MatrixFunctionWorkspace, NdarrayCholeskyResult,
    NdarrayComplexPolarResult, NdarrayComplexSVD, NdarrayEigenResult,
    NdarrayGeneralizedEigenResult, NdarrayLUResult, NdarrayNonsymmetricEigenResult,
    NdarrayPolarResult, NdarraySVD, NdarraySchurResult, OrthogonalizationError,
    PairwiseCosineWorkspace, PolarError, PseudoInverseConfig, QRConfig, QRError, QRResult,
    SVDError, SchurError, SchurWorkspace, SparseError, SylvesterError, SylvesterWorkspace,
    TensorError, TriangularError, VectorError, accelerator, cholesky, eigen, lu, matrix,
    matrix_functions, orthogonalization, polar, qr, schur, sparse, svd, sylvester, tensor,
    triangular, vector,
};
pub use nabled_ml::{
    AdamConfig, IterativeConfig, IterativeError, JacobianConfig, JacobianError, LineSearchConfig,
    MomentumConfig, NdarrayPCAResult, NdarrayRegressionResult, OptimizationError, PCAError,
    RMSPropConfig, RegressionError, SGDConfig, StatsError, iterative, jacobian, optimization, pca,
    regression, stats,
};

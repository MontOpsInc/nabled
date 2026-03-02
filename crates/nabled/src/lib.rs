//! # nabled
//!
//! Workspace facade crate for ndarray-first numerical modules.

pub use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
pub use nabled_core::prelude;
pub use nabled_linalg::{
    AcceleratorError, BackendKind, CholeskyError, CooMatrix, CpuBackend, CscMatrix, CsrMatrix,
    CudaBackend, DistributedBackend, DistributedConfig, DistributedSchedule, EigenError,
    Hosvd3Result, IC0Factorization, ILDL0Factorization, ILU0Factorization, ILUKConfig,
    ILUKFactorization, ILUTConfig, ILUTFactorization, JacobiPreconditioner, LUError, LogDetResult,
    MatrixError, MatrixFunctionError, MatrixFunctionWorkspace, NdarrayCholeskyResult,
    NdarrayComplexPolarResult, NdarrayComplexSVD, NdarrayEigenResult,
    NdarrayGeneralizedEigenResult, NdarrayLUResult, NdarrayNonsymmetricBiEigenResult,
    NdarrayNonsymmetricEigenResult, NdarrayPolarResult, NdarraySVD, NdarraySchurResult,
    NonsymmetricEigenConfig, OrthogonalizationError, PairwiseCosineWorkspace, PolarError,
    PseudoInverseConfig, QRConfig, QRError, QRResult, SVDError, SchurError, SchurWorkspace,
    SparseError, SparseLUFactorization, SylvesterError, SylvesterWorkspace, TensorError,
    TriangularError, VectorError, accelerator, batched, cholesky, eigen, lu, matrix,
    matrix_functions, orthogonalization, polar, qr, schur, sparse, svd, sylvester, tensor,
    triangular, vector,
};
pub use nabled_ml::{
    AdamConfig, BFGSConfig, IterativeConfig, IterativeError, JacobianConfig, JacobianError,
    LineSearchConfig, MomentumConfig, NdarrayPCAResult, NdarrayRegressionResult, OptimizationError,
    PCAError, ProjectedGradientConfig, RMSPropConfig, RegressionError, SGDConfig, StatsError,
    iterative, jacobian, optimization, pca, regression, stats,
};

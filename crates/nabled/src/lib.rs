//! # nabled
//!
//! Workspace facade crate for ndarray-first numerical modules.

pub use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
pub use nabled_core::prelude;
pub use nabled_linalg::accelerator::backends::{
    AcceleratorError, BackendKind, CpuBackend, CudaBackend,
};
pub use nabled_linalg::cholesky::{CholeskyError, NdarrayCholeskyResult};
pub use nabled_linalg::eigen::{
    EigenError, NdarrayEigenResult, NdarrayGeneralizedEigenResult,
    NdarrayNonsymmetricBiEigenResult, NdarrayNonsymmetricEigenResult, NonsymmetricEigenConfig,
};
pub use nabled_linalg::lu::{LUError, LogDetResult, NdarrayLUResult};
pub use nabled_linalg::matrix::MatrixError;
pub use nabled_linalg::matrix_functions::{MatrixFunctionError, MatrixFunctionWorkspace};
pub use nabled_linalg::orthogonalization::OrthogonalizationError;
pub use nabled_linalg::polar::{NdarrayComplexPolarResult, NdarrayPolarResult, PolarError};
pub use nabled_linalg::qr::{QRConfig, QRError, QRResult};
pub use nabled_linalg::schur::{NdarraySchurResult, SchurError, SchurWorkspace};
pub use nabled_linalg::sparse::{
    CooMatrix, CscMatrix, CsrMatrix, IC0Factorization, ILDL0Factorization, ILU0Factorization,
    ILUKConfig, ILUKFactorization, ILUTConfig, ILUTFactorization, JacobiPreconditioner,
    SparseError, SparseLUFactorization,
};
pub use nabled_linalg::svd::{NdarrayComplexSVD, NdarraySVD, PseudoInverseConfig, SVDError};
pub use nabled_linalg::sylvester::{SylvesterError, SylvesterWorkspace};
pub use nabled_linalg::tensor::{Hosvd3Result, TensorError};
pub use nabled_linalg::triangular::TriangularError;
pub use nabled_linalg::vector::{PairwiseCosineWorkspace, VectorError};
pub use nabled_linalg::{
    accelerator, batched, cholesky, eigen, lu, matrix, matrix_functions, orthogonalization, polar,
    qr, schur, sparse, svd, sylvester, tensor, triangular, vector,
};
pub use nabled_ml::iterative::{IterativeConfig, IterativeError};
pub use nabled_ml::jacobian::{JacobianConfig, JacobianError};
pub use nabled_ml::optimization::{
    AdamConfig, BFGSConfig, LineSearchConfig, MomentumConfig, OptimizationError,
    ProjectedGradientConfig, RMSPropConfig, SGDConfig,
};
pub use nabled_ml::pca::{NdarrayPCAResult, PCAError};
pub use nabled_ml::regression::{NdarrayRegressionResult, RegressionError};
pub use nabled_ml::stats::StatsError;
pub use nabled_ml::{iterative, jacobian, optimization, pca, regression, stats};

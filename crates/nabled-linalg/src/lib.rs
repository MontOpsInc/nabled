//! Ndarray-native linear algebra domain crate.

pub use nabled_core::prelude;

mod internal;

pub mod cholesky;
pub mod eigen;
pub mod lu;
pub mod matrix_functions;
pub mod orthogonalization;
pub mod polar;
pub mod qr;
pub mod schur;
pub mod sparse;
pub mod svd;
pub mod sylvester;
pub mod triangular;
pub mod vector;

pub use cholesky::{CholeskyError, NdarrayCholeskyResult};
pub use eigen::{EigenError, NdarrayEigenResult, NdarrayGeneralizedEigenResult};
pub use lu::{LUError, LogDetResult, NdarrayLUResult};
pub use matrix_functions::{MatrixFunctionError, MatrixFunctionWorkspace};
pub use orthogonalization::OrthogonalizationError;
pub use polar::{NdarrayPolarResult, PolarError};
pub use qr::{QRConfig, QRError, QRResult};
pub use schur::{NdarraySchurResult, SchurError, SchurWorkspace};
pub use sparse::{CsrMatrix, SparseError};
pub use svd::{NdarraySVD, PseudoInverseConfig, SVDError};
pub use sylvester::{SylvesterError, SylvesterWorkspace};
pub use triangular::TriangularError;
pub use vector::{PairwiseCosineWorkspace, VectorError};

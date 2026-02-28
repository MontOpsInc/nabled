//! Ndarray-native ML domain crate.

pub use nabled_core::prelude;
pub use nabled_linalg as linalg;

pub mod iterative;
pub mod jacobian;
pub mod optimization;
pub mod pca;
pub mod regression;
pub mod stats;

pub use iterative::{IterativeConfig, IterativeError};
pub use jacobian::{JacobianConfig, JacobianError};
pub use optimization::{AdamConfig, LineSearchConfig, OptimizationError, SGDConfig};
pub use pca::{NdarrayPCAResult, PCAError};
pub use regression::{NdarrayRegressionResult, RegressionError};
pub use stats::StatsError;

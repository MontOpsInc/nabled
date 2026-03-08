//! Provider-specific integration modules.

#[cfg(feature = "magma-system")]
pub(crate) mod magma;
#[cfg(feature = "magma-system")]
pub(crate) mod magma_runtime;
#[cfg(feature = "magma-system")]
pub(crate) mod magma_sparse;
#[cfg(feature = "magma-system")]
pub(crate) mod policy;

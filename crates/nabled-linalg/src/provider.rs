//! Provider-specific integration modules.

#[cfg(feature = "magma-system")]
pub(crate) mod magma;
#[cfg(feature = "magma-system")]
pub(crate) mod magma_sparse;

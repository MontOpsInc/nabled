use std::fmt;

/// Backend category for compile-time kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// CPU backend.
    Cpu,
    /// GPU backend.
    Cuda,
}

/// Error type for backend orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorError {
    /// Selected backend is not currently available.
    UnsupportedBackend(BackendKind),
    /// Invalid chunking policy.
    InvalidChunkSize,
    /// Matrix dimensions are incompatible.
    DimensionMismatch,
    /// Optional accelerator feature was not enabled at compile time.
    FeatureNotEnabled,
    /// No suitable GPU device was found.
    DeviceUnavailable,
    /// GPU kernel execution failed.
    KernelExecutionFailed,
}

impl fmt::Display for AcceleratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorError::UnsupportedBackend(kind) => {
                write!(f, "backend {kind:?} is not currently available")
            }
            AcceleratorError::InvalidChunkSize => write!(f, "chunk size must be greater than zero"),
            AcceleratorError::DimensionMismatch => {
                write!(f, "matrix dimensions are incompatible")
            }
            AcceleratorError::FeatureNotEnabled => {
                write!(f, "requested accelerator feature is not enabled")
            }
            AcceleratorError::DeviceUnavailable => write!(f, "no suitable GPU device is available"),
            AcceleratorError::KernelExecutionFailed => {
                write!(f, "GPU kernel execution failed")
            }
        }
    }
}

impl std::error::Error for AcceleratorError {}

/// Compile-time backend contract.
pub trait ComputeBackend {
    /// Backend kind.
    const KIND: BackendKind;
}

/// CPU backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    const KIND: BackendKind = BackendKind::Cpu;
}

/// CUDA backend placeholder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    const KIND: BackendKind = BackendKind::Cuda;
}

/// Execute a closure with compile-time backend selection.
///
/// # Errors
/// Returns an error if the selected backend is not currently supported.
pub fn execute<B, T, F>(operation: F) -> Result<T, AcceleratorError>
where
    B: ComputeBackend,
    F: FnOnce() -> T,
{
    match B::KIND {
        BackendKind::Cpu => Ok(operation()),
        BackendKind::Cuda => Err(AcceleratorError::UnsupportedBackend(B::KIND)),
    }
}

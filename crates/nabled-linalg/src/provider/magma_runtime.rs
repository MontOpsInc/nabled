//! Shared MAGMA runtime initialization and device binding helpers.

use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

const MAGMA_SUCCESS: i32 = 0;

static MAGMA_INIT_STATUS: OnceLock<i32> = OnceLock::new();
#[cfg(test)]
static MAGMA_RUNTIME_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
#[inline]
fn mark_magma_runtime_call() {
    let _previous = MAGMA_RUNTIME_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline]
fn mark_magma_runtime_call() {}

#[cfg(test)]
pub(crate) fn reset_magma_runtime_call_count() {
    MAGMA_RUNTIME_CALLS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn magma_runtime_call_count() -> u64 {
    MAGMA_RUNTIME_CALLS.load(Ordering::Relaxed)
}

#[link(name = "magma")]
unsafe extern "C" {
    fn magma_init() -> i32;
    fn magma_getdevice(dev: *mut i32);
    fn magma_setdevice(device: i32);
}

/// Ensure process-global MAGMA runtime is initialized exactly once.
pub(crate) fn ensure_magma_initialized() -> Result<(), &'static str> {
    mark_magma_runtime_call();
    let status = *MAGMA_INIT_STATUS.get_or_init(|| {
        // SAFETY: MAGMA global initialization is process-global and idempotent on success.
        unsafe { magma_init() }
    });
    if status == MAGMA_SUCCESS { Ok(()) } else { Err("provider_init_failed") }
}

/// Bind the current host thread to MAGMA's current device and return the device id.
pub(crate) fn bind_current_thread_to_magma_device() -> Result<i32, &'static str> {
    mark_magma_runtime_call();
    ensure_magma_initialized()?;
    let mut device = 0_i32;
    // SAFETY: `device` is a valid out-pointer and setting the returned device for this thread
    // keeps queue creation/device allocations on a consistent CUDA context.
    unsafe {
        magma_getdevice(&raw mut device);
        magma_setdevice(device);
    }
    Ok(device)
}

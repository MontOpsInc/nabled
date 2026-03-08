//! Provider routing policy for MAGMA-backed decomposition paths.

use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

/// MAGMA-specific decomposition policy.
pub(crate) struct MagmaProviderPolicy;

#[cfg(test)]
static VERIFY_FORCE_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

impl MagmaProviderPolicy {
    const BATCH_MIN_DECOMPOSITION_COUNT: usize = 32;
    const BATCH_MIN_DECOMPOSITION_COUNT_FLOOR: usize = 8;
    const BATCH_MIN_DECOMPOSITION_DIM: usize = 32;
    const BATCH_MIN_DECOMPOSITION_DIM_FLOOR: usize = 16;
    const BATCH_MIN_DECOMPOSITION_WORK: usize = 524_288;
    const BATCH_MIN_DECOMPOSITION_WORK_FLOOR: usize = 8_192;
    const MIN_DECOMPOSITION_DIM: usize = 128;
    const MIN_DECOMPOSITION_DIM_FLOOR: usize = 16;
    const MIN_DECOMPOSITION_WORK: usize = 131_072;
    const MIN_DECOMPOSITION_WORK_FLOOR: usize = 4_096;

    fn env_positive_usize(name: &str) -> Option<usize> {
        let raw = std::env::var(name).ok()?;
        let parsed = raw.parse::<usize>().ok()?;
        (parsed > 0).then_some(parsed)
    }

    fn env_truthy(name: &str) -> bool {
        std::env::var(name).ok().is_some_and(|raw| {
            let value = raw.trim();
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
        })
    }

    #[must_use]
    pub(crate) fn strict_mode() -> bool {
        static VALUE: OnceLock<bool> = OnceLock::new();
        *VALUE.get_or_init(|| Self::env_truthy("NABLED_MAGMA_STRICT"))
    }

    #[must_use]
    pub(crate) fn verify_force_mode() -> bool {
        static VALUE: OnceLock<bool> = OnceLock::new();
        #[cfg(test)]
        {
            match VERIFY_FORCE_OVERRIDE.load(Ordering::Relaxed) {
                0 => return false,
                1 => return true,
                _ => {}
            }
        }
        *VALUE.get_or_init(|| Self::env_truthy("NABLED_MAGMA_VERIFY_FORCE"))
    }

    #[must_use]
    pub(crate) fn fail_fast_mode() -> bool { Self::strict_mode() }

    #[must_use]
    fn min_decomposition_dim() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_MIN_DECOMPOSITION_DIM")
                .unwrap_or(Self::MIN_DECOMPOSITION_DIM)
                .max(Self::MIN_DECOMPOSITION_DIM_FLOOR)
        })
    }

    #[must_use]
    fn min_decomposition_work() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_MIN_DECOMPOSITION_WORK")
                .unwrap_or(Self::MIN_DECOMPOSITION_WORK)
                .max(Self::MIN_DECOMPOSITION_WORK_FLOOR)
        })
    }

    #[must_use]
    fn batch_min_decomposition_count() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_COUNT")
                .unwrap_or(Self::BATCH_MIN_DECOMPOSITION_COUNT)
                .max(Self::BATCH_MIN_DECOMPOSITION_COUNT_FLOOR)
        })
    }

    #[must_use]
    fn batch_min_decomposition_dim() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_DIM")
                .unwrap_or(Self::BATCH_MIN_DECOMPOSITION_DIM)
                .max(Self::BATCH_MIN_DECOMPOSITION_DIM_FLOOR)
        })
    }

    #[must_use]
    fn batch_min_decomposition_work() -> usize {
        static VALUE: OnceLock<usize> = OnceLock::new();
        *VALUE.get_or_init(|| {
            Self::env_positive_usize("NABLED_MAGMA_BATCH_MIN_DECOMPOSITION_WORK")
                .unwrap_or(Self::BATCH_MIN_DECOMPOSITION_WORK)
                .max(Self::BATCH_MIN_DECOMPOSITION_WORK_FLOOR)
        })
    }

    #[must_use]
    pub(crate) fn prefer_decomposition(rows: usize, cols: usize) -> bool {
        if Self::verify_force_mode() {
            return rows > 0 && cols > 0;
        }
        let min_dim = rows.min(cols);
        if min_dim < Self::min_decomposition_dim() {
            return false;
        }

        rows.saturating_mul(cols) >= Self::min_decomposition_work()
    }

    #[must_use]
    pub(crate) fn prefer_batched_decomposition(batch: usize, rows: usize, cols: usize) -> bool {
        if batch == 0 || rows == 0 || cols == 0 {
            return false;
        }
        if Self::verify_force_mode() {
            return true;
        }

        let min_dim = rows.min(cols);
        if min_dim >= Self::min_decomposition_dim() {
            // A few very large matrices are still good MAGMA candidates.
            return true;
        }

        let batch_count_ok = batch >= Self::batch_min_decomposition_count();
        let batch_dim_ok = min_dim >= Self::batch_min_decomposition_dim();
        let work = batch.saturating_mul(rows).saturating_mul(cols);
        let batch_work_ok = work >= Self::batch_min_decomposition_work();

        batch_count_ok && batch_dim_ok && batch_work_ok
    }

    #[cfg(test)]
    pub(crate) fn set_verify_force_override(value: Option<bool>) {
        let encoded = match value {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };
        VERIFY_FORCE_OVERRIDE.store(encoded, Ordering::Relaxed);
    }
}

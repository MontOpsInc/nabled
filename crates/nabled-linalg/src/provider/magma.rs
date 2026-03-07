//! MAGMA provider bindings for CUDA-backed dense decompositions.
//!
//! These kernels are compiled only when `magma-system` is enabled.

use std::ffi::{c_char, c_void};
use std::mem::size_of;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use num_complex::{Complex32, Complex64};

use super::magma_runtime;
use crate::internal::DenseKernelPolicy;

const MAGMA_SUCCESS: i32 = 0;
const MAGMA_LOWER: i32 = 122;
const MAGMA_NO_TRANS: i32 = 111;
const MAGMA_NO_VEC: i32 = 301;
const MAGMA_VEC: i32 = 302;
const MAGMA_ALL_VEC: i32 = 304;

const CALLSITE_FUNC: &[u8] = b"nabled\0";
const CALLSITE_FILE: &[u8] = b"provider/magma.rs\0";

type MagmaQueue = *mut c_void;

#[cfg(test)]
static MAGMA_PROVIDER_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
#[inline]
fn mark_magma_provider_call() {
    let _previous = MAGMA_PROVIDER_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(not(test))]
#[inline]
fn mark_magma_provider_call() {}

#[cfg(test)]
pub(crate) fn reset_magma_provider_call_count() {
    MAGMA_PROVIDER_CALLS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn magma_provider_call_count() -> u64 { MAGMA_PROVIDER_CALLS.load(Ordering::Relaxed) }

#[link(name = "magma")]
unsafe extern "C" {
    fn magma_queue_create_internal(
        device: i32,
        queue_ptr: *mut MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: i32,
    );
    fn magma_queue_destroy_internal(
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: i32,
    );
    fn magma_queue_sync_internal(
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: i32,
    );

    fn magma_malloc(ptr_ptr: *mut *mut c_void, bytes: usize) -> i32;
    fn magma_free_internal(ptr: *mut c_void, func: *const c_char, file: *const c_char, line: i32);

    fn magma_setmatrix_internal(
        m: i32,
        n: i32,
        elem_size: i32,
        h_src: *const c_void,
        lda: i32,
        d_dst: *mut c_void,
        ldd: i32,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: i32,
    );
    fn magma_getmatrix_internal(
        m: i32,
        n: i32,
        elem_size: i32,
        d_src: *const c_void,
        ldd: i32,
        h_dst: *mut c_void,
        ldh: i32,
        queue: MagmaQueue,
        func: *const c_char,
        file: *const c_char,
        line: i32,
    );
    fn magma_sgesv(
        n: i32,
        nrhs: i32,
        a: *mut f32,
        lda: i32,
        ipiv: *mut i32,
        b: *mut f32,
        ldb: i32,
        info: *mut i32,
    );
    fn magma_dgesv(
        n: i32,
        nrhs: i32,
        a: *mut f64,
        lda: i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: i32,
        info: *mut i32,
    );
    fn magma_zgesv(
        n: i32,
        nrhs: i32,
        a: *mut Complex64,
        lda: i32,
        ipiv: *mut i32,
        b: *mut Complex64,
        ldb: i32,
        info: *mut i32,
    );
    fn magma_dsgesv_gpu(
        trans: i32,
        n: i32,
        nrhs: i32,
        d_a: *mut f64,
        ldda: i32,
        ipiv: *mut i32,
        d_ipiv: *mut i32,
        d_b: *mut f64,
        lddb: i32,
        d_x: *mut f64,
        lddx: i32,
        d_workd: *mut f64,
        d_works: *mut f32,
        iter: *mut i32,
        info: *mut i32,
    );
    fn magma_zcgesv_gpu(
        trans: i32,
        n: i32,
        nrhs: i32,
        d_a: *mut Complex64,
        ldda: i32,
        ipiv: *mut i32,
        d_ipiv: *mut i32,
        d_b: *mut Complex64,
        lddb: i32,
        d_x: *mut Complex64,
        lddx: i32,
        d_workd: *mut Complex64,
        d_works: *mut Complex32,
        iter: *mut i32,
        info: *mut i32,
    );

    fn magma_sgetrf(m: i32, n: i32, a: *mut f32, lda: i32, ipiv: *mut i32, info: *mut i32);
    fn magma_dgetrf(m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32, info: *mut i32);
    fn magma_zgetrf(m: i32, n: i32, a: *mut Complex64, lda: i32, ipiv: *mut i32, info: *mut i32);

    fn magma_spotrf(uplo: i32, n: i32, a: *mut f32, lda: i32, info: *mut i32);
    fn magma_dpotrf(uplo: i32, n: i32, a: *mut f64, lda: i32, info: *mut i32);
    fn magma_zpotrf(uplo: i32, n: i32, a: *mut Complex64, lda: i32, info: *mut i32);

    fn magma_sgeqrf(
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        tau: *mut f32,
        work: *mut f32,
        lwork: i32,
        info: *mut i32,
    );
    fn magma_dgeqrf(
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: i32,
        info: *mut i32,
    );
    fn magma_zgeqrf(
        m: i32,
        n: i32,
        a: *mut Complex64,
        lda: i32,
        tau: *mut Complex64,
        work: *mut Complex64,
        lwork: i32,
        info: *mut i32,
    );

    fn magma_sorgqr2(m: i32, n: i32, k: i32, a: *mut f32, lda: i32, tau: *mut f32, info: *mut i32);
    fn magma_dorgqr2(m: i32, n: i32, k: i32, a: *mut f64, lda: i32, tau: *mut f64, info: *mut i32);
    fn magma_zungqr2(
        m: i32,
        n: i32,
        k: i32,
        a: *mut Complex64,
        lda: i32,
        tau: *mut Complex64,
        info: *mut i32,
    );

    fn magma_sgesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        s: *mut f32,
        u: *mut f32,
        ldu: i32,
        vt: *mut f32,
        ldvt: i32,
        work: *mut f32,
        lwork: i32,
        info: *mut i32,
    );
    fn magma_dgesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        s: *mut f64,
        u: *mut f64,
        ldu: i32,
        vt: *mut f64,
        ldvt: i32,
        work: *mut f64,
        lwork: i32,
        info: *mut i32,
    );
    fn magma_zgesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut Complex64,
        lda: i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: i32,
        vt: *mut Complex64,
        ldvt: i32,
        work: *mut Complex64,
        lwork: i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    fn magma_ssyevd(
        jobz: i32,
        uplo: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        w: *mut f32,
        work: *mut f32,
        lwork: i32,
        iwork: *mut i32,
        liwork: i32,
        info: *mut i32,
    );
    fn magma_dsyevd(
        jobz: i32,
        uplo: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        w: *mut f64,
        work: *mut f64,
        lwork: i32,
        iwork: *mut i32,
        liwork: i32,
        info: *mut i32,
    );
    fn magma_zgeev(
        jobvl: i32,
        jobvr: i32,
        n: i32,
        a: *mut Complex64,
        lda: i32,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: i32,
        vr: *mut Complex64,
        ldvr: i32,
        work: *mut Complex64,
        lwork: i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    fn magma_sset_pointer(
        output_array: *mut *mut f32,
        input: *mut f32,
        lda: i32,
        row: i32,
        column: i32,
        batch_offset: i32,
        batch_count: i32,
        queue: MagmaQueue,
    );
    fn magma_dset_pointer(
        output_array: *mut *mut f64,
        input: *mut f64,
        lda: i32,
        row: i32,
        column: i32,
        batch_offset: i32,
        batch_count: i32,
        queue: MagmaQueue,
    );
    fn magma_sgetrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut f32,
        lda: i32,
        ipiv_array: *mut *mut i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dgetrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut f64,
        lda: i32,
        ipiv_array: *mut *mut i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;

    fn magma_spotrf_batched(
        uplo: i32,
        n: i32,
        d_a_array: *mut *mut f32,
        lda: i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dpotrf_batched(
        uplo: i32,
        n: i32,
        d_a_array: *mut *mut f64,
        lda: i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;

    fn magma_sgeqrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut f32,
        lda: i32,
        dtau_array: *mut *mut f32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dgeqrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut f64,
        lda: i32,
        dtau_array: *mut *mut f64,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
}

#[allow(unnameable_types)]
#[doc(hidden)]
#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
pub trait MagmaReal: NabledReal + Copy + 'static {
    #[allow(clippy::too_many_arguments)]
    fn magma_gesv(
        n: i32,
        nrhs: i32,
        a: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        b: *mut Self,
        ldb: i32,
        info: *mut i32,
    );
    fn magma_getrf(m: i32, n: i32, a: *mut Self, lda: i32, ipiv: *mut i32, info: *mut i32);
    fn magma_potrf(uplo: i32, n: i32, a: *mut Self, lda: i32, info: *mut i32);
    fn magma_geqrf(
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    );
    fn magma_orgqr2(m: i32, n: i32, k: i32, a: *mut Self, lda: i32, tau: *mut Self, info: *mut i32);
    fn magma_gesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn magma_syevd(
        jobz: i32,
        uplo: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        iwork: *mut i32,
        liwork: i32,
        info: *mut i32,
    );
}

#[allow(unnameable_types)]
#[doc(hidden)]
#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
pub trait MagmaRealBatched: MagmaReal {
    fn magma_set_pointer(
        output_array: *mut *mut Self,
        input: *mut Self,
        lda: i32,
        row: i32,
        column: i32,
        batch_offset: i32,
        batch_count: i32,
        queue: MagmaQueue,
    );
    fn magma_getrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        ipiv_array: *mut *mut i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_potrf_batched(
        uplo: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_geqrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        dtau_array: *mut *mut Self,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32;
}

type LuDecomposition<T> = (Array2<T>, Array2<T>, Vec<usize>, i8);

#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
impl MagmaReal for f32 {
    fn magma_gesv(
        n: i32,
        nrhs: i32,
        a: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        b: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_sgesv(n, nrhs, a, lda, ipiv, b, ldb, info) };
    }

    fn magma_getrf(m: i32, n: i32, a: *mut Self, lda: i32, ipiv: *mut i32, info: *mut i32) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_sgetrf(m, n, a, lda, ipiv, info) };
    }

    fn magma_potrf(uplo: i32, n: i32, a: *mut Self, lda: i32, info: *mut i32) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_spotrf(uplo, n, a, lda, info) };
    }

    fn magma_geqrf(
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_sgeqrf(m, n, a, lda, tau, work, lwork, info) };
    }

    fn magma_orgqr2(
        m: i32,
        n: i32,
        k: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_sorgqr2(m, n, k, a, lda, tau, info) };
    }

    fn magma_gesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) };
    }

    fn magma_syevd(
        jobz: i32,
        uplo: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        iwork: *mut i32,
        liwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_ssyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info) };
    }
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
impl MagmaRealBatched for f32 {
    fn magma_set_pointer(
        output_array: *mut *mut Self,
        input: *mut Self,
        lda: i32,
        row: i32,
        column: i32,
        batch_offset: i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_sset_pointer(
                output_array,
                input,
                lda,
                row,
                column,
                batch_offset,
                batch_count,
                queue,
            );
        };
    }

    fn magma_getrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        ipiv_array: *mut *mut i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_sgetrf_batched(m, n, d_a_array, lda, ipiv_array, info_array, batch_count, queue)
        }
    }

    fn magma_potrf_batched(
        uplo: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_spotrf_batched(uplo, n, d_a_array, lda, info_array, batch_count, queue) }
    }

    fn magma_geqrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        dtau_array: *mut *mut Self,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_sgeqrf_batched(m, n, d_a_array, lda, dtau_array, info_array, batch_count, queue)
        }
    }
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
impl MagmaReal for f64 {
    fn magma_gesv(
        n: i32,
        nrhs: i32,
        a: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        b: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dgesv(n, nrhs, a, lda, ipiv, b, ldb, info) };
    }

    fn magma_getrf(m: i32, n: i32, a: *mut Self, lda: i32, ipiv: *mut i32, info: *mut i32) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dgetrf(m, n, a, lda, ipiv, info) };
    }

    fn magma_potrf(uplo: i32, n: i32, a: *mut Self, lda: i32, info: *mut i32) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dpotrf(uplo, n, a, lda, info) };
    }

    fn magma_geqrf(
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dgeqrf(m, n, a, lda, tau, work, lwork, info) };
    }

    fn magma_orgqr2(
        m: i32,
        n: i32,
        k: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dorgqr2(m, n, k, a, lda, tau, info) };
    }

    fn magma_gesvd(
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) };
    }

    fn magma_syevd(
        jobz: i32,
        uplo: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        iwork: *mut i32,
        liwork: i32,
        info: *mut i32,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info) };
    }
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments, clippy::similar_names)]
impl MagmaRealBatched for f64 {
    fn magma_set_pointer(
        output_array: *mut *mut Self,
        input: *mut Self,
        lda: i32,
        row: i32,
        column: i32,
        batch_offset: i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_dset_pointer(
                output_array,
                input,
                lda,
                row,
                column,
                batch_offset,
                batch_count,
                queue,
            );
        };
    }

    fn magma_getrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        ipiv_array: *mut *mut i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_dgetrf_batched(m, n, d_a_array, lda, ipiv_array, info_array, batch_count, queue)
        }
    }

    fn magma_potrf_batched(
        uplo: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe { magma_dpotrf_batched(uplo, n, d_a_array, lda, info_array, batch_count, queue) }
    }

    fn magma_geqrf_batched(
        m: i32,
        n: i32,
        d_a_array: *mut *mut Self,
        lda: i32,
        dtau_array: *mut *mut Self,
        info_array: *mut i32,
        batch_count: i32,
        queue: MagmaQueue,
    ) -> i32 {
        // SAFETY: Call sites guarantee valid dimensions and pointers.
        unsafe {
            magma_dgeqrf_batched(m, n, d_a_array, lda, dtau_array, info_array, batch_count, queue)
        }
    }
}

fn ensure_magma_initialized() -> Result<(), &'static str> {
    mark_magma_provider_call();
    magma_runtime::ensure_magma_initialized()
}

#[inline]
fn callsite_func() -> *const c_char { CALLSITE_FUNC.as_ptr().cast() }

#[inline]
fn callsite_file() -> *const c_char { CALLSITE_FILE.as_ptr().cast() }

#[inline]
fn magma_trace_alloc_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NABLED_MAGMA_TRACE_ALLOC").ok().is_some_and(|raw| {
            let value = raw.trim();
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
        })
    })
}

#[inline]
fn trace_magma_alloc(message: &str) {
    if magma_trace_alloc_enabled() {
        eprintln!("{message}");
    }
}

struct MagmaQueueGuard {
    raw: MagmaQueue,
}

impl MagmaQueueGuard {
    fn new() -> Result<Self, &'static str> {
        let device = magma_runtime::bind_current_thread_to_magma_device()?;
        let mut queue = std::ptr::null_mut();
        // SAFETY: Queue creation with valid device id and output pointer.
        unsafe {
            magma_queue_create_internal(
                device,
                &raw mut queue,
                callsite_func(),
                callsite_file(),
                0,
            );
        };
        if queue.is_null() {
            return Err("provider_init_failed");
        }
        Ok(Self { raw: queue })
    }

    #[inline]
    fn as_raw(&self) -> MagmaQueue { self.raw }

    fn sync(&self) {
        // SAFETY: Queue handle is valid for lifetime of this guard.
        unsafe { magma_queue_sync_internal(self.raw, callsite_func(), callsite_file(), 0) };
    }
}

impl Drop for MagmaQueueGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: Queue handle is valid and is released once in Drop.
            unsafe { magma_queue_destroy_internal(self.raw, callsite_func(), callsite_file(), 0) };
        }
    }
}

struct MagmaDeviceMem {
    ptr: *mut c_void,
}

impl MagmaDeviceMem {
    fn alloc(bytes: usize) -> Result<Self, &'static str> { Self::alloc_labeled("unnamed", bytes) }

    fn alloc_labeled(label: &str, bytes: usize) -> Result<Self, &'static str> {
        let requested = bytes.max(1);
        let _device = magma_runtime::bind_current_thread_to_magma_device()?;
        trace_magma_alloc(&format!("[magma-alloc] begin label={label} bytes={requested}"));
        let mut ptr = std::ptr::null_mut();
        // SAFETY: MAGMA allocator writes a device allocation pointer on success.
        let status = unsafe { magma_malloc(&raw mut ptr, requested) };
        if status != MAGMA_SUCCESS || ptr.is_null() {
            trace_magma_alloc(&format!(
                "[magma-alloc] fail label={label} bytes={requested} status={status} ptr={ptr:p}"
            ));
            return Err("provider_alloc_failed");
        }
        trace_magma_alloc(&format!("[magma-alloc] ok label={label} bytes={requested} ptr={ptr:p}"));
        Ok(Self { ptr })
    }

    #[inline]
    fn as_ptr<T>(&self) -> *mut T { self.ptr.cast() }
}

impl Drop for MagmaDeviceMem {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: Device pointer is valid and released once in Drop.
            unsafe { magma_free_internal(self.ptr, callsite_func(), callsite_file(), 0) };
        }
    }
}

#[inline]
fn as_i32(value: usize) -> Result<i32, &'static str> {
    i32::try_from(value).map_err(|_| "invalid_dimensions")
}

#[inline]
fn workspace_len_from_query<T: NabledReal>(query: T) -> Result<usize, &'static str> {
    let as_f64 = query.to_f64().ok_or("invalid_dimensions")?;
    if !as_f64.is_finite() || as_f64 < 1.0 {
        return Ok(1);
    }
    let rounded = as_f64.ceil();
    let rounded_u128 = num_traits::ToPrimitive::to_u128(&rounded).ok_or("invalid_dimensions")?;
    usize::try_from(rounded_u128).map_err(|_| "invalid_dimensions")
}

#[inline]
fn workspace_len_from_complex_query(query: Complex64) -> Result<usize, &'static str> {
    let as_f64 = query.re;
    if !as_f64.is_finite() || as_f64 < 1.0 {
        return Ok(1);
    }
    let rounded = as_f64.ceil();
    let rounded_u128 = num_traits::ToPrimitive::to_u128(&rounded).ok_or("invalid_dimensions")?;
    usize::try_from(rounded_u128).map_err(|_| "invalid_dimensions")
}

#[inline]
fn pivot_to_index(pivot_1_based: i32, n: usize) -> Result<usize, &'static str> {
    if pivot_1_based <= 0 {
        return Err("invalid_input");
    }
    let pivot = usize::try_from(pivot_1_based - 1).map_err(|_| "invalid_input")?;
    if pivot >= n {
        return Err("invalid_input");
    }
    Ok(pivot)
}

fn to_col_major<T: Copy>(matrix: &ArrayView2<'_, T>) -> Vec<T> {
    let (rows, cols) = matrix.dim();
    let mut output = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            output.push(matrix[[i, j]]);
        }
    }
    output
}

fn from_col_major<T: NabledReal>(data: &[T], rows: usize, cols: usize) -> Array2<T> {
    let mut output = Array2::<T>::zeros((rows, cols));
    for j in 0..cols {
        for i in 0..rows {
            output[[i, j]] = data[i + j * rows];
        }
    }
    output
}

fn from_col_major_complex(data: &[Complex64], rows: usize, cols: usize) -> Array2<Complex64> {
    let mut output = Array2::<Complex64>::zeros((rows, cols));
    for j in 0..cols {
        for i in 0..rows {
            output[[i, j]] = data[i + j * rows];
        }
    }
    output
}

fn lower_from_col_major<T: NabledReal>(data: &[T], n: usize) -> Array2<T> {
    let mut output = Array2::<T>::zeros((n, n));
    for j in 0..n {
        for i in j..n {
            output[[i, j]] = data[i + j * n];
        }
    }
    output
}

fn lower_from_col_major_complex(data: &[Complex64], n: usize) -> Array2<Complex64> {
    let mut output = Array2::<Complex64>::zeros((n, n));
    for j in 0..n {
        for i in j..n {
            output[[i, j]] = data[i + j * n];
        }
    }
    output
}

fn validate_batched_square_finite<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<(usize, usize), &'static str> {
    let (batch_count, rows, cols) = matrices.dim();
    if matrices.is_empty() || batch_count == 0 {
        return Err("empty");
    }
    if rows != cols {
        return Err("not_square");
    }
    if matrices.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok((batch_count, rows))
}

fn validate_batched_finite<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<(usize, usize, usize), &'static str> {
    let (batch_count, rows, cols) = matrices.dim();
    if matrices.is_empty() || batch_count == 0 {
        return Err("empty");
    }
    if matrices.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok((batch_count, rows, cols))
}

#[allow(clippy::type_complexity)]
pub(crate) fn qr_decompose_batched<T: MagmaRealBatched>(
    matrices: &ArrayView3<'_, T>,
    rank_tolerance: T,
) -> Result<Vec<(Array2<T>, Array2<T>, usize)>, &'static str> {
    let (batch_count, rows, cols) = validate_batched_finite(matrices)?;
    if rows < cols {
        return Err("unsupported_shape");
    }
    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(qr_decompose(&matrix, rank_tolerance)?);
    }

    Ok(output)
}

#[allow(clippy::type_complexity)]
pub(crate) fn lu_decompose_batched<T: MagmaRealBatched>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<(Array2<T>, Array2<T>)>, &'static str> {
    let (batch_count, _n) = validate_batched_square_finite(matrices)?;
    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        let (l, u, _pivots, _sign) = lu_decompose(&matrix)?;
        output.push((l, u));
    }

    Ok(output)
}

pub(crate) fn cholesky_decompose_batched<T: MagmaRealBatched>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<Array2<T>>, &'static str> {
    let (batch_count, _n) = validate_batched_square_finite(matrices)?;
    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(cholesky_decompose(&matrix)?);
    }
    Ok(output)
}

fn lu_factor_raw<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, Vec<i32>, i8), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut data = to_col_major(matrix);
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;

    T::magma_getrf(n_i32, n_i32, data.as_mut_ptr(), n_i32, pivots.as_mut_ptr(), &raw mut info);

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }

    let mut sign = 1_i8;
    for (k, pivot_1_based) in pivots.iter().enumerate() {
        if pivot_to_index(*pivot_1_based, n)? != k {
            sign = -sign;
        }
    }

    Ok((data, pivots, sign))
}

#[allow(dead_code)]
pub(crate) fn lu_decompose<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<LuDecomposition<T>, &'static str> {
    let (data, pivots_1_based, sign) = lu_factor_raw(matrix)?;
    let n = matrix.nrows();

    let mut permutation: Vec<usize> = (0..n).collect();
    for (k, pivot_1_based) in pivots_1_based.iter().enumerate() {
        let pivot = pivot_to_index(*pivot_1_based, n)?;
        permutation.swap(k, pivot);
    }

    let mut l = Array2::<T>::zeros((n, n));
    let mut u = Array2::<T>::zeros((n, n));
    for i in 0..n {
        l[[i, i]] = T::one();
    }
    for j in 0..n {
        for i in 0..n {
            let value = data[i + j * n];
            if i > j {
                l[[i, j]] = value;
            } else {
                u[[i, j]] = value;
            }
        }
    }

    Ok((l, u, permutation, sign))
}

pub(crate) fn lu_solve<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, &'static str> {
    if rhs.len() != matrix.nrows() {
        return Err("bad_dimensions");
    }
    if matrix.iter().any(|value| !value.is_finite()) || rhs.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut b = rhs.to_vec();
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;

    T::magma_gesv(
        n_i32,
        1_i32,
        a.as_mut_ptr(),
        n_i32,
        pivots.as_mut_ptr(),
        b.as_mut_ptr(),
        n_i32,
        &raw mut info,
    );

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }
    Ok(Array1::from_vec(b))
}

fn validate_mixed_lu_real_inputs(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<(usize, i32), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if rhs.len() != matrix.nrows() {
        return Err("bad_dimensions");
    }
    if matrix.iter().any(|value| !value.is_finite()) || rhs.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok((matrix.nrows(), as_i32(matrix.nrows())?))
}

fn validate_mixed_lu_complex_inputs(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<(usize, i32), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if rhs.len() != matrix.nrows() {
        return Err("bad_dimensions");
    }
    if !matrix_is_finite_complex(matrix) || !vector_is_finite_complex(rhs) {
        return Err("non_finite");
    }
    Ok((matrix.nrows(), as_i32(matrix.nrows())?))
}

fn mixed_lu_iterations(info: i32, iterations: i32) -> Result<usize, &'static str> {
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }
    if iterations < 0 {
        return Err("convergence_failed");
    }
    usize::try_from(iterations).map_err(|_| "invalid_input")
}

pub(crate) fn lu_solve_mixed_f64(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, usize), &'static str> {
    let (n, n_i32) = validate_mixed_lu_real_inputs(matrix, rhs)?;
    let queue = MagmaQueueGuard::new()?;
    let nrhs_i32 = 1_i32;
    let matrix_elems = n.saturating_mul(n);
    let rhs_elems = n;
    let work_len = n.saturating_mul(n.saturating_add(1));
    let elem_size_f64 = i32::try_from(size_of::<f64>()).map_err(|_| "invalid_dimensions")?;

    let a_host = to_col_major(matrix);
    let b_host = rhs.to_vec();
    let device_a = MagmaDeviceMem::alloc(matrix_elems.saturating_mul(size_of::<f64>()))?;
    let device_b = MagmaDeviceMem::alloc(rhs_elems.saturating_mul(size_of::<f64>()))?;
    let device_x = MagmaDeviceMem::alloc(rhs_elems.saturating_mul(size_of::<f64>()))?;
    let device_work_d = MagmaDeviceMem::alloc(work_len.saturating_mul(size_of::<f64>()))?;
    let device_work_s = MagmaDeviceMem::alloc(work_len.saturating_mul(size_of::<f32>()))?;
    let device_pivots = MagmaDeviceMem::alloc(n.saturating_mul(size_of::<i32>()))?;

    // SAFETY: Source and destination buffers are valid and sized for one square matrix.
    unsafe {
        magma_setmatrix_internal(
            n_i32,
            n_i32,
            elem_size_f64,
            a_host.as_ptr().cast::<c_void>(),
            n_i32,
            device_a.as_ptr::<f64>().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
        magma_setmatrix_internal(
            n_i32,
            nrhs_i32,
            elem_size_f64,
            b_host.as_ptr().cast::<c_void>(),
            n_i32,
            device_b.as_ptr::<f64>().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
    }

    let mut pivots = vec![0_i32; n];
    let mut iterations = 0_i32;
    let mut info = 0_i32;

    // SAFETY: Pointers refer to valid device/host buffers matching MAGMA's mixed-solve contract.
    unsafe {
        magma_dsgesv_gpu(
            MAGMA_NO_TRANS,
            n_i32,
            nrhs_i32,
            device_a.as_ptr::<f64>(),
            n_i32,
            pivots.as_mut_ptr(),
            device_pivots.as_ptr::<i32>(),
            device_b.as_ptr::<f64>(),
            n_i32,
            device_x.as_ptr::<f64>(),
            n_i32,
            device_work_d.as_ptr::<f64>(),
            device_work_s.as_ptr::<f32>(),
            &raw mut iterations,
            &raw mut info,
        );
    }
    queue.sync();
    let refinement_iterations = mixed_lu_iterations(info, iterations)?;

    let mut solution = vec![0.0_f64; rhs_elems];
    // SAFETY: Device output vector is valid and copied into a host buffer of matching size.
    unsafe {
        magma_getmatrix_internal(
            n_i32,
            nrhs_i32,
            elem_size_f64,
            device_x.as_ptr::<f64>().cast::<c_void>().cast_const(),
            n_i32,
            solution.as_mut_ptr().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
    }
    queue.sync();

    if solution.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }

    Ok((Array1::from_vec(solution), refinement_iterations))
}

pub(crate) fn lu_solve_mixed_complex(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<(Array1<Complex64>, usize), &'static str> {
    let (n, n_i32) = validate_mixed_lu_complex_inputs(matrix, rhs)?;
    let queue = MagmaQueueGuard::new()?;
    let nrhs_i32 = 1_i32;
    let matrix_elems = n.saturating_mul(n);
    let rhs_elems = n;
    let work_len = n.saturating_mul(n.saturating_add(1));
    let elem_size_z = i32::try_from(size_of::<Complex64>()).map_err(|_| "invalid_dimensions")?;

    let a_host = to_col_major(matrix);
    let b_host = rhs.to_vec();
    let device_a = MagmaDeviceMem::alloc(matrix_elems.saturating_mul(size_of::<Complex64>()))?;
    let device_b = MagmaDeviceMem::alloc(rhs_elems.saturating_mul(size_of::<Complex64>()))?;
    let device_x = MagmaDeviceMem::alloc(rhs_elems.saturating_mul(size_of::<Complex64>()))?;
    let device_work_d = MagmaDeviceMem::alloc(work_len.saturating_mul(size_of::<Complex64>()))?;
    let device_work_s = MagmaDeviceMem::alloc(work_len.saturating_mul(size_of::<Complex32>()))?;
    let device_pivots = MagmaDeviceMem::alloc(n.saturating_mul(size_of::<i32>()))?;

    // SAFETY: Source and destination buffers are valid and sized for one square matrix.
    unsafe {
        magma_setmatrix_internal(
            n_i32,
            n_i32,
            elem_size_z,
            a_host.as_ptr().cast::<c_void>(),
            n_i32,
            device_a.as_ptr::<Complex64>().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
        magma_setmatrix_internal(
            n_i32,
            nrhs_i32,
            elem_size_z,
            b_host.as_ptr().cast::<c_void>(),
            n_i32,
            device_b.as_ptr::<Complex64>().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
    }

    let mut pivots = vec![0_i32; n];
    let mut iterations = 0_i32;
    let mut info = 0_i32;

    // SAFETY: Pointers refer to valid device/host buffers matching MAGMA's mixed-solve contract.
    unsafe {
        magma_zcgesv_gpu(
            MAGMA_NO_TRANS,
            n_i32,
            nrhs_i32,
            device_a.as_ptr::<Complex64>(),
            n_i32,
            pivots.as_mut_ptr(),
            device_pivots.as_ptr::<i32>(),
            device_b.as_ptr::<Complex64>(),
            n_i32,
            device_x.as_ptr::<Complex64>(),
            n_i32,
            device_work_d.as_ptr::<Complex64>(),
            device_work_s.as_ptr::<Complex32>(),
            &raw mut iterations,
            &raw mut info,
        );
    }
    queue.sync();
    let refinement_iterations = mixed_lu_iterations(info, iterations)?;

    let mut solution = vec![Complex64::new(0.0, 0.0); rhs_elems];
    // SAFETY: Device output vector is valid and copied into a host buffer of matching size.
    unsafe {
        magma_getmatrix_internal(
            n_i32,
            nrhs_i32,
            elem_size_z,
            device_x.as_ptr::<Complex64>().cast::<c_void>().cast_const(),
            n_i32,
            solution.as_mut_ptr().cast::<c_void>(),
            n_i32,
            queue.as_raw(),
            callsite_func(),
            callsite_file(),
            0,
        );
    }
    queue.sync();

    if solution.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err("non_finite");
    }

    Ok((Array1::from_vec(solution), refinement_iterations))
}

pub(crate) fn lu_inverse<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, &'static str> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("not_square");
    }
    if n == 0 {
        return Err("empty");
    }
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut identity_col_major = vec![T::zero(); n * n];
    for i in 0..n {
        identity_col_major[i + i * n] = T::one();
    }
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;

    T::magma_gesv(
        n_i32,
        n_i32,
        a.as_mut_ptr(),
        n_i32,
        pivots.as_mut_ptr(),
        identity_col_major.as_mut_ptr(),
        n_i32,
        &raw mut info,
    );

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }
    Ok(from_col_major(&identity_col_major, n, n))
}

pub(crate) fn lu_determinant<T: MagmaReal>(matrix: &ArrayView2<'_, T>) -> Result<T, &'static str> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    let (data, _, sign) = lu_factor_raw(matrix)?;
    let n = matrix.nrows();
    let mut determinant = if sign >= 0 { T::one() } else { -T::one() };
    for i in 0..n {
        determinant *= data[i + i * n];
    }
    if !determinant.is_finite() {
        return Err("non_finite");
    }
    Ok(determinant)
}

#[allow(clippy::type_complexity)]
pub(crate) fn qr_decompose<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<(Array2<T>, Array2<T>, usize), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let (rows, cols) = matrix.dim();
    if rows < cols {
        return Err("unsupported_shape");
    }
    let rows_i32 = as_i32(rows)?;
    let cols_i32 = as_i32(cols)?;
    let k_i32 = as_i32(rows.min(cols))?;

    let mut a = to_col_major(matrix);
    let mut tau = vec![T::zero(); rows.min(cols)];
    let mut geqrf_query = vec![T::zero(); 1];
    let mut info = 0_i32;

    T::magma_geqrf(
        rows_i32,
        cols_i32,
        a.as_mut_ptr(),
        rows_i32,
        tau.as_mut_ptr(),
        geqrf_query.as_mut_ptr(),
        -1,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }
    let lwork_geqrf = workspace_len_from_query(geqrf_query[0])?;
    let mut geqrf_work = vec![T::zero(); lwork_geqrf];

    T::magma_geqrf(
        rows_i32,
        cols_i32,
        a.as_mut_ptr(),
        rows_i32,
        tau.as_mut_ptr(),
        geqrf_work.as_mut_ptr(),
        as_i32(lwork_geqrf)?,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let factorized = a.clone();
    T::magma_orgqr2(
        rows_i32,
        cols_i32,
        k_i32,
        a.as_mut_ptr(),
        rows_i32,
        tau.as_mut_ptr(),
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let q = from_col_major(&a, rows, cols);
    let mut r = Array2::<T>::zeros((cols, cols));
    for j in 0..cols {
        for i in 0..=j.min(cols - 1) {
            r[[i, j]] = factorized[i + j * rows];
        }
    }

    let rank = (0..cols).filter(|&index| r[[index, index]].abs() > tolerance).count();
    Ok((q, r, rank))
}

#[allow(clippy::type_complexity)]
#[cfg_attr(feature = "lapack-provider", allow(dead_code))]
pub(crate) fn svd_decompose<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Array2<T>, Array1<T>, Array2<T>), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let (rows, cols) = matrix.dim();
    let keep = rows.min(cols);
    let rows_i32 = as_i32(rows)?;
    let cols_i32 = as_i32(cols)?;

    let mut a = to_col_major(matrix);
    let mut singular_values = vec![T::zero(); keep];
    let mut u = vec![T::zero(); rows * rows];
    let mut vt = vec![T::zero(); cols * cols];
    let mut work_query = vec![T::zero(); 1];
    let mut info = 0_i32;

    T::magma_gesvd(
        MAGMA_ALL_VEC,
        MAGMA_ALL_VEC,
        rows_i32,
        cols_i32,
        a.as_mut_ptr(),
        rows_i32,
        singular_values.as_mut_ptr(),
        u.as_mut_ptr(),
        rows_i32,
        vt.as_mut_ptr(),
        cols_i32,
        work_query.as_mut_ptr(),
        -1,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }

    let lwork = workspace_len_from_query(work_query[0])?;
    let mut work = vec![T::zero(); lwork];
    T::magma_gesvd(
        MAGMA_ALL_VEC,
        MAGMA_ALL_VEC,
        rows_i32,
        cols_i32,
        a.as_mut_ptr(),
        rows_i32,
        singular_values.as_mut_ptr(),
        u.as_mut_ptr(),
        rows_i32,
        vt.as_mut_ptr(),
        cols_i32,
        work.as_mut_ptr(),
        as_i32(lwork)?,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let u_full = from_col_major(&u, rows, rows);
    let vt_full = from_col_major(&vt, cols, cols);
    let u_econ = u_full.slice(ndarray::s![.., ..keep]).to_owned();
    let vt_econ = vt_full.slice(ndarray::s![..keep, ..]).to_owned();
    Ok((u_econ, Array1::from_vec(singular_values), vt_econ))
}

pub(crate) fn symmetric_eigen<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Array1<T>, Array2<T>), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut eigenvalues = vec![T::zero(); n];
    let mut work_query = vec![T::zero(); 1];
    let mut iwork_query = vec![0_i32; 1];
    let mut info = 0_i32;

    T::magma_syevd(
        MAGMA_VEC,
        MAGMA_LOWER,
        n_i32,
        a.as_mut_ptr(),
        n_i32,
        eigenvalues.as_mut_ptr(),
        work_query.as_mut_ptr(),
        -1,
        iwork_query.as_mut_ptr(),
        -1,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }

    let lwork = workspace_len_from_query(work_query[0])?;
    let int_workspace_len =
        usize::try_from(iwork_query[0]).map_err(|_| "invalid_dimensions")?.max(1);
    let mut work = vec![T::zero(); lwork];
    let mut int_workspace = vec![0_i32; int_workspace_len];

    T::magma_syevd(
        MAGMA_VEC,
        MAGMA_LOWER,
        n_i32,
        a.as_mut_ptr(),
        n_i32,
        eigenvalues.as_mut_ptr(),
        work.as_mut_ptr(),
        as_i32(lwork)?,
        int_workspace.as_mut_ptr(),
        as_i32(int_workspace_len)?,
        &raw mut info,
    );
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    Ok((Array1::from_vec(eigenvalues), from_col_major(&a, n, n)))
}

pub(crate) fn cholesky_decompose<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut data = to_col_major(matrix);
    let mut info = 0_i32;
    T::magma_potrf(MAGMA_LOWER, n_i32, data.as_mut_ptr(), n_i32, &raw mut info);

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("not_positive_definite");
    }
    Ok(lower_from_col_major(&data, n))
}

fn cholesky_solve_from_factor<T: NabledReal>(
    lower_factor: &Array2<T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, &'static str> {
    let size = lower_factor.nrows();
    if rhs.len() != size {
        return Err("bad_dimensions");
    }

    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());

    let mut y = Array1::<T>::zeros(size);
    for i in 0..size {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower_factor[[i, j]] * y[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.abs() <= tolerance {
            return Err("not_positive_definite");
        }
        y[i] = sum / diagonal;
    }

    let mut x = Array1::<T>::zeros(size);
    for i_rev in 0..size {
        let i = size - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= lower_factor[[j, i]] * x[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.abs() <= tolerance {
            return Err("not_positive_definite");
        }
        x[i] = sum / diagonal;
    }

    Ok(x)
}

pub(crate) fn cholesky_solve<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, &'static str> {
    let lower = cholesky_decompose(matrix)?;
    cholesky_solve_from_factor(&lower, rhs)
}

pub(crate) fn cholesky_inverse<T: MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, &'static str> {
    let lower = cholesky_decompose(matrix)?;
    let size = lower.nrows();
    let mut inverse = Array2::<T>::zeros((size, size));

    for col in 0..size {
        let mut basis = Array1::<T>::zeros(size);
        basis[col] = T::one();
        let solution = cholesky_solve_from_factor(&lower, &basis.view())?;
        for row in 0..size {
            inverse[[row, col]] = solution[row];
        }
    }

    Ok(inverse)
}

fn matrix_is_finite_complex(matrix: &ArrayView2<'_, Complex64>) -> bool {
    matrix.iter().all(|value| value.re.is_finite() && value.im.is_finite())
}

fn vector_is_finite_complex(vector: &ArrayView1<'_, Complex64>) -> bool {
    vector.iter().all(|value| value.re.is_finite() && value.im.is_finite())
}

fn lu_factor_raw_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Vec<Complex64>, Vec<i32>, i8), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut data = to_col_major(matrix);
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;
    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgetrf(n_i32, n_i32, data.as_mut_ptr(), n_i32, pivots.as_mut_ptr(), &raw mut info);
    };

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }

    let mut sign = 1_i8;
    for (k, pivot_1_based) in pivots.iter().enumerate() {
        if pivot_to_index(*pivot_1_based, n)? != k {
            sign = -sign;
        }
    }

    Ok((data, pivots, sign))
}

pub(crate) fn lu_solve_complex(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, &'static str> {
    if rhs.len() != matrix.nrows() {
        return Err("bad_dimensions");
    }
    if !matrix_is_finite_complex(matrix) || !vector_is_finite_complex(rhs) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("not_square");
    }
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut b = rhs.to_vec();
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgesv(
            n_i32,
            1_i32,
            a.as_mut_ptr(),
            n_i32,
            pivots.as_mut_ptr(),
            b.as_mut_ptr(),
            n_i32,
            &raw mut info,
        );
    };

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }
    Ok(Array1::from_vec(b))
}

pub(crate) fn lu_inverse_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, &'static str> {
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("not_square");
    }
    if n == 0 {
        return Err("empty");
    }
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut identity_col_major = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        identity_col_major[i + i * n] = Complex64::new(1.0, 0.0);
    }
    let mut pivots = vec![0_i32; n];
    let mut info = 0_i32;

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgesv(
            n_i32,
            n_i32,
            a.as_mut_ptr(),
            n_i32,
            pivots.as_mut_ptr(),
            identity_col_major.as_mut_ptr(),
            n_i32,
            &raw mut info,
        );
    };

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("singular");
    }
    Ok(from_col_major_complex(&identity_col_major, n, n))
}

pub(crate) fn lu_determinant_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Complex64, &'static str> {
    let (data, _, sign) = lu_factor_raw_complex(matrix)?;
    let n = matrix.nrows();
    let mut determinant = Complex64::new(f64::from(sign), 0.0);
    for i in 0..n {
        determinant *= data[i + i * n];
    }
    if !determinant.re.is_finite() || !determinant.im.is_finite() {
        return Err("non_finite");
    }
    Ok(determinant)
}

pub(crate) fn cholesky_decompose_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut data = to_col_major(matrix);
    let mut info = 0_i32;
    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe { magma_zpotrf(MAGMA_LOWER, n_i32, data.as_mut_ptr(), n_i32, &raw mut info) };

    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("not_positive_definite");
    }
    Ok(lower_from_col_major_complex(&data, n))
}

fn cholesky_solve_from_factor_complex(
    lower_factor: &Array2<Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, &'static str> {
    let size = lower_factor.nrows();
    if rhs.len() != size {
        return Err("bad_dimensions");
    }

    let mut y = Array1::<Complex64>::zeros(size);
    for i in 0..size {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower_factor[[i, j]] * y[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err("not_positive_definite");
        }
        y[i] = sum / diagonal;
    }

    let mut x = Array1::<Complex64>::zeros(size);
    for i_rev in 0..size {
        let i = size - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= lower_factor[[j, i]].conj() * x[j];
        }
        let diagonal = lower_factor[[i, i]].conj();
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err("not_positive_definite");
        }
        x[i] = sum / diagonal;
    }

    Ok(x)
}

pub(crate) fn cholesky_solve_complex(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, &'static str> {
    if !vector_is_finite_complex(rhs) {
        return Err("non_finite");
    }
    let lower = cholesky_decompose_complex(matrix)?;
    cholesky_solve_from_factor_complex(&lower, rhs)
}

pub(crate) fn cholesky_inverse_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, &'static str> {
    let lower = cholesky_decompose_complex(matrix)?;
    let size = lower.nrows();
    let mut inverse = Array2::<Complex64>::zeros((size, size));

    for col in 0..size {
        let mut basis = Array1::<Complex64>::zeros(size);
        basis[col] = Complex64::new(1.0, 0.0);
        let solution = cholesky_solve_from_factor_complex(&lower, &basis.view())?;
        for row in 0..size {
            inverse[[row, col]] = solution[row];
        }
    }

    Ok(inverse)
}

#[allow(clippy::type_complexity)]
pub(crate) fn qr_decompose_complex(
    matrix: &ArrayView2<'_, Complex64>,
    tolerance: f64,
) -> Result<(Array2<Complex64>, Array2<Complex64>, usize), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let (rows, cols) = matrix.dim();
    if rows < cols {
        return Err("unsupported_shape");
    }
    let rows_i32 = as_i32(rows)?;
    let cols_i32 = as_i32(cols)?;
    let k_i32 = as_i32(rows.min(cols))?;

    let mut a = to_col_major(matrix);
    let mut tau = vec![Complex64::new(0.0, 0.0); rows.min(cols)];
    let mut geqrf_query = vec![Complex64::new(0.0, 0.0); 1];
    let mut info = 0_i32;

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgeqrf(
            rows_i32,
            cols_i32,
            a.as_mut_ptr(),
            rows_i32,
            tau.as_mut_ptr(),
            geqrf_query.as_mut_ptr(),
            -1,
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }
    let lwork_geqrf = workspace_len_from_complex_query(geqrf_query[0])?;
    let mut geqrf_work = vec![Complex64::new(0.0, 0.0); lwork_geqrf];

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgeqrf(
            rows_i32,
            cols_i32,
            a.as_mut_ptr(),
            rows_i32,
            tau.as_mut_ptr(),
            geqrf_work.as_mut_ptr(),
            as_i32(lwork_geqrf)?,
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let factorized = a.clone();
    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zungqr2(
            rows_i32,
            cols_i32,
            k_i32,
            a.as_mut_ptr(),
            rows_i32,
            tau.as_mut_ptr(),
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let q = from_col_major_complex(&a, rows, cols);
    let mut r = Array2::<Complex64>::zeros((cols, cols));
    for j in 0..cols {
        for i in 0..=j.min(cols - 1) {
            r[[i, j]] = factorized[i + j * rows];
        }
    }

    let rank = (0..cols).filter(|&index| r[[index, index]].norm() > tolerance).count();
    Ok((q, r, rank))
}

#[allow(clippy::type_complexity)]
pub(crate) fn svd_decompose_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let (rows, cols) = matrix.dim();
    let keep = rows.min(cols);
    let rows_i32 = as_i32(rows)?;
    let cols_i32 = as_i32(cols)?;

    let mut a = to_col_major(matrix);
    let mut singular_values = vec![0.0_f64; keep];
    let mut u = vec![Complex64::new(0.0, 0.0); rows * rows];
    let mut vt = vec![Complex64::new(0.0, 0.0); cols * cols];
    let mut work_query = vec![Complex64::new(0.0, 0.0); 1];
    let mut rwork_query = vec![0.0_f64; 1];
    let mut info = 0_i32;

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgesvd(
            MAGMA_ALL_VEC,
            MAGMA_ALL_VEC,
            rows_i32,
            cols_i32,
            a.as_mut_ptr(),
            rows_i32,
            singular_values.as_mut_ptr(),
            u.as_mut_ptr(),
            rows_i32,
            vt.as_mut_ptr(),
            cols_i32,
            work_query.as_mut_ptr(),
            -1,
            rwork_query.as_mut_ptr(),
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }

    let lwork = workspace_len_from_complex_query(work_query[0])?;
    let mut work = vec![Complex64::new(0.0, 0.0); lwork];
    let mut rwork = vec![0.0_f64; (5 * keep).max(1)];
    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgesvd(
            MAGMA_ALL_VEC,
            MAGMA_ALL_VEC,
            rows_i32,
            cols_i32,
            a.as_mut_ptr(),
            rows_i32,
            singular_values.as_mut_ptr(),
            u.as_mut_ptr(),
            rows_i32,
            vt.as_mut_ptr(),
            cols_i32,
            work.as_mut_ptr(),
            as_i32(lwork)?,
            rwork.as_mut_ptr(),
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    let u_full = from_col_major_complex(&u, rows, rows);
    let vt_full = from_col_major_complex(&vt, cols, cols);
    let u_econ = u_full.slice(ndarray::s![.., ..keep]).to_owned();
    let vt_econ = vt_full.slice(ndarray::s![..keep, ..]).to_owned();
    Ok((u_econ, Array1::from_vec(singular_values), vt_econ))
}

pub(crate) fn nonsymmetric_eigen_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array1<Complex64>, Array2<Complex64>), &'static str> {
    if matrix.is_empty() {
        return Err("empty");
    }
    if matrix.nrows() != matrix.ncols() {
        return Err("not_square");
    }
    if !matrix_is_finite_complex(matrix) {
        return Err("non_finite");
    }
    ensure_magma_initialized()?;

    let n = matrix.nrows();
    let n_i32 = as_i32(n)?;
    let mut a = to_col_major(matrix);
    let mut eigenvalues = vec![Complex64::new(0.0, 0.0); n];
    let mut left_vectors = vec![Complex64::new(0.0, 0.0); n * n];
    let mut right_vectors = vec![Complex64::new(0.0, 0.0); n * n];
    let mut work_query = vec![Complex64::new(0.0, 0.0); 1];
    let mut rwork_query = vec![0.0_f64; 1];
    let mut info = 0_i32;

    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgeev(
            MAGMA_NO_VEC,
            MAGMA_VEC,
            n_i32,
            a.as_mut_ptr(),
            n_i32,
            eigenvalues.as_mut_ptr(),
            left_vectors.as_mut_ptr(),
            n_i32,
            right_vectors.as_mut_ptr(),
            n_i32,
            work_query.as_mut_ptr(),
            -1,
            rwork_query.as_mut_ptr(),
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }

    let lwork = workspace_len_from_complex_query(work_query[0])?;
    let mut work = vec![Complex64::new(0.0, 0.0); lwork];
    let mut rwork = vec![0.0_f64; (2 * n).max(1)];
    // SAFETY: Call sites guarantee valid dimensions and pointers.
    unsafe {
        magma_zgeev(
            MAGMA_NO_VEC,
            MAGMA_VEC,
            n_i32,
            a.as_mut_ptr(),
            n_i32,
            eigenvalues.as_mut_ptr(),
            left_vectors.as_mut_ptr(),
            n_i32,
            right_vectors.as_mut_ptr(),
            n_i32,
            work.as_mut_ptr(),
            as_i32(lwork)?,
            rwork.as_mut_ptr(),
            &raw mut info,
        );
    };
    if info < 0 {
        return Err("invalid_input");
    }
    if info > 0 {
        return Err("convergence_failed");
    }

    Ok((Array1::from_vec(eigenvalues), from_col_major_complex(&right_vectors, n, n)))
}

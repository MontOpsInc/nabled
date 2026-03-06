//! MAGMA sparse provider bindings used for opt-in sparse acceleration paths.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use ndarray::{Array1, Array2};

const MAGMA_SUCCESS: i32 = 0;
const MAGMA_CSR: i32 = 611;
const MAGMA_DENSE: i32 = 614;
const MAGMA_CPU: i32 = 571;
const MAGMA_DEV: i32 = 572;
const MAGMA_ROW_MAJOR: i32 = 101;
const MAGMA_COL_MAJOR: i32 = 102;

const CALLSITE_FUNC: &[u8] = b"nabled\0";
const CALLSITE_FILE: &[u8] = b"provider/magma_sparse.rs\0";

type MagmaQueue = *mut c_void;

static MAGMA_SPARSE_INIT_STATUS: OnceLock<i32> = OnceLock::new();

#[repr(C)]
#[derive(Clone, Copy)]
struct MagmaDMatrix {
    storage_type:            i32,
    memory_location:         i32,
    sym:                     i32,
    diagorder_type:          i32,
    fill_mode:               i32,
    num_rows:                i32,
    num_cols:                i32,
    nnz:                     i32,
    max_nnz_row:             i32,
    diameter:                i32,
    true_nnz:                i32,
    ownership:               i32,
    val:                     *mut c_void,
    diag:                    *mut c_void,
    row:                     *mut c_void,
    rowidx:                  *mut c_void,
    col:                     *mut c_void,
    list:                    *mut c_void,
    tile_ptr:                *mut c_void,
    tile_desc:               *mut c_void,
    tile_desc_offset_ptr:    *mut c_void,
    tile_desc_offset:        *mut c_void,
    calibrator:              *mut c_void,
    blockinfo:               *mut c_void,
    blocksize:               i32,
    numblocks:               i32,
    alignment:               i32,
    csr5_sigma:              i32,
    csr5_bit_y_offset:       i32,
    csr5_bit_scansum_offset: i32,
    csr5_num_packets:        i32,
    csr5_p:                  i32,
    csr5_num_offsets:        i32,
    csr5_tail_tile_start:    i32,
    major:                   i32,
    ld:                      i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MagmaSMatrix {
    storage_type:            i32,
    memory_location:         i32,
    sym:                     i32,
    diagorder_type:          i32,
    fill_mode:               i32,
    num_rows:                i32,
    num_cols:                i32,
    nnz:                     i32,
    max_nnz_row:             i32,
    diameter:                i32,
    true_nnz:                i32,
    ownership:               i32,
    val:                     *mut c_void,
    diag:                    *mut c_void,
    row:                     *mut c_void,
    rowidx:                  *mut c_void,
    col:                     *mut c_void,
    list:                    *mut c_void,
    tile_ptr:                *mut c_void,
    tile_desc:               *mut c_void,
    tile_desc_offset_ptr:    *mut c_void,
    tile_desc_offset:        *mut c_void,
    calibrator:              *mut c_void,
    blockinfo:               *mut c_void,
    blocksize:               i32,
    numblocks:               i32,
    alignment:               i32,
    csr5_sigma:              i32,
    csr5_bit_y_offset:       i32,
    csr5_bit_scansum_offset: i32,
    csr5_num_packets:        i32,
    csr5_p:                  i32,
    csr5_num_offsets:        i32,
    csr5_tail_tile_start:    i32,
    major:                   i32,
    ld:                      i32,
}

#[link(name = "magma")]
unsafe extern "C" {
    fn magma_init() -> i32;
    fn magma_getdevice(dev: *mut i32);
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
}

#[link(name = "magma_sparse")]
unsafe extern "C" {
    fn magma_dcsrset(
        m: i32,
        n: i32,
        row: *mut i32,
        col: *mut i32,
        val: *mut f64,
        a: *mut MagmaDMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_scsrset(
        m: i32,
        n: i32,
        row: *mut i32,
        col: *mut i32,
        val: *mut f32,
        a: *mut MagmaSMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dvset(m: i32, n: i32, val: *mut f64, v: *mut MagmaDMatrix, queue: MagmaQueue) -> i32;
    fn magma_svset(m: i32, n: i32, val: *mut f32, v: *mut MagmaSMatrix, queue: MagmaQueue) -> i32;
    fn magma_dvget(
        v: MagmaDMatrix,
        m: *mut i32,
        n: *mut i32,
        val: *mut *mut f64,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_svget(
        v: MagmaSMatrix,
        m: *mut i32,
        n: *mut i32,
        val: *mut *mut f32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dmtransfer(
        a: MagmaDMatrix,
        b: *mut MagmaDMatrix,
        src: i32,
        dst: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_smtransfer(
        a: MagmaSMatrix,
        b: *mut MagmaSMatrix,
        src: i32,
        dst: i32,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_d_spmv(
        alpha: f64,
        a: MagmaDMatrix,
        x: MagmaDMatrix,
        beta: f64,
        y: MagmaDMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_s_spmv(
        alpha: f32,
        a: MagmaSMatrix,
        x: MagmaSMatrix,
        beta: f32,
        y: MagmaSMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_d_spmm(
        alpha: f64,
        a: MagmaDMatrix,
        b: MagmaDMatrix,
        c: *mut MagmaDMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_s_spmm(
        alpha: f32,
        a: MagmaSMatrix,
        b: MagmaSMatrix,
        c: *mut MagmaSMatrix,
        queue: MagmaQueue,
    ) -> i32;
    fn magma_dmfree(a: *mut MagmaDMatrix, queue: MagmaQueue) -> i32;
    fn magma_smfree(a: *mut MagmaSMatrix, queue: MagmaQueue) -> i32;
}

fn callsite_func() -> *const c_char { CALLSITE_FUNC.as_ptr().cast::<c_char>() }
fn callsite_file() -> *const c_char { CALLSITE_FILE.as_ptr().cast::<c_char>() }

fn as_i32(value: usize) -> Result<i32, &'static str> {
    i32::try_from(value).map_err(|_| "invalid_dimensions")
}

fn check_status(status: i32) -> Result<(), &'static str> {
    if status == MAGMA_SUCCESS { Ok(()) } else { Err("provider_failure") }
}

fn ensure_magma_sparse_init() -> Result<(), &'static str> {
    let status = *MAGMA_SPARSE_INIT_STATUS.get_or_init(|| {
        // SAFETY: Process-global MAGMA initialization is idempotent on success paths.
        unsafe { magma_init() }
    });
    if status == MAGMA_SUCCESS { Ok(()) } else { Err("provider_init_failed") }
}

struct MagmaQueueGuard {
    queue: MagmaQueue,
}

impl MagmaQueueGuard {
    fn new() -> Result<Self, &'static str> {
        ensure_magma_sparse_init()?;
        let mut device = 0_i32;
        // SAFETY: `device` is a valid out-pointer.
        unsafe {
            magma_getdevice(&raw mut device);
        }
        let mut queue: MagmaQueue = std::ptr::null_mut();
        // SAFETY: Valid current device and out-pointer; callsite pointers are static C strings.
        unsafe {
            magma_queue_create_internal(
                device,
                &raw mut queue,
                callsite_func(),
                callsite_file(),
                0,
            );
        }
        if queue.is_null() { Err("provider_init_failed") } else { Ok(Self { queue }) }
    }

    fn as_raw(&self) -> MagmaQueue { self.queue }

    fn sync(&self) {
        // SAFETY: Queue is valid for the lifetime of this guard.
        unsafe {
            magma_queue_sync_internal(self.queue, callsite_func(), callsite_file(), 0);
        }
    }
}

impl Drop for MagmaQueueGuard {
    fn drop(&mut self) {
        if self.queue.is_null() {
            return;
        }
        // SAFETY: Queue was created by MAGMA and is owned by this guard.
        unsafe {
            magma_queue_destroy_internal(self.queue, callsite_func(), callsite_file(), 0);
        }
    }
}

struct DMatrixHandle {
    matrix:      MagmaDMatrix,
    queue:       MagmaQueue,
    initialized: bool,
}

impl DMatrixHandle {
    fn new(queue: MagmaQueue) -> Self {
        // SAFETY: `MagmaDMatrix` is plain-data in C and zero-init is a valid empty state.
        let matrix = unsafe { std::mem::zeroed::<MagmaDMatrix>() };
        Self { matrix, queue, initialized: false }
    }

    fn as_mut_ptr(&mut self) -> *mut MagmaDMatrix { &raw mut self.matrix }

    fn value(&self) -> MagmaDMatrix { self.matrix }

    fn mark_initialized(&mut self) { self.initialized = true; }
}

impl Drop for DMatrixHandle {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }
        // SAFETY: Matrix is MAGMA-initialized and this handle owns its lifecycle.
        unsafe {
            let _ = magma_dmfree(&raw mut self.matrix, self.queue);
        }
    }
}

struct SMatrixHandle {
    matrix:      MagmaSMatrix,
    queue:       MagmaQueue,
    initialized: bool,
}

impl SMatrixHandle {
    fn new(queue: MagmaQueue) -> Self {
        // SAFETY: `MagmaSMatrix` is plain-data in C and zero-init is a valid empty state.
        let matrix = unsafe { std::mem::zeroed::<MagmaSMatrix>() };
        Self { matrix, queue, initialized: false }
    }

    fn as_mut_ptr(&mut self) -> *mut MagmaSMatrix { &raw mut self.matrix }

    fn value(&self) -> MagmaSMatrix { self.matrix }

    fn mark_initialized(&mut self) { self.initialized = true; }
}

impl Drop for SMatrixHandle {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }
        // SAFETY: Matrix is MAGMA-initialized and this handle owns its lifecycle.
        unsafe {
            let _ = magma_smfree(&raw mut self.matrix, self.queue);
        }
    }
}

fn validate_sparse_structure(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[i32],
    col_indices: &[i32],
    values_len: usize,
) -> Result<(), &'static str> {
    if nrows == 0 || ncols == 0 {
        return Err("empty");
    }
    if row_ptrs.len() != nrows + 1 || col_indices.len() != values_len {
        return Err("invalid_structure");
    }
    if row_ptrs.first().copied().unwrap_or(i32::MIN) != 0 {
        return Err("invalid_structure");
    }

    let nnz_i32 = as_i32(values_len)?;
    if row_ptrs.last().copied().unwrap_or(i32::MIN) != nnz_i32 {
        return Err("invalid_structure");
    }

    for bounds in row_ptrs.windows(2) {
        let start = bounds[0];
        let end = bounds[1];
        if start < 0 || end < start || end > nnz_i32 {
            return Err("invalid_structure");
        }
    }

    let ncols_i32 = as_i32(ncols)?;
    for &column in col_indices {
        if column < 0 || column >= ncols_i32 {
            return Err("invalid_structure");
        }
    }

    Ok(())
}

/// Compute sparse matrix-vector multiplication via MAGMA sparse (`f64`).
///
/// # Errors
/// Returns an error for invalid sparse structure, shape mismatch, non-finite inputs, or provider
/// failures.
#[allow(clippy::too_many_lines)]
pub(crate) fn spmv_f64(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[i32],
    col_indices: &[i32],
    values: &[f64],
    vector: &Array1<f64>,
) -> Result<Array1<f64>, &'static str> {
    validate_sparse_structure(nrows, ncols, row_ptrs, col_indices, values.len())?;
    if vector.len() != ncols {
        return Err("bad_dimensions");
    }
    if values.iter().any(|value| !value.is_finite())
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err("non_finite");
    }

    let nrows_i32 = as_i32(nrows)?;
    let ncols_i32 = as_i32(ncols)?;
    let queue = MagmaQueueGuard::new()?;

    let mut row = row_ptrs.to_vec();
    let mut col = col_indices.to_vec();
    let mut val = values.to_vec();
    let mut x_host = vector.to_vec();
    let mut y_host = vec![0.0_f64; nrows];

    let mut a_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Buffers and out-matrix pointer are valid for the duration of the call.
        unsafe {
            magma_dcsrset(
                nrows_i32,
                ncols_i32,
                row.as_mut_ptr(),
                col.as_mut_ptr(),
                val.as_mut_ptr(),
                a_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    a_cpu.mark_initialized();
    a_cpu.matrix.storage_type = MAGMA_CSR;

    let mut a_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                a_cpu.value(),
                a_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    a_dev.mark_initialized();

    let mut x_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Vector host buffer and output matrix pointer are valid.
        unsafe {
            magma_dvset(ncols_i32, 1, x_host.as_mut_ptr(), x_cpu.as_mut_ptr(), queue.as_raw())
        },
    )?;
    x_cpu.mark_initialized();
    x_cpu.matrix.storage_type = MAGMA_DENSE;
    x_cpu.matrix.major = MAGMA_ROW_MAJOR;

    let mut x_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source vector matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                x_cpu.value(),
                x_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    x_dev.mark_initialized();

    let mut y_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Output host buffer and output matrix pointer are valid.
        unsafe {
            magma_dvset(nrows_i32, 1, y_host.as_mut_ptr(), y_cpu.as_mut_ptr(), queue.as_raw())
        },
    )?;
    y_cpu.mark_initialized();
    y_cpu.matrix.storage_type = MAGMA_DENSE;
    y_cpu.matrix.major = MAGMA_ROW_MAJOR;

    let mut y_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source vector matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                y_cpu.value(),
                y_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    y_dev.mark_initialized();

    check_status(
        // SAFETY: All MAGMA matrices are valid and live for the call duration.
        unsafe {
            magma_d_spmv(1.0, a_dev.value(), x_dev.value(), 0.0, y_dev.value(), queue.as_raw())
        },
    )?;
    queue.sync();

    let mut y_out_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                y_dev.value(),
                y_out_cpu.as_mut_ptr(),
                MAGMA_DEV,
                MAGMA_CPU,
                queue.as_raw(),
            )
        },
    )?;
    y_out_cpu.mark_initialized();

    let mut out_rows = 0_i32;
    let mut out_cols = 0_i32;
    let mut out_ptr: *mut f64 = std::ptr::null_mut();
    check_status(
        // SAFETY: Output matrix is valid and out-pointers are writable.
        unsafe {
            magma_dvget(
                y_out_cpu.value(),
                &raw mut out_rows,
                &raw mut out_cols,
                &raw mut out_ptr,
                queue.as_raw(),
            )
        },
    )?;
    if out_rows != nrows_i32 || out_cols != 1 || out_ptr.is_null() {
        return Err("provider_failure");
    }

    // SAFETY: `out_ptr` is validated non-null and MAGMA reports `out_rows` elements.
    let output = unsafe { std::slice::from_raw_parts(out_ptr, nrows).to_vec() };
    if output.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok(Array1::from_vec(output))
}

/// Compute sparse matrix-vector multiplication via MAGMA sparse (`f32`).
///
/// # Errors
/// Returns an error for invalid sparse structure, shape mismatch, non-finite inputs, or provider
/// failures.
#[allow(clippy::too_many_lines)]
pub(crate) fn spmv_f32(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[i32],
    col_indices: &[i32],
    values: &[f32],
    vector: &Array1<f32>,
) -> Result<Array1<f32>, &'static str> {
    validate_sparse_structure(nrows, ncols, row_ptrs, col_indices, values.len())?;
    if vector.len() != ncols {
        return Err("bad_dimensions");
    }
    if values.iter().any(|value| !value.is_finite())
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err("non_finite");
    }

    let nrows_i32 = as_i32(nrows)?;
    let ncols_i32 = as_i32(ncols)?;
    let queue = MagmaQueueGuard::new()?;

    let mut row = row_ptrs.to_vec();
    let mut col = col_indices.to_vec();
    let mut val = values.to_vec();
    let mut x_host = vector.to_vec();
    let mut y_host = vec![0.0_f32; nrows];

    let mut a_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Buffers and out-matrix pointer are valid for the duration of the call.
        unsafe {
            magma_scsrset(
                nrows_i32,
                ncols_i32,
                row.as_mut_ptr(),
                col.as_mut_ptr(),
                val.as_mut_ptr(),
                a_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    a_cpu.mark_initialized();
    a_cpu.matrix.storage_type = MAGMA_CSR;

    let mut a_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                a_cpu.value(),
                a_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    a_dev.mark_initialized();

    let mut x_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Vector host buffer and output matrix pointer are valid.
        unsafe {
            magma_svset(ncols_i32, 1, x_host.as_mut_ptr(), x_cpu.as_mut_ptr(), queue.as_raw())
        },
    )?;
    x_cpu.mark_initialized();
    x_cpu.matrix.storage_type = MAGMA_DENSE;
    x_cpu.matrix.major = MAGMA_ROW_MAJOR;

    let mut x_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source vector matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                x_cpu.value(),
                x_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    x_dev.mark_initialized();

    let mut y_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Output host buffer and output matrix pointer are valid.
        unsafe {
            magma_svset(nrows_i32, 1, y_host.as_mut_ptr(), y_cpu.as_mut_ptr(), queue.as_raw())
        },
    )?;
    y_cpu.mark_initialized();
    y_cpu.matrix.storage_type = MAGMA_DENSE;
    y_cpu.matrix.major = MAGMA_ROW_MAJOR;

    let mut y_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source vector matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                y_cpu.value(),
                y_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    y_dev.mark_initialized();

    check_status(
        // SAFETY: All MAGMA matrices are valid and live for the call duration.
        unsafe {
            magma_s_spmv(1.0, a_dev.value(), x_dev.value(), 0.0, y_dev.value(), queue.as_raw())
        },
    )?;
    queue.sync();

    let mut y_out_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                y_dev.value(),
                y_out_cpu.as_mut_ptr(),
                MAGMA_DEV,
                MAGMA_CPU,
                queue.as_raw(),
            )
        },
    )?;
    y_out_cpu.mark_initialized();

    let mut out_rows = 0_i32;
    let mut out_cols = 0_i32;
    let mut out_ptr: *mut f32 = std::ptr::null_mut();
    check_status(
        // SAFETY: Output matrix is valid and out-pointers are writable.
        unsafe {
            magma_svget(
                y_out_cpu.value(),
                &raw mut out_rows,
                &raw mut out_cols,
                &raw mut out_ptr,
                queue.as_raw(),
            )
        },
    )?;
    if out_rows != nrows_i32 || out_cols != 1 || out_ptr.is_null() {
        return Err("provider_failure");
    }

    // SAFETY: `out_ptr` is validated non-null and MAGMA reports `out_rows` elements.
    let output = unsafe { std::slice::from_raw_parts(out_ptr, nrows).to_vec() };
    if output.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Ok(Array1::from_vec(output))
}

fn dense_row_major_to_col_major_f64(dense: &Array2<f64>) -> Vec<f64> {
    let mut converted = vec![0.0_f64; dense.len()];
    for col in 0..dense.ncols() {
        for row in 0..dense.nrows() {
            converted[col * dense.nrows() + row] = dense[[row, col]];
        }
    }
    converted
}

fn dense_row_major_to_col_major_f32(dense: &Array2<f32>) -> Vec<f32> {
    let mut converted = vec![0.0_f32; dense.len()];
    for col in 0..dense.ncols() {
        for row in 0..dense.nrows() {
            converted[col * dense.nrows() + row] = dense[[row, col]];
        }
    }
    converted
}

fn dense_col_major_to_row_major_f64(col_major: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut converted = vec![0.0_f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            converted[row * cols + col] = col_major[col * rows + row];
        }
    }
    converted
}

fn dense_col_major_to_row_major_f32(col_major: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut converted = vec![0.0_f32; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            converted[row * cols + col] = col_major[col * rows + row];
        }
    }
    converted
}

/// Compute sparse-dense multiplication via MAGMA sparse (`f64`).
///
/// # Errors
/// Returns an error for invalid sparse structure, shape mismatch, non-finite inputs, or provider
/// failures.
#[allow(clippy::too_many_lines)]
pub(crate) fn spmm_f64(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[i32],
    col_indices: &[i32],
    values: &[f64],
    dense: &Array2<f64>,
) -> Result<Array2<f64>, &'static str> {
    validate_sparse_structure(nrows, ncols, row_ptrs, col_indices, values.len())?;
    if dense.nrows() != ncols {
        return Err("bad_dimensions");
    }
    if values.iter().any(|value| !value.is_finite()) || dense.iter().any(|value| !value.is_finite())
    {
        return Err("non_finite");
    }

    let nrows_i32 = as_i32(nrows)?;
    let ncols_i32 = as_i32(ncols)?;
    let rhs_cols_i32 = as_i32(dense.ncols())?;
    let queue = MagmaQueueGuard::new()?;

    let mut row = row_ptrs.to_vec();
    let mut col = col_indices.to_vec();
    let mut val = values.to_vec();
    let mut b_host = dense_row_major_to_col_major_f64(dense);

    let mut a_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Buffers and out-matrix pointer are valid for the duration of the call.
        unsafe {
            magma_dcsrset(
                nrows_i32,
                ncols_i32,
                row.as_mut_ptr(),
                col.as_mut_ptr(),
                val.as_mut_ptr(),
                a_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    a_cpu.mark_initialized();
    a_cpu.matrix.storage_type = MAGMA_CSR;

    let mut a_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                a_cpu.value(),
                a_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    a_dev.mark_initialized();

    let mut b_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Host dense buffer and output matrix pointer are valid.
        unsafe {
            magma_dvset(
                ncols_i32,
                rhs_cols_i32,
                b_host.as_mut_ptr(),
                b_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    b_cpu.mark_initialized();
    b_cpu.matrix.storage_type = MAGMA_DENSE;
    b_cpu.matrix.major = MAGMA_COL_MAJOR;

    let mut b_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                b_cpu.value(),
                b_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    b_dev.mark_initialized();

    let mut c_dev = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Input matrices are valid and output pointer is writable.
        unsafe {
            magma_d_spmm(1.0, a_dev.value(), b_dev.value(), c_dev.as_mut_ptr(), queue.as_raw())
        },
    )?;
    c_dev.mark_initialized();
    queue.sync();

    let mut c_cpu = DMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_dmtransfer(
                c_dev.value(),
                c_cpu.as_mut_ptr(),
                MAGMA_DEV,
                MAGMA_CPU,
                queue.as_raw(),
            )
        },
    )?;
    c_cpu.mark_initialized();

    let mut out_rows = 0_i32;
    let mut out_cols = 0_i32;
    let mut out_ptr: *mut f64 = std::ptr::null_mut();
    check_status(
        // SAFETY: Output matrix is valid and out-pointers are writable.
        unsafe {
            magma_dvget(
                c_cpu.value(),
                &raw mut out_rows,
                &raw mut out_cols,
                &raw mut out_ptr,
                queue.as_raw(),
            )
        },
    )?;
    if out_rows != nrows_i32 || out_cols != rhs_cols_i32 || out_ptr.is_null() {
        return Err("provider_failure");
    }

    // SAFETY: `out_ptr` is validated non-null and `out_rows*out_cols` elements are available.
    let raw = unsafe {
        std::slice::from_raw_parts(out_ptr, nrows.saturating_mul(dense.ncols())).to_vec()
    };
    let output = dense_col_major_to_row_major_f64(&raw, nrows, dense.ncols());
    if output.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Array2::from_shape_vec((nrows, dense.ncols()), output).map_err(|_| "provider_failure")
}

/// Compute sparse-dense multiplication via MAGMA sparse (`f32`).
///
/// # Errors
/// Returns an error for invalid sparse structure, shape mismatch, non-finite inputs, or provider
/// failures.
#[allow(clippy::too_many_lines)]
pub(crate) fn spmm_f32(
    nrows: usize,
    ncols: usize,
    row_ptrs: &[i32],
    col_indices: &[i32],
    values: &[f32],
    dense: &Array2<f32>,
) -> Result<Array2<f32>, &'static str> {
    validate_sparse_structure(nrows, ncols, row_ptrs, col_indices, values.len())?;
    if dense.nrows() != ncols {
        return Err("bad_dimensions");
    }
    if values.iter().any(|value| !value.is_finite()) || dense.iter().any(|value| !value.is_finite())
    {
        return Err("non_finite");
    }

    let nrows_i32 = as_i32(nrows)?;
    let ncols_i32 = as_i32(ncols)?;
    let rhs_cols_i32 = as_i32(dense.ncols())?;
    let queue = MagmaQueueGuard::new()?;

    let mut row = row_ptrs.to_vec();
    let mut col = col_indices.to_vec();
    let mut val = values.to_vec();
    let mut b_host = dense_row_major_to_col_major_f32(dense);

    let mut a_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Buffers and out-matrix pointer are valid for the duration of the call.
        unsafe {
            magma_scsrset(
                nrows_i32,
                ncols_i32,
                row.as_mut_ptr(),
                col.as_mut_ptr(),
                val.as_mut_ptr(),
                a_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    a_cpu.mark_initialized();
    a_cpu.matrix.storage_type = MAGMA_CSR;

    let mut a_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                a_cpu.value(),
                a_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    a_dev.mark_initialized();

    let mut b_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Host dense buffer and output matrix pointer are valid.
        unsafe {
            magma_svset(
                ncols_i32,
                rhs_cols_i32,
                b_host.as_mut_ptr(),
                b_cpu.as_mut_ptr(),
                queue.as_raw(),
            )
        },
    )?;
    b_cpu.mark_initialized();
    b_cpu.matrix.storage_type = MAGMA_DENSE;
    b_cpu.matrix.major = MAGMA_COL_MAJOR;

    let mut b_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                b_cpu.value(),
                b_dev.as_mut_ptr(),
                MAGMA_CPU,
                MAGMA_DEV,
                queue.as_raw(),
            )
        },
    )?;
    b_dev.mark_initialized();

    let mut c_dev = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Input matrices are valid and output pointer is writable.
        unsafe {
            magma_s_spmm(1.0, a_dev.value(), b_dev.value(), c_dev.as_mut_ptr(), queue.as_raw())
        },
    )?;
    c_dev.mark_initialized();
    queue.sync();

    let mut c_cpu = SMatrixHandle::new(queue.as_raw());
    check_status(
        // SAFETY: Source matrix is valid and destination pointer is writable.
        unsafe {
            magma_smtransfer(
                c_dev.value(),
                c_cpu.as_mut_ptr(),
                MAGMA_DEV,
                MAGMA_CPU,
                queue.as_raw(),
            )
        },
    )?;
    c_cpu.mark_initialized();

    let mut out_rows = 0_i32;
    let mut out_cols = 0_i32;
    let mut out_ptr: *mut f32 = std::ptr::null_mut();
    check_status(
        // SAFETY: Output matrix is valid and out-pointers are writable.
        unsafe {
            magma_svget(
                c_cpu.value(),
                &raw mut out_rows,
                &raw mut out_cols,
                &raw mut out_ptr,
                queue.as_raw(),
            )
        },
    )?;
    if out_rows != nrows_i32 || out_cols != rhs_cols_i32 || out_ptr.is_null() {
        return Err("provider_failure");
    }

    // SAFETY: `out_ptr` is validated non-null and `out_rows*out_cols` elements are available.
    let raw = unsafe {
        std::slice::from_raw_parts(out_ptr, nrows.saturating_mul(dense.ncols())).to_vec()
    };
    let output = dense_col_major_to_row_major_f32(&raw, nrows, dense.ncols());
    if output.iter().any(|value| !value.is_finite()) {
        return Err("non_finite");
    }
    Array2::from_shape_vec((nrows, dense.ncols()), output).map_err(|_| "provider_failure")
}

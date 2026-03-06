//! Runtime GPU routing thresholds for `GpuBackend` kernel dispatch.
//!
//! Compile-time backend selection stays unchanged: callers pick `CpuBackend` or `GpuBackend`.
//! This policy only decides whether `GpuBackend` should attempt GPU execution for a specific
//! workload size, or route directly to CPU for the same backend path.
//!
//! Environment overrides:
//! - `NABLED_GPU_FORCE` / `NABLED_GPU_DISABLE`
//! - `NABLED_GPU_MIN_MATMAT_FLOPS`
//! - `NABLED_GPU_MIN_MATVEC_FLOPS`
//! - `NABLED_GPU_MIN_BATCHED_MATMAT_FLOPS`
//! - `NABLED_GPU_MIN_BATCHED_ROW_MATVEC_FLOPS`
//! - `NABLED_GPU_MIN_DOT_LEN`
//! - `NABLED_GPU_MIN_PAIRWISE_FLOPS`
//! - `NABLED_GPU_MIN_SPARSE_MATVEC_WORK`
//! - `NABLED_GPU_MIN_SPARSE_MATMAT_DENSE_WORK`
//! - `NABLED_GPU_MIN_SPARSE_MATMAT_SPARSE_NNZ`
//! - `NABLED_GPU_MIN_TRIANGULAR_SOLVE_VEC_FLOPS`
//! - `NABLED_GPU_MIN_TRIANGULAR_SOLVE_MAT_FLOPS`
//! - `NABLED_GPU_MIN_TENSOR_CONTRACT_FLOPS`
//! - `NABLED_GPU_MIN_TENSOR_BATCHED_MATMUL_FLOPS`
//! - `NABLED_GPU_MIN_TENSOR_SUM_WORK`

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuRoutingMode {
    Auto,
    ForceGpu,
    ForceCpu,
}

#[derive(Debug, Clone, Copy)]
struct GpuRoutingPolicy {
    mode:                            GpuRoutingMode,
    matmat_min_flops:                usize,
    matvec_min_flops:                usize,
    batched_matmat_min_flops:        usize,
    batched_row_matvec_min_flops:    usize,
    dot_min_len:                     usize,
    pairwise_min_flops:              usize,
    sparse_matvec_min_work:          usize,
    sparse_matmat_dense_min_work:    usize,
    sparse_matmat_sparse_min_nnz:    usize,
    triangular_solve_vec_min_flops:  usize,
    triangular_solve_mat_min_flops:  usize,
    tensor_contract_min_flops:       usize,
    tensor_batched_matmul_min_flops: usize,
    tensor_sum_min_work:             usize,
}

impl Default for GpuRoutingPolicy {
    fn default() -> Self {
        Self {
            mode:                            GpuRoutingMode::Auto,
            // Defaults are tuned for release-profile 4090 measurements where small/medium
            // workloads are often faster on CPU because GPU launch/staging dominates.
            matmat_min_flops:                1_200_000_000,
            matvec_min_flops:                120_000_000,
            batched_matmat_min_flops:        800_000_000,
            batched_row_matvec_min_flops:    200_000_000,
            dot_min_len:                     1_048_576,
            pairwise_min_flops:              800_000_000,
            sparse_matvec_min_work:          2_000_000,
            sparse_matmat_dense_min_work:    8_000_000,
            sparse_matmat_sparse_min_nnz:    1_000_000,
            triangular_solve_vec_min_flops:  120_000_000,
            triangular_solve_mat_min_flops:  160_000_000,
            tensor_contract_min_flops:       1_000_000_000,
            tensor_batched_matmul_min_flops: 1_200_000_000,
            tensor_sum_min_work:             8_000_000,
        }
    }
}

const VAR_FORCE_GPU: &str = "NABLED_GPU_FORCE";
const VAR_FORCE_CPU: &str = "NABLED_GPU_DISABLE";
const VAR_MATMAT_FLOPS: &str = "NABLED_GPU_MIN_MATMAT_FLOPS";
const VAR_MATVEC_FLOPS: &str = "NABLED_GPU_MIN_MATVEC_FLOPS";
const VAR_BATCHED_MATMAT_FLOPS: &str = "NABLED_GPU_MIN_BATCHED_MATMAT_FLOPS";
const VAR_BATCHED_ROW_MATVEC_FLOPS: &str = "NABLED_GPU_MIN_BATCHED_ROW_MATVEC_FLOPS";
const VAR_DOT_LEN: &str = "NABLED_GPU_MIN_DOT_LEN";
const VAR_PAIRWISE_FLOPS: &str = "NABLED_GPU_MIN_PAIRWISE_FLOPS";
const VAR_SPARSE_MATVEC_WORK: &str = "NABLED_GPU_MIN_SPARSE_MATVEC_WORK";
const VAR_SPARSE_MATMAT_DENSE_WORK: &str = "NABLED_GPU_MIN_SPARSE_MATMAT_DENSE_WORK";
const VAR_SPARSE_MATMAT_SPARSE_NNZ: &str = "NABLED_GPU_MIN_SPARSE_MATMAT_SPARSE_NNZ";
const VAR_TRI_SOLVE_VEC_FLOPS: &str = "NABLED_GPU_MIN_TRIANGULAR_SOLVE_VEC_FLOPS";
const VAR_TRI_SOLVE_MAT_FLOPS: &str = "NABLED_GPU_MIN_TRIANGULAR_SOLVE_MAT_FLOPS";
const VAR_TENSOR_CONTRACT_FLOPS: &str = "NABLED_GPU_MIN_TENSOR_CONTRACT_FLOPS";
const VAR_TENSOR_BATCHED_MATMUL_FLOPS: &str = "NABLED_GPU_MIN_TENSOR_BATCHED_MATMUL_FLOPS";
const VAR_TENSOR_SUM_WORK: &str = "NABLED_GPU_MIN_TENSOR_SUM_WORK";

static GPU_POLICY: OnceLock<GpuRoutingPolicy> = OnceLock::new();

fn parse_bool_env(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_usize_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<usize>().ok()
}

fn default_policy() -> GpuRoutingPolicy {
    let mut policy = GpuRoutingPolicy::default();

    let gpu_forced_on = parse_bool_env(VAR_FORCE_GPU).unwrap_or(false);
    let gpu_forced_off = parse_bool_env(VAR_FORCE_CPU).unwrap_or(false);
    policy.mode = if gpu_forced_on && !gpu_forced_off {
        GpuRoutingMode::ForceGpu
    } else if gpu_forced_off {
        GpuRoutingMode::ForceCpu
    } else {
        GpuRoutingMode::Auto
    };

    policy.matmat_min_flops = parse_usize_env(VAR_MATMAT_FLOPS).unwrap_or(policy.matmat_min_flops);
    policy.matvec_min_flops = parse_usize_env(VAR_MATVEC_FLOPS).unwrap_or(policy.matvec_min_flops);
    policy.batched_matmat_min_flops =
        parse_usize_env(VAR_BATCHED_MATMAT_FLOPS).unwrap_or(policy.batched_matmat_min_flops);
    policy.batched_row_matvec_min_flops = parse_usize_env(VAR_BATCHED_ROW_MATVEC_FLOPS)
        .unwrap_or(policy.batched_row_matvec_min_flops);
    policy.dot_min_len = parse_usize_env(VAR_DOT_LEN).unwrap_or(policy.dot_min_len);
    policy.pairwise_min_flops =
        parse_usize_env(VAR_PAIRWISE_FLOPS).unwrap_or(policy.pairwise_min_flops);
    policy.sparse_matvec_min_work =
        parse_usize_env(VAR_SPARSE_MATVEC_WORK).unwrap_or(policy.sparse_matvec_min_work);
    policy.sparse_matmat_dense_min_work = parse_usize_env(VAR_SPARSE_MATMAT_DENSE_WORK)
        .unwrap_or(policy.sparse_matmat_dense_min_work);
    policy.sparse_matmat_sparse_min_nnz = parse_usize_env(VAR_SPARSE_MATMAT_SPARSE_NNZ)
        .unwrap_or(policy.sparse_matmat_sparse_min_nnz);
    policy.triangular_solve_vec_min_flops =
        parse_usize_env(VAR_TRI_SOLVE_VEC_FLOPS).unwrap_or(policy.triangular_solve_vec_min_flops);
    policy.triangular_solve_mat_min_flops =
        parse_usize_env(VAR_TRI_SOLVE_MAT_FLOPS).unwrap_or(policy.triangular_solve_mat_min_flops);
    policy.tensor_contract_min_flops =
        parse_usize_env(VAR_TENSOR_CONTRACT_FLOPS).unwrap_or(policy.tensor_contract_min_flops);
    policy.tensor_batched_matmul_min_flops = parse_usize_env(VAR_TENSOR_BATCHED_MATMUL_FLOPS)
        .unwrap_or(policy.tensor_batched_matmul_min_flops);
    policy.tensor_sum_min_work =
        parse_usize_env(VAR_TENSOR_SUM_WORK).unwrap_or(policy.tensor_sum_min_work);

    policy
}

fn policy() -> &'static GpuRoutingPolicy { GPU_POLICY.get_or_init(default_policy) }

fn flops2(rows: usize, inner: usize, cols: usize) -> usize {
    rows.saturating_mul(inner).saturating_mul(cols).saturating_mul(2)
}

fn tri_flops(n: usize, rhs_cols: usize) -> usize { n.saturating_mul(n).saturating_mul(rhs_cols) }

fn mode_gate(auto_decision: bool) -> bool {
    match policy().mode {
        GpuRoutingMode::ForceGpu => true,
        GpuRoutingMode::ForceCpu => false,
        GpuRoutingMode::Auto => auto_decision,
    }
}

pub(crate) fn should_attempt_gpu_matmat(rows: usize, inner: usize, cols: usize) -> bool {
    mode_gate(flops2(rows, inner, cols) >= policy().matmat_min_flops)
}

pub(crate) fn should_attempt_gpu_matvec(rows: usize, cols: usize) -> bool {
    mode_gate(flops2(rows, cols, 1) >= policy().matvec_min_flops)
}

pub(crate) fn should_attempt_gpu_batched_matmat(
    batch: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) -> bool {
    mode_gate(batch.saturating_mul(flops2(rows, inner, cols)) >= policy().batched_matmat_min_flops)
}

pub(crate) fn should_attempt_gpu_batched_row_matvec(
    batch: usize,
    inner: usize,
    cols: usize,
) -> bool {
    mode_gate(batch.saturating_mul(flops2(1, inner, cols)) >= policy().batched_row_matvec_min_flops)
}

pub(crate) fn should_attempt_gpu_dot(len: usize) -> bool { mode_gate(len >= policy().dot_min_len) }

pub(crate) fn should_attempt_gpu_pairwise(rows_left: usize, rows_right: usize, dim: usize) -> bool {
    mode_gate(flops2(rows_left, dim, rows_right) >= policy().pairwise_min_flops)
}

pub(crate) fn should_attempt_gpu_sparse_matvec(nnz: usize) -> bool {
    mode_gate(nnz >= policy().sparse_matvec_min_work)
}

pub(crate) fn should_attempt_gpu_sparse_matmat_dense(nnz: usize, cols: usize) -> bool {
    mode_gate(nnz.saturating_mul(cols) >= policy().sparse_matmat_dense_min_work)
}

pub(crate) fn should_attempt_gpu_sparse_matmat_sparse(left_nnz: usize, right_nnz: usize) -> bool {
    mode_gate(left_nnz.saturating_add(right_nnz) >= policy().sparse_matmat_sparse_min_nnz)
}

pub(crate) fn should_attempt_gpu_triangular_solve_vec(n: usize) -> bool {
    mode_gate(tri_flops(n, 1) >= policy().triangular_solve_vec_min_flops)
}

pub(crate) fn should_attempt_gpu_triangular_solve_mat(n: usize, rhs_cols: usize) -> bool {
    mode_gate(tri_flops(n, rhs_cols) >= policy().triangular_solve_mat_min_flops)
}

pub(crate) fn should_attempt_gpu_tensor_contract(left_len: usize, right_len: usize) -> bool {
    mode_gate(left_len.saturating_mul(right_len) >= policy().tensor_contract_min_flops)
}

pub(crate) fn should_attempt_gpu_tensor_batched_matmul(
    batch: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) -> bool {
    mode_gate(
        batch.saturating_mul(flops2(rows, inner, cols)) >= policy().tensor_batched_matmul_min_flops,
    )
}

pub(crate) fn should_attempt_gpu_tensor_sum(len: usize) -> bool {
    mode_gate(len >= policy().tensor_sum_min_work)
}

#[cfg(test)]
mod tests {
    use super::{
        should_attempt_gpu_batched_matmat, should_attempt_gpu_dot, should_attempt_gpu_matmat,
        should_attempt_gpu_sparse_matvec,
    };

    #[test]
    fn tiny_dense_workloads_default_to_cpu() {
        assert!(!should_attempt_gpu_matmat(256, 256, 256));
        assert!(!should_attempt_gpu_batched_matmat(4, 256, 256, 256));
        assert!(!should_attempt_gpu_dot(1_024));
    }

    #[test]
    fn large_dense_workloads_attempt_gpu() {
        assert!(should_attempt_gpu_matmat(1_024, 1_024, 1_024));
        assert!(should_attempt_gpu_batched_matmat(16, 512, 512, 512));
    }

    #[test]
    fn sparse_threshold_defaults_are_nontrivial() {
        assert!(!should_attempt_gpu_sparse_matvec(10_000));
        assert!(should_attempt_gpu_sparse_matvec(2_000_000));
    }
}

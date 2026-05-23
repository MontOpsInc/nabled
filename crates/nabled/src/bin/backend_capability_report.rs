#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, io};

use serde::Serialize;

#[cfg_attr(coverage_nightly, coverage(off))]
#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
#[expect(dead_code)]
enum ProviderPath {
    Native,
    Fallback,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum GpuKernelPath {
    NativeOrFallback,
    FallbackOnly,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[derive(Debug, Serialize)]
struct DomainCapability {
    tier: &'static str,
    domain: &'static str,
    baseline_kernels: bool,
    provider_path: ProviderPath,
    notes: &'static str,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[derive(Debug, Serialize)]
struct KernelCapability {
    family: &'static str,
    operation: &'static str,
    dtype: &'static str,
    gpu_path: GpuKernelPath,
    gpu_native_possible: bool,
    gpu_fallback: bool,
    notes: &'static str,
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[derive(Debug, Serialize)]
struct CapabilityReport {
    generated_at_unix: u64,
    git_sha: String,
    rustc_version: String,
    target_os: &'static str,
    target_arch: &'static str,
    provider_feature: &'static str,
    provider_feature_enabled: bool,
    provider_build_active: bool,
    native_provider_domains: usize,
    fallback_provider_domains: usize,
    gpu_native_or_fallback: usize,
    gpu_fallback_only: usize,
    domains: Vec<DomainCapability>,
    kernels: Vec<KernelCapability>,
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn main() -> io::Result<()> {
    let output_dir = parse_output_dir()?;
    fs::create_dir_all(&output_dir)?;

    let report = build_report();
    write_summary_json(&output_dir, &report)?;
    write_summary_markdown(&output_dir, &report)?;

    println!("Wrote backend capability report to {}", output_dir.canonicalize()?.display());
    Ok(())
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn parse_output_dir() -> io::Result<PathBuf> {
    let mut args = env::args().skip(1);
    let mut output_dir = PathBuf::from("coverage/backend-capabilities");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                let Some(path) = args.next() else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "missing value for --output-dir",
                    ));
                };
                output_dir = PathBuf::from(path);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {arg}"),
                ));
            }
        }
    }

    Ok(output_dir)
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn build_report() -> CapabilityReport {
    let mut domains = tier_a_domains();
    domains.extend(tier_b_domains());
    let kernels = kernel_capabilities();

    let native_provider_domains = domains
        .iter()
        .filter(|domain| matches!(domain.provider_path, ProviderPath::Native))
        .count();
    let fallback_provider_domains = domains
        .iter()
        .filter(|domain| matches!(domain.provider_path, ProviderPath::Fallback))
        .count();
    let gpu_native_or_fallback = kernels
        .iter()
        .filter(|kernel| matches!(kernel.gpu_path, GpuKernelPath::NativeOrFallback))
        .count();
    let gpu_fallback_only = kernels
        .iter()
        .filter(|kernel| matches!(kernel.gpu_path, GpuKernelPath::FallbackOnly))
        .count();

    CapabilityReport {
        generated_at_unix: now_unix_secs(),
        git_sha: command_output("git", &["rev-parse", "--short", "HEAD"]),
        rustc_version: command_output("rustc", &["-V"]),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        provider_feature: selected_provider_feature(),
        provider_feature_enabled: provider_feature_enabled(),
        provider_build_active: provider_feature_enabled(),
        native_provider_domains,
        fallback_provider_domains,
        gpu_native_or_fallback,
        gpu_fallback_only,
        domains,
        kernels,
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn provider_feature_enabled() -> bool {
    cfg!(feature = "openblas-system")
        || cfg!(feature = "openblas-static")
        || cfg!(feature = "netlib-system")
        || cfg!(feature = "netlib-static")
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn selected_provider_feature() -> &'static str {
    let openblas_system = cfg!(feature = "openblas-system");
    let openblas_static = cfg!(feature = "openblas-static");
    let netlib_system = cfg!(feature = "netlib-system");
    let netlib_static = cfg!(feature = "netlib-static");
    let selected_count = [openblas_system, openblas_static, netlib_system, netlib_static]
        .into_iter()
        .filter(|selected| *selected)
        .count();

    if selected_count > 1 {
        "mixed"
    } else if openblas_system {
        "openblas-system"
    } else if openblas_static {
        "openblas-static"
    } else if netlib_system {
        "netlib-system"
    } else if netlib_static {
        "netlib-static"
    } else {
        "internal"
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn tier_a_domains() -> Vec<DomainCapability> {
    vec![
        DomainCapability {
            tier: "tier_a",
            domain: "svd",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/svd.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "qr",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/qr.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "lu",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/lu.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "cholesky",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/cholesky.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "eigen",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/eigen.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "schur",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/schur.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "triangular_solve",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/triangular.rs",
        },
        DomainCapability {
            tier: "tier_a",
            domain: "vector_primitives",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/vector.rs",
        },
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn tier_b_domains() -> Vec<DomainCapability> {
    vec![
        DomainCapability {
            tier: "tier_b",
            domain: "polar",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/polar.rs",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "pca",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-ml/src/pca.rs",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "regression",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-ml/src/regression.rs",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "sylvester_lyapunov",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/sylvester.rs",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "matrix_functions",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/matrix_functions.rs (complex and real \
                               paths are available in internal and provider-enabled builds; \
                               Taylor paths remain baseline by design)",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "sparse",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/sparse.rs (CSR/CSC/COO, iterative \
                               solvers, and ILU0/IC0/ILUT/ILUK/ILDL0 preconditioned workflows)",
        },
        DomainCapability {
            tier: "tier_b",
            domain: "accelerator",
            baseline_kernels: true,
            provider_path: ProviderPath::Native,
            notes: "crates/nabled-linalg/src/accelerator.rs (CPU kernels + bounded \
                               GPU-backend `f32` kernel support via `wgpu` across \
                               dense/vector/tensor families with explicit CPU fallback outside v1 \
                               GPU scope)",
        },
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn gpu_path(native_when_wgpu: bool) -> GpuKernelPath {
    if native_when_wgpu && cfg!(feature = "accelerator-wgpu") {
        GpuKernelPath::NativeOrFallback
    } else {
        GpuKernelPath::FallbackOnly
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn kernel_capabilities() -> Vec<KernelCapability> {
    let mut kernels = Vec::new();
    kernels.extend(dense_kernel_capabilities());
    kernels.extend(sparse_kernel_capabilities());
    kernels.extend(vector_kernel_capabilities());
    kernels.extend(tensor_kernel_capabilities());
    kernels.extend(triangular_kernel_capabilities());
    kernels.extend(complex_kernel_capabilities());
    kernels
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn kernel_capability(
    family: &'static str,
    operation: &'static str,
    dtype: &'static str,
    native_when_wgpu: bool,
    notes: &'static str,
) -> KernelCapability {
    let wgpu_enabled = cfg!(feature = "accelerator-wgpu");
    KernelCapability {
        family,
        operation,
        dtype,
        gpu_path: gpu_path(native_when_wgpu),
        gpu_native_possible: wgpu_enabled && native_when_wgpu,
        gpu_fallback: true,
        notes,
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn dense_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "dense",
            "matmat",
            "f32",
            true,
            "compile-time backend dispatch; native wgpu path when enabled, CPU fallback otherwise",
        ),
        kernel_capability(
            "dense",
            "matmat",
            "f64",
            true,
            "native wgpu path requires SHADER_F64 at runtime; CPU fallback retained",
        ),
        kernel_capability(
            "dense",
            "matvec",
            "f32",
            true,
            "native via gpu matmat composition with CPU fallback",
        ),
        kernel_capability(
            "dense",
            "matvec",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "dense",
            "batched_matmat",
            "f32",
            true,
            "per-batch native wgpu matmat composition with CPU fallback",
        ),
        kernel_capability(
            "dense",
            "batched_matmat",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "dense",
            "batched_row_matvec",
            "f32",
            true,
            "native via gpu matmat composition with CPU fallback",
        ),
        kernel_capability(
            "dense",
            "batched_row_matvec",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn sparse_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "sparse",
            "matvec_csr",
            "f32",
            true,
            "phase-1 native CSR matvec compute shader with CPU fallback",
        ),
        kernel_capability(
            "sparse",
            "matvec_csr",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "sparse",
            "matmat_dense",
            "f32",
            true,
            "phase-1 native via repeated CSR matvec kernels with reusable output buffers; CPU \
             fallback retained",
        ),
        kernel_capability(
            "sparse",
            "matmat_dense",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "sparse",
            "matmat_sparse",
            "f32",
            true,
            "native via sparse-dense GPU composition and CSR rebuild with CPU fallback",
        ),
        kernel_capability(
            "sparse",
            "matmat_sparse",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn vector_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "vector",
            "dot",
            "f32",
            true,
            "native via gpu matmat composition with CPU fallback",
        ),
        kernel_capability(
            "vector",
            "dot",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "vector",
            "pairwise_l2",
            "f32",
            true,
            "native via gpu matmat cross term with CPU fallback",
        ),
        kernel_capability(
            "vector",
            "pairwise_l2",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "vector",
            "pairwise_cosine",
            "f32",
            true,
            "native via gpu matmat cross term with CPU fallback",
        ),
        kernel_capability(
            "vector",
            "pairwise_cosine",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn tensor_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "tensor",
            "batched_matmul_last_two",
            "f32",
            true,
            "native per-batch gpu matmat composition with CPU fallback",
        ),
        kernel_capability(
            "tensor",
            "batched_matmul_last_two",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "tensor",
            "contract_axes_single_axis",
            "f32",
            true,
            "native via 2D reshape + gpu matmat with CPU fallback",
        ),
        kernel_capability(
            "tensor",
            "contract_axes_single_axis",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "tensor",
            "sum_last_axis",
            "f32",
            true,
            "native via matmul reduction kernel with CPU fallback",
        ),
        kernel_capability(
            "tensor",
            "sum_last_axis",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn triangular_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "triangular",
            "solve_vec",
            "f32",
            true,
            "native triangular compute kernel with CPU fallback",
        ),
        kernel_capability(
            "triangular",
            "solve_vec",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
        kernel_capability(
            "triangular",
            "solve_mat",
            "f32",
            true,
            "native triangular compute kernel with CPU fallback",
        ),
        kernel_capability(
            "triangular",
            "solve_mat",
            "f64",
            true,
            "native path is SHADER_F64-dependent; CPU fallback retained",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn complex_kernel_capabilities() -> Vec<KernelCapability> {
    vec![
        kernel_capability(
            "complex",
            "tensor_batched_matmul_last_two",
            "complex64",
            true,
            "native via complex decomposition over f64 GPU kernels with CPU fallback",
        ),
        kernel_capability(
            "complex",
            "tensor_contract_axes_single_axis",
            "complex64",
            true,
            "native via complex decomposition over f64 GPU kernels with CPU fallback",
        ),
        kernel_capability(
            "complex",
            "tensor_sum_last_axis",
            "complex64",
            true,
            "native via split real/imag reductions over f64 GPU kernels with CPU fallback",
        ),
    ]
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn now_unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_secs())
}

#[cfg_attr(coverage_nightly, coverage(off))]
fn command_output(program: &str, args: &[&str]) -> String {
    let output = Command::new(program).args(args).output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn write_summary_json(output_dir: &Path, report: &CapabilityReport) -> io::Result<()> {
    let path = output_dir.join("summary.json");
    let content = serde_json::to_string_pretty(report).map_err(io::Error::other)?;
    fs::write(path, content)
}

fn write_summary_markdown(output_dir: &Path, report: &CapabilityReport) -> io::Result<()> {
    let mut markdown = String::new();
    markdown.push_str("# Backend Capability Report\n\n");
    let _ = writeln!(markdown, "- generated_at_unix: `{}`", report.generated_at_unix);
    let _ = writeln!(markdown, "- git_sha: `{}`", report.git_sha);
    let _ = writeln!(markdown, "- rustc: `{}`", report.rustc_version);
    let _ = writeln!(markdown, "- target: `{}-{}`", report.target_arch, report.target_os);
    let _ = writeln!(markdown, "- provider_feature_enabled: `{}`", report.provider_feature_enabled);
    let _ = writeln!(markdown, "- provider_build_active: `{}`", report.provider_build_active);
    let _ = writeln!(markdown, "- native_provider_domains: `{}`", report.native_provider_domains);
    let _ =
        writeln!(markdown, "- fallback_provider_domains: `{}`", report.fallback_provider_domains);
    let _ =
        writeln!(markdown, "- gpu_native_or_fallback_rows: `{}`", report.gpu_native_or_fallback);
    let _ = writeln!(markdown, "- gpu_fallback_only_rows: `{}`", report.gpu_fallback_only);
    markdown.push('\n');
    markdown.push_str("| Tier | Domain | Baseline Kernels | Provider Path | Notes |\n");
    markdown.push_str("|---|---|---|---|---|\n");

    for domain in &report.domains {
        let baseline = if domain.baseline_kernels { "yes" } else { "no" };
        let provider = match domain.provider_path {
            ProviderPath::Native => "native",
            ProviderPath::Fallback => "fallback",
        };
        let _ = writeln!(
            markdown,
            "| {} | {} | {} | {} | {} |",
            domain.tier, domain.domain, baseline, provider, domain.notes
        );
    }

    markdown.push('\n');
    markdown.push_str(
        "| Family | Operation | DType | GPU Path | Native Possible | CPU Fallback | Notes |\n",
    );
    markdown.push_str("|---|---|---|---|---|---|---|\n");
    for kernel in &report.kernels {
        let gpu_path = match kernel.gpu_path {
            GpuKernelPath::NativeOrFallback => "native_or_fallback",
            GpuKernelPath::FallbackOnly => "fallback_only",
        };
        let native_possible = if kernel.gpu_native_possible { "yes" } else { "no" };
        let fallback = if kernel.gpu_fallback { "yes" } else { "no" };
        let _ = writeln!(
            markdown,
            "| {} | {} | {} | {} | {} | {} | {} |",
            kernel.family,
            kernel.operation,
            kernel.dtype,
            gpu_path,
            native_possible,
            fallback,
            kernel.notes
        );
    }

    fs::write(output_dir.join("summary.md"), markdown)
}

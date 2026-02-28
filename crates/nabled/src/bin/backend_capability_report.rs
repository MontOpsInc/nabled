use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, io};

use serde::Serialize;

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
enum ProviderPath {
    Native,
    Fallback,
}

#[derive(Debug, Serialize)]
struct DomainCapability {
    tier:             &'static str,
    domain:           &'static str,
    baseline_kernels: bool,
    provider_path:    ProviderPath,
    notes:            &'static str,
}

#[derive(Debug, Serialize)]
struct CapabilityReport {
    generated_at_unix:         u64,
    git_sha:                   String,
    rustc_version:             String,
    target_os:                 &'static str,
    target_arch:               &'static str,
    provider_feature_enabled:  bool,
    provider_build_active:     bool,
    native_provider_domains:   usize,
    fallback_provider_domains: usize,
    domains:                   Vec<DomainCapability>,
}

fn main() -> io::Result<()> {
    let output_dir = parse_output_dir()?;
    fs::create_dir_all(&output_dir)?;

    let report = build_report();
    write_summary_json(&output_dir, &report)?;
    write_summary_markdown(&output_dir, &report)?;

    println!("Wrote backend capability report to {}", output_dir.canonicalize()?.display());
    Ok(())
}

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

fn build_report() -> CapabilityReport {
    let mut domains = tier_a_domains();
    domains.extend(tier_b_domains());

    let native_provider_domains = domains
        .iter()
        .filter(|domain| matches!(domain.provider_path, ProviderPath::Native))
        .count();
    let fallback_provider_domains = domains
        .iter()
        .filter(|domain| matches!(domain.provider_path, ProviderPath::Fallback))
        .count();

    CapabilityReport {
        generated_at_unix: now_unix_secs(),
        git_sha: command_output("git", &["rev-parse", "--short", "HEAD"]),
        rustc_version: command_output("rustc", &["-V"]),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        provider_feature_enabled: cfg!(feature = "openblas-system"),
        provider_build_active: cfg!(feature = "openblas-system"),
        native_provider_domains,
        fallback_provider_domains,
        domains,
    }
}

fn tier_a_domains() -> Vec<DomainCapability> {
    vec![
        DomainCapability {
            tier:             "tier_a",
            domain:           "svd",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/svd.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "qr",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/qr.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "lu",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/lu.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "cholesky",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/cholesky.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "eigen",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/eigen.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "schur",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/schur.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "triangular_solve",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/triangular.rs",
        },
        DomainCapability {
            tier:             "tier_a",
            domain:           "vector_primitives",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/vector.rs",
        },
    ]
}

fn tier_b_domains() -> Vec<DomainCapability> {
    vec![
        DomainCapability {
            tier:             "tier_b",
            domain:           "polar",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/polar.rs",
        },
        DomainCapability {
            tier:             "tier_b",
            domain:           "pca",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-ml/src/pca.rs",
        },
        DomainCapability {
            tier:             "tier_b",
            domain:           "regression",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-ml/src/regression.rs",
        },
        DomainCapability {
            tier:             "tier_b",
            domain:           "sylvester_lyapunov",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/sylvester.rs",
        },
        DomainCapability {
            tier:             "tier_b",
            domain:           "matrix_functions",
            baseline_kernels: true,
            provider_path:    ProviderPath::Native,
            notes:            "crates/nabled-linalg/src/matrix_functions.rs (provider-backed \
                               paths for eigen/SVD operations; Taylor paths remain baseline by \
                               design)",
        },
    ]
}

fn now_unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_secs())
}

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

    fs::write(output_dir.join("summary.md"), markdown)
}

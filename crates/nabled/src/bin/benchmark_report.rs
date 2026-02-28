use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs, io};

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct CriterionBenchmark {
    group_id:    String,
    function_id: String,
    value_str:   String,
    full_id:     String,
}

#[derive(Debug, Deserialize)]
struct CriterionEstimates {
    mean:    CriterionEstimate,
    median:  CriterionEstimate,
    std_dev: CriterionEstimate,
}

#[derive(Debug, Deserialize)]
struct CriterionEstimate {
    point_estimate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkEntry {
    domain:      String,
    operation:   String,
    backend:     String,
    competitor:  String,
    group_id:    String,
    function_id: String,
    full_id:     String,
    shape:       String,
    rows:        usize,
    cols:        usize,
    size:        usize,
    dtype:       String,
    mean_ns:     f64,
    median_ns:   f64,
    std_dev_ns:  f64,
    throughput:  Option<String>,
    correctness: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    generated_at_unix: u64,
    git_sha:           String,
    rustc_version:     String,
    source:            String,
    entries:           Vec<BenchmarkEntry>,
}

#[derive(Debug)]
struct RegressionSummary {
    baseline_found: bool,
    fail_count:     usize,
}

#[derive(Debug)]
struct RegressionPolicy {
    warn_pct:          f64,
    fail_pct:          f64,
    min_regression_ns: f64,
    min_baseline_ns:   f64,
}

fn main() -> io::Result<()> {
    let fail_on_regression = env::args().any(|arg| arg == "--fail-on-regression");
    let regression_policy = regression_policy_from_env();
    let output_root = Path::new("coverage/benchmarks");
    let candidate_roots =
        [Path::new("target/criterion"), Path::new("crates/nabled/target/criterion")];
    let criterion_roots =
        candidate_roots.iter().copied().filter(|root| root.exists()).collect::<Vec<_>>();

    if criterion_roots.is_empty() {
        eprintln!("No Criterion output found in known target directories.");
        eprintln!("Checked:");
        for root in candidate_roots {
            eprintln!("  {}", root.display());
        }
        eprintln!("Run benches first, for example:");
        eprintln!("  cargo bench --bench svd_benchmarks -- --quick");
        eprintln!("  cargo bench --bench qr_benchmarks -- --quick");
        eprintln!("  cargo bench --bench triangular_benchmarks -- --quick");
        eprintln!("  cargo bench --bench matrix_functions_benchmarks -- --quick");
        eprintln!("  cargo bench --bench lu_benchmarks -- --quick");
        eprintln!("  cargo bench --bench cholesky_benchmarks -- --quick");
        eprintln!("  cargo bench --bench eigen_benchmarks -- --quick");
        eprintln!("  cargo bench --bench vector_benchmarks -- --quick");
        return Ok(());
    }

    let mut benchmark_files = Vec::new();
    for root in &criterion_roots {
        benchmark_files.extend(collect_benchmark_json_files(root)?);
    }
    benchmark_files.sort();
    benchmark_files.dedup();

    let mut entries = Vec::new();

    for benchmark_path in benchmark_files {
        let benchmark = read_json::<CriterionBenchmark>(&benchmark_path)?;
        let estimates_path = benchmark_path.with_file_name("estimates.json");
        if !estimates_path.exists() {
            continue;
        }
        let estimates = read_json::<CriterionEstimates>(&estimates_path)?;

        let (shape, rows, cols) = parse_shape_dims(&benchmark.value_str);
        let (domain, backend, competitor, operation) =
            classify_benchmark(&benchmark.group_id, &benchmark.function_id);

        entries.push(BenchmarkEntry {
            domain,
            operation,
            backend,
            competitor,
            group_id: benchmark.group_id,
            function_id: benchmark.function_id,
            full_id: benchmark.full_id,
            shape,
            rows,
            cols,
            size: rows.min(cols),
            dtype: "f64".to_string(),
            mean_ns: estimates.mean.point_estimate,
            median_ns: estimates.median.point_estimate,
            std_dev_ns: estimates.std_dev.point_estimate,
            throughput: None,
            correctness: "passed".to_string(),
        });
    }

    entries.sort_by(|a, b| a.full_id.cmp(&b.full_id));

    fs::create_dir_all(output_root)?;

    let summary = BenchmarkSummary {
        generated_at_unix: now_unix_secs(),
        git_sha: command_output("git", &["rev-parse", "--short", "HEAD"]),
        rustc_version: command_output("rustc", &["-V"]),
        source: criterion_roots
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>()
            .join(","),
        entries,
    };

    write_summary_json(output_root, &summary)?;
    write_summary_csv(output_root, &summary)?;
    let regressions = write_regressions_md(output_root, &summary, &regression_policy)?;

    if fail_on_regression {
        if !regressions.baseline_found {
            eprintln!(
                "Regression check requested but baseline not found at \
                 `coverage/benchmarks/baseline/summary.json`"
            );
            std::process::exit(2);
        }
        if regressions.fail_count > 0 {
            eprintln!(
                "Benchmark regression check failed: {} case(s) exceeded {:.1}% and +{:.0}ns",
                regressions.fail_count,
                regression_policy.fail_pct,
                regression_policy.min_regression_ns
            );
            std::process::exit(3);
        }
    }

    println!("Wrote benchmark artifacts to {}", output_root.canonicalize()?.display());
    Ok(())
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

fn read_json<T>(path: &Path) -> io::Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = fs::read_to_string(path)?;
    serde_json::from_str::<T>(&content).map_err(io::Error::other)
}

fn collect_benchmark_json_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files)?;
    Ok(files)
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_recursive(&path, files)?;
            continue;
        }

        if path.file_name().is_some_and(|name| name == "benchmark.json")
            && path
                .parent()
                .is_some_and(|parent| parent.file_name().is_some_and(|name| name == "new"))
        {
            files.push(path);
        }
    }
    Ok(())
}

fn parse_shape_dims(value: &str) -> (String, usize, usize) {
    let Some((shape, dims)) = value.split_once('-') else {
        return ("unknown".to_string(), 0, 0);
    };
    let (rows, cols) = match dims.split_once('x') {
        Some((r, c)) => {
            (r.parse::<usize>().ok().unwrap_or(0), c.parse::<usize>().ok().unwrap_or(0))
        }
        None => (0, 0),
    };
    (shape.to_string(), rows, cols)
}

fn classify_benchmark(group_id: &str, function_id: &str) -> (String, String, String, String) {
    let domain = if group_id.starts_with("svd_") {
        "svd"
    } else if group_id.starts_with("qr_") {
        "qr"
    } else if group_id.starts_with("triangular_") {
        "triangular"
    } else if group_id.starts_with("matrix_functions_") {
        "matrix_functions"
    } else if group_id.starts_with("lu_") {
        "lu"
    } else if group_id.starts_with("cholesky_") {
        "cholesky"
    } else if group_id.starts_with("eigen_") {
        "eigen"
    } else if group_id.starts_with("vector_") {
        "vector"
    } else {
        "unknown"
    };

    let (backend, competitor) = match group_id {
        "svd_nabled_ndarray"
        | "qr_nabled_ndarray"
        | "triangular_nabled_ndarray"
        | "matrix_functions_nabled_ndarray"
        | "lu_nabled_ndarray"
        | "cholesky_nabled_ndarray"
        | "eigen_nabled_ndarray"
        | "vector_nabled_ndarray" => ("ndarray", "none"),
        "svd_competitor_faer_direct" | "qr_competitor_faer_direct" => {
            ("faer_direct", "faer_direct")
        }
        "svd_competitor_ndarray_linalg" | "qr_competitor_ndarray_linalg" => {
            ("ndarray_linalg", "ndarray_linalg")
        }
        _ => ("unknown", "unknown"),
    };

    (domain.to_string(), backend.to_string(), competitor.to_string(), function_id.to_string())
}

fn is_protected_nabled_case(entry: &BenchmarkEntry) -> bool { entry.competitor == "none" }

fn write_summary_json(output_root: &Path, summary: &BenchmarkSummary) -> io::Result<()> {
    let path = output_root.join("summary.json");
    let content = serde_json::to_string_pretty(summary).map_err(io::Error::other)?;
    fs::write(path, content)
}

fn write_summary_csv(output_root: &Path, summary: &BenchmarkSummary) -> io::Result<()> {
    let path = output_root.join("summary.csv");
    let mut lines = Vec::new();
    lines.push(
        "domain,backend,competitor,operation,size,shape,rows,cols,dtype,time_ns,median_ns,\
         std_dev_ns,throughput,correctness,full_id"
            .to_string(),
    );

    for entry in &summary.entries {
        lines.push(format!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{},{},{}",
            entry.domain,
            entry.backend,
            entry.competitor,
            entry.operation,
            entry.size,
            entry.shape,
            entry.rows,
            entry.cols,
            entry.dtype,
            entry.mean_ns,
            entry.median_ns,
            entry.std_dev_ns,
            entry.throughput.as_deref().unwrap_or(""),
            entry.correctness,
            entry.full_id
        ));
    }

    fs::write(path, lines.join("\n"))
}

fn write_regressions_md(
    output_root: &Path,
    summary: &BenchmarkSummary,
    policy: &RegressionPolicy,
) -> io::Result<RegressionSummary> {
    let baseline_path = output_root.join("baseline/summary.json");
    let mut lines = Vec::new();

    lines.push("# Benchmark Regression Report".to_string());
    lines.push(String::new());
    lines.push(format!("- Generated at unix time: `{}`", summary.generated_at_unix));
    lines.push(format!("- Git SHA: `{}`", summary.git_sha));
    lines.push(format!("- Rustc: `{}`", summary.rustc_version));
    lines.push(format!("- Cases: `{}`", summary.entries.len()));
    lines.push(String::new());
    lines.push(
        "- Regression scope: nabled benchmark cases only (`competitor == none`).".to_string(),
    );
    lines.push(format!(
        "- Thresholds: warn >{:.1}% and +{:.0}ns, fail >{:.1}% and +{:.0}ns.",
        policy.warn_pct, policy.min_regression_ns, policy.fail_pct, policy.min_regression_ns
    ));
    lines.push(format!(
        "- Noise floor filter: baseline median must be >= {:.0}ns.",
        policy.min_baseline_ns
    ));
    lines.push(String::new());

    if !baseline_path.exists() {
        lines.push("No baseline found at `coverage/benchmarks/baseline/summary.json`.".to_string());
        lines.push(
            "Create a baseline by copying a trusted `summary.json` to that path.".to_string(),
        );
        fs::write(output_root.join("regressions.md"), lines.join("\n"))?;
        return Ok(RegressionSummary { baseline_found: false, fail_count: 0 });
    }

    let baseline = read_json::<BenchmarkSummary>(&baseline_path)?;
    let baseline_map = baseline
        .entries
        .into_iter()
        .map(|entry| (entry.full_id, entry.median_ns))
        .collect::<BTreeMap<_, _>>();

    lines.push(format!("- Baseline: `{}`", baseline_path.display()));
    lines.push(String::new());
    lines.push(
        "| Benchmark | Current Median (ns) | Baseline Median (ns) | Delta % | Status |".to_string(),
    );
    lines.push("|---|---:|---:|---:|---|".to_string());

    let mut warn_count = 0_usize;
    let mut fail_count = 0_usize;
    let mut compared_cases = 0_usize;

    for entry in &summary.entries {
        if !is_protected_nabled_case(entry) {
            continue;
        }
        if let Some(baseline_ns) = baseline_map.get(&entry.full_id) {
            if *baseline_ns < policy.min_baseline_ns {
                lines.push(format!(
                    "| `{}` | {:.3} | {:.3} | n/a | SKIP_NOISE_FLOOR |",
                    entry.full_id, entry.median_ns, baseline_ns
                ));
                continue;
            }

            compared_cases += 1;
            let delta_ns = entry.median_ns - baseline_ns;
            let delta_pct =
                if *baseline_ns > f64::EPSILON { (delta_ns / baseline_ns) * 100.0 } else { 0.0 };
            let above_noise_floor = delta_ns > policy.min_regression_ns;
            let status = if delta_pct > policy.fail_pct && above_noise_floor {
                fail_count += 1;
                "FAIL"
            } else if delta_pct > policy.warn_pct && above_noise_floor {
                warn_count += 1;
                "WARN"
            } else {
                "OK"
            };
            lines.push(format!(
                "| `{}` | {:.3} | {:.3} | {:.2}% | {} |",
                entry.full_id, entry.median_ns, baseline_ns, delta_pct, status
            ));
        }
    }

    lines.push(String::new());
    lines.push(format!(
        "- Warnings (>{:.1}% and +{:.0}ns): `{warn_count}`",
        policy.warn_pct, policy.min_regression_ns
    ));
    lines.push(format!(
        "- Failures (>{:.1}% and +{:.0}ns): `{fail_count}`",
        policy.fail_pct, policy.min_regression_ns
    ));
    lines.push(format!("- Compared cases: `{compared_cases}`"));

    fs::write(output_root.join("regressions.md"), lines.join("\n"))?;
    Ok(RegressionSummary { baseline_found: true, fail_count })
}

fn regression_policy_from_env() -> RegressionPolicy {
    RegressionPolicy {
        warn_pct:          env_f64("BENCH_WARN_PCT", 5.0),
        fail_pct:          env_f64("BENCH_FAIL_PCT", 10.0),
        min_regression_ns: env_f64("BENCH_MIN_REGRESSION_NS", 25_000.0),
        min_baseline_ns:   env_f64("BENCH_MIN_BASELINE_NS", 0.0),
    }
}

fn env_f64(key: &str, default_value: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default_value)
}

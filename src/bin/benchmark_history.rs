use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct BenchmarkSummary {
    generated_at_unix: u64,
    git_sha:           String,
    entries:           Vec<BenchmarkEntry>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkEntry {
    competitor: String,
    full_id:    String,
    median_ns:  f64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct BenchmarkHistory {
    runs: Vec<HistoryRun>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HistoryRun {
    run_id:            String,
    branch:            String,
    git_sha:           String,
    generated_at_unix: u64,
    case_count:        usize,
    geometric_mean_ns: f64,
    medians:           BTreeMap<String, f64>,
}

#[derive(Debug)]
struct CliArgs {
    history_path: PathBuf,
    max_runs:     usize,
    output_md:    PathBuf,
    summary_path: PathBuf,
}

fn main() -> io::Result<()> {
    let args = parse_args()?;
    ensure_parent_dirs(&args)?;

    let summary = read_json::<BenchmarkSummary>(&args.summary_path)?;
    let medians = collect_nabled_medians(&summary);
    let current_run = build_current_run(&summary, medians)?;

    let mut history = load_history(&args.history_path)?;
    upsert_run(&mut history, current_run);
    trim_history(&mut history, args.max_runs);

    write_json_pretty(&args.history_path, &history)?;
    write_history_markdown(&args.output_md, &history)?;

    println!("Updated benchmark history at {}", args.history_path.display());
    println!("Wrote benchmark trend markdown to {}", args.output_md.display());
    Ok(())
}

fn parse_args() -> io::Result<CliArgs> {
    let mut history_path = PathBuf::from("coverage/benchmarks/baseline/history.json");
    let mut max_runs = 20_usize;
    let mut output_md = PathBuf::from("coverage/benchmarks/history.md");
    let mut summary_path = PathBuf::from("coverage/benchmarks/summary.json");

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--history" => history_path = required_path_arg(&mut args, "--history")?,
            "--max-runs" => max_runs = required_usize_arg(&mut args, "--max-runs")?,
            "--output-md" => output_md = required_path_arg(&mut args, "--output-md")?,
            "--summary" => summary_path = required_path_arg(&mut args, "--summary")?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {arg}"),
                ));
            }
        }
    }

    Ok(CliArgs { history_path, max_runs, output_md, summary_path })
}

fn required_path_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<PathBuf> {
    args.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("missing value for {flag}"))
    })
}

fn required_usize_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> io::Result<usize> {
    let value = args.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("missing value for {flag}"))
    })?;
    value.parse::<usize>().map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("invalid usize for {flag}: {value}"))
    })
}

fn ensure_parent_dirs(args: &CliArgs) -> io::Result<()> {
    if let Some(parent) = args.history_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.output_md.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn collect_nabled_medians(summary: &BenchmarkSummary) -> BTreeMap<String, f64> {
    summary
        .entries
        .iter()
        .filter(|entry| entry.competitor == "none")
        .map(|entry| (entry.full_id.clone(), entry.median_ns))
        .collect::<BTreeMap<_, _>>()
}

fn build_current_run(
    summary: &BenchmarkSummary,
    medians: BTreeMap<String, f64>,
) -> io::Result<HistoryRun> {
    let geometric_mean_ns = geometric_mean_ns(medians.values().copied())?;
    let run_id = env::var("GITHUB_RUN_ID")
        .unwrap_or_else(|_| format!("local-{}", summary.generated_at_unix));
    let branch = env::var("GITHUB_REF_NAME").unwrap_or_else(|_| "local".to_string());
    let git_sha = env::var("GITHUB_SHA").unwrap_or_else(|_| summary.git_sha.clone());

    Ok(HistoryRun {
        run_id,
        branch,
        git_sha,
        generated_at_unix: summary.generated_at_unix,
        case_count: medians.len(),
        geometric_mean_ns,
        medians,
    })
}

fn geometric_mean_ns(values: impl Iterator<Item = f64>) -> io::Result<f64> {
    let positives = values.filter(|value| value.is_finite() && *value > 0.0).collect::<Vec<_>>();
    if positives.is_empty() {
        return Ok(0.0);
    }

    let sum_ln = positives.iter().map(|value| value.ln()).sum::<f64>();
    let count = positives
        .len()
        .to_f64()
        .ok_or_else(|| io::Error::other("failed to convert value count to f64"))?;
    Ok((sum_ln / count).exp())
}

fn load_history(path: &Path) -> io::Result<BenchmarkHistory> {
    if !path.exists() {
        return Ok(BenchmarkHistory::default());
    }
    read_json::<BenchmarkHistory>(path)
}

fn upsert_run(history: &mut BenchmarkHistory, run: HistoryRun) {
    if let Some(existing) = history.runs.iter_mut().find(|existing| existing.run_id == run.run_id) {
        *existing = run;
    } else {
        history.runs.push(run);
    }
    history.runs.sort_by_key(|run| run.generated_at_unix);
}

fn trim_history(history: &mut BenchmarkHistory, max_runs: usize) {
    if history.runs.len() <= max_runs {
        return;
    }
    let drop_count = history.runs.len().saturating_sub(max_runs);
    drop(history.runs.drain(0..drop_count));
}

fn write_history_markdown(path: &Path, history: &BenchmarkHistory) -> io::Result<()> {
    let mut markdown = String::new();
    markdown.push_str("# Benchmark Trend (Last N Runs)\n\n");
    let _ = writeln!(markdown, "- runs in window: `{}`", history.runs.len());
    markdown.push('\n');

    if history.runs.is_empty() {
        markdown.push_str("No benchmark history available yet.\n");
        return fs::write(path, markdown);
    }

    markdown.push_str("## Run Summary\n\n");
    markdown.push_str(
        "| Run ID | Branch | SHA | Cases | Geomean ns | Delta vs Prev | Delta vs First |\n",
    );
    markdown.push_str("|---|---|---|---:|---:|---:|---:|\n");

    let first_geomean = history.runs.first().map_or(0.0, |run| run.geometric_mean_ns);
    let mut previous_geomean: Option<f64> = None;
    for run in &history.runs {
        let delta_prev = format_percent_delta(run.geometric_mean_ns, previous_geomean);
        let delta_first = format_percent_delta(run.geometric_mean_ns, Some(first_geomean));
        let short_sha = &run.git_sha[..run.git_sha.len().min(12)];
        let _ = writeln!(
            markdown,
            "| `{}` | `{}` | `{}` | {} | {:.3} | {} | {} |",
            run.run_id,
            run.branch,
            short_sha,
            run.case_count,
            run.geometric_mean_ns,
            delta_prev,
            delta_first
        );
        previous_geomean = Some(run.geometric_mean_ns);
    }

    markdown.push('\n');
    markdown.push_str("## Latest Case Snapshot\n\n");
    markdown.push_str(
        "| Benchmark | Latest Median (ns) | Prev Median (ns) | Delta vs Prev | Window Min-Max \
         (ns) |\n",
    );
    markdown.push_str("|---|---:|---:|---:|---|\n");

    let Some(latest_run) = history.runs.last() else {
        return fs::write(path, markdown);
    };
    let previous_run = history.runs.iter().rev().nth(1);

    for (benchmark, latest_ns) in &latest_run.medians {
        let prev_ns = previous_run.and_then(|run| run.medians.get(benchmark).copied());
        let delta_prev = format_percent_delta(*latest_ns, prev_ns);
        let (window_min, window_max) = min_max_for_benchmark(&history.runs, benchmark);
        let prev_str = prev_ns.map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"));
        let _ = writeln!(
            markdown,
            "| `{benchmark}` | {latest_ns:.3} | {prev_str} | {delta_prev} | {window_min:.3} - \
             {window_max:.3} |"
        );
    }

    fs::write(path, markdown)
}

fn min_max_for_benchmark(runs: &[HistoryRun], benchmark: &str) -> (f64, f64) {
    let values =
        runs.iter().filter_map(|run| run.medians.get(benchmark).copied()).collect::<Vec<_>>();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min.is_finite() && max.is_finite() { (min, max) } else { (0.0, 0.0) }
}

fn format_percent_delta(current: f64, reference: Option<f64>) -> String {
    let Some(reference) = reference else {
        return "n/a".to_string();
    };
    if reference <= f64::EPSILON {
        return "n/a".to_string();
    }
    let delta = ((current - reference) / reference) * 100.0;
    format!("{delta:.2}%")
}

fn read_json<T>(path: &Path) -> io::Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let content = fs::read_to_string(path)?;
    serde_json::from_str::<T>(&content).map_err(io::Error::other)
}

fn write_json_pretty<T>(path: &Path, value: &T) -> io::Result<()>
where
    T: Serialize,
{
    let content = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    fs::write(path, content)
}

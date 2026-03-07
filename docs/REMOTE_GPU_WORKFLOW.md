# Remote GPU Workflow

Last updated: 2026-03-07

## Purpose

Define a deterministic, agent-friendly workflow for CUDA/MAGMA tasks on remote NVIDIA hosts with zero manual in-host setup.

Goals:

1. Provision and prepare remote hosts with one command.
2. Run long jobs in tmux by default.
3. Keep all work observable (GPU monitor + live logs + work pane).
4. Keep commands identical for human and agent execution.

## Image Baseline

`docker/Dockerfile.nvidia` is the canonical CUDA development image for this repository.

It includes:

1. Users: `root` (Vast-injected SSH access) and `agent` (sudo-enabled)
2. Rust stable + `clippy` + `rustfmt`
3. `just`, `ripgrep`, `tmux`, `neovim`, `jq`
4. Python/PyO3 build prerequisites (`python3-dev`, `venv`, `pip`, `maturin`)
5. MAGMA/OpenBLAS/LAPACK dev packages + `gfortran` toolchain
6. Vulkan tooling/runtime for `wgpu` probing
7. SSH hardening defaults (key-only auth with `PermitRootLogin prohibit-password`)
8. `/etc/nabled/nvidia-image` marker used by remote prepare scripts to skip redundant bootstrap
9. `/root/.no_auto_tmux` is pre-created so Vast's root auto-tmux hook does not break non-interactive SSH automation

Build and push:

```bash
docker build -f docker/Dockerfile.nvidia -t montopsinc/nabled:nvidia-cuda12.9-rust-stable .
docker push montopsinc/nabled:nvidia-cuda12.9-rust-stable
```

## Scripted Flow

All scripts default to:

1. `SSH_USER=root`
2. `SSH_PORT=18800`
3. `SSH_KEY=~/.ssh/nabled_vast_4090`

SSH reproducibility guarantee:

1. Scripts force `ssh -F /dev/null` and set explicit options (`ControlMaster=no`, `ControlPath=none`, `ControlPersist=no`, `IdentitiesOnly=yes`).
2. Local `~/.ssh/config` cannot override workflow behavior.
3. Non-interactive scripts use stdin-driven remote execution (`ssh ... <<EOF`) instead of `ssh host "..."` command mode to avoid host-wrapper drift (including Vast command-mode tmux interception).

For Vast template startup script:

1. Leave it empty for this image (recommended).
2. Use startup script only for host-specific overrides (for example custom mount/setup), not base toolchain installs.

### 0) Single-entrypoint wrapper (preferred)

```bash
scripts/gpu_remote.sh up <host>
scripts/gpu_remote.sh one <host> magma-verify
scripts/gpu_remote.sh one <host> magma-capability
scripts/gpu_remote.sh one <host> magma-provider-bench
scripts/gpu_remote.sh run <host> "just checks"
scripts/gpu_remote.sh attach <host>
```

### 1) Prepare host + tmux session (low-level scripts)

```bash
scripts/gpu_remote_prepare.sh <host>
scripts/gpu_remote_tmux_session.sh <host>
```

### 2) One-command launch for standard jobs

```bash
scripts/gpu_remote.sh one <host> magma-verify
scripts/gpu_remote.sh one <host> magma-capability
scripts/gpu_remote.sh one <host> magma-provider-bench
scripts/gpu_remote.sh one <host> gpu-probe
scripts/gpu_remote.sh one <host> checks
```

### 3) Attach and observe

```bash
scripts/gpu_remote_tmux_attach.sh <host>
```

Session defaults:

1. Session: `nabled-agent`
2. Window `work`: active command execution
3. Window `gpu`: `watch -n 1 nvidia-smi`
4. Window `logs`: tail of current job log

### Tmux UX troubleshooting

If `Ctrl+B` is echoed as `^B` or pane layout is unexpectedly tiny:

1. Always attach through `scripts/gpu_remote_tmux_attach.sh`; it now uses a direct interactive SSH command and `tmux attach -d` to avoid stale-client sizing issues.
2. Check for nested tmux (`echo $TMUX`). If nested, use `Ctrl+B Ctrl+B` for inner-session commands or detach outer tmux first.
3. Detach stale clients that can constrain dimensions:
   `tmux detach-client -a -t nabled-agent`
4. Verify live client dimensions:
   `tmux list-clients -F '#{client_tty} #{client_width}x#{client_height} #{session_name}'`

### 4) Run any custom command in `work`

```bash
scripts/gpu_remote_tmux_run.sh <host> "just bench-smoke-report-provider"
```

The run script writes:

1. `~/.cache/nabled-agent/logs/job-<timestamp>.log`
2. `~/.cache/nabled-agent/logs/current.log` (symlink to latest)
3. pane transcript: `~/.cache/nabled-agent/logs/tmux-work-pane.log`

## Remote job scripts

Reusable remote jobs (called by wrapper scripts and tmux run flow):

1. `scripts/remote_jobs/magma_verify_job.sh`
2. `scripts/remote_jobs/magma_capability_job.sh`
3. `scripts/remote_jobs/magma_provider_bench_job.sh`
4. `scripts/remote_jobs/gpu_probe_job.sh`

## MAGMA expansion tracking

Current MAGMA provider coverage and future expansion must be tracked in:

1. `docs/GPU_V2_TRACKER.md`
2. `docs/EXECUTION_TRACKER.md`
3. `docs/BENCHMARK_TRACKER.md`

## Notes

1. Provider and backend are orthogonal; this workflow is infrastructure-only and does not change API semantics.
2. If a host is pre-baked from `Dockerfile.nvidia`, prepare time is minimized and mostly syncs repository state.
3. `scripts/remote_jobs/gpu_probe_job.sh` writes a temporary Vulkan ICD JSON at runtime (`/tmp/nvidia_icd_egl.json`) so `wgpu` selects NVIDIA Vulkan deterministically across heterogeneous host images. This is host-runtime wiring, not image baseline configuration.

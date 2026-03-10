LOG := env('RUST_LOG', '')
features := 'arrow blas lapack-provider openblas-system openblas-static netlib-system netlib-static accelerator-rayon accelerator-wgpu'
provider_env_prefix := if os() == "macos" { "env PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}} OPENBLAS_DIR=/opt/homebrew/opt/openblas" } else { "env" }
provider_features := env('NABLED_PROVIDER_FEATURES', 'openblas-system')
provider_bench_features := env('NABLED_PROVIDER_BENCH_FEATURES', 'openblas-system')
coverage_line_threshold := "90"
coverage_ignore_regex := "crates/nabled-linalg/src/accelerator/gpu.rs"
gpu_remote_user := env('NABLED_GPU_REMOTE_USER', 'root')
gpu_remote_port := env('NABLED_GPU_REMOTE_PORT', '40637')
gpu_remote_key := env('NABLED_GPU_REMOTE_KEY', '${HOME}/.ssh/nabled_vast_4090')
docker_nvidia_tag := env('NABLED_DOCKER_NVIDIA_TAG', 'montopsinc/nabled:nvidia-cuda12.9-rust-stable-amd64')

# List of Examples

examples := ''

default:
    @just --list

# --- REMOTE GPU WORKFLOW ---
docker-nvidia-build tag=docker_nvidia_tag:
    docker build -f docker/Dockerfile.nvidia -t "{{ tag }}" .

docker-nvidia-push tag=docker_nvidia_tag:
    docker push "{{ tag }}"

gpu-remote-up host:
    SSH_USER={{ gpu_remote_user }} SSH_PORT={{ gpu_remote_port }} SSH_KEY={{ gpu_remote_key }} scripts/gpu_remote.sh up "{{ host }}"

gpu-remote-one host job='magma-verify':
    SSH_USER={{ gpu_remote_user }} SSH_PORT={{ gpu_remote_port }} SSH_KEY={{ gpu_remote_key }} scripts/gpu_remote.sh one "{{ host }}" "{{ job }}"

gpu-remote-run host command:
    SSH_USER={{ gpu_remote_user }} SSH_PORT={{ gpu_remote_port }} SSH_KEY={{ gpu_remote_key }} scripts/gpu_remote.sh run "{{ host }}" "{{ command }}"

gpu-remote-attach host:
    SSH_USER={{ gpu_remote_user }} SSH_PORT={{ gpu_remote_port }} SSH_KEY={{ gpu_remote_key }} scripts/gpu_remote.sh attach "{{ host }}"

# --- TESTS ---
test:
    just -f {{ justfile() }} test-unit
    just -f {{ justfile() }} test-integration-all

test-provider:
    {{ provider_env_prefix }} RUST_LOG={{ LOG }} cargo test --workspace --lib --features {{ provider_features }} -- --nocapture --show-output
    {{ provider_env_prefix }} RUST_LOG={{ LOG }} cargo test -p nabled --features {{ provider_features }} --tests -- --nocapture --show-output

test-unit:
    RUST_LOG={{ LOG }} cargo test --workspace --lib -- --nocapture --show-output

test-all-targets:
    RUST_LOG={{ LOG }} cargo test --workspace --all-targets -- --nocapture --show-output

test-one test_name:
    RUST_LOG={{ LOG }} cargo test --workspace "{{ test_name }}" -- --nocapture --show-output

test-integration test_name:
    RUST_LOG={{ LOG }} cargo test -p nabled --test "{{ test_name }}" -- --nocapture --show-output

test-integration-all:
    RUST_LOG={{ LOG }} cargo test -p nabled --tests -- --nocapture --show-output

coverage:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-default-features --no-report --exclude 'nabled'
    {{ provider_env_prefix }} cargo llvm-cov --workspace --lib --tests --no-default-features --features {{ provider_features }} --no-report --exclude 'nabled'
    cargo llvm-cov report -vv --html --output-dir coverage --open --ignore-filename-regex {{ coverage_ignore_regex }}

coverage-json:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-default-features --no-report --exclude 'nabled'
    {{ provider_env_prefix }} cargo llvm-cov --workspace --lib --tests --no-default-features --features {{ provider_features }} --no-report --exclude 'nabled'
    cargo llvm-cov report --json --output-path coverage/cov.json --ignore-filename-regex {{ coverage_ignore_regex }}

coverage-lcov:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-default-features --no-report --exclude 'nabled'
    {{ provider_env_prefix }} cargo llvm-cov --workspace --lib --tests --no-default-features --features {{ provider_features }} --no-report --exclude 'nabled'
    cargo llvm-cov report --lcov --output-path coverage/lcov.info --ignore-filename-regex {{ coverage_ignore_regex }}

coverage-check:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-default-features --no-report --exclude 'nabled'
    {{ provider_env_prefix }} cargo llvm-cov --workspace --lib --tests --no-default-features --features {{ provider_features }} --no-report --exclude 'nabled'
    cargo llvm-cov report --summary-only --fail-under-lines {{ coverage_line_threshold }} --ignore-filename-regex {{ coverage_ignore_regex }}

# --- DOCS ---
docs:
    cargo doc --workspace --open

# --- BENCHES ---
[confirm('Delete all benchmark reports?')]
clear-benches:
    rm -rf target/criterion/*

bench:
    RUST_LOG={{ LOG }} cargo bench -p nabled --profile=release && open target/criterion/report/index.html

bench-lto:
    RUST_LOG={{ LOG }} cargo bench -p nabled --profile=release-lto && open target/criterion/report/index.html

bench-one bench:
    RUST_LOG={{ LOG }} cargo bench -p nabled \
     --profile=release \
     --bench "{{ bench }}" && \
     open target/criterion/report/index.html

bench-one-lto bench:
    RUST_LOG={{ LOG }} cargo bench -p nabled \
     --profile=release-lto \
     --bench "{{ bench }}" && \
     open target/criterion/report/index.html

bench-smoke:
    rm -rf target/criterion crates/nabled/target/criterion
    cargo bench -p nabled --bench svd_benchmarks -- --quick
    cargo bench -p nabled --bench qr_benchmarks -- --quick
    cargo bench -p nabled --bench triangular_benchmarks -- --quick
    cargo bench -p nabled --bench matrix_functions_benchmarks -- --quick
    cargo bench -p nabled --bench lu_benchmarks -- --quick
    cargo bench -p nabled --bench cholesky_benchmarks -- --quick
    cargo bench -p nabled --bench eigen_benchmarks -- --quick
    cargo bench -p nabled --bench vector_benchmarks -- --quick
    cargo bench -p nabled --bench matrix_benchmarks -- --quick
    cargo bench -p nabled --bench sparse_benchmarks -- --quick
    cargo bench -p nabled --bench tensor_benchmarks -- --quick
    cargo bench -p nabled --bench accelerator_benchmarks -- --quick
    cargo bench -p nabled --no-default-features --features accelerator-wgpu --bench accelerator_benchmarks -- --quick
    cargo bench -p nabled --bench schur_benchmarks -- --quick
    cargo bench -p nabled --bench sylvester_benchmarks -- --quick
    cargo bench -p nabled --bench optimization_benchmarks -- --quick
    cargo bench -p nabled --bench polar_benchmarks -- --quick
    cargo bench -p nabled --bench orthogonalization_benchmarks -- --quick

bench-smoke-provider:
    rm -rf target/criterion crates/nabled/target/criterion
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench svd_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench qr_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench triangular_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench matrix_functions_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench lu_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench cholesky_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench eigen_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench vector_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench matrix_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench sparse_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench tensor_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench accelerator_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench schur_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench sylvester_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench optimization_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench polar_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench orthogonalization_benchmarks -- --quick

bench-report:
    cargo run -p nabled --bin benchmark_report

bench-report-check:
    cargo run -p nabled --bin benchmark_report -- --fail-on-regression

bench-history:
    cargo run -p nabled --bin benchmark_history

bench-history-window max_runs:
    cargo run -p nabled --bin benchmark_history -- --max-runs "{{ max_runs }}"

bench-baseline-update:
    mkdir -p coverage/benchmarks/baseline
    cp coverage/benchmarks/summary.json coverage/benchmarks/baseline/summary.json

bench-smoke-report:
    just -f {{ justfile() }} bench-smoke
    just -f {{ justfile() }} bench-report

bench-smoke-report-provider:
    just -f {{ justfile() }} bench-smoke-provider
    just -f {{ justfile() }} bench-report

bench-smoke-provider-decomposition:
    rm -rf target/criterion crates/nabled/target/criterion
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench svd_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench qr_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench matrix_functions_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench lu_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench cholesky_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench eigen_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench schur_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench sylvester_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench polar_benchmarks -- --quick

bench-smoke-provider-decomposition-lto:
    rm -rf target/criterion crates/nabled/target/criterion
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench svd_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench qr_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench matrix_functions_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench lu_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench cholesky_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench eigen_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench schur_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench sylvester_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --profile=release-lto --features {{ provider_bench_features }} --bench polar_benchmarks -- --quick

bench-smoke-report-provider-decomposition:
    just -f {{ justfile() }} bench-smoke-provider-decomposition
    just -f {{ justfile() }} bench-report

bench-smoke-report-provider-decomposition-lto:
    just -f {{ justfile() }} bench-smoke-provider-decomposition-lto
    just -f {{ justfile() }} bench-report

bench-smoke-check:
    just -f {{ justfile() }} bench-smoke
    just -f {{ justfile() }} bench-report-check

# --- BACKEND CAPABILITY REPORTING ---
backend-capability-report:
    cargo run -p nabled --bin backend_capability_report -- --output-dir coverage/backend-capabilities/baseline

backend-capability-report-provider:
    {{ provider_env_prefix }} cargo run -p nabled --features {{ provider_features }} --bin backend_capability_report -- --output-dir coverage/backend-capabilities/provider

backend-capability-report-all:
    just -f {{ justfile() }} backend-capability-report
    just -f {{ justfile() }} backend-capability-report-provider

# --- EXAMPLES ---

debug-profile example:
    RUSTFLAGS='-g' cargo build -p nabled --example "{{ example }}"

release-debug example:
    RUSTFLAGS='-g' cargo build -p nabled --profile=release-with-debug --example "{{ example }}"
    codesign -s - -v -f --entitlements assets/mac.entitlements "target/release-with-debug/examples/{{ example }}"

release-lto example:
    cargo build -p nabled --profile=release-lto --example "{{ example }}"
    codesign -s - -v -f --entitlements assets/mac.entitlements "target/release-lto/examples/{{ example }}"

example example:
    cargo run -p nabled --example "{{ example }}"

example-lto example:
    cargo run -p nabled --profile=release-lto --example "{{ example }}"

example-release-debug example:
    cargo run -p nabled --profile=release-with-debug --example "{{ example }}"

examples:
    @for ex in {{ examples }}; do \
        echo "Running example: $ex"; \
        cargo run -p nabled --example "$ex"; \
    done

# --- PROFILING ---
flamegraph example *args='':
    CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root --flamechart --open \
     -p nabled \
     --profile=release-with-debug \
     --min-width="0.0001" \
     --example "{{ example }}" -- "{{ args }}"

samply example *args='': (release-debug example)
    # TODO: Add install check here
    samply record -r 100000 "target/release-with-debug/examples/{{ example }}" "{{ args }}"

# --- CLIPPY AND FORMATTING ---

# Check all feature combinations
check-features *ARGS=features:
    @echo "Checking no features..."
    cargo clippy -p nabled --no-default-features --all-targets -- -D warnings
    @echo "Building no features..."
    cargo check -p nabled --no-default-features --all-targets
    @echo "Checking default features..."
    cargo clippy -p nabled --all-targets -- -D warnings
    @echo "Building default features..."
    cargo check -p nabled --all-targets
    @echo "Checking provider + accelerator feature set..."
    {{ provider_env_prefix }} cargo clippy -p nabled --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings
    @echo "Building provider + accelerator feature set..."
    {{ provider_env_prefix }} cargo check -p nabled --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets
    @echo "Checking each feature..."
    @for feature in {{ ARGS }}; do \
        echo "Checking & Building feature: $feature"; \
        cargo clippy -p nabled --no-default-features --features $feature --all-targets -- -D warnings; \
        cargo check -p nabled --no-default-features --features $feature --all-targets; \
    done
    @echo "Checking each feature with defaults..."
    @for feature in {{ ARGS }}; do \
        echo "Checking feature (with defaults): $feature"; \
        cargo clippy -p nabled --features $feature --all-targets -- -D warnings; \
        cargo check -p nabled --features $feature --all-targets; \
    done
    @echo "Checking all provided features..."
    cargo clippy -p nabled --no-default-features --features "{{ ARGS }}" --all-targets -- -D warnings
    cargo check -p nabled --no-default-features --features "{{ ARGS }}" --all-targets

fmt:
    @echo "Running rustfmt..."
    cargo +nightly fmt --check -- --config-path ./rustfmt.toml

fmt-fix:
    @echo "Running rustfmt..."
    cargo +nightly fmt -- --config-path ./rustfmt.toml

fix:
    {{ provider_env_prefix }} cargo clippy --fix --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets --allow-dirty

# --- MAINTENANCE ---

# Run checks CI will
checks:
    cargo +nightly fmt --all -- --check --config-path ./rustfmt.toml
    cargo +nightly clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +nightly clippy --workspace --no-default-features --features lapack-provider --all-targets -- -D warnings
    cargo +nightly clippy --workspace --no-default-features --features arrow --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +nightly clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features lapack-provider --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features arrow --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features accelerator-rayon --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features accelerator-wgpu --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings
    just -f {{ justfile() }} check-provider-clippy
    just -f {{ justfile() }} check-provider-netlib
    just -f {{ justfile() }} check-arrow
    just -f {{ justfile() }} test
    just -f {{ justfile() }} test-provider
    just -f {{ justfile() }} test-arrow
    just -f {{ justfile() }} check-accelerator
    just -f {{ justfile() }} test-accelerator
    just -f {{ justfile() }} coverage-check
    just -f {{ justfile() }} check-provider
    just -f {{ justfile() }} backend-capability-report

# Verify provider-gated lint paths are checked locally.
check-provider-clippy:
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features {{ provider_features }} --all-targets -- -D warnings

# Verify Arrow interop compiles and lints in internal/provider modes.
check-arrow:
    cargo +stable clippy --workspace --no-default-features --features arrow --all-targets -- -D warnings
    cargo +stable check --workspace --no-default-features --features arrow --all-targets
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features "{{ provider_features }} arrow" --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +stable check --workspace --no-default-features --features "{{ provider_features }} arrow" --all-targets

# Verify provider-enabled code paths compile under stable.
check-provider:
    just -f {{ justfile() }} check-provider-clippy
    {{ provider_env_prefix }} cargo +stable check --workspace --features {{ provider_features }} --all-targets

# Verify alternate provider-gated lint/compile paths (no OpenBLAS env required).
check-provider-netlib:
    cargo +stable clippy --workspace --no-default-features --features netlib-system --all-targets -- -D warnings
    cargo +stable check --workspace --no-default-features --features netlib-system --all-targets

# Optional static-provider compile checks (toolchain-dependent: gcc/gfortran/make).
check-provider-static:
    cargo +stable check --workspace --no-default-features --features openblas-static --all-targets
    cargo +stable check --workspace --no-default-features --features netlib-static --all-targets

# Verify accelerator feature permutations compile under stable.
check-accelerator:
    cargo +stable clippy --workspace --no-default-features --features accelerator-rayon --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features accelerator-wgpu --all-targets -- -D warnings
    cargo +stable check --workspace --no-default-features --features accelerator-rayon --all-targets
    cargo +stable check --workspace --no-default-features --features accelerator-wgpu --all-targets
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon" --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-wgpu" --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +stable check --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon" --all-targets
    {{ provider_env_prefix }} cargo +stable check --workspace --no-default-features --features "{{ provider_features }} accelerator-wgpu" --all-targets

# Verify accelerator contract tests in feature-gated paths.
test-accelerator:
    cargo +stable test -p nabled-linalg --no-default-features --features accelerator-rayon --lib accelerated_matmat_matches_serial -- --nocapture --show-output
    cargo +stable test -p nabled-linalg --no-default-features --features accelerator-wgpu --lib gpu_ -- --nocapture --show-output
    {{ provider_env_prefix }} cargo +stable test -p nabled-linalg --no-default-features --features "{{ provider_features }} accelerator-wgpu" --lib gpu_ -- --nocapture --show-output

# Verify Arrow interop integration tests in internal/provider modes.
test-arrow:
    cargo +stable test -p nabled --no-default-features --features arrow --test arrow_interop -- --nocapture --show-output
    {{ provider_env_prefix }} cargo +stable test -p nabled --no-default-features --features "{{ provider_features }} arrow" --test arrow_interop -- --nocapture --show-output

# Initialize development environment for maintainers
init-dev:
    @echo "Installing development tools..."
    cargo install cargo-release || true
    cargo install git-cliff || true
    cargo install cargo-edit || true
    cargo install cargo-outdated || true
    cargo install cargo-audit || true
    @echo ""
    @echo "✅ Development tools installed!"
    @echo ""
    @echo "Next steps:"
    @echo "1. Get your crates.io API token from https://crates.io/settings/tokens"
    @echo "2. Add it as CARGO_REGISTRY_TOKEN in GitHub repo settings → Secrets"
    @echo "3. Use 'just prepare-release <X.Y.Z>' to prepare release PRs"
    @echo ""
    @echo "Useful commands:"
    @echo "  just release-dry 0.1.0  # Preview what would happen"
    @echo "  just check-outdated     # Check for outdated dependencies"
    @echo "  just audit              # Security audit"

# Check for outdated dependencies
check-outdated:
    cargo outdated

# Run security audit
audit:
    cargo audit

# Prepare a release (creates PR with version bumps and changelog)
prepare-release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Validate version format
    if ! [[ "{{ version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
        exit 1
    fi

    # Require clean tree for deterministic release prep.
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: Working tree is not clean. Commit or stash changes first."
        exit 1
    fi

    # Create release branch
    git checkout -b "release-v{{ version }}"

    # Update version in root Cargo.toml (in [workspace.package] section)
    awk '/^\[workspace\.package\]/ {in_workspace_package=1} in_workspace_package && /^version = / {gsub(/"[^"]*"/, "\"{{ version }}\""); in_workspace_package=0} {print}' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

    # Update internal workspace dependency versions for publish/package correctness.
    sed -i '' "s/^nabled = { path = \"crates\\/nabled\", version = \"=[^\"]*\" }/nabled = { path = \"crates\\/nabled\", version = \"={{ version }}\" }/" Cargo.toml
    sed -i '' "s/^nabled-core = { path = \"crates\\/nabled-core\", version = \"=[^\"]*\" }/nabled-core = { path = \"crates\\/nabled-core\", version = \"={{ version }}\" }/" Cargo.toml
    sed -i '' "s/^nabled-linalg = { path = \"crates\\/nabled-linalg\", version = \"=[^\"]*\" }/nabled-linalg = { path = \"crates\\/nabled-linalg\", version = \"={{ version }}\" }/" Cargo.toml
    sed -i '' "s/^nabled-ml = { path = \"crates\\/nabled-ml\", version = \"=[^\"]*\" }/nabled-ml = { path = \"crates\\/nabled-ml\", version = \"={{ version }}\" }/" Cargo.toml

    # Update nabled crate version references in README files (if they exist).
    # Look for patterns like: nabled = "0.1.1" or nabled = { version = "0.1.1" }.
    for readme in README.md crates/nabled-core/README.md crates/nabled-linalg/README.md crates/nabled-ml/README.md; do
        if [ -f "$readme" ]; then
            for dep in nabled nabled-core nabled-linalg nabled-ml; do
                # Update simple dependency format
                sed -i '' "s/$dep = \"[0-9]*\.[0-9]*\.[0-9]*\"/$dep = \"{{ version }}\"/" "$readme" || true
                # Update version field in dependency table format
                sed -i '' "s/$dep = { version = \"[0-9]*\.[0-9]*\.[0-9]*\"/$dep = { version = \"{{ version }}\"/" "$readme" || true
            done
        fi
    done

    # Update Cargo.lock
    cargo update --workspace

    # Verify leaf crate packages locally.
    # Dependent crates resolve internal dependencies via crates.io at package time,
    # so full package verification is performed in release workflow publish order.
    cargo package --allow-dirty -p nabled-core

    # Generate full changelog
    echo "Generating changelog..."
    git cliff -o CHANGELOG.md

    # Generate release notes for this version
    echo "Generating release notes..."
    git cliff --unreleased --tag v{{ version }} --strip header -o RELEASE_NOTES.md

    # Stage all changes.
    # Cargo.lock is ignored in this repository, so stage it only if it is tracked.
    git add Cargo.toml CHANGELOG.md RELEASE_NOTES.md
    if git ls-files --error-unmatch Cargo.lock >/dev/null 2>&1; then
        git add Cargo.lock
    fi
    # Also add README files if they were modified
    git add README.md crates/nabled-core/README.md crates/nabled-linalg/README.md crates/nabled-ml/README.md 2>/dev/null || true

    # Commit
    git commit -m "chore: prepare release v{{ version }}"

    # Push branch and set upstream so later `git push` works without extra flags.
    git push --set-upstream origin "release-v{{ version }}"

    echo ""
    echo "✅ Release preparation complete!"
    echo ""
    echo "Release notes preview:"
    echo "----------------------"
    head -20 RELEASE_NOTES.md
    echo ""
    echo "Next steps:"
    echo "1. Create a PR from the 'release-v{{ version }}' branch"
    echo "2. Review and merge the PR"
    echo "3. After merge, run: just tag-release {{ version }}"
    echo ""

# Tag a release after the PR is merged
tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Ensure we're on main and up to date
    git checkout main
    git pull origin main

    # Verify the version in Cargo.toml matches requested version
    CARGO_VERSION=$(awk '/^\[workspace\.package\]/ {in_workspace_package=1; next} in_workspace_package && /^version = / {gsub(/"/,"",$3); print $3; exit}' Cargo.toml)
    if [ "$CARGO_VERSION" != "{{ version }}" ]; then
        echo "Error: Cargo.toml version ($CARGO_VERSION) does not match requested version ({{ version }})"
        echo "Did the release PR merge successfully?"
        exit 1
    fi

    # Verify leaf publish path works.
    cargo publish --dry-run -p nabled-core --no-verify

    # Create and push tag
    git tag -a "v{{ version }}" -m "Release v{{ version }}"
    git push origin "v{{ version }}"

    echo ""
    echo "✅ Tag v{{ version }} created and pushed!"
    echo "The release workflow will now run automatically."
    echo ""

# Preview what a release would do (dry run)
release-dry version:
    @echo "This would:"
    @echo "1. Create branch: release-v{{ version }}"
    @echo "2. Update version to {{ version }} in:"
    @echo "   - Cargo.toml [workspace.package] version"
    @echo "   - Cargo.toml [workspace.dependencies] internal crate versions"
    @echo "   - README files (if they contain nabled version references)"
    @echo "3. Run local package check for nabled-core (leaf crate)"
    @echo "   - dependent crates are verified by ordered publish in release workflow"
    @echo "4. Update Cargo.lock"
    @echo "5. Generate CHANGELOG.md"
    @echo "6. Generate RELEASE_NOTES.md"
    @echo "7. Create commit and push branch"
    @echo ""
    @echo "After PR merge, 'just tag-release {{ version }}' would:"
    @echo "1. Tag the merged commit as v{{ version }}"
    @echo "2. Verify leaf dry-run (nabled-core)"
    @echo "3. Push the tag (triggering ordered release workflow)"

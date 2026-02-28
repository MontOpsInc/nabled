LOG := env('RUST_LOG', '')
features := 'blas openblas-system'
provider_env_prefix := if os() == "macos" { "env PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}} OPENBLAS_DIR=/opt/homebrew/opt/openblas" } else { "env" }
provider_features := 'openblas-system'
provider_bench_features := 'openblas-system'
coverage_ignore_regex := "(errors|examples|benches|src/bin).*"
coverage_line_threshold := "90"

# List of Examples

examples := ''

default:
    @just --list

# --- TESTS ---
test:
    just -f {{ justfile() }} test-unit
    just -f {{ justfile() }} test-integration integration

test-provider:
    {{ provider_env_prefix }} RUST_LOG={{ LOG }} cargo test --workspace --lib --features {{ provider_features }} -- --nocapture --show-output
    {{ provider_env_prefix }} RUST_LOG={{ LOG }} cargo test -p nabled --features {{ provider_features }} --test integration -- --nocapture --show-output

test-unit:
    RUST_LOG={{ LOG }} cargo test --workspace --lib -- --nocapture --show-output

test-all-targets:
    RUST_LOG={{ LOG }} cargo test --workspace --all-targets -- --nocapture --show-output

test-one test_name:
    RUST_LOG={{ LOG }} cargo test --workspace "{{ test_name }}" -- --nocapture --show-output

test-integration test_name:
    RUST_LOG={{ LOG }} cargo test -p nabled --test "{{ test_name }}" -- --nocapture --show-output

coverage:
    cargo llvm-cov --workspace --lib --tests --html \
     --ignore-filename-regex "{{ coverage_ignore_regex }}" \
     --output-dir coverage --open

coverage-json:
    cargo llvm-cov --workspace --lib --tests --json \
    --ignore-filename-regex "{{ coverage_ignore_regex }}" \
    --output-path coverage/cov.json

coverage-lcov:
    cargo llvm-cov --workspace --lib --tests --lcov \
    --ignore-filename-regex "{{ coverage_ignore_regex }}" \
    --output-path coverage/lcov.info

coverage-check:
    cargo llvm-cov --workspace --lib --tests --summary-only \
    --fail-under-lines {{ coverage_line_threshold }} \
    --ignore-filename-regex "{{ coverage_ignore_regex }}"

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
    cargo bench -p nabled --bench svd_benchmarks -- --quick
    cargo bench -p nabled --bench qr_benchmarks -- --quick
    cargo bench -p nabled --bench triangular_benchmarks -- --quick
    cargo bench -p nabled --bench matrix_functions_benchmarks -- --quick
    cargo bench -p nabled --bench lu_benchmarks -- --quick
    cargo bench -p nabled --bench cholesky_benchmarks -- --quick
    cargo bench -p nabled --bench eigen_benchmarks -- --quick
    cargo bench -p nabled --bench vector_benchmarks -- --quick
    cargo bench -p nabled --bench sparse_benchmarks -- --quick
    cargo bench -p nabled --bench schur_benchmarks -- --quick
    cargo bench -p nabled --bench sylvester_benchmarks -- --quick
    cargo bench -p nabled --bench optimization_benchmarks -- --quick

bench-smoke-provider:
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench svd_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench qr_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench triangular_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench matrix_functions_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench lu_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench cholesky_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench eigen_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench vector_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench sparse_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench schur_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench sylvester_benchmarks -- --quick
    {{ provider_env_prefix }} cargo bench -p nabled --features {{ provider_bench_features }} --bench optimization_benchmarks -- --quick

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
    @echo "Checking all features..."
    cargo clippy -p nabled --all-features --all-targets -- -D warnings
    @echo "Building all features..."
    cargo check -p nabled --all-features --all-targets
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
    cargo clippy --fix --all-features --all-targets --allow-dirty

# --- MAINTENANCE ---

# Run checks CI will
checks:
    cargo +nightly fmt --all -- --check --config-path ./rustfmt.toml
    cargo +nightly clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +nightly clippy --workspace --all-features --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +stable clippy --workspace --all-features --all-targets -- -D warnings
    just -f {{ justfile() }} check-provider-clippy
    just -f {{ justfile() }} test
    just -f {{ justfile() }} test-provider
    just -f {{ justfile() }} coverage-check
    just -f {{ justfile() }} check-provider
    just -f {{ justfile() }} backend-capability-report

# Verify provider-gated lint paths are checked locally.
check-provider-clippy:
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features {{ provider_features }} --all-targets -- -D warnings

# Verify provider-enabled code paths compile under stable.
check-provider:
    just -f {{ justfile() }} check-provider-clippy
    {{ provider_env_prefix }} cargo +stable check --workspace --features {{ provider_features }} --all-targets

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
    @echo "3. Use 'cargo release patch/minor/major' to create releases"
    @echo ""
    @echo "Useful commands:"
    @echo "  just release-dry patch  # Preview what would happen"
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

    # Parse version components
    IFS='.' read -r MAJOR MINOR PATCH <<< "{{ version }}"

    # Get current version for release notes
    CURRENT_VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

    # Create release branch
    git checkout -b "release-v{{ version }}"

    # Update version in root Cargo.toml (in [workspace.package] section)
    awk '/^\[workspace\.package\]/ {in_workspace_package=1} in_workspace_package && /^version = / {gsub(/"[^"]*"/, "\"{{ version }}\""); in_workspace_package=0} {print}' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

    # Update nabled version references in README files (if they exist)
    # Look for patterns like: nabled = "0.1.1" or nabled = { version = "0.1.1"
    for readme in README.md; do
        if [ -f "$readme" ]; then
            # Update simple dependency format
            sed -i '' "s/nabled = \"[0-9]*\.[0-9]*\.[0-9]*\"/nabled = \"{{ version }}\"/" "$readme" || true
            # Update version field in dependency table format
            sed -i '' "s/nabled = { version = \"[0-9]*\.[0-9]*\.[0-9]*\"/nabled = { version = \"{{ version }}\"/" "$readme" || true
        fi
    done

    # Update Cargo.lock
    cargo update --workspace

    # Generate full changelog
    echo "Generating changelog..."
    git cliff -o CHANGELOG.md

    # Generate release notes for this version
    echo "Generating release notes..."
    git cliff --unreleased --tag v{{ version }} --strip header -o RELEASE_NOTES.md

    # Stage all changes
    git add Cargo.toml Cargo.lock CHANGELOG.md RELEASE_NOTES.md
    # Also add README files if they were modified
    git add README.md 2>/dev/null || true

    # Commit
    git commit -m "chore: prepare release v{{ version }}"

    # Push branch
    git push origin "release-v{{ version }}"

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

    # Verify the version in Cargo.toml matches
    CARGO_VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    if [ "$CARGO_VERSION" != "{{ version }}" ]; then
        echo "Error: Cargo.toml version ($CARGO_VERSION) does not match requested version ({{ version }})"
        echo "Did the release PR merge successfully?"
        exit 1
    fi

    # Verify publish will work
    cargo publish --dry-run -p nabled --no-verify

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
    @echo "   - Cargo.toml (workspace.package section only)"
    @echo "   - README files (if they contain nabled version references)"
    @echo "3. Update Cargo.lock (usually done automatically with Cargo.toml change)"
    @echo "4. Generate CHANGELOG.md"
    @echo "5. Generate RELEASE_NOTES.md"
    @echo "6. Create commit and push branch"
    @echo ""
    @echo "After PR merge, 'just tag-release {{ version }}' would:"
    @echo "1. Tag the merged commit as v{{ version }}"
    @echo "2. Push the tag (triggering release workflow)"

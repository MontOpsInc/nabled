LOG := env('RUST_LOG', '')
features := ''

# List of Examples

examples := ''

default:
    @just --list

# --- TESTS ---
test:
    RUST_LOG={{ LOG }} cargo test -- --nocapture --show-output

test-one test_name:
    RUST_LOG={{ LOG }} cargo test "{{ test_name }}" -- --nocapture --show-output

test-integration test_name:
    RUST_LOG={{ LOG }} cargo test --test "{{ test_name }}" -- --nocapture --show-output

coverage:
    cargo llvm-cov --html \
     --ignore-filename-regex "(errors|examples).*" \
     --output-dir coverage --open

coverage-json:
    cargo llvm-cov --json \
    --ignore-filename-regex "(errors|examples|benches).*" \
    --output-path coverage/cov.json

coverage-lcov:
    cargo llvm-cov --lcov \
    --ignore-filename-regex "(errors|examples|benches).*" \
    --output-path coverage/lcov.info

# --- DOCS ---
docs:
    cargo doc --open

# --- BENCHES ---
[confirm('Delete all benchmark reports?')]
clear-benches:
    rm -rf target/criterion/*

bench:
    RUST_LOG={{ LOG }} cargo bench --profile=release && open ../target/criterion/report/index.html

bench-lto:
    RUST_LOG={{ LOG }} cargo bench --profile=release-lto && open ../target/criterion/report/index.html

bench-one bench:
    RUST_LOG={{ LOG }} cargo bench \
     --profile=release \
     --bench "{{ bench }}" && \
     open ../target/criterion/report/index.html

bench-one-lto bench:
    RUST_LOG={{ LOG }} cargo bench \
     --profile=release-lto \
     --bench "{{ bench }}" && \
     open ../target/criterion/report/index.html

bench-smoke:
    cargo bench --bench svd_benchmarks -- --quick
    cargo bench --bench qr_benchmarks -- --quick

bench-report:
    cargo run --bin benchmark_report

bench-smoke-report:
    just -f {{ justfile() }} bench-smoke
    just -f {{ justfile() }} bench-report

# --- EXAMPLES ---

debug-profile example:
    RUSTFLAGS='-g' cargo build --example "{{ example }}"

release-debug example:
    RUSTFLAGS='-g' cargo build --profile=release-with-debug --example "{{ example }}"
    codesign -s - -v -f --entitlements assets/mac.entitlements "target/release-with-debug/examples/{{ example }}"

release-lto example:
    cargo build --profile=release-lto --example "{{ example }}"
    codesign -s - -v -f --entitlements assets/mac.entitlements "target/release-lto/examples/{{ example }}"

example example:
    cargo run --example "{{ example }}"

example-lto example:
    cargo run --profile=release-lto --example "{{ example }}"

example-release-debug example:
    cargo run --profile=release-with-debug --example "{{ example }}"

examples:
    @for ex in {{ examples }}; do \
        echo "Running example: $ex"; \
        cargo run --example "$ex"; \
    done

# --- PROFILING ---
flamegraph example *args='':
    CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root --flamechart --open \
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
    cargo clippy --no-default-features --all-targets
    @echo "Building no features..."
    cargo check --no-default-features --all-targets
    @echo "Checking default features..."
    cargo clippy --all-targets
    @echo "Building default features..."
    cargo check --all-targets
    @echo "Checking all features..."
    cargo clippy --all-features --all-targets
    @echo "Building all features..."
    cargo check --all-features --all-targets
    @echo "Checking each feature..."
    @for feature in {{ ARGS }}; do \
        echo "Checking & Building feature: $feature"; \
        cargo clippy --no-default-features --features $feature --all-targets; \
        cargo check --no-default-features --features $feature --all-targets; \
    done
    @echo "Checking each feature with defaults..."
    @for feature in {{ ARGS }}; do \
        echo "Checking feature (with defaults): $feature"; \
        cargo clippy --features $feature --all-targets; \
        cargo check --features $feature --all-targets; \
    done
    @echo "Checking all provided features..."
    cargo clippy --no-default-features --features "{{ ARGS }}" --all-targets
    cargo check --no-default-features --features "{{ ARGS }}" --all-targets

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
    cargo +nightly fmt -- --check
    cargo +nightly clippy --all-features --all-targets
    cargo +stable clippy --all-features --all-targets -- -D warnings
    just -f {{ justfile() }} test

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

    # Update version in root Cargo.toml (in [package] section)
    # This uses a more specific pattern to only match the version under [package]
    awk '/^\[package\]/ {in_package=1} in_package && /^version = / {gsub(/"[^"]*"/, "\"{{ version }}\""); in_package=0} {print}' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

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

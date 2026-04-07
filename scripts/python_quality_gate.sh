#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
    else
        PYTHON_BIN="python3"
    fi
fi
VENV_ROOT="${PYNABLED_VENV_ROOT:-/tmp/pynabled-quality}"
COVERAGE_THRESHOLD="${PYNABLED_PYTHON_COVERAGE_THRESHOLD:-90}"
COVERAGE_DIR="${PYNABLED_PYTHON_COVERAGE_DIR:-${ROOT_DIR}/coverage/python}"
DIST_DIR="${PYNABLED_PYTHON_DIST_DIR:-${ROOT_DIR}/dist/python-quality}"
CARGO_HOME="${PYNABLED_CARGO_HOME:-/tmp/pynabled-quality-cargo-home}"
CARGO_TARGET_DIR="${PYNABLED_CARGO_TARGET_DIR:-/tmp/pynabled-quality-target}"
SMOKE_SCRIPT="${ROOT_DIR}/scripts/pynabled_smoke.py"

create_venv() {
    local venv_dir="$1"
    rm -rf "${venv_dir}"
    "${PYTHON_BIN}" -m venv "${venv_dir}"
    "${venv_dir}/bin/pip" install --upgrade pip >/dev/null
}

install_dev_tools() {
    local venv_dir="$1"
    "${venv_dir}/bin/pip" install \
        "maturin>=1.12" \
        "pytest>=7" \
        "pytest-cov>=7" \
        "pyarrow>=14" >/dev/null
}

install_packaging_tools() {
    local venv_dir="$1"
    local require_arrow="${2:-0}"
    "${venv_dir}/bin/pip" install "maturin>=1.12" >/dev/null
    if [[ "${require_arrow}" == "1" ]]; then
        "${venv_dir}/bin/pip" install "pyarrow>=14" >/dev/null
    fi
}

run_in_repo() {
    (
        cd "${ROOT_DIR}"
        "$@"
    )
}

build_and_smoke() {
    local label="$1"
    local artifact_kind="$2"
    local require_arrow="$3"
    local features="${4:-}"
    local venv_dir="${VENV_ROOT}/${label}"
    local out_dir="${DIST_DIR}/${label}"
    local feature_args=()

    create_venv "${venv_dir}"
    install_packaging_tools "${venv_dir}" "${require_arrow}"
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    if [[ -n "${features}" ]]; then
        feature_args=(--features "${features}")
    fi

    if [[ "${artifact_kind}" == "wheel" ]]; then
        if [[ "${#feature_args[@]}" -gt 0 ]]; then
            run_in_repo \
                "${venv_dir}/bin/maturin" build \
                --release \
                --interpreter "${venv_dir}/bin/python" \
                --out "${out_dir}" \
                "${feature_args[@]}"
        else
            run_in_repo \
                "${venv_dir}/bin/maturin" build \
                --release \
                --interpreter "${venv_dir}/bin/python" \
                --out "${out_dir}"
        fi
        "${venv_dir}/bin/pip" install "${out_dir}"/pynabled-*.whl >/dev/null
    else
        if [[ "${#feature_args[@]}" -gt 0 ]]; then
            run_in_repo "${venv_dir}/bin/maturin" sdist --out "${out_dir}" "${feature_args[@]}"
        else
            run_in_repo "${venv_dir}/bin/maturin" sdist --out "${out_dir}"
        fi
        "${venv_dir}/bin/pip" install "${out_dir}"/pynabled-*.tar.gz >/dev/null
    fi

    if [[ "${require_arrow}" == "1" ]]; then
        "${venv_dir}/bin/python" "${SMOKE_SCRIPT}" --require-arrow
    else
        "${venv_dir}/bin/python" "${SMOKE_SCRIPT}"
    fi
}

main() {
    local dev_venv="${VENV_ROOT}/dev"
    local python_version
    local coverage_file

    python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "pynabled quality gate requires Python >= 3.10; found ${python_version} via ${PYTHON_BIN}" >&2
        exit 1
    fi

    mkdir -p "${COVERAGE_DIR}" "${DIST_DIR}"
    mkdir -p "${CARGO_HOME}" "${CARGO_TARGET_DIR}"
    coverage_file="${COVERAGE_DIR}/.coverage"
    rm -f "${coverage_file}" "${COVERAGE_DIR}/python-coverage.xml"

    export CARGO_HOME
    export CARGO_TARGET_DIR
    export COVERAGE_FILE="${coverage_file}"

    create_venv "${dev_venv}"
    install_dev_tools "${dev_venv}"

    run_in_repo env VIRTUAL_ENV="${dev_venv}" PATH="${dev_venv}/bin:${PATH}" "${dev_venv}/bin/maturin" develop
    run_in_repo \
        "${dev_venv}/bin/python" -m pytest python/tests \
        --cov=pynabled \
        --cov-report= \
        --cov-fail-under=0 \
        -q

    run_in_repo \
        env VIRTUAL_ENV="${dev_venv}" PATH="${dev_venv}/bin:${PATH}" \
        "${dev_venv}/bin/maturin" develop --features arrow
    run_in_repo \
        "${dev_venv}/bin/python" -m pytest python/tests/test_arrow.py \
        --cov=pynabled \
        --cov-append \
        --cov-report= \
        --cov-fail-under=0 \
        -q
    run_in_repo \
        "${dev_venv}/bin/python" -m coverage xml -o "${COVERAGE_DIR}/python-coverage.xml"
    run_in_repo \
        "${dev_venv}/bin/python" -m coverage report \
        --show-missing \
        --fail-under="${COVERAGE_THRESHOLD}"

    build_and_smoke "wheel-default" "wheel" "0"
    build_and_smoke "sdist-default" "sdist" "0"
    build_and_smoke "wheel-provider" "wheel" "0" "openblas-system"
    build_and_smoke "wheel-accelerator" "wheel" "0" "accelerator-rayon"
    build_and_smoke "wheel-combined" "wheel" "1" "openblas-system accelerator-rayon arrow"
}

main "$@"

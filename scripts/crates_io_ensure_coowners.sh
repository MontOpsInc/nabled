#!/usr/bin/env bash
# Ensure a crates.io user is co-owner of every workspace publish crate.
# Requires CARGO_REGISTRY_TOKEN with change-owners scope.
set -euo pipefail

USER="${1:-${CRATES_IO_COOWNER:-NiklausParcell}}"
TOKEN="${CARGO_REGISTRY_TOKEN:?CARGO_REGISTRY_TOKEN must be set}"
# CI sets CARGO_TERM_COLOR=always; disable colors so grep can match cargo search output.
export CARGO_TERM_COLOR=never

# Keep in sync with docs/PUBLISH_CHECKLIST.md publish order.
WORKSPACE_CRATES=(
  nabled-core
  nabled-linalg
  nabled-ml
  nabled-embeddings
  nabled-kinematics
  nabled-model
  nabled-dynamics
  nabled-control
  nabled-sensor
  nabled-sim
  nabled
)

crate_exists() {
  local pkg="$1"
  if cargo search "${pkg}" --limit 1 --token "${TOKEN}" 2>/dev/null | grep -q "^${pkg} = "; then
    return 0
  fi
  return 1
}

add_owner() {
  local pkg="$1"

  if ! crate_exists "${pkg}"; then
    echo "Skip: ${pkg} is not published on crates.io yet"
    return 0
  fi

  echo "Adding owner ${USER} to ${pkg}..."
  if cargo owner --add "${USER}" "${pkg}" --token "${TOKEN}"; then
    echo "OK: ${pkg}"
    return 0
  fi
  if cargo owner --list "${pkg}" --token "${TOKEN}" | grep -Fq "${USER}"; then
    echo "Skip: ${USER} is already an owner of ${pkg}"
    return 0
  fi
  echo "Error: could not add owner to ${pkg}" >&2
  return 1
}

FAILED=0
for pkg in "${WORKSPACE_CRATES[@]}"; do
  add_owner "${pkg}" || FAILED=1
done

echo ""
if [ "${FAILED}" -eq 1 ]; then
  echo "One or more crates failed owner-add for ${USER}." >&2
  echo "Verify CARGO_REGISTRY_TOKEN has change-owners scope, or run the crates-io-add-owner workflow." >&2
  exit 1
fi
echo "Done. Verify Owners on crates.io for: ${USER}"

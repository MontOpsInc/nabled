#!/usr/bin/env bash
# Ensure a crates.io user is co-owner of every workspace publish crate.
# Requires CARGO_REGISTRY_TOKEN with change-owners scope.
set -euo pipefail

USER="${1:-${CRATES_IO_COOWNER:-NiklausParcell}}"
TOKEN="${CARGO_REGISTRY_TOKEN:?CARGO_REGISTRY_TOKEN must be set}"
export CARGO_TERM_COLOR=never
export NO_COLOR=1

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

strip_ansi() {
  sed $'s/\x1b\\[[0-9;]*m//g'
}

crate_exists() {
  local pkg="$1"
  local line
  # `cargo search` is public-index only; it does not accept `--token` (unlike `cargo owner`).
  line="$(cargo search "${pkg}" --limit 1 2>/dev/null | strip_ansi | head -1 || true)"
  [[ "${line}" == "${pkg} = "* ]]
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
  if cargo owner --list "${pkg}" --token "${TOKEN}" 2>/dev/null | strip_ansi | grep -Fq "${USER}"; then
    echo "Skip: ${USER} is already an owner of ${pkg}"
    return 0
  fi
  echo "Error: could not add owner to ${pkg}" >&2
  return 1
}

FAILED=0
PUBLISHED=0
for pkg in "${WORKSPACE_CRATES[@]}"; do
  if crate_exists "${pkg}"; then
    PUBLISHED=$((PUBLISHED + 1))
  fi
  add_owner "${pkg}" || FAILED=1
done

echo ""
if [ "${PUBLISHED}" -eq 0 ]; then
  echo "Error: no workspace crates found on crates.io; check cargo search / index sync." >&2
  exit 1
fi
if [ "${FAILED}" -eq 1 ]; then
  echo "One or more crates failed owner-add for ${USER}." >&2
  echo "Verify CARGO_REGISTRY_TOKEN has change-owners scope, or run the crates-io-add-owner workflow." >&2
  exit 1
fi
echo "Done. Verify Owners on crates.io for: ${USER}"

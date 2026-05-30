# Release v0.0.10

## Physical AI crates.io publish fix

- Expanded README and `description` metadata for all six Physical AI domain crates.
- Hardened `.github/workflows/release.yml` with rate-limit pacing, publish resume, 429 retry, and split GitHub Release job.
- Added `.github/workflows/crates-io-add-owner.yml` for one-shot co-owner add on crates.io.
- Updated `docs/PUBLISH_CHECKLIST.md` with publish order, rate limits, and co-ownership policy.

## Maintainer actions after merge

1. Run **crates.io add owner** workflow on `main` (verify George + NiklausParcell on Owners).
2. `just tag-release 0.0.10`
3. After Release workflow completes, `just tag-pypi-release`

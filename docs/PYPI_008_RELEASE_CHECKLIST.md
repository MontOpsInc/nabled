# PyPI 0.0.8 Release-Day Checklist

Last updated: 2026-05-23

Maintainer-only steps for shipping `pynabled` **0.0.8**. Physical AI Python depth ships in **0.0.9** (`N-PY-PAI-006`).

## Preconditions

1. Branch merged to release target (`main` or release branch per `docs/EXECUTION_TRACKER.md` D-253).
2. `pyproject.toml` version is `0.0.8` on the release commit (do **not** include 0.0.9 Physical AI bumps in this tag).
3. Local gates green:
   - `just checks` (OpenBLAS or `NABLED_PROVIDER_FEATURES=netlib-system` per `BUILD.md`)
   - `just python-quality` (dense/ML scope for 0.0.8)

## Release steps

| Step | Command / action | Expected |
|------|------------------|----------|
| 1 | Final gate | `just checks` exit 0 |
| 2 | Python quality | `just python-quality` exit 0 |
| 3 | Tag | `git tag -a pypi-v0.0.8 -m "pynabled 0.0.8"` |
| 4 | Push tag | `git push origin pypi-v0.0.8` |
| 5 | Observe CI | GitHub publish workflow (Trusted Publishing / OIDC) completes |
| 6 | Install smoke | `pip install pynabled==0.0.8 && python scripts/pynabled_smoke.py` |
| 7 | Record | Update `docs/EXECUTION_TRACKER.md` publish observation |

## Post-0.0.8

1. Bump `pyproject.toml` to `0.0.9` on development branch for Physical AI parity (`N-PY-PAI-006`).
2. Do not advertise Physical AI tree FK / `IkWorkspace` in 0.0.8 release notes unless already present in the published wheel.

## Rollback

If publish fails, fix workflow or artifact issues, delete erroneous tag locally/remotely only with maintainer approval, and re-tag after gates pass.

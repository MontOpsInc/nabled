from __future__ import annotations

import pytest

from build_backend import pynabled_backend


def test_translates_provider_and_accelerator_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYNABLED_PROVIDER", raising=False)
    monkeypatch.delenv("PYNABLED_ACCELERATORS", raising=False)
    monkeypatch.delenv("PYNABLED_FEATURES", raising=False)
    translated = pynabled_backend._translated_config_settings(
        {
            "pynabled-provider": "openblas-system,magma-system",
            "pynabled-accelerators": "rayon",
        }
    )

    assert translated == {
        "pynabled-provider": "openblas-system,magma-system",
        "pynabled-accelerators": "rayon",
        "build-args": "--features=openblas-system,magma-system,accelerator-rayon",
    }


def test_env_aliases_feed_build_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYNABLED_PROVIDER", "netlib-static")
    monkeypatch.setenv("PYNABLED_ACCELERATORS", "wgpu")
    monkeypatch.setenv("PYNABLED_FEATURES", "arrow")

    translated = pynabled_backend._translated_config_settings(None)

    assert translated == {
        "build-args": "--features=netlib-static,accelerator-wgpu,arrow"
    }


def test_conflicting_lapack_provider_selection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYNABLED_PROVIDER", raising=False)
    with pytest.raises(pynabled_backend.PynabledBuildConfigError, match="at most one LAPACK"):
        pynabled_backend._translated_config_settings(
            {"pynabled-provider": "openblas-system netlib-system"}
        )


def test_raw_feature_args_conflict_with_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MATURIN_PEP517_ARGS", "--features=arrow")
    with pytest.raises(pynabled_backend.PynabledBuildConfigError, match="not both"):
        pynabled_backend._translated_config_settings({"pynabled-accelerators": "rayon"})


def test_non_feature_build_args_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MATURIN_PEP517_ARGS", raising=False)
    translated = pynabled_backend._translated_config_settings(
        {
            "build-args": "--compatibility=linux",
            "pynabled-features": "arrow",
        }
    )

    assert translated == {
        "build-args": "--compatibility=linux --features=arrow",
        "pynabled-features": "arrow",
    }

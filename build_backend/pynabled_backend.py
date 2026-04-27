"""Thin PEP 517 shim that translates friendly pynabled feature settings to maturin."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

_LAPACK_PROVIDER_ORDER = (
    "openblas-system",
    "openblas-static",
    "netlib-system",
    "netlib-static",
)
_PROVIDER_ORDER = _LAPACK_PROVIDER_ORDER + ("magma-system",)
_ACCELERATOR_ORDER = ("accelerator-rayon", "accelerator-wgpu")
_EXTRA_FEATURE_ORDER = ("arrow",)
_ALL_FEATURE_ORDER = _PROVIDER_ORDER + _ACCELERATOR_ORDER + _EXTRA_FEATURE_ORDER

_PROVIDER_ALIASES = {
    "openblas": "openblas-system",
    "openblas-system": "openblas-system",
    "openblas-static": "openblas-static",
    "netlib": "netlib-system",
    "netlib-system": "netlib-system",
    "netlib-static": "netlib-static",
    "magma": "magma-system",
    "magma-system": "magma-system",
}
_ACCELERATOR_ALIASES = {
    "rayon": "accelerator-rayon",
    "accelerator-rayon": "accelerator-rayon",
    "wgpu": "accelerator-wgpu",
    "accelerator-wgpu": "accelerator-wgpu",
}
_FEATURE_ALIASES = {
    **_PROVIDER_ALIASES,
    **_ACCELERATOR_ALIASES,
    "arrow": "arrow",
}

_SHIM_KEYS = {
    "provider": ("pynabled-provider", "pynabled_provider"),
    "accelerators": ("pynabled-accelerators", "pynabled_accelerators"),
    "features": ("pynabled-features", "pynabled_features"),
}


class PynabledBuildConfigError(ValueError):
    """Raised when friendly pynabled build settings are invalid."""


def _maturin_backend():
    import maturin

    return maturin


def _flatten_config_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_config_value(item))
        return values
    return [str(value)]


def _config_values(config_settings: Mapping[str, Any] | None, keys: Sequence[str]) -> list[str]:
    if not config_settings:
        return []
    values: list[str] = []
    for key in keys:
        if key in config_settings:
            values.extend(_flatten_config_value(config_settings[key]))
    return values


def _split_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for token in value.replace(",", " ").split():
            normalized = token.strip()
            if normalized:
                tokens.append(normalized)
    return tokens


def _normalize_tokens(
    values: Sequence[str],
    *,
    aliases: Mapping[str, str],
    field_name: str,
) -> set[str]:
    normalized: set[str] = set()
    for token in _split_tokens(values):
        canonical = aliases.get(token.lower())
        if canonical is None:
            allowed = ", ".join(sorted(aliases))
            raise PynabledBuildConfigError(
                f"Unknown {field_name} value `{token}`. Allowed values: {allowed}."
            )
        normalized.add(canonical)
    return normalized


def _requested_features(config_settings: Mapping[str, Any] | None) -> list[str]:
    provider_values = [os.getenv("PYNABLED_PROVIDER", "")]
    provider_values.extend(_config_values(config_settings, _SHIM_KEYS["provider"]))

    accelerator_values = [os.getenv("PYNABLED_ACCELERATORS", "")]
    accelerator_values.extend(_config_values(config_settings, _SHIM_KEYS["accelerators"]))

    feature_values = [os.getenv("PYNABLED_FEATURES", "")]
    feature_values.extend(_config_values(config_settings, _SHIM_KEYS["features"]))

    provider_features = _normalize_tokens(
        provider_values,
        aliases=_PROVIDER_ALIASES,
        field_name="provider feature",
    )
    accelerator_features = _normalize_tokens(
        accelerator_values,
        aliases=_ACCELERATOR_ALIASES,
        field_name="accelerator feature",
    )
    explicit_features = _normalize_tokens(
        feature_values,
        aliases=_FEATURE_ALIASES,
        field_name="feature",
    )

    combined = provider_features | accelerator_features | explicit_features
    lapack_providers = [name for name in _LAPACK_PROVIDER_ORDER if name in combined]
    if len(lapack_providers) > 1:
        choices = ", ".join(lapack_providers)
        raise PynabledBuildConfigError(
            "pynabled source builds can select at most one LAPACK provider feature; "
            f"got {choices}."
        )

    ordered: list[str] = []
    ordered.extend(lapack_providers)
    for feature in _PROVIDER_ORDER[len(_LAPACK_PROVIDER_ORDER) :]:
        if feature in combined:
            ordered.append(feature)
    for feature in _ACCELERATOR_ORDER:
        if feature in combined:
            ordered.append(feature)
    for feature in _EXTRA_FEATURE_ORDER:
        if feature in combined:
            ordered.append(feature)

    extra = combined.difference(ordered)
    if extra:
        ordered.extend(sorted(extra))
    return ordered


def _contains_explicit_feature_selector(raw_args: str) -> bool:
    if not raw_args.strip():
        return False
    try:
        tokens = shlex.split(raw_args)
    except ValueError:
        tokens = raw_args.split()
    for token in tokens:
        if token in {"--features", "--all-features", "-F"}:
            return True
        if token.startswith("--features=") or token.startswith("-F="):
            return True
    return False


def _translated_config_settings(config_settings: Mapping[str, Any] | None) -> dict[str, Any] | None:
    features = _requested_features(config_settings)
    if not features:
        return dict(config_settings) if config_settings is not None else None

    raw_build_args_values = _config_values(config_settings, ("build-args",))
    raw_build_args = " ".join(raw_build_args_values).strip()
    raw_env_build_args = os.getenv("MATURIN_PEP517_ARGS", "")

    if _contains_explicit_feature_selector(raw_build_args) or _contains_explicit_feature_selector(
        raw_env_build_args
    ):
        raise PynabledBuildConfigError(
            "Use either the friendly `pynabled-*` build settings / `PYNABLED_*` environment "
            "variables or raw maturin feature arguments (`build-args` / `MATURIN_PEP517_ARGS`), "
            "but not both."
        )

    translated = dict(config_settings) if config_settings is not None else {}
    feature_arg = f"--features={','.join(features)}"
    if raw_build_args:
        translated["build-args"] = f"{raw_build_args} {feature_arg}"
    else:
        translated["build-args"] = feature_arg
    return translated


def get_requires_for_build_wheel(config_settings: Mapping[str, Any] | None = None) -> list[str]:
    return _maturin_backend().get_requires_for_build_wheel(_translated_config_settings(config_settings))


def get_requires_for_build_editable(config_settings: Mapping[str, Any] | None = None) -> list[str]:
    return _maturin_backend().get_requires_for_build_editable(
        _translated_config_settings(config_settings)
    )


def get_requires_for_build_sdist(config_settings: Mapping[str, Any] | None = None) -> list[str]:
    return _maturin_backend().get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return _maturin_backend().prepare_metadata_for_build_wheel(
        metadata_directory,
        _translated_config_settings(config_settings),
    )


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _maturin_backend().build_wheel(
        wheel_directory,
        _translated_config_settings(config_settings),
        metadata_directory,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return _maturin_backend().prepare_metadata_for_build_editable(
        metadata_directory,
        _translated_config_settings(config_settings),
    )


def build_editable(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _maturin_backend().build_editable(
        wheel_directory,
        _translated_config_settings(config_settings),
        metadata_directory,
    )


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return _maturin_backend().build_sdist(sdist_directory, config_settings)

from __future__ import annotations

import pynabled


KNOWN_BUILD_FEATURES = {
    "accelerator-rayon",
    "accelerator-wgpu",
    "arrow",
    "magma-system",
    "netlib-static",
    "netlib-system",
    "openblas-static",
    "openblas-system",
    "physical-ai",
    "signal",
}


def test_build_features_are_sorted_known_names():
    features = pynabled.build_features()
    assert features == tuple(sorted(features))
    assert len(features) == len(set(features))
    assert set(features) <= KNOWN_BUILD_FEATURES

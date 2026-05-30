"""Signal processing bindings (requires `signal` feature at build time)."""

from __future__ import annotations

try:
    from pynabled._pynabled import autocorrelation_full_py as _autocorrelation_full
    from pynabled._pynabled import bin_to_hz_py as bin_to_hz
    from pynabled._pynabled import dominant_frequency_py as _dominant_frequency
    from pynabled._pynabled import irfft_into_py as _irfft_into
    from pynabled._pynabled import irfft_py as _irfft
    from pynabled._pynabled import rfft_into_py as _rfft_into
    from pynabled._pynabled import rfft_py as _rfft
except ImportError as exc:  # pragma: no cover - exercised when extension lacks signal
    raise ImportError(
        "pynabled.signal requires a build compiled with the `signal` feature",
    ) from exc


def rfft(signal, *, out=None):
    if out is None:
        return _rfft(signal)
    _rfft_into(signal, out)
    return out


def irfft(spectrum, *, out=None):
    if out is None:
        return _irfft(spectrum)
    _irfft_into(spectrum, out)
    return out


def dominant_frequency(signal):
    return _dominant_frequency(signal)


def autocorrelation_full(signal, *, out=None):
    if out is None:
        return _autocorrelation_full(signal)
    out[:] = _autocorrelation_full(signal)
    return out

"""Control synthesis bindings."""

from __future__ import annotations

from pynabled._pynabled import LqrResult as LqrResult
from pynabled._pynabled import controllability_gramian_py as controllability_gramian
from pynabled._pynabled import dare_residual_norm_py as dare_residual_norm
from pynabled._pynabled import dare_solve_py as dare_solve
from pynabled._pynabled import discrete_lqr_into as _discrete_lqr_into
from pynabled._pynabled import discrete_lqr_py as _discrete_lqr
from pynabled._pynabled import luenberger_gain_py as luenberger_gain
from pynabled._pynabled import place_poles_py as place_poles


def discrete_lqr(a, b, q, r, *, out: LqrResult | None = None) -> LqrResult:
    if out is None:
        return _discrete_lqr(a, b, q, r)
    _discrete_lqr_into(a, b, q, r, out.gain, out.riccati)
    return out

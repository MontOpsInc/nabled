"""Rigid-body dynamics bindings."""

from __future__ import annotations

from pynabled._pynabled import DynamicsConfig as DynamicsConfig
from pynabled._pynabled import forward_dynamics as _forward_dynamics
from pynabled._pynabled import forward_dynamics_into as _forward_dynamics_into
from pynabled._pynabled import mass_matrix_py as _mass_matrix
from pynabled._pynabled import rnea as _rnea
from pynabled._pynabled import rnea_into as _rnea_into


def rnea(model, chain, q, qd, qdd, *, config=None, out=None):
    if out is None:
        return _rnea(model, chain, q, qd, qdd, config)
    _rnea_into(model, chain, q, qd, qdd, out, config)
    return out


def mass_matrix(model, chain, q, *, out=None):
    if out is None:
        return _mass_matrix(model, chain, q)
    out[:] = _mass_matrix(model, chain, q)
    return out


def forward_dynamics(model, chain, q, qd, tau, *, config=None, out=None):
    if out is None:
        return _forward_dynamics(model, chain, q, qd, tau, config)
    _forward_dynamics_into(model, chain, q, qd, tau, out, config)
    return out

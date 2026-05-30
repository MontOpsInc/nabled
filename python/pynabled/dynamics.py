"""Rigid-body dynamics bindings."""

from __future__ import annotations

from pynabled._pynabled import DynamicsConfig as DynamicsConfig
from pynabled._pynabled import forward_dynamics as _forward_dynamics
from pynabled._pynabled import forward_dynamics_into as _forward_dynamics_into
from pynabled._pynabled import forward_dynamics_tree_into_py as _forward_dynamics_tree_into
from pynabled._pynabled import forward_dynamics_tree_py as _forward_dynamics_tree
from pynabled._pynabled import mass_matrix_py as _mass_matrix
from pynabled._pynabled import mass_matrix_tree_into_py as _mass_matrix_tree_into
from pynabled._pynabled import mass_matrix_tree_py as _mass_matrix_tree
from pynabled._pynabled import rnea as _rnea
from pynabled._pynabled import rnea_into as _rnea_into
from pynabled._pynabled import rnea_tree_into_py as _rnea_tree_into
from pynabled._pynabled import rnea_tree_py as _rnea_tree


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


def rnea_tree(model, base_link, ee_link, q, qd, qdd, *, config=None, out=None):
    """Branch-routed RNEA on a tree model.

    Inputs ``q``, ``qd``, ``qdd`` are in full-model actuated ordering. Returns a
    full-ordering torque vector with non-branch entries zero. Passing ``out``
    writes into that buffer in-place.
    """
    if out is None:
        return _rnea_tree(model, base_link, ee_link, q, qd, qdd, config)
    _rnea_tree_into(model, base_link, ee_link, q, qd, qdd, out, config)
    return out


def mass_matrix_tree(model, base_link, ee_link, q, *, config=None, out=None):
    """Branch-routed CRBA mass matrix on a tree model."""
    if out is None:
        return _mass_matrix_tree(model, base_link, ee_link, q, config)
    _mass_matrix_tree_into(model, base_link, ee_link, q, out, config)
    return out


def forward_dynamics_tree(
    model, base_link, ee_link, q, qd, tau, *, config=None, out=None
):
    """Branch-routed forward dynamics on a tree model."""
    if out is None:
        return _forward_dynamics_tree(model, base_link, ee_link, q, qd, tau, config)
    _forward_dynamics_tree_into(model, base_link, ee_link, q, qd, tau, out, config)
    return out

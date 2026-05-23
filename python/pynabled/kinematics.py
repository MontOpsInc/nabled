"""Kinematics bindings (FK, Jacobian, IK)."""

from __future__ import annotations

from pynabled._pynabled import ChainSpec as ChainSpec
from pynabled._pynabled import IkConfig as IkConfig
from pynabled._pynabled import IkResult as IkResult
from pynabled._pynabled import end_effector_pose_py as _end_effector_pose
from pynabled._pynabled import fk as _fk
from pynabled._pynabled import inverse_kinematics_dls_py as _inverse_kinematics_dls
from pynabled._pynabled import jacobian_py as _jacobian
from pynabled._pynabled import jacobian_translation_py as _jacobian_translation
from pynabled._pynabled import pose_error_py as _pose_error


def end_effector_pose(chain: ChainSpec, q):
    return _end_effector_pose(chain, q)


def fk(chain: ChainSpec, q):
    return _fk(chain, q)


def jacobian(chain: ChainSpec, q, *, out=None):
    if out is None:
        return _jacobian(chain, q)
    out[:] = _jacobian(chain, q)
    return out


def jacobian_translation(chain: ChainSpec, q, *, out=None):
    if out is None:
        return _jacobian_translation(chain, q)
    out[:] = _jacobian_translation(chain, q)
    return out


def pose_error(achieved, target):
    return _pose_error(achieved, target)


def inverse_kinematics_dls(chain: ChainSpec, q_init, target, config=None) -> IkResult:
    return _inverse_kinematics_dls(chain, q_init, target, config)

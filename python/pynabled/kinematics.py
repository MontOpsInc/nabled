"""Kinematics bindings (FK, Jacobian, IK)."""

from __future__ import annotations

from pynabled._pynabled import ChainSpec as ChainSpec
from pynabled._pynabled import IkConfig as IkConfig
from pynabled._pynabled import IkResult as IkResult
from pynabled._pynabled import IkWorkspace as IkWorkspace
from pynabled._pynabled import end_effector_pose_py as _end_effector_pose
from pynabled._pynabled import end_effector_pose_tree_py as _end_effector_pose_tree
from pynabled._pynabled import fk as _fk
from pynabled._pynabled import inverse_kinematics_dls_into_py as _inverse_kinematics_dls_into
from pynabled._pynabled import inverse_kinematics_dls_py as _inverse_kinematics_dls
from pynabled._pynabled import inverse_kinematics_tree_dls_py as _inverse_kinematics_tree_dls
from pynabled._pynabled import jacobian_py as _jacobian
from pynabled._pynabled import jacobian_tree_py as _jacobian_tree
from pynabled._pynabled import jacobian_translation_py as _jacobian_translation
from pynabled._pynabled import link_transforms_tree_py as _link_transforms_tree
from pynabled._pynabled import pose_error_into_py as _pose_error_into
from pynabled._pynabled import pose_error_py as _pose_error


def end_effector_pose(chain: ChainSpec, q):
    return _end_effector_pose(chain, q)


def end_effector_pose_tree(model, base_link: str, ee_link: str, q):
    return _end_effector_pose_tree(model, base_link, ee_link, q)


def fk(chain: ChainSpec, q):
    return _fk(chain, q)


def jacobian(chain: ChainSpec, q, *, out=None):
    if out is None:
        return _jacobian(chain, q)
    out[:] = _jacobian(chain, q)
    return out


def jacobian_tree(model, base_link: str, ee_link: str, q, *, out=None):
    if out is None:
        return _jacobian_tree(model, base_link, ee_link, q)
    out[:] = _jacobian_tree(model, base_link, ee_link, q)
    return out


def jacobian_translation(chain: ChainSpec, q, *, out=None):
    if out is None:
        return _jacobian_translation(chain, q)
    out[:] = _jacobian_translation(chain, q)
    return out


def link_transforms_tree(model, q):
    return _link_transforms_tree(model, q)


def pose_error(achieved, target, *, out=None):
    if out is None:
        return _pose_error(achieved, target)
    _pose_error_into(achieved, target, out)
    return out


def inverse_kinematics_dls(
    chain: ChainSpec,
    q_init,
    target,
    config=None,
    *,
    workspace: IkWorkspace | None = None,
    out=None,
) -> IkResult:
    if out is None:
        return _inverse_kinematics_dls(chain, q_init, target, config)
    return _inverse_kinematics_dls_into(chain, q_init, target, out, config, workspace)


def inverse_kinematics_tree_dls(
    model,
    base_link: str,
    ee_link: str,
    q_init,
    target,
    config=None,
) -> IkResult:
    return _inverse_kinematics_tree_dls(model, base_link, ee_link, q_init, target, config)

"""Rigid-body geometry bindings."""

from __future__ import annotations

from pynabled._pynabled import Transform3 as Transform3
from pynabled._pynabled import quat_from_axis_angle as _quat_from_axis_angle
from pynabled._pynabled import quat_to_rotation_matrix as _quat_to_rotation_matrix
from pynabled._pynabled import se3_compose as _se3_compose
from pynabled._pynabled import se3_compose_into as _se3_compose_into
from pynabled._pynabled import se3_exp as _se3_exp
from pynabled._pynabled import se3_log as _se3_log
from pynabled._pynabled import se3_log_into as _se3_log_into
from pynabled._pynabled import so3_compose as _so3_compose
from pynabled._pynabled import transform3_from_parts as _transform3_from_parts
from pynabled._pynabled import transform3_to_parts as _transform3_to_parts


def transform3_from_parts(rotation, translation):
    return _transform3_from_parts(rotation, translation)


def transform3_to_parts(transform):
    return _transform3_to_parts(transform)


def se3_compose(left: Transform3, right: Transform3, *, out: Transform3 | None = None) -> Transform3:
    if out is None:
        return _se3_compose(left, right)
    _se3_compose_into(left, right, out)
    return out


def se3_log(transform: Transform3, *, out=None):
    if out is None:
        return _se3_log(transform)
    _se3_log_into(transform, out)
    return out


def se3_exp(twist):
    return _se3_exp(twist)


def quat_from_axis_angle(axis, angle):
    return _quat_from_axis_angle(axis, angle)


def quat_to_rotation_matrix(q):
    return _quat_to_rotation_matrix(q)


def so3_compose(left, right, *, out=None):
    if out is None:
        return _so3_compose(left, right)
    out[:] = _so3_compose(left, right)
    return out

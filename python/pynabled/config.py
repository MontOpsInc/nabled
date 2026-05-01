"""Typed Python config objects for pynabled iterative, Jacobian, and optimizer APIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IterativeConfig:
    tolerance: float | None = None
    max_iterations: int | None = None


@dataclass(slots=True)
class JacobianConfig:
    step_size: float | None = None
    tolerance: float | None = None
    max_iterations: int | None = None


@dataclass(slots=True)
class LineSearchConfig:
    initial_step: float | None = None
    contraction: float | None = None
    sufficient_decrease: float | None = None
    max_iterations: int | None = None


@dataclass(slots=True)
class GradientDescentConfig:
    learning_rate: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass(slots=True)
class AdamConfig:
    learning_rate: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass(slots=True)
class MomentumConfig:
    learning_rate: float | None = None
    momentum: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass(slots=True)
class RMSPropConfig:
    learning_rate: float | None = None
    rho: float | None = None
    epsilon: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass(slots=True)
class ProjectedGradientConfig:
    learning_rate: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None


@dataclass(slots=True)
class BFGSConfig:
    step_size: float | None = None
    max_iterations: int | None = None
    tolerance: float | None = None
    curvature_tolerance: float | None = None


__all__ = [
    "AdamConfig",
    "BFGSConfig",
    "GradientDescentConfig",
    "IterativeConfig",
    "JacobianConfig",
    "LineSearchConfig",
    "MomentumConfig",
    "ProjectedGradientConfig",
    "RMSPropConfig",
]

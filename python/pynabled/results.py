"""Typed Python result objects for pynabled workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SvdResult:
    u: np.ndarray
    singular_values: np.ndarray
    vt: np.ndarray


@dataclass(slots=True)
class QrResult:
    q: np.ndarray
    r: np.ndarray
    rank: int
    p: np.ndarray | None = None


@dataclass(slots=True)
class LuResult:
    l: np.ndarray
    u: np.ndarray
    pivots: np.ndarray | None = None
    permutation_sign: int | None = None


@dataclass(slots=True)
class MixedSolveResult:
    solution: np.ndarray
    refinement_iterations: int


@dataclass(slots=True)
class LogDetResult:
    sign: int
    ln_abs_det: float


@dataclass(slots=True)
class CholeskyResult:
    l: np.ndarray


@dataclass(slots=True)
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(slots=True)
class GeneralizedEigenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(slots=True)
class NonsymmetricEigenResult:
    eigenvalues: np.ndarray
    schur_vectors: np.ndarray


@dataclass(slots=True)
class BalancedNonsymmetricResult:
    balanced_matrix: np.ndarray
    balancing_diagonal: np.ndarray


@dataclass(slots=True)
class NonsymmetricBiEigenResult:
    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray
    balancing_diagonal: np.ndarray
    balanced_matrix: np.ndarray


@dataclass(slots=True)
class SchurResult:
    q: np.ndarray
    t: np.ndarray


@dataclass(slots=True)
class PolarResult:
    u: np.ndarray
    p: np.ndarray


@dataclass(slots=True)
class PcaResult:
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    scores: np.ndarray


@dataclass(slots=True)
class RegressionResult:
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float


@dataclass(slots=True)
class MixedSylvesterResult:
    solution: np.ndarray
    refinement_iterations: int


@dataclass(slots=True)
class Hosvd3Result:
    core: np.ndarray
    u0: np.ndarray
    u1: np.ndarray
    u2: np.ndarray


@dataclass(slots=True)
class HosvdNdResult:
    core: np.ndarray
    factors: list[np.ndarray]


@dataclass(slots=True)
class CpAls3Result:
    weights: np.ndarray
    factor_0: np.ndarray
    factor_1: np.ndarray
    factor_2: np.ndarray


@dataclass(slots=True)
class CpAlsNdResult:
    weights: np.ndarray
    factors: list[np.ndarray]
    shape: tuple[int, ...]


@dataclass(slots=True)
class CpErrorMetrics:
    signal_norm: float
    residual_norm: float
    relative_error: float
    fit: float


@dataclass(slots=True)
class CpConvergenceReport:
    iterations_run: int
    converged: bool
    final_max_factor_change: float


@dataclass(slots=True)
class CpAlsReport:
    convergence: CpConvergenceReport
    metrics: CpErrorMetrics


@dataclass(slots=True)
class TensorTrainResult:
    cores: list[np.ndarray]


__all__ = [
    "CholeskyResult",
    "CpAls3Result",
    "CpAlsNdResult",
    "CpAlsReport",
    "CpConvergenceReport",
    "CpErrorMetrics",
    "BalancedNonsymmetricResult",
    "EigenResult",
    "GeneralizedEigenResult",
    "Hosvd3Result",
    "HosvdNdResult",
    "LogDetResult",
    "LuResult",
    "MixedSolveResult",
    "MixedSylvesterResult",
    "NonsymmetricBiEigenResult",
    "NonsymmetricEigenResult",
    "PcaResult",
    "PolarResult",
    "QrResult",
    "RegressionResult",
    "SchurResult",
    "SvdResult",
    "TensorTrainResult",
]

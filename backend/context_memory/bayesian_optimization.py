"""
Bayesian Optimization for Parameter Tuning.

Implements a lightweight, from-scratch Bayesian optimisation loop using
a Gaussian Process (GP) surrogate model and Expected Improvement (EI)
as the acquisition function.

This module is self-contained and uses only Python's standard library
plus numpy-free math.  The GP uses an RBF kernel and is limited to
small parameter spaces (suitable for tuning a handful of retrieval
or ranking weights).

Limitations:
    - Designed for low-dimensional (1-5 parameters), small-budget
      optimisation (tens of evaluations).
    - The GP implementation inverts the kernel matrix directly
      (O(n^3)), which is fine for n < 200 evaluations.
    - No noise modelling; assumes deterministic objectives.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Callable, Optional, Dict


# -------------------------------------------------------------------
# Kernel
# -------------------------------------------------------------------

def rbf_kernel(
    x1: List[float],
    x2: List[float],
    length_scale: float = 1.0,
    variance: float = 1.0,
) -> float:
    """Radial Basis Function (squared-exponential) kernel."""
    sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return variance * math.exp(-0.5 * sq_dist / (length_scale ** 2))


def kernel_matrix(
    X: List[List[float]],
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-6,
) -> List[List[float]]:
    """Compute the kernel matrix K with a small diagonal noise term."""
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = rbf_kernel(X[i], X[j], length_scale, variance)
            if i == j:
                K[i][j] += noise
    return K


# -------------------------------------------------------------------
# Linear algebra helpers (stdlib-only)
# -------------------------------------------------------------------

def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition A = L L^T. Returns lower-triangular L."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val < 0:
                    val = 1e-10  # numerical fix
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] != 0 else 0.0
    return L


def _solve_triangular_lower(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve L x = b where L is lower-triangular."""
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / L[i][i]
    return x


def _solve_triangular_upper(U: List[List[float]], b: List[float]) -> List[float]:
    """Solve U x = b where U is upper-triangular."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def _transpose(M: List[List[float]]) -> List[List[float]]:
    n = len(M)
    return [[M[j][i] for j in range(n)] for i in range(n)]


def _solve_via_cholesky(K: List[List[float]], y: List[float]) -> List[float]:
    """Solve K alpha = y using Cholesky decomposition."""
    L = _cholesky(K)
    z = _solve_triangular_lower(L, y)
    alpha = _solve_triangular_upper(_transpose(L), z)
    return alpha


# -------------------------------------------------------------------
# GP prediction
# -------------------------------------------------------------------

def gp_predict(
    X_train: List[List[float]],
    y_train: List[float],
    x_test: List[float],
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise: float = 1e-6,
) -> Tuple[float, float]:
    """Predict mean and variance at x_test given training data.

    Returns (mu, sigma^2).
    """
    if not X_train:
        return 0.0, variance

    K = kernel_matrix(X_train, length_scale, variance, noise)
    alpha = _solve_via_cholesky(K, y_train)

    # k_star: kernel between x_test and each training point
    k_star = [rbf_kernel(x_test, xi, length_scale, variance) for xi in X_train]

    # Predictive mean
    mu = sum(a * k for a, k in zip(alpha, k_star))

    # Predictive variance
    L = _cholesky(K)
    v = _solve_triangular_lower(L, k_star)
    k_ss = rbf_kernel(x_test, x_test, length_scale, variance) + noise
    sigma2 = k_ss - sum(vi * vi for vi in v)
    sigma2 = max(sigma2, 1e-10)

    return mu, sigma2


# -------------------------------------------------------------------
# Acquisition function: Expected Improvement
# -------------------------------------------------------------------

def _standard_normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def expected_improvement(
    mu: float,
    sigma2: float,
    best_y: float,
    xi: float = 0.01,
) -> float:
    """Expected Improvement acquisition function.

    Parameters
    ----------
    mu, sigma2 : float
        GP predictive mean and variance at the candidate point.
    best_y : float
        Best objective value observed so far.
    xi : float
        Exploration-exploitation trade-off parameter.
    """
    sigma = math.sqrt(sigma2)
    if sigma < 1e-10:
        return 0.0
    z = (mu - best_y - xi) / sigma
    return sigma * (z * _standard_normal_cdf(z) + _standard_normal_pdf(z))


# -------------------------------------------------------------------
# Bayesian Optimiser
# -------------------------------------------------------------------

class BayesianOptimiser:
    """Lightweight Bayesian optimisation for parameter tuning.

    Parameters
    ----------
    param_bounds : list of (min, max) tuples
        Bounds for each parameter dimension.
    objective_fn : callable
        Function mapping a parameter vector to a scalar value to maximise.
    n_initial : int
        Number of random initial evaluations.
    n_iterations : int
        Number of optimisation iterations after the initial phase.
    n_candidates : int
        Number of random candidate points evaluated per iteration.
    seed : int
        Random seed for reproducibility.
    length_scale : float
        GP kernel length scale.
    """

    def __init__(
        self,
        param_bounds: List[Tuple[float, float]],
        objective_fn: Callable[[List[float]], float],
        n_initial: int = 5,
        n_iterations: int = 20,
        n_candidates: int = 100,
        seed: int = 42,
        length_scale: float = 0.5,
    ) -> None:
        if not param_bounds:
            raise ValueError("param_bounds must be non-empty")
        for lo, hi in param_bounds:
            if lo >= hi:
                raise ValueError(f"Invalid bound: ({lo}, {hi})")

        self.param_bounds = param_bounds
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.length_scale = length_scale
        self._rng = random.Random(seed)

        self.X: List[List[float]] = []
        self.y: List[float] = []
        self.best_params: Optional[List[float]] = None
        self.best_value: float = float("-inf")
        self.history: List[Dict] = []

    def _random_point(self) -> List[float]:
        return [
            self._rng.uniform(lo, hi) for lo, hi in self.param_bounds
        ]

    def _evaluate(self, params: List[float]) -> float:
        value = self.objective_fn(params)
        self.X.append(params)
        self.y.append(value)
        if value > self.best_value:
            self.best_value = value
            self.best_params = list(params)
        self.history.append({"params": list(params), "value": value})
        return value

    def optimise(self) -> Tuple[List[float], float]:
        """Run the full optimisation loop.

        Returns (best_params, best_value).
        """
        # Phase 1: random initialisation
        for _ in range(self.n_initial):
            self._evaluate(self._random_point())

        # Phase 2: GP-guided search
        for iteration in range(self.n_iterations):
            best_y = max(self.y)

            # Generate candidate points and pick the one with highest EI
            best_ei = -1.0
            best_candidate: Optional[List[float]] = None

            for _ in range(self.n_candidates):
                candidate = self._random_point()
                mu, sigma2 = gp_predict(
                    self.X, self.y, candidate,
                    length_scale=self.length_scale,
                )
                ei = expected_improvement(mu, sigma2, best_y)
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = candidate

            if best_candidate is not None:
                self._evaluate(best_candidate)

        assert self.best_params is not None
        return self.best_params, self.best_value

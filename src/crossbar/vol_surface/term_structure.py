"""Term structure -> instantaneous forward vols.

The Monte Carlo engine evolves the underlying under a piecewise-constant
``sigma(t)``.  These helpers turn the quoted implied term structure into
forward vols whose cumulative variance reproduces the quoted implied
variance, either at fixed delta (sticky-delta) or at a fixed strike
(sticky-strike).
"""

from __future__ import annotations

import numpy as np

from ..params import BSParams
from .strike import sigma_for_strike
from .surface import VolSurface

__all__ = [
    "stepwise_sigmas_from_surface",
    "stepwise_sigmas_for_strike",
]


def _forward_vols(variance: np.ndarray, T: float, n_steps: int) -> np.ndarray:
    """Piecewise-constant forward vols from a total-variance path."""
    fwd_var = np.diff(np.asarray(variance, dtype=float))
    fwd_var = np.maximum(fwd_var, 0.0)  # guard tiny negative round-off
    return np.sqrt(fwd_var / (float(T) / n_steps))


def stepwise_sigmas_from_surface(
    surface: VolSurface, T: float, n_steps: int, delta: float = 0.0
) -> np.ndarray:
    """Piecewise-constant instantaneous forward vols to maturity ``T``.

    The implied total variance ``w(t) = sigma_impl(t)^2 t`` is read off
    the surface at the ``n_steps + 1`` time nodes for the chosen
    ``delta``; the forward vol on each step is ``sqrt(dw / dt)``, so
    cumulative variance matches the quoted term structure.  This is a
    *sticky-delta* (fixed moneyness) term structure.
    """
    T = float(T)
    n_steps = int(n_steps)
    if T <= 0:
        raise ValueError("maturity must be strictly positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.array(
        [surface.total_variance(t[i], delta) for i in range(n_steps + 1)]
    )
    return _forward_vols(variance, T, n_steps)


def stepwise_sigmas_for_strike(
    surface: VolSurface, params: BSParams, K: float, T: float, n_steps: int
) -> np.ndarray:
    """Forward vols at a *fixed strike* up to maturity ``T``.

    The implied variance at ``K`` is read off :func:`sigma_for_strike` at
    every time node, so the cumulative variance matches the quoted
    smile at that strike.  This is the *sticky-strike* term structure,
    the natural input for a barrier whose level pins the relevant vol.
    """
    T = float(T)
    n_steps = int(n_steps)
    if T <= 0:
        raise ValueError("maturity must be strictly positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        sig = float(sigma_for_strike(surface, params, K, t[i]))
        variance[i] = sig * sig * t[i]
    return _forward_vols(variance, T, n_steps)

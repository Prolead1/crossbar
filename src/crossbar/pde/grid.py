"""Asset/time grid construction and boundary conditions."""

from __future__ import annotations

import numpy as np

from ..params import BarrierSpec, BSParams

__all__ = [
    "build_grid",
    "boundary_values",
    "enforce_barrier_dirichlet",
]


def build_grid(
    bs: BSParams, bar: BarrierSpec, M: int = 400, N: int = 400, truncate: bool = True
):
    """Build a non-uniform asset grid with the barrier snapped to a node.

    Returns ``(S, t)`` with ``M + 1`` space nodes and ``N + 1`` time
    nodes.  With ``truncate=True`` the domain stops at half the barrier
    for a down barrier (the region beyond it is dead under continuous
    monitoring); pass ``truncate=False`` for discrete monitoring, where
    the price can diffuse below the barrier between observations.
    """
    if not bar.is_up and truncate:
        Smin = max(1e-8, 0.5 * bar.H)
        Smax = max(6.0 * bs.S0, 6.0 * bar.K)
    else:
        Smin = 1e-8
        Smax = max(1.5 * bar.H, 6.0 * bs.S0, 6.0 * bar.K)

    S = np.linspace(Smin, Smax, M + 1)
    j = int(np.argmin(np.abs(S - bar.H)))
    S[j] = bar.H
    t = np.linspace(0.0, bs.T, N + 1)
    return S, t


def boundary_values(bs: BSParams, bar: BarrierSpec, Smin, Smax, tau):
    """Call/put boundary values at the extreme grid nodes."""
    if bar.is_call:
        left = 0.0
        right = Smax * np.exp(-bs.q * tau) - bar.K * np.exp(-bs.r * tau)
    else:
        left = bar.K * np.exp(-bs.r * tau)
        right = 0.0
    return left, right


def enforce_barrier_dirichlet(S: np.ndarray, V: np.ndarray, bar: BarrierSpec):
    """Set the option value to the rebate beyond the barrier."""
    if not bar.is_up:
        V[S <= bar.H] = bar.rebate
    else:
        V[S >= bar.H] = bar.rebate
    return V

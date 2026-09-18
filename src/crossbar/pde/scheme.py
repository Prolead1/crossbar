"""Theta-scheme stencil assembly and time stepping.

The Black-Scholes PDE is discretised with the theta scheme: ``theta=0.5``
is Crank-Nicolson and ``theta=1.0`` backward Euler.  ``_operator_plan``
assembles and factors the time-invariant stencil, ``_step`` applies one
step, and :func:`theta_step` / :func:`cn_step` / :func:`rannacher_pair`
are the convenience wrappers used by the pricing loop and the tests.
"""

from __future__ import annotations

import numpy as np

from ..params import BarrierSpec, BSParams
from .grid import boundary_values, enforce_barrier_dirichlet
from .tridiagonal import thomas_factor, thomas_solve_factored

__all__ = [
    "theta_step",
    "cn_step",
    "rannacher_pair",
]


def _operator_plan(
    bs: BSParams,
    bar: BarrierSpec,
    S: np.ndarray,
    dt: float,
    theta: float,
    impose_barrier: bool = True,
    sigma=None,
):
    """Assemble and factor the time-invariant theta-scheme stencil.

    ``impose_barrier`` pins the barrier node to the rebate inside the
    solve (continuous monitoring).  With ``False`` the barrier is a plain
    interior node and is only enforced at the discrete observation dates.

    Returns ``(aL, cL, aR, bR, cR, factors, barrier_row)`` where
    ``factors`` is the Thomas factorisation of the implicit matrix and
    ``barrier_row`` is the interior row pinned to the rebate (``-1`` when
    the barrier is not an interior node).
    """
    M = len(S) - 1
    Sm, S0, Sp = S[:-2], S[1:-1], S[2:]
    hm, hp = S0 - Sm, Sp - S0

    a2 = 2.0 / (hm * (hm + hp))
    b2 = -2.0 / (hm * hp)
    c2 = 2.0 / (hp * (hm + hp))
    a1 = -hp / (hm * (hm + hp))
    b1 = (hp - hm) / (hm * hp)
    c1 = hm / (hp * (hm + hp))

    if sigma is None:
        z = 0.5 * bs.sigma**2 * S0**2
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.shape != S0.shape:
            raise ValueError(
                f"sigma must have shape {S0.shape} (one per interior node), "
                f"got {sig.shape}"
            )
        z = 0.5 * sig**2 * S0**2
    a = z * a2 + (bs.r - bs.q) * S0 * a1
    b = z * b2 + (bs.r - bs.q) * S0 * b1 - bs.r
    c = z * c2 + (bs.r - bs.q) * S0 * c1

    aL = -theta * dt * a
    bL = 1.0 - theta * dt * b
    cL = -theta * dt * c
    aR = (1.0 - theta) * dt * a
    bR = 1.0 + (1.0 - theta) * dt * b
    cR = (1.0 - theta) * dt * c

    # Impose the barrier as a Dirichlet node inside the linear solve so the
    # live region is decoupled from the (unconstrained) region beyond it.
    j = int(np.argmin(np.abs(S - bar.H)))
    barrier_row = -1
    if impose_barrier and 0 < j < M and np.isclose(S[j], bar.H):
        barrier_row = j - 1
        aL[barrier_row] = 0.0
        bL[barrier_row] = 1.0
        cL[barrier_row] = 0.0

    return aL, cL, aR, bR, cR, thomas_factor(aL, bL, cL), barrier_row


def _step(
    bs: BSParams,
    bar: BarrierSpec,
    S: np.ndarray,
    Vnext: np.ndarray,
    plan,
    tnow: float,
) -> np.ndarray:
    """One implicit/explicit step using a pre-factored stencil ``plan``."""
    M = len(S) - 1
    aL, cL, aR, bR, cR, factors, barrier_row = plan

    rhs = bR * Vnext[1:M] + aR * Vnext[: M - 1] + cR * Vnext[2 : M + 1]

    left, right = boundary_values(bs, bar, S[0], S[-1], bs.T - tnow)
    if bar.monitor == "discrete":
        # Between observations the far side of the barrier is still alive
        # but will almost surely be knocked out at the next date, so pin
        # it to the rebate rather than the vanilla boundary value.
        if bar.is_up:
            right = bar.rebate
        else:
            left = bar.rebate
    else:
        # Nodes at or beyond the barrier are knocked out and hold the
        # rebate, so they must not take the vanilla boundary value.
        if not bar.is_up and S[0] <= bar.H:
            left = bar.rebate
        if bar.is_up and S[-1] >= bar.H:
            right = bar.rebate

    rhs[0] -= aL[0] * left
    rhs[-1] -= cL[-1] * right
    if barrier_row >= 0:
        rhs[barrier_row] = bar.rebate

    V = np.empty_like(Vnext)
    V[0] = left
    V[-1] = right
    V[1:M] = thomas_solve_factored(rhs, factors)
    return V


def theta_step(
    bs: BSParams,
    bar: BarrierSpec,
    S: np.ndarray,
    Vnext: np.ndarray,
    dt: float,
    tnow: float,
    theta: float,
    sigma=None,
) -> np.ndarray:
    """One implicit/explicit time step of the Black-Scholes PDE.

    ``theta = 0.5`` is Crank-Nicolson and ``theta = 1.0`` is backward
    Euler.  ``tnow`` is the earlier time being solved for.
    """
    plan = _operator_plan(bs, bar, S, dt, theta, sigma=sigma)
    return _step(bs, bar, S, Vnext, plan, tnow)


def cn_step(bs, bar, S, Vn, dt, t_after):
    """A single Crank-Nicolson step from ``t_after`` to ``t_after - dt``."""
    Vnow = theta_step(bs, bar, S, Vn, dt=dt, tnow=t_after - dt, theta=0.5)
    return enforce_barrier_dirichlet(S, Vnow, bar)


def rannacher_pair(bs, bar, S, Vn, dt_full, t_after):
    """Two backward-Euler half-steps of size ``dt_full / 2``."""
    h = 0.5 * dt_full
    Vmid = theta_step(bs, bar, S, Vn, dt=h, tnow=t_after - h, theta=1.0)
    Vmid = enforce_barrier_dirichlet(S, Vmid, bar)
    Vnow = theta_step(bs, bar, S, Vmid, dt=h, tnow=t_after - 2.0 * h, theta=1.0)
    return enforce_barrier_dirichlet(S, Vnow, bar)

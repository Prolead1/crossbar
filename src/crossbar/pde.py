"""Crank-Nicolson finite-difference pricing with Rannacher time stepping.

The barrier is placed exactly on a grid node and enforced with Dirichlet
conditions, which prices the continuously monitored contract.  Knock-ins
are obtained from the knock-out value through in-out parity.  The first
time step from maturity is split into two backward-Euler half-steps
(Rannacher smoothing) to damp the high-frequency oscillations that a
plain Crank-Nicolson scheme produces at the payoff kink.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .analytic import barrier_rebate_terms, price_vanilla, vanilla_payoff
from .params import BarrierSpec, BSParams


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


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Solve a tridiagonal system with the Thomas algorithm."""
    n = len(b)
    cp = np.empty(n - 1)
    dp = np.empty(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den

    den = b[n - 1] - a[n - 1] * cp[n - 2]
    dp[n - 1] = (d[n - 1] - a[n - 1] * dp[n - 2]) / den

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def thomas_factor(a, b, c):
    """Pre-factor a tridiagonal matrix for repeated Thomas solves.

    The forward-elimination ratios depend only on the matrix, so a
    stencil reused across many right-hand sides (the time-invariant
    Crank-Nicolson/BE matrices here) can be factorised once.  The factors
    are plain Python lists so the sweeps in
    :func:`thomas_solve_factored` avoid per-element numpy overhead.
    """
    a = np.asarray(a, dtype=float).tolist()
    b = np.asarray(b, dtype=float).tolist()
    c = np.asarray(c, dtype=float).tolist()
    n = len(b)
    cp = [0.0] * n
    den = [0.0] * n
    den[0] = b[0]
    cp[0] = c[0] / b[0]
    for i in range(1, n - 1):
        den[i] = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den[i]
    den[n - 1] = b[n - 1] - a[n - 1] * cp[n - 2]
    return a, cp, den


def thomas_solve_factored(d, factors):
    """Solve a system against a matrix from :func:`thomas_factor`."""
    a, cp, den = factors
    d = d.tolist()
    n = len(d)
    dp = [0.0] * n
    dp[0] = d[0] / den[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den[i]
    x = [0.0] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.asarray(x)


def boundary_values(bs: BSParams, bar: BarrierSpec, Smin, Smax, tau):
    """Call/put boundary values at the extreme grid nodes."""
    if bar.is_call:
        left = 0.0
        right = Smax * np.exp(-bs.q * tau) - bar.K * np.exp(-bs.r * tau)
    else:
        left = bar.K * np.exp(-bs.r * tau)
        right = 0.0
    return left, right


def _operator_plan(
    bs: BSParams,
    bar: BarrierSpec,
    S: np.ndarray,
    dt: float,
    theta: float,
    impose_barrier: bool = True,
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

    z = 0.5 * bs.sigma**2 * S0**2
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
) -> np.ndarray:
    """One implicit/explicit time step of the Black-Scholes PDE.

    ``theta = 0.5`` is Crank-Nicolson and ``theta = 1.0`` is backward
    Euler.  ``tnow`` is the earlier time being solved for.
    """
    plan = _operator_plan(bs, bar, S, dt, theta)
    return _step(bs, bar, S, Vnext, plan, tnow)


def enforce_barrier_dirichlet(S: np.ndarray, V: np.ndarray, bar: BarrierSpec):
    """Set the option value to the rebate beyond the barrier."""
    if not bar.is_up:
        V[S <= bar.H] = bar.rebate
    else:
        V[S >= bar.H] = bar.rebate
    return V


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


def _knock_out_surface(
    bs: BSParams,
    bar: BarrierSpec,
    M: int,
    N: int,
    rannacher_pairs: int,
    monitor_steps: int | None = None,
):
    """Backward-induct the PDE and return the full ``(S, V)`` surface.

    Continuous barriers are imposed at every time step.  Discrete
    barriers diffuse freely between observations and are only reset at
    ``monitor_steps + 1`` equally spaced dates (default: every node).
    """
    discrete = bar.monitor == "discrete"
    S, t = build_grid(bs, bar, M, N, truncate=not discrete)
    V = vanilla_payoff(S, bar.K, bar.is_call)
    V = enforce_barrier_dirichlet(S, V, bar)

    if discrete:
        K = N if monitor_steps is None else int(monitor_steps)
        obs = np.zeros(N + 1, dtype=bool)
        for ot in np.linspace(0.0, bs.T, K + 1):
            obs[int(np.argmin(np.abs(t - ot)))] = True

    # The stencil only depends on (theta, dt); ``linspace`` step sizes differ
    # by a few ULP, so cache a factorisation per distinct dt to keep the
    # arithmetic identical to rebuilding the stencil every step.
    plans: dict = {}

    def plan_for(theta: float, dt: float):
        key = (theta, dt)
        cached = plans.get(key)
        if cached is None:
            cached = _operator_plan(
                bs, bar, S, dt, theta, impose_barrier=not discrete
            )
            plans[key] = cached
        return cached

    for n in range(N, 0, -1):
        dt = t[n] - t[n - 1]
        if (N - n) < rannacher_pairs:
            h = 0.5 * dt
            V = _step(bs, bar, S, V, plan_for(1.0, h), tnow=t[n] - h)
            if not discrete:
                V = enforce_barrier_dirichlet(S, V, bar)
            V = _step(bs, bar, S, V, plan_for(1.0, h), tnow=t[n] - 2.0 * h)
        else:
            V = _step(bs, bar, S, V, plan_for(0.5, dt), tnow=t[n] - dt)
        if not discrete or obs[n - 1]:
            V = enforce_barrier_dirichlet(S, V, bar)

    return S, V


def pde_knock_out(
    bs: BSParams,
    bar: BarrierSpec,
    M: int = 400,
    N: int = 400,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
) -> float:
    """Price a knock-out by backward PDE induction (continuous or discrete)."""
    S, V = _knock_out_surface(bs, bar, M, N, rannacher_pairs, monitor_steps)
    return float(np.interp(bs.S0, S, V))


def pde_surface(
    bs: BSParams,
    bar: BarrierSpec,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
):
    """Barrier value surface ``(S, V)``.

    Knock-outs are solved directly; knock-ins use in-out parity against
    the analytic vanilla surface plus the closed-form rebate correction,
    ``V_in = V_vanilla - V_out + E + F`` (exact for continuous monitoring).
    """
    if not bar.is_in:
        return _knock_out_surface(bs, bar, M, N, rannacher_pairs, monitor_steps)

    if bar.monitor == "discrete" and bar.rebate != 0.0:
        raise NotImplementedError(
            "PDE knock-in rebates are only supported for continuous monitoring"
        )
    out_spec = replace(bar, barrier_type=bar.barrier_type.replace("in", "out"))
    S, V_out = _knock_out_surface(bs, out_spec, M, N, rannacher_pairs, monitor_steps)
    E, F = barrier_rebate_terms(bs, bar)
    V = price_vanilla(
        S, bar.K, bs.r, bs.q, bs.sigma, bs.T, bar.is_call
    ) - V_out + E + F
    return S, V


def price_barrier_pde(
    bs: BSParams,
    bar: BarrierSpec,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
) -> float:
    """Price a barrier option with the PDE solver (continuous or discrete).

    Knock-outs are solved directly; knock-ins use in-out parity plus the
    closed-form rebate correction ``V_in = V_vanilla - V_out + E + F``.
    """
    if not bar.is_in:
        return pde_knock_out(
            bs,
            bar,
            M=M,
            N=N,
            rannacher_pairs=rannacher_pairs,
            monitor_steps=monitor_steps,
        )

    if bar.monitor == "discrete" and bar.rebate != 0.0:
        raise NotImplementedError(
            "PDE knock-in rebates are only supported for continuous monitoring"
        )
    vanilla = price_vanilla(bs.S0, bar.K, bs.r, bs.q, bs.sigma, bs.T, bar.is_call)
    out_spec = replace(bar, barrier_type=bar.barrier_type.replace("in", "out"))
    v_out = pde_knock_out(
        bs,
        out_spec,
        M=M,
        N=N,
        rannacher_pairs=rannacher_pairs,
        monitor_steps=monitor_steps,
    )
    E, F = barrier_rebate_terms(bs, bar)
    return float(vanilla) - v_out + E + F

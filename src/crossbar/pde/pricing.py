"""Barrier pricing by backward PDE induction.

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

from ..analytic import barrier_rebate_terms, price_vanilla, vanilla_payoff
from ..params import BarrierSpec, BSParams
from .grid import build_grid, enforce_barrier_dirichlet
from .scheme import _operator_plan, _step

__all__ = [
    "pde_knock_out",
    "pde_surface",
    "price_barrier_pde",
]


def _knock_out_surface(
    bs: BSParams,
    bar: BarrierSpec,
    M: int,
    N: int,
    rannacher_pairs: int,
    monitor_steps: int | None = None,
    surface=None,
):
    """Backward-induct the PDE and return the full ``(S, V)`` surface.

    Continuous barriers are imposed at every time step.  Discrete
    barriers diffuse freely between observations and are only reset at
    ``monitor_steps + 1`` equally spaced dates (default: every node).
    """
    discrete = bar.monitor == "discrete"
    S, t = build_grid(bs, bar, M, N, truncate=not discrete)
    if surface is not None:
        from ..vol_surface import build_sigma_grid

        sigma_grid = build_sigma_grid(surface, bs, bar, S, t)
    else:
        sigma_grid = None
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

    def plan_for(theta: float, dt: float, sigma):
        if sigma is not None:
            return _operator_plan(
                bs, bar, S, dt, theta, impose_barrier=not discrete, sigma=sigma
            )
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
        # Earlier time row: the local-vol grid feeds the diffusion at the
        # start of the step (first order in time, as for a time-dependent
        # sigma).
        sig = None if sigma_grid is None else sigma_grid[n - 1, 1:M]
        if (N - n) < rannacher_pairs:
            h = 0.5 * dt
            V = _step(bs, bar, S, V, plan_for(1.0, h, sig), tnow=t[n] - h)
            if not discrete:
                V = enforce_barrier_dirichlet(S, V, bar)
            V = _step(bs, bar, S, V, plan_for(1.0, h, sig), tnow=t[n] - 2.0 * h)
        else:
            V = _step(bs, bar, S, V, plan_for(0.5, dt, sig), tnow=t[n] - dt)
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
    surface=None,
) -> float:
    """Price a knock-out by backward PDE induction (continuous or discrete)."""
    S, V = _knock_out_surface(bs, bar, M, N, rannacher_pairs, monitor_steps, surface)
    return float(np.interp(bs.S0, S, V))


def _vanilla_leg_vol(bs: BSParams, bar: BarrierSpec, surface=None) -> float:
    """Implied vol for the knock-in parity vanilla leg.

    Under a local-vol surface the model vanilla price is the market one,
    i.e. Black-Scholes at the surface vol for the strike, so knock-in
    parity uses that rather than the ATM vol.  Without a surface the flat
    ``bs.sigma`` is returned.
    """
    if surface is None:
        return bs.sigma
    from ..vol_surface import sigma_for_strike

    return float(sigma_for_strike(surface, bs, bar.K, bs.T))


def pde_surface(
    bs: BSParams,
    bar: BarrierSpec,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
    surface=None,
):
    """Barrier value surface ``(S, V)``.

    Knock-outs are solved directly; knock-ins use in-out parity against
    the analytic vanilla surface plus the closed-form rebate correction,
    ``V_in = V_vanilla - V_out + E + F`` (exact for continuous monitoring).
    """
    if not bar.is_in:
        return _knock_out_surface(
            bs, bar, M, N, rannacher_pairs, monitor_steps, surface
        )

    if bar.monitor == "discrete" and bar.rebate != 0.0:
        raise NotImplementedError(
            "PDE knock-in rebates are only supported for continuous monitoring"
        )
    out_spec = replace(bar, barrier_type=bar.barrier_type.replace("in", "out"))
    S, V_out = _knock_out_surface(
        bs, out_spec, M, N, rannacher_pairs, monitor_steps, surface
    )
    E, F = barrier_rebate_terms(bs, bar)
    V = price_vanilla(
        S, bar.K, bs.r, bs.q, _vanilla_leg_vol(bs, bar, surface), bs.T, bar.is_call
    ) - V_out + E + F
    return S, V


def price_barrier_pde(
    bs: BSParams,
    bar: BarrierSpec,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
    surface=None,
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
            surface=surface,
        )

    if bar.monitor == "discrete" and bar.rebate != 0.0:
        raise NotImplementedError(
            "PDE knock-in rebates are only supported for continuous monitoring"
        )
    vanilla = price_vanilla(
        bs.S0, bar.K, bs.r, bs.q, _vanilla_leg_vol(bs, bar, surface), bs.T, bar.is_call
    )
    out_spec = replace(bar, barrier_type=bar.barrier_type.replace("in", "out"))
    v_out = pde_knock_out(
        bs,
        out_spec,
        M=M,
        N=N,
        rannacher_pairs=rannacher_pairs,
        monitor_steps=monitor_steps,
        surface=surface,
    )
    E, F = barrier_rebate_terms(bs, bar)
    return float(vanilla) - v_out + E + F

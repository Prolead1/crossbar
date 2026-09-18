"""PDE risk profiles from a single value-surface solve."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from ..analytic import barrier_rebate_terms, price_vanilla
from ..params import BarrierSpec, BSParams, barrier_variants
from ..pde import _vanilla_leg_vol, pde_surface
from .surface import _surface_derivatives, _vanilla_greeks, surface_risk_profile

__all__ = ["pde_risk_profile", "greeks_by_variant_pde"]


def pde_risk_profile(
    bs: BSParams,
    bar: BarrierSpec,
    spots,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
    surface=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PDE price/delta/gamma profile from a single surface solve.

    Knock-ins combine the analytic vanilla leg with the solved knock-out
    surface (plus the closed-form rebate correction), so their prices
    agree with :func:`crossbar.price_barrier_pde`.  An optional
    ``surface`` is forwarded to :func:`crossbar.pde.pde_surface` for the
    local-vol grid.
    """
    spots = np.asarray(spots, dtype=float)
    if not bar.is_in:
        S, V = pde_surface(
            bs,
            bar,
            M=M,
            N=N,
            rannacher_pairs=rannacher_pairs,
            monitor_steps=monitor_steps,
            surface=surface,
        )
        return surface_risk_profile(S, V, spots)

    if bar.monitor == "discrete" and bar.rebate != 0.0:
        raise NotImplementedError(
            "PDE knock-in rebates are only supported for continuous monitoring"
        )
    out_spec = replace(bar, barrier_type=bar.barrier_type.replace("in", "out"))
    S, V_out = pde_surface(
        bs,
        out_spec,
        M=M,
        N=N,
        rannacher_pairs=rannacher_pairs,
        monitor_steps=monitor_steps,
        surface=surface,
    )
    delta_out, gamma_out = _surface_derivatives(S, V_out)
    E, F = barrier_rebate_terms(bs, bar)
    sigma_leg = _vanilla_leg_vol(bs, bar, surface)

    prices = price_vanilla(
        spots, bar.K, bs.r, bs.q, sigma_leg, bs.T, bar.is_call
    ) - np.interp(spots, S, V_out) + E + F
    v_delta, v_gamma = _vanilla_greeks(bs, bar, spots, sigma=sigma_leg)
    deltas = v_delta - np.interp(spots, S, delta_out)
    gammas = v_gamma - np.interp(spots, S, gamma_out)
    return prices, deltas, gammas


def greeks_by_variant_pde(
    bs: BSParams,
    spots,
    M: int = 500,
    N: int = 500,
    rannacher_pairs: int = 1,
    monitor_steps: int | None = None,
    **variant_kwargs,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """PDE delta/gamma profiles for all eight single-barrier variants.

    One surface solve per variant, versus three solves per spot for
    :func:`crossbar.greeks.greeks_by_variant`.
    """
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, spec in barrier_variants(bs, **variant_kwargs):
        profiles[label] = pde_risk_profile(
            bs,
            spec,
            spots,
            M=M,
            N=N,
            rannacher_pairs=rannacher_pairs,
            monitor_steps=monitor_steps,
        )
    return profiles

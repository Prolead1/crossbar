"""The ``crossbar price`` command: run the engines and emit results."""

from __future__ import annotations

import argparse

from ..analytic import price_barrier_closed_form, price_vanilla
from ..greeks import mc_greek_bumps, mc_risk_profile, pde_risk_profile, risk_profile
from ..monte_carlo import gen_normals, price_barrier_mc
from ..params import BarrierSpec, BSParams, validate_inputs
from ..vol_surface import stepwise_sigmas_for_strike
from .constants import DELTA_BUMP, PDE_MESH
from .output import (
    _barrier_contract,
    _emit,
    _load_surface,
    _market_dict,
    _render_results,
)

__all__ = ["_cmd_price"]


def _bar_from_args(a: argparse.Namespace) -> BarrierSpec:
    return BarrierSpec(
        barrier_type=a.barrier_type,
        monitor=a.monitor,
        H=a.barrier,
        K=a.strike,
        is_call=a.is_call,
        rebate=a.rebate,
    )


def _run_engines(bs: BSParams, bar: BarrierSpec, a, surface) -> list:
    """Price and greeks with all three engines, skipping inapplicable ones."""
    spot = float(bs.S0)
    results = []

    prices, deltas, gammas = risk_profile(
        price_barrier_closed_form, bs, bar, [spot], bump=DELTA_BUMP
    )
    results.append(
        {
            "engine": "analytic",
            "price": float(prices[0]),
            "delta": float(deltas[0]),
            "gamma": float(gammas[0]),
        }
    )

    if surface is None:
        steps = None
    else:
        # Sticky-strike term structure at the barrier level, the vol that
        # dominates a barrier option's value.  The PDE instead uses the
        # full Dupire local-vol surface built from the same quotes.
        steps = stepwise_sigmas_for_strike(surface, bs, bar.H, bs.T, a.steps)

    # Draw the paths once and share them between the price and the delta
    # bumps.  ``mc_risk_profile`` reuses a single simulation for all bumps,
    # so the expensive path generation runs once instead of many times.
    # The greek stencils scale with the spot: a fixed absolute bump makes
    # the second-difference gamma meaningless for large spots.
    Z = gen_normals(a.paths, a.steps, seed=a.seed)
    price, se = price_barrier_mc(
        bs,
        bar,
        n_paths=a.paths,
        n_steps=a.steps,
        seed=a.seed,
        control_variate=a.control_variate,
        Z=Z,
        sigma=steps,
    )
    delta_bump, gamma_bump = mc_greek_bumps(bs.S0, bs.sigma, bs.T)
    _, deltas, gammas, delta_errors, gamma_errors = mc_risk_profile(
        bs,
        bar,
        [spot],
        bump=delta_bump,
        gamma_bump=gamma_bump,
        seed=a.seed,
        control_variate=a.control_variate,
        Z=Z,
        sigma=steps,
        return_errors=True,
    )
    results.append(
        {
            "engine": "mc",
            "price": float(price),
            "delta": float(deltas[0]),
            "gamma": float(gammas[0]),
            "std_error": float(se),
            "delta_std_error": float(delta_errors[0]),
            "gamma_std_error": float(gamma_errors[0]),
        }
    )

    prices, deltas, gammas = pde_risk_profile(
        bs,
        bar,
        [spot],
        M=a.M,
        N=a.N,
        rannacher_pairs=a.rannacher,
        surface=surface,
    )
    results.append(
        {
            "engine": "pde",
            "price": float(prices[0]),
            "delta": float(deltas[0]),
            "gamma": float(gammas[0]),
        }
    )
    return results


def _cmd_price(a: argparse.Namespace) -> None:
    surface = _load_surface(a.surface)
    if surface is None:
        sigma = a.sigma
        vol_source = "constant"
    else:
        sigma = float(surface.interp_sigma(a.maturity, 0.0))
        vol_source = "ATM from surface"
    bs = BSParams(S0=a.spot, r=a.rate, q=a.div, sigma=sigma, T=a.maturity)
    bar = _bar_from_args(a)
    validate_inputs(bs, bar)

    vanilla = float(
        price_vanilla(bs.S0, bar.K, bs.r, bs.q, bs.sigma, bs.T, is_call=bar.is_call)
    )
    payload = {
        "contract": _barrier_contract(bar),
        "market": _market_dict(a, sigma),
        "vol_source": vol_source,
        "vanilla_benchmark": vanilla,
        "quotes": a.surface,
        "pde_options": dict(PDE_MESH),
        "prices": _run_engines(bs, bar, a, surface),
        "mc_options": {
            "paths": a.paths,
            "steps": a.steps,
            "seed": a.seed,
            "control_variate": a.control_variate,
        },
    }

    _emit(payload, a.json, _render_results)

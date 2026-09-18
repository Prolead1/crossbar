"""Delta and gamma profiles for barrier options.

:func:`risk_profile` is the engine-agnostic bump-and-revalue profile
used by the Monte Carlo pricer.  :func:`mc_risk_profile` is its Monte
Carlo counterpart: it draws the paths once and reuses the same
simulation for every bump, so the common random numbers come for free.
:func:`pde_risk_profile` instead solves the PDE value surface once and
reads price, delta and gamma at every spot from it -- far cheaper, and
much smoother than bumping the interpolant.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from .analytic import barrier_rebate_terms, price_vanilla
from .monte_carlo import (
    gen_normals,
    log_price_increments,
    paths_from_log_increments,
    price_barrier_mc,
)
from .params import BarrierSpec, BSParams, barrier_variants
from .pde import _vanilla_leg_vol, pde_surface


def _extract_price(result) -> float:
    """Return the price from a pricer result.

    Pricers may return a bare price or a ``(price, standard_error)``
    pair (as :func:`crossbar.price_barrier_mc` does); only the price is
    used for the bump-and-revalue profile.
    """
    if isinstance(result, tuple):
        result = result[0]
    return float(result)


def risk_profile(
    pricer: Callable,
    bs: BSParams,
    bar: BarrierSpec,
    spots,
    bump: float = 0.5,
    pricer_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference price, delta and gamma across a spot grid.

    Returns ``(prices, deltas, gammas)`` evaluated at each spot in
    ``spots`` using a symmetric bump of size ``bump``.  ``pricer`` may
    return either a bare price or a ``(price, standard_error)`` pair, so
    both :func:`crossbar.price_barrier_mc` and scalar pricers such as
    :func:`crossbar.price_barrier_pde` can be passed directly.

    For a stochastic pricer, pass the same seed through
    ``pricer_kwargs`` so the three revaluations share random numbers and
    the finite differences are not swamped by simulation noise.
    """
    kwargs = pricer_kwargs or {}
    spots = np.asarray(spots, dtype=float)
    prices = np.empty_like(spots)
    deltas = np.empty_like(spots)
    gammas = np.empty_like(spots)

    for i, s in enumerate(spots):
        s_up = float(s + bump)
        s_dn = float(max(s - bump, 1e-8))

        p0 = _extract_price(
            pricer(BSParams(float(s), bs.r, bs.q, bs.sigma, bs.T), bar, **kwargs)
        )
        p_up = _extract_price(
            pricer(BSParams(s_up, bs.r, bs.q, bs.sigma, bs.T), bar, **kwargs)
        )
        p_dn = _extract_price(
            pricer(BSParams(s_dn, bs.r, bs.q, bs.sigma, bs.T), bar, **kwargs)
        )

        prices[i] = p0
        deltas[i] = (p_up - p_dn) / (2.0 * bump)
        gammas[i] = (p_up - 2.0 * p0 + p_dn) / (bump**2)

    return prices, deltas, gammas


def greeks_by_variant(
    pricer: Callable,
    bs: BSParams,
    spots,
    bump: float = 0.5,
    pricer_kwargs: Optional[dict] = None,
    **variant_kwargs,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Delta/gamma profiles for all eight single-barrier variants.

    Extra keyword arguments are forwarded to
    :func:`crossbar.params.barrier_variants`.
    """
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, spec in barrier_variants(bs, **variant_kwargs):
        profiles[label] = risk_profile(
            pricer, bs, spec, spots, bump=bump, pricer_kwargs=pricer_kwargs
        )
    return profiles


def mc_risk_profile(
    bs: BSParams,
    bar: BarrierSpec,
    spots,
    bump: float = 0.5,
    n_paths: int = 200_000,
    n_steps: int = 252,
    seed: int = 0,
    control_variate: bool = True,
    Z=None,
    sigma=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo price, delta and gamma from one set of random paths.

    This is the Monte Carlo counterpart of :func:`risk_profile`, but the
    normal draws are generated once and reused for every bump and spot.
    Reusing the paths is exactly the common-random-numbers requirement
    the generic bump-and-revalue profile documents, so the finite
    differences are the same as calling :func:`risk_profile` with a fixed
    ``seed`` -- without redrawing (and re-simulating) the paths three
    times per spot.  Pass ``Z`` to share the draws with a separate
    :func:`crossbar.price_barrier_mc` call, and ``sigma`` for a scalar or
    length-``n_steps`` instantaneous-vol term structure.
    """
    spots = np.asarray(spots, dtype=float)
    if Z is None:
        Z = gen_normals(n_paths, n_steps, seed=seed)
    # Simulate the log-return increments once.  GBM paths are linear in the
    # spot, so every bumped revaluation is the same paths with a different
    # ``log(S0)`` added -- no redraw and no re-simulation per bump.
    log_paths = log_price_increments(bs, Z, sigma=sigma)

    prices = np.empty_like(spots)
    deltas = np.empty_like(spots)
    gammas = np.empty_like(spots)

    def _price(spot: float) -> float:
        bs_spot = BSParams(spot, bs.r, bs.q, bs.sigma, bs.T)
        S = paths_from_log_increments(log_paths, spot)
        return _extract_price(
            price_barrier_mc(
                bs_spot,
                bar,
                seed=seed,
                control_variate=control_variate,
                S=S,
                sigma=sigma,
            )
        )

    for i, s in enumerate(spots):
        p0 = _price(float(s))
        p_up = _price(float(s + bump))
        p_dn = _price(float(max(s - bump, 1e-8)))

        prices[i] = p0
        deltas[i] = (p_up - p_dn) / (2.0 * bump)
        gammas[i] = (p_up - 2.0 * p0 + p_dn) / (bump**2)

    return prices, deltas, gammas


def _surface_derivatives(S: np.ndarray, V: np.ndarray):
    """First and second spot derivatives of ``V`` on the (non-uniform) grid."""
    Sm, S0, Sp = S[:-2], S[1:-1], S[2:]
    hm, hp = S0 - Sm, Sp - S0

    a1 = -hp / (hm * (hm + hp))
    b1 = (hp - hm) / (hm * hp)
    c1 = hm / (hp * (hm + hp))
    a2 = 2.0 / (hm * (hm + hp))
    b2 = -2.0 / (hm * hp)
    c2 = 2.0 / (hp * (hm + hp))

    delta = np.empty_like(S)
    gamma = np.empty_like(S)
    delta[1:-1] = a1 * V[:-2] + b1 * V[1:-1] + c1 * V[2:]
    gamma[1:-1] = a2 * V[:-2] + b2 * V[1:-1] + c2 * V[2:]
    delta[0] = (V[1] - V[0]) / (S[1] - S[0])
    delta[-1] = (V[-1] - V[-2]) / (S[-1] - S[-2])
    gamma[0] = gamma[-1] = 0.0
    return delta, gamma


def surface_risk_profile(S: np.ndarray, V: np.ndarray, spots) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Price, delta and gamma read off a solved value surface.

    ``S``/``V`` come from :func:`crossbar.pde.pde_surface`.  Delta and
    gamma are the surface's spot derivatives on the grid, interpolated at
    ``spots``.
    """
    spots = np.asarray(spots, dtype=float)
    delta, gamma = _surface_derivatives(S, V)
    prices = np.interp(spots, S, V)
    deltas = np.interp(spots, S, delta)
    gammas = np.interp(spots, S, gamma)
    return prices, deltas, gammas


def _vanilla_greeks(bs: BSParams, bar: BarrierSpec, spots, sigma=None):
    """Analytic Black-Scholes delta and gamma for the vanilla leg."""
    spots = np.asarray(spots, dtype=float)
    sigma = bs.sigma if sigma is None else sigma
    sqrtT = np.sqrt(bs.T)
    d1 = (np.log(spots / bar.K) + (bs.r - bs.q + 0.5 * sigma**2) * bs.T) / (
        sigma * sqrtT
    )
    disc_q = np.exp(-bs.q * bs.T)
    if bar.is_call:
        delta = disc_q * norm.cdf(d1)
    else:
        delta = -disc_q * norm.cdf(-d1)
    gamma = disc_q * norm.pdf(d1) / (spots * sigma * sqrtT)
    return delta, gamma


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
    :func:`greeks_by_variant`.
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

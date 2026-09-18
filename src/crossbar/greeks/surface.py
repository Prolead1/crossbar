"""Greeks read off a solved value surface (and the analytic vanilla leg)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm

from ..params import BarrierSpec, BSParams

__all__ = ["surface_risk_profile"]


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

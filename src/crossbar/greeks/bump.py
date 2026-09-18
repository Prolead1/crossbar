"""Engine-agnostic bump-and-revalue delta/gamma profiles."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..params import BarrierSpec, BSParams, barrier_variants
from ._common import _extract_price

__all__ = ["risk_profile", "greeks_by_variant"]


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

"""Monte Carlo risk profile with common random numbers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..monte_carlo import (
    gen_normals,
    log_price_increments,
    paths_from_log_increments,
    price_barrier_mc,
)
from ..params import BarrierSpec, BSParams
from ._common import _extract_price

__all__ = ["mc_risk_profile"]


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

    This is the Monte Carlo counterpart of
    :func:`crossbar.greeks.risk_profile`, but the normal draws are
    generated once and reused for every bump and spot.  Reusing the paths
    is exactly the common-random-numbers requirement the generic
    bump-and-revalue profile documents, so the finite differences are the
    same as calling :func:`risk_profile` with a fixed ``seed`` -- without
    redrawing (and re-simulating) the paths three times per spot.  Pass
    ``Z`` to share the draws with a separate
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

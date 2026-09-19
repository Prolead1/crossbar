"""Monte Carlo risk profile with common random numbers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..monte_carlo import (
    gen_normals,
    log_price_increments,
    paths_from_log_increments,
    price_barrier,
    price_barrier_cv,
)
from ..params import BarrierSpec, BSParams

__all__ = ["mc_risk_profile", "mc_greek_bumps"]


def mc_greek_bumps(S0: float, sigma: float, T: float) -> Tuple[float, float]:
    """Spot bumps for first- and second-order Monte Carlo Greeks.

    A barrier payoff is discontinuous in the spot, because shifting ``S0``
    moves the barrier crossing of every path.  The second difference of a
    bump-and-revalue estimate therefore contains a spike whenever a
    path's crossing level crosses the stencil, and dividing by the square
    of the bump amplifies it.  An absolute bump (say ``0.01``) is far too
    small for a spot of 100; the stencil is instead tied to the natural
    spot scale ``S0 * sigma * sqrt(T)`` with a floor as a fraction of the
    spot for short-dated, low-vol contracts.  Gamma needs a wider stencil
    than delta because it divides by the bump squared.

    Returns ``(delta_bump, gamma_bump)``.
    """
    scale = S0 * sigma * np.sqrt(T)
    delta_bump = max(0.002 * S0, 0.1 * scale)
    gamma_bump = max(0.01 * S0, 0.2 * scale)
    return float(delta_bump), float(gamma_bump)


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
    gamma_bump: float | None = None,
    return_errors: bool = False,
):
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

    ``gamma_bump`` uses a separate (usually wider) stencil for the second
    difference, see :func:`mc_greek_bumps`; it defaults to ``bump``.  With
    ``return_errors`` the delta and gamma Monte Carlo standard errors are
    appended to the return value as ``(prices, deltas, gammas,
    delta_errors, gamma_errors)``.

    Note on the second difference: the barrier payoff is discontinuous in
    the spot, so the gamma estimate is noisy however it is computed.  The
    standard errors returned here are the honest way to judge it -- the
    stencil choice only keeps the point estimate from being dominated by
    the spike noise.
    """
    spots = np.asarray(spots, dtype=float)
    if Z is None:
        Z = gen_normals(n_paths, n_steps, seed=seed)
    # Simulate the log-return increments once.  GBM paths are linear in the
    # spot, so every bumped revaluation is the same paths with a different
    # ``log(S0)`` added -- no redraw and no re-simulation per bump.
    log_paths = log_price_increments(bs, Z, sigma=sigma)
    n_paths, n_steps = log_paths.shape
    if gamma_bump is None:
        gamma_bump = bump

    prices = np.empty_like(spots)
    deltas = np.empty_like(spots)
    gammas = np.empty_like(spots)
    delta_errors = np.empty_like(spots) if return_errors else None
    gamma_errors = np.empty_like(spots) if return_errors else None

    def _samples(spot: float) -> np.ndarray:
        """Per-path control-variate prices at ``spot`` on the shared paths."""
        bs_spot = BSParams(spot, bs.r, bs.q, bs.sigma, bs.T)
        S = paths_from_log_increments(log_paths, spot)
        if control_variate:
            return price_barrier_cv(
                bs_spot, bar, n_paths, n_steps, seed=seed, S=S, sigma=sigma
            )
        return price_barrier(
            bs_spot, bar, n_paths, n_steps, seed=seed, S=S, sigma=sigma
        )[0]

    for i, s in enumerate(spots):
        s = float(s)
        s_dn = float(max(s - bump, 1e-8))
        p0 = _samples(s)
        p_up = _samples(s + bump)
        p_dn = _samples(s_dn)

        prices[i] = p0.mean()
        d_samples = (p_up - p_dn) / (2.0 * bump)
        deltas[i] = d_samples.mean()
        if return_errors:
            delta_errors[i] = d_samples.std(ddof=1) / np.sqrt(n_paths)

        if gamma_bump == bump:
            g_samples = (p_up - 2.0 * p0 + p_dn) / (bump**2)
        else:
            gs_dn = float(max(s - gamma_bump, 1e-8))
            g_up = _samples(s + gamma_bump)
            g_dn = _samples(gs_dn)
            g_samples = (g_up - 2.0 * p0 + g_dn) / (gamma_bump**2)
        gammas[i] = g_samples.mean()
        if return_errors:
            gamma_errors[i] = g_samples.std(ddof=1) / np.sqrt(n_paths)

    if return_errors:
        return prices, deltas, gammas, delta_errors, gamma_errors
    return prices, deltas, gammas

"""Delta <-> strike conversion and strike-conditional implied vol.

The smile is quoted in deltas but the engines (and the Dupire local-vol
grid) work in strikes, so this module converts between the two.  The
mapping is implicit when the vol is read off the surface, which is why
:func:`strike_from_delta` uses a short fixed-point iteration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ..params import BSParams
from .surface import VolSmile, VolSurface, _interp_total_variance

__all__ = [
    "bs_delta",
    "strike_from_delta",
    "sigma_for_strike",
]


def bs_delta(
    S: float, K: float, r: float, q: float, sigma: float, T: float, is_call: bool
) -> float:
    """Unadjusted Black-Scholes spot delta (signed for puts)."""
    if T <= 0:
        if is_call:
            return float(np.exp(-q * T) * (S > K))
        return float(-np.exp(-q * T) * (S < K))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(-np.exp(-q * T) * norm.cdf(-d1))


def _atm_strike(S: float, r: float, q: float, sigma: float, T: float) -> float:
    """Strike of the delta-neutral straddle (``d1 = 0``)."""
    forward = S * np.exp((r - q) * T)
    return float(forward * np.exp(0.5 * sigma * sigma * T))


def _strike_from_bs_delta(
    S: float, r: float, q: float, sigma: float, T: float, delta_mag: float, is_call: bool
) -> float:
    """Closed-form strike whose spot delta magnitude is ``delta_mag``."""
    p = delta_mag / np.exp(-q * T)
    if not 0.0 < p < 1.0:
        raise ValueError("delta is inconsistent with the carry q")
    d1 = norm.ppf(p) if is_call else -norm.ppf(p)
    return float(S * np.exp((r - q + 0.5 * sigma * sigma) * T - d1 * sigma * np.sqrt(T)))


def _strike_for_delta(
    params: BSParams, delta: float, T: float, sigma: float
) -> float:
    """Strike for a signed delta at a *fixed* vol ``sigma``."""
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between -1 and 1")
    if delta == 0.0:
        return _atm_strike(params.S0, params.r, params.q, sigma, T)
    return _strike_from_bs_delta(
        params.S0, params.r, params.q, sigma, T, abs(delta), delta > 0
    )


def _pillar_strikes(
    smile: VolSmile, params: BSParams, T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``(log-strike, vol)`` at the smile pillars, sorted by strike."""
    deltas, vols = smile.pillars()
    strikes = np.array(
        [_strike_for_delta(params, d, T, s) for d, s in zip(deltas, vols)]
    )
    order = np.argsort(strikes)
    return np.log(strikes[order]), vols[order]


def strike_from_delta(
    surface: VolSurface,
    params: BSParams,
    delta: float,
    T: float,
    sigma: Optional[float] = None,
) -> float:
    """Strike whose Black-Scholes spot delta equals the signed ``delta``.

    With ``sigma=None`` the vol is taken from the surface, which makes the
    mapping implicit, so a short fixed-point iteration is used: invert the
    delta at the current vol, re-read the surface vol at that strike, and
    repeat.  Passing an explicit ``sigma`` performs a single inversion.
    """
    delta = float(delta)
    T = float(T)
    if T <= 0:
        raise ValueError("maturity must be strictly positive")
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between -1 and 1")

    if sigma is not None:
        return _strike_for_delta(params, delta, T, float(sigma))

    sig = float(surface.interp_sigma(T, delta))
    strike = _strike_for_delta(params, delta, T, sig)
    for _ in range(50):
        new_sig = float(sigma_for_strike(surface, params, strike, T))
        if abs(new_sig - sig) <= 1e-14:
            break
        sig = new_sig
        strike = _strike_for_delta(params, delta, T, sig)
    return strike


def _strike_vols_per_maturity(surface: VolSurface, params: BSParams, strikes):
    """Implied vol at ``strikes`` for every quoted maturity.

    Returns a ``(n_maturities, ...)`` array: at each maturity the pillar
    deltas are converted into strikes with their own quoted vols, and the
    smile is read linearly in log-strike (clamped).
    """
    log_strikes = np.log(np.asarray(strikes, dtype=float))
    curves = [
        _pillar_strikes(surface.smiles[float(t)], params, float(t))
        for t in surface.maturities
    ]
    return np.stack(
        [np.interp(log_strikes, logK, vols) for logK, vols in curves], axis=0
    )


def sigma_for_strike(
    surface: VolSurface, params: BSParams, K, T: float
) -> float:
    """Approximate implied vol at strike ``K`` and maturity ``T``.

    At every quoted maturity the pillar deltas are converted into strikes
    using their own quoted vols, so the vol can be interpolated linearly
    in log-strike.  The resulting *total variance* is then interpolated
    linearly in maturity (flat vol outside the quoted range).
    """
    K_arr = np.asarray(K, dtype=float)
    per_mat = _strike_vols_per_maturity(surface, params, K_arr)
    if float(T) <= 0.0:
        return per_mat[0]
    w_stack = per_mat**2 * surface.maturities.reshape((-1,) + (1,) * K_arr.ndim)
    return np.sqrt(_interp_total_variance(T, surface.maturities, w_stack) / float(T))

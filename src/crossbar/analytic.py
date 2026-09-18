"""Closed-form prices used as the validation benchmark.

Contains the Black-Scholes vanilla formula and the Reiner-Rubinstein /
Haug single-barrier formulas for continuously monitored options with a
cash rebate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .params import BarrierSpec, BSParams


def vanilla_payoff(ST, K, is_call):
    """European payoff ``max(ST - K, 0)`` (call) or ``max(K - ST, 0)``."""
    ST = np.asarray(ST, dtype=float)
    if is_call:
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def price_vanilla(S0, K, r, q, sigma, T, is_call=True):
    """Black-Scholes price of a European vanilla option.

    Broadcasts over array inputs and handles the deterministic
    (``T <= 0`` or ``sigma <= 0``) limits.
    """
    S0b, Kb, Tb, sigb = np.broadcast_arrays(
        np.asarray(S0, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    res = np.empty_like(S0b, dtype=float)

    mask_T = Tb <= 0
    if mask_T.any():
        res[mask_T] = vanilla_payoff(S0b[mask_T], Kb[mask_T], is_call)

    mask_sigma = (~mask_T) & (sigb <= 0)
    if mask_sigma.any():
        ST_det = S0b[mask_sigma] * np.exp((r - q) * Tb[mask_sigma])
        res[mask_sigma] = np.exp(-r * Tb[mask_sigma]) * vanilla_payoff(
            ST_det, Kb[mask_sigma], is_call
        )

    mask_main = ~(mask_T | mask_sigma)
    if mask_main.any():
        sqrtT = np.sqrt(Tb[mask_main])
        sig = sigb[mask_main]
        d1 = (
            np.log(S0b[mask_main] / Kb[mask_main])
            + (r - q + 0.5 * sig**2) * Tb[mask_main]
        ) / (sig * sqrtT)
        d2 = d1 - sig * sqrtT
        disc_r = np.exp(-r * Tb[mask_main])
        disc_q = np.exp(-q * Tb[mask_main])
        if is_call:
            res[mask_main] = S0b[mask_main] * disc_q * norm.cdf(d1) - Kb[
                mask_main
            ] * disc_r * norm.cdf(d2)
        else:
            res[mask_main] = Kb[mask_main] * disc_r * norm.cdf(-d2) - S0b[
                mask_main
            ] * disc_q * norm.cdf(-d1)

    return float(res) if res.shape == () else res


def barrier_rebate_terms(bs: BSParams, bar: BarrierSpec):
    """Closed-form rebate corrections ``(E, F)`` for a single barrier.

    ``E`` values the knock-in rebate (paid at maturity when the barrier is
    never hit) and ``F`` the knock-out rebate (paid at the first hitting
    time).  Both are zero for a zero rebate or at expiry.
    """
    S, X, H = bs.S0, bar.K, bar.H
    r, q, sigma, T = bs.r, bs.q, bs.sigma, bs.T
    rebate = bar.rebate
    if rebate == 0.0 or T <= 0:
        return 0.0, 0.0

    sqrtT = np.sqrt(T)
    mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = np.sqrt(mu * mu + 2.0 * r / (sigma * sigma))
    eta = 1.0 if not bar.is_up else -1.0  # +1 down, -1 up

    ln_HS = np.log(H / S)
    ln_SH = -ln_HS
    h2 = (ln_SH / (sigma * sqrtT)) + mu * sigma * sqrtT
    y4 = (ln_HS / (sigma * sqrtT)) + mu * sigma * sqrtT
    z = (ln_HS / (sigma * sqrtT)) + lam * sigma * sqrtT

    H_over_S = H / S
    E = rebate * np.exp(-r * T) * (
        norm.cdf(eta * h2) - H_over_S ** (2.0 * mu) * norm.cdf(eta * y4)
    )
    F = rebate * (
        H_over_S ** (mu + lam) * norm.cdf(eta * z)
        + H_over_S ** (mu - lam)
        * norm.cdf(eta * (z - 2.0 * lam * sigma * sqrtT))
    )
    return float(E), float(F)


def price_barrier_closed_form(bs: BSParams, bar: BarrierSpec) -> float:
    """Reiner-Rubinstein / Haug single-barrier price.

    Assumes continuous monitoring and ``S0`` on the standard side of the
    barrier (below an up barrier, above a down barrier).  The cash rebate
    is paid at the first hitting time for knock-outs and at maturity for
    knock-ins, matching the Monte Carlo convention in this package.
    """
    S = bs.S0
    X = bar.K
    H = bar.H
    r = bs.r
    q = bs.q
    sigma = bs.sigma
    T = bs.T
    rebate = bar.rebate

    if T <= 0:
        if not bar.is_in:
            return float(vanilla_payoff(S, X, bar.is_call))
        return float(rebate)

    sqrtT = np.sqrt(T)

    # Haug parameters
    mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = np.sqrt(mu * mu + 2.0 * r / (sigma * sigma))

    phi = 1.0 if bar.is_call else -1.0  # +1 call, -1 put
    eta = 1.0 if not bar.is_up else -1.0  # +1 down, -1 up

    ln_S_over_X = np.log(S / X)
    ln_S_over_H = np.log(S / H)
    ln_H2_over_SX = np.log(H * H / (S * X))
    ln_H_over_S = np.log(H / S)

    d1 = (ln_S_over_X / (sigma * sqrtT)) + (mu + 1.0) * sigma * sqrtT
    d2 = d1 - sigma * sqrtT

    h1 = (ln_S_over_H / (sigma * sqrtT)) + (mu + 1.0) * sigma * sqrtT
    h2 = h1 - sigma * sqrtT

    y1 = (ln_H2_over_SX / (sigma * sqrtT)) + (mu + 1.0) * sigma * sqrtT
    y2 = y1 - sigma * sqrtT

    y3 = (ln_H_over_S / (sigma * sqrtT)) + (mu + 1.0) * sigma * sqrtT
    y4 = y3 - sigma * sqrtT

    H_over_S = H / S
    H_over_S_pow_2mu = H_over_S ** (2.0 * mu)
    H_over_S_pow_2mu1 = H_over_S ** (2.0 * (mu + 1.0))

    # Building blocks A..F (Haug 4.17.1)
    A = phi * S * np.exp(-q * T) * norm.cdf(phi * d1) - phi * X * np.exp(
        -r * T
    ) * norm.cdf(phi * d2)

    B = phi * S * np.exp(-q * T) * norm.cdf(phi * h1) - phi * X * np.exp(
        -r * T
    ) * norm.cdf(phi * h2)

    C = (
        phi * S * np.exp(-q * T) * H_over_S_pow_2mu1 * norm.cdf(eta * y1)
        - phi * X * np.exp(-r * T) * H_over_S_pow_2mu * norm.cdf(eta * y2)
    )

    D = (
        phi * S * np.exp(-q * T) * H_over_S_pow_2mu1 * norm.cdf(eta * y3)
        - phi * X * np.exp(-r * T) * H_over_S_pow_2mu * norm.cdf(eta * y4)
    )

    # Rebate corrections: E (knock-in, paid at maturity if never hit) and
    # F (knock-out, paid at the first hitting time).
    E, F = barrier_rebate_terms(bs, bar)

    is_call = bar.is_call
    is_in = bar.is_in
    is_up = bar.is_up
    is_out = not is_in
    is_down = not is_up
    X_gt_H = X > H

    if is_in and is_down and is_call:  # down-and-in call
        price = (C + E) if X_gt_H else (A - B + D + E)
    elif is_in and is_up and is_call:  # up-and-in call
        price = (A + E) if X_gt_H else (B - C + D + E)
    elif is_in and is_down and not is_call:  # down-and-in put
        price = (B - C + D + E) if X_gt_H else (A + E)
    elif is_in and is_up and not is_call:  # up-and-in put
        price = (A - B + D + E) if X_gt_H else (C + E)
    elif is_out and is_down and is_call:  # down-and-out call
        price = (A - C + F) if X_gt_H else (B - D + F)
    elif is_out and is_up and is_call:  # up-and-out call
        price = F if X_gt_H else (A - B + C - D + F)
    elif is_out and is_down and not is_call:  # down-and-out put
        price = (A - B + C - D + F) if X_gt_H else F
    elif is_out and is_up and not is_call:  # up-and-out put
        price = (B - D + F) if X_gt_H else (A - C + F)
    else:
        raise ValueError("unrecognised barrier configuration")  # pragma: no cover

    return float(price)

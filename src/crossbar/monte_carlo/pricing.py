"""Barrier payoff sampling and Monte Carlo price estimates.

The payoff sampler applies knock-out rebates at the first hitting time
and knock-in rebates at maturity, and a Black-Scholes control variate
removes most of the terminal-payoff variance.
"""

from __future__ import annotations

import numpy as np

from ..analytic import price_vanilla, vanilla_payoff
from ..params import BarrierSpec, BSParams
from .barrier import barrier_hits
from .simulate import gen_normals, monte_carlo_paths

__all__ = [
    "price_barrier",
    "price_barrier_cv",
    "price_barrier_mc",
]


def _payoff_samples(
    bs: BSParams, bar: BarrierSpec, S: np.ndarray, seed: int, sigma=None
):
    """Discounted barrier and vanilla payoff samples for path matrix ``S``."""
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    hit, step = barrier_hits(S, bs, bar, seed=seed, sigma=sigma)

    disc = np.exp(-bs.r * bs.T)
    vanilla_T = vanilla_payoff(S[:, -1], bar.K, bar.is_call)
    dt = bs.T / n_steps

    if not bar.is_in:
        t_hit = np.clip(step, 1, n_steps) * dt
        rebate_pv = bar.rebate * np.exp(-bs.r * t_hit)
        X = np.where(hit, rebate_pv, disc * vanilla_T)
    else:
        X = np.where(hit, disc * vanilla_T, disc * bar.rebate)

    return X, disc * vanilla_T


def price_barrier(
    bs: BSParams,
    bar: BarrierSpec,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
    Z=None,
    S=None,
    sigma=None,
):
    """Discounted barrier and vanilla payoff samples.

    Returns ``(X, Y)`` where ``X`` are the barrier-option payoffs and
    ``Y`` the vanilla payoffs on the same paths (used by the control
    variate).  Knock-out rebates are paid at the first hitting time;
    knock-in rebates are paid at maturity when the barrier is never hit.

    ``Z`` supplies pre-drawn normals and ``S`` an already-simulated path
    matrix; either overrides the sizes above.  Passing ``S`` lets several
    revaluations share one simulation.  ``sigma`` may be a scalar or a
    length-``n_steps`` sequence of instantaneous vols; it is used both to
    simulate (when ``S`` is not given) and for the bridge variance.
    """
    if S is None:
        if Z is None:
            Z = gen_normals(n_paths, n_steps, seed=seed)
        else:
            n_paths, n_steps = Z.shape
        S = monte_carlo_paths(bs, Z, sigma)

    return _payoff_samples(bs, bar, S, seed, sigma=sigma)


def price_barrier_cv(
    bs: BSParams,
    bar: BarrierSpec,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
    Z=None,
    S=None,
    sigma=None,
) -> np.ndarray:
    """Control-variate adjusted barrier payoff samples.

    Uses the analytic Black-Scholes vanilla price as the control mean, so
    the sample mean of the returned array is the control-variate price
    estimate.  Pass a pre-drawn ``Z`` or an already-simulated path matrix
    ``S`` to reuse the same paths across revaluations (common random
    numbers).  For a time-varying ``sigma`` the control mean is the
    Black-Scholes price at the effective vol ``sqrt(mean(sigma^2))``,
    which is exact for deterministic vol.
    """
    X, Y = price_barrier(
        bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S, sigma=sigma
    )
    if sigma is None:
        sigma_eff = bs.sigma
    else:
        sig = np.asarray(sigma, dtype=float)
        sigma_eff = float(sig) if sig.ndim == 0 else float(np.sqrt(np.mean(sig**2)))
    EY = price_vanilla(
        bs.S0, bar.K, bs.r, bs.q, sigma_eff, bs.T, is_call=bar.is_call
    )

    Ym = Y - Y.mean()
    Xm = X - X.mean()
    n = len(Y)
    varY = float(Ym @ Ym) / (n - 1)
    covXY = float(Xm @ Ym) / (n - 1)
    beta = 0.0 if varY == 0.0 else covXY / varY
    return X - beta * (Y - EY)


def price_barrier_mc(
    bs: BSParams,
    bar: BarrierSpec,
    n_paths: int = 200_000,
    n_steps: int = 252,
    seed: int = 0,
    control_variate: bool = True,
    Z=None,
    S=None,
    sigma=None,
):
    """Monte Carlo barrier price and standard error.

    Returns ``(price, standard_error)``.  Pass a pre-drawn normal matrix
    ``Z`` of shape ``(n_paths, n_steps)``, or an already-simulated path
    matrix ``S``, to reuse a fixed set of paths -- for example to share
    common random numbers across the bumps of a finite-difference Greek.
    ``sigma`` may be a scalar or a length-``n_steps`` sequence of
    instantaneous vols (see :func:`crossbar.vol_surface.stepwise_sigmas_for_strike`).
    """
    if control_variate:
        samples = price_barrier_cv(
            bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S,
            sigma=sigma,
        )
    else:
        samples = price_barrier(
            bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S,
            sigma=sigma,
        )[0]

    price = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return price, se

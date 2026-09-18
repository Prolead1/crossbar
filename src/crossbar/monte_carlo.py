"""Monte Carlo pricing of single-barrier options.

Variance reduction combines Latin-hypercube stratification, antithetic
variates and moment matching, with a Black-Scholes control variate on the
discounted European payoff.  When the contract is continuously monitored
the discrete sampling bias is corrected with Brownian-bridge
barrier-crossing probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .analytic import price_vanilla, vanilla_payoff
from .params import BarrierSpec, BSParams


def gen_normals(
    n_paths: int,
    n_steps: int,
    seed: int = 0,
    antithetic: bool = True,
    lhs: bool = True,
    moment_match: bool = True,
) -> np.ndarray:
    """Draw an ``(n_paths, n_steps)`` array of standard normals.

    ``lhs`` applies Latin-hypercube stratification per time step,
    ``antithetic`` appends negated paths, and ``moment_match`` standardises
    each column to zero mean and unit variance.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be at least 1")

    rng = np.random.default_rng(seed)
    m = (n_paths + 1) // 2 if antithetic else n_paths

    if lhs:
        strata = (np.arange(m).reshape(-1, 1) + rng.random((m, n_steps))) / m
        U = np.empty((m, n_steps))
        for j in range(n_steps):
            U[:, j] = rng.permutation(strata[:, j])
        Z = ndtri(U)
    else:
        Z = rng.standard_normal((m, n_steps))

    if antithetic:
        Z = np.vstack([Z, -Z])
    if Z.shape[0] > n_paths:
        Z = Z[:n_paths]

    if moment_match:
        Z -= Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, ddof=1, keepdims=True)
        std[std == 0.0] = 1.0
        Z /= std

    return Z


def log_price_increments(bs: BSParams, Z: np.ndarray, sigma=None) -> np.ndarray:
    """Cumulative log-return paths, before adding ``log(S0)``.

    Returns an ``(n_paths, n_steps)`` array ``L`` whose column ``k`` is
    the sum of the risk-neutral log increments up to step ``k + 1``.  By
    factoring out the spot, one draw of ``Z`` can drive the simulated
    paths for many spots: add ``log(S0)`` and exponentiate (see
    :func:`paths_from_log_increments`).  This is what makes the
    bump-and-revalue Monte Carlo Greeks cheap.
    """
    n_paths, n_steps = Z.shape
    dt = bs.T / n_steps

    if sigma is None:
        sig = np.full(n_steps, float(bs.sigma))
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 0:
            sig = np.full(n_steps, float(sig))
        elif sig.shape != (n_steps,):
            raise ValueError(
                f"sigma must be scalar or have shape ({n_steps},), got {sig.shape}"
            )

    drift = (bs.r - bs.q - 0.5 * sig**2) * dt
    vol = sig * np.sqrt(dt)

    # Build the log-price increments in place to avoid several
    # full-size ``(n_paths, n_steps)`` temporaries.
    logS = np.multiply(Z, vol)
    logS += drift
    np.cumsum(logS, axis=1, out=logS)
    return logS


def paths_from_log_increments(logS: np.ndarray, S0: float) -> np.ndarray:
    """Turn cumulative log returns into ``(n_paths, n_steps + 1)`` paths."""
    n_paths, n_steps = logS.shape
    scaled = logS + np.log(S0)
    np.exp(scaled, out=scaled)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = scaled
    return S


def monte_carlo_paths(bs: BSParams, Z: np.ndarray, sigma=None) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure.

    Returns an ``(n_paths, n_steps + 1)`` array whose first column is
    ``bs.S0``.  ``sigma`` may be ``None`` (fall back to the flat
    ``bs.sigma``), a scalar, or a length-``n_steps`` sequence of
    instantaneous vols -- for instance the output of
    :func:`crossbar.vol_surface.stepwise_sigmas_from_surface` -- so the
    paths can follow a market-implied term structure.
    """
    n_paths, n_steps = Z.shape
    # ``log_price_increments`` returns a fresh array, so scale it in place
    # rather than paying for a copy through ``paths_from_log_increments``.
    logS = log_price_increments(bs, Z, sigma)
    logS += np.log(bs.S0)
    np.exp(logS, out=logS)

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = bs.S0
    S[:, 1:] = logS
    return S


def step_bridge_cross_prob(S_t, S_tp1, H, sigma, dt, up=True):
    """Probability that a Brownian bridge crosses ``H`` on one step.

    Conditional probability that ``H`` is breached between two simulated
    log-price endpoints, given that neither endpoint has breached it.
    """
    x = np.log(S_t)
    y = np.log(S_tp1)
    h = np.log(H)
    var = sigma**2 * dt

    if up:
        safe = h >= np.maximum(x, y)
        p = np.exp(-2.0 * (h - x) * (h - y) / var)
    else:
        safe = h <= np.minimum(x, y)
        p = np.exp(-2.0 * (x - h) * (y - h) / var)

    p = np.where(safe, p, 0.0)
    return np.clip(p, 0.0, 1.0)


def barrier_hits(
    S: np.ndarray,
    bs: BSParams,
    bar: BarrierSpec,
    seed: int = 0,
    chunk_size: int | None = None,
):
    """Locate the first barrier hit for each path.

    Returns ``(hit, step)`` where ``hit`` is a boolean array and ``step``
    is the first monitored time index (``1..n_steps``, or ``n_steps + 1``
    when the path never hits).  For continuous monitoring, Brownian-bridge
    crossings between grid points are sampled in addition to discrete
    hits.

    Paths are processed in blocks of ``chunk_size`` so the bridge
    probabilities never materialise as a full ``(n_paths, n_steps)``
    array.  ``None`` picks a block that bounds that working set to a few
    tens of megabytes.
    """
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    dt = bs.T / n_steps
    up = bar.is_up

    crossed = (S >= bar.H) if up else (S <= bar.H)
    crossed = crossed.copy()
    crossed[:, 0] = False  # S0 is on the safe side by construction

    disc_hit = crossed.any(axis=1)
    disc_step = np.argmax(crossed, axis=1)
    disc_step = np.where(disc_hit, disc_step, n_steps + 1)

    if bar.monitor == "discrete":
        return disc_hit, disc_step

    if chunk_size is None:
        chunk_size = max(1, 2_000_000 // n_steps)

    log_H = np.log(bar.H)
    var = bs.sigma**2 * dt
    u = np.random.default_rng(seed).random(n_paths)
    thresh = 1.0 - u

    hit = disc_hit
    step = disc_step
    for lo in range(0, n_paths, chunk_size):
        hi = min(lo + chunk_size, n_paths)
        # Log each block once; the two step endpoints are shifted views.
        logS = np.log(S[lo:hi])
        x = logS[:, :-1]
        y = logS[:, 1:]
        if up:
            safe = log_H >= np.maximum(x, y)
            p = np.exp(-2.0 * (log_H - x) * (log_H - y) / var)
        else:
            safe = log_H <= np.minimum(x, y)
            p = np.exp(-2.0 * (x - log_H) * (y - log_H) / var)
        p = np.where(safe, p, 0.0)
        np.clip(p, 0.0, 1.0, out=p)
        np.subtract(1.0, p, out=p)
        np.cumprod(p, axis=1, out=p)

        bridge_crossed = p <= thresh[lo:hi, None]
        bridge_hit = bridge_crossed.any(axis=1)
        bridge_step = np.argmax(bridge_crossed, axis=1) + 1
        bridge_step = np.where(bridge_hit, bridge_step, n_steps + 1)

        hit[lo:hi] |= bridge_hit
        np.minimum(step[lo:hi], bridge_step, out=step[lo:hi])

    return hit, step


def _payoff_samples(bs: BSParams, bar: BarrierSpec, S: np.ndarray, seed: int):
    """Discounted barrier and vanilla payoff samples for path matrix ``S``."""
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    hit, step = barrier_hits(S, bs, bar, seed=seed)

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
):
    """Discounted barrier and vanilla payoff samples.

    Returns ``(X, Y)`` where ``X`` are the barrier-option payoffs and
    ``Y`` the vanilla payoffs on the same paths (used by the control
    variate).  Knock-out rebates are paid at the first hitting time;
    knock-in rebates are paid at maturity when the barrier is never hit.

    ``Z`` supplies pre-drawn normals and ``S`` an already-simulated path
    matrix; either overrides the sizes above.  Passing ``S`` lets several
    revaluations share one simulation.
    """
    if S is None:
        if Z is None:
            Z = gen_normals(n_paths, n_steps, seed=seed)
        else:
            n_paths, n_steps = Z.shape
        S = monte_carlo_paths(bs, Z)

    return _payoff_samples(bs, bar, S, seed)


def price_barrier_cv(
    bs: BSParams,
    bar: BarrierSpec,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
    Z=None,
    S=None,
) -> np.ndarray:
    """Control-variate adjusted barrier payoff samples.

    Uses the analytic Black-Scholes vanilla price as the control mean, so
    the sample mean of the returned array is the control-variate price
    estimate.  Pass a pre-drawn ``Z`` or an already-simulated path matrix
    ``S`` to reuse the same paths across revaluations (common random
    numbers).
    """
    X, Y = price_barrier(
        bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S
    )
    EY = price_vanilla(bs.S0, bar.K, bs.r, bs.q, bs.sigma, bs.T, is_call=bar.is_call)

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
):
    """Monte Carlo barrier price and standard error.

    Returns ``(price, standard_error)``.  Pass a pre-drawn normal matrix
    ``Z`` of shape ``(n_paths, n_steps)``, or an already-simulated path
    matrix ``S``, to reuse a fixed set of paths -- for example to share
    common random numbers across the bumps of a finite-difference Greek.
    """
    if control_variate:
        samples = price_barrier_cv(
            bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S
        )
    else:
        samples = price_barrier(
            bs, bar, n_steps=n_steps, n_paths=n_paths, seed=seed, Z=Z, S=S
        )[0]

    price = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return price, se

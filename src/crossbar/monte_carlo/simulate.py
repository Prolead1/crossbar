"""Path generation and variance reduction.

Variance reduction combines Latin-hypercube stratification, antithetic
variates and moment matching.  Paths are built from a shared
``(n_paths, n_steps)`` normal matrix, and GBM is linear in the spot, so
one draw can price many bumped spots -- which is what makes the
bump-and-revalue Monte Carlo Greeks cheap.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from ..params import BSParams

__all__ = [
    "gen_normals",
    "log_price_increments",
    "paths_from_log_increments",
    "monte_carlo_paths",
]


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

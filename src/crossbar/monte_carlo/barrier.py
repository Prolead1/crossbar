"""Barrier-hit detection with Brownian-bridge crossing correction.

For continuously monitored barriers the discrete sampling bias is
corrected with the conditional Brownian-bridge crossing probability
between two simulated endpoints, sampled along the path.
"""

from __future__ import annotations

import numpy as np

from ..params import BarrierSpec, BSParams

__all__ = ["step_bridge_cross_prob", "barrier_hits"]


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
    sigma=None,
):
    """Locate the first barrier hit for each path.

    Returns ``(hit, step)`` where ``hit`` is a boolean array and ``step``
    is the first monitored time index (``1..n_steps``, or ``n_steps + 1``
    when the path never hits).  For continuous monitoring, Brownian-bridge
    crossings between grid points are sampled in addition to discrete
    hits.  ``sigma`` may be a scalar or a length-``n_steps`` sequence of
    instantaneous vols, which is used for the per-step bridge variance.

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
    if sigma is None:
        var = bs.sigma**2 * dt
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 0:
            sig = np.full(n_steps, float(sig))
        elif sig.shape != (n_steps,):
            raise ValueError(
                f"sigma must be scalar or have shape ({n_steps},), got {sig.shape}"
            )
        var = sig**2 * dt
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

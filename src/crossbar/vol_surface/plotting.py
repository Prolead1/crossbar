"""Surface visualisation helpers.

:func:`total_variance_grid` builds the implied total-variance grid that
the plot is drawn from; :func:`plot_vol_surface` renders it as a 3-D
surface (requires the optional :mod:`matplotlib` dependency).
"""

from __future__ import annotations

import numpy as np

from ..params import BSParams
from .strike import _strike_vols_per_maturity
from .surface import VolSurface, _interp_total_variance

__all__ = ["total_variance_grid", "plot_vol_surface"]


def total_variance_grid(surface: VolSurface, params: BSParams, strikes, maturities):
    """Total implied variance ``w(T, K) = sigma(T, K)^2 T`` on a grid.

    Returns an ``(len(maturities), len(strikes))`` array.  At every quoted
    maturity the smile is read linearly in log-strike (clamped); the
    resulting total variance is then interpolated linearly in maturity at
    fixed strike (flat vol outside the quoted range).  A flat vol surface
    yields ``w = sigma^2 T`` at every maturity.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    per_mat = _strike_vols_per_maturity(surface, params, strikes)
    w_stack = per_mat**2 * surface.maturities.reshape((-1,) + (1,) * strikes.ndim)
    out = np.empty((maturities.size, strikes.size))
    for i, T in enumerate(maturities):
        out[i] = _interp_total_variance(float(T), surface.maturities, w_stack)
    return out


def plot_vol_surface(
    surface: VolSurface,
    params: BSParams,
    strikes=None,
    maturities=None,
    ax=None,
):
    """Plot the implied vol surface over ``(maturity, strike)``.

    Builds the total-variance grid, converts it back to vol, and draws a
    3-D surface.  Requires :mod:`matplotlib` (the ``plot`` extra).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional dependency

    strikes = (
        np.linspace(0.7 * params.S0, 1.3 * params.S0, 61)
        if strikes is None
        else np.asarray(strikes, dtype=float)
    )
    maturities = (
        np.linspace(surface.maturities[0], surface.maturities[-1], 41)
        if maturities is None
        else np.asarray(maturities, dtype=float)
    )
    var = total_variance_grid(surface, params, strikes, maturities)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = np.sqrt(var / maturities[:, None])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    K, T = np.meshgrid(strikes, maturities)
    ax.plot_surface(K, T, vol, cmap="viridis", linewidth=0)
    ax.set_xlabel("strike")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol")
    return ax
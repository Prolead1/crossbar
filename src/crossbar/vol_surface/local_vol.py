"""Dupire local volatility and arbitrage screening.

:class:`LocalVolSurface` converts a delta-quoted implied surface into a
proper Dupire local-volatility surface.  :func:`build_sigma_grid` feeds
its ``sigma_loc(S, t)`` grid to the finite-difference scheme, and
:func:`check_arbitrage` screens the implied surface for calendar and
butterfly arbitrage.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..params import BSParams, BarrierSpec
from .strike import _pillar_strikes
from .surface import VolSurface

__all__ = [
    "LocalVolSurface",
    "local_volatility",
    "local_vol_grid",
    "check_arbitrage",
    "build_sigma_grid",
]


class LocalVolSurface:
    """Dupire local volatility implied by a delta-quoted :class:`VolSurface`.

    The implied total variance ``w = sigma_imp^2 T`` is represented as a
    function of log-strike on each quoted maturity (a clamped cubic spline
    through the five pillars, held flat in vol outside them) and
    interpolated linearly in maturity at fixed strike.  Dupire's formula
    in log-forward-moneyness ``y = log(K / F_T)`` then gives the local
    variance

    .. math::

        \\sigma_{loc}^2 = \\frac{\\partial_T w}
        {1 - \\frac{y}{w}\\partial_y w
         + \\frac14\\left(-\\frac14 - \\frac1w + \\frac{y^2}{w^2}\\right)
           (\\partial_y w)^2
         + \\frac12 \\partial^2_y w}.

    where ``\\partial_T w`` is taken at fixed moneyness and all
    ``\\partial_y`` derivatives at fixed maturity.  A flat implied-vol
    surface reproduces that constant exactly, and a strike-independent
    term structure reproduces the quoted forward variance.  The
    denominator is the same quantity whose positivity is the
    butterfly-arbitrage condition (see :func:`check_arbitrage`); where it
    turns negative the local variance is floored at a small positive
    value.
    """

    def __init__(self, surface: VolSurface, params: BSParams) -> None:
        self.surface = surface
        self.params = params
        self.maturities = np.asarray(surface.maturities, dtype=float)
        self._logK = []
        self._splines = []
        for t in self.maturities:
            logK, vols = _pillar_strikes(surface.smiles[float(t)], params, float(t))
            w = vols**2 * float(t)
            self._logK.append(logK)
            # Clamped zero slope at the 10-delta wings matches the flat-vol
            # extrapolation used everywhere else and is far better behaved
            # than a natural spline on the tiny short-dated variance range.
            self._splines.append(
                CubicSpline(logK, w, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)
            )

    def _strike_derivatives(self, K, deriv: int) -> np.ndarray:
        """Per-maturity ``d^deriv w / d(log K)^deriv`` at ``K``.

        The spline is evaluated on the pillar log-strike range and held
        flat in vol (so the value is the boundary variance and every
        derivative is zero) outside it.
        """
        lk = np.log(np.asarray(K, dtype=float))
        out = np.empty((self.maturities.size,) + lk.shape)
        for i, (logK, spline) in enumerate(zip(self._logK, self._splines)):
            clamped = np.clip(lk, logK[0], logK[-1])
            value = spline(clamped, deriv)
            if deriv:
                outside = (lk < logK[0]) | (lk > logK[-1])
                value = np.where(outside, 0.0, value)
            out[i] = value
        return out

    def _state(self, K, T):
        """``(w, w_y/w, w_y^2/w, w_yy, w_T)`` at ``K`` and scalar ``T``.

        Quantities are returned in the form that stays finite as ``T``
        tends to zero, so the same expression serves the local variance
        and the butterfly density test.
        """
        K = np.asarray(K, dtype=float)
        T = float(T)
        mats = self.maturities
        wv = self._strike_derivatives(K, 0)
        wy = self._strike_derivatives(K, 1)
        wyy = self._strike_derivatives(K, 2)

        if T <= mats[0] or T >= mats[-1]:
            i = 0 if T <= mats[0] else -1
            unit, du, duu = wv[i], wy[i], wyy[i]
            nu = T / mats[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(unit > 0.0, du / unit, 0.0)
            return unit * nu, ratio, nu * du * ratio, nu * duu, unit / mats[i]

        i = int(np.searchsorted(mats, T))
        t0, t1 = mats[i - 1], mats[i]
        a = (T - t0) / (t1 - t0)
        w = (1.0 - a) * wv[i - 1] + a * wv[i]
        dw = (1.0 - a) * wy[i - 1] + a * wy[i]
        dwv = (1.0 - a) * wyy[i - 1] + a * wyy[i]
        wT = (wv[i] - wv[i - 1]) / (t1 - t0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(w > 0.0, dw / w, 0.0)
        return w, ratio, dw * ratio, dwv, wT

    def sigma(self, K, T):
        """Dupire local volatility at ``K`` (scalar or array), scalar ``T``."""
        params = self.params
        T = float(T)
        w, ratio, dwsq_over_w, dww, wT = self._state(K, T)
        forward = params.S0 * np.exp((params.r - params.q) * T)
        y = np.log(np.asarray(K, dtype=float) / forward)
        denom = (
            1.0
            - y * ratio
            + 0.25
            * (-0.25 * w * w * ratio * ratio - dwsq_over_w + (y * ratio) ** 2)
            + 0.5 * dww
        )
        # w_T in the formula is at fixed moneyness; ``wT`` comes from the
        # fixed-strike interpolation, so shift by (r - q) w_y as the
        # forward moneyness drifts with maturity.
        numerator = wT + (params.r - params.q) * w * ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(
                denom > 0.0, numerator / np.where(denom > 0.0, denom, 1.0), 0.0
            )
        return np.sqrt(np.maximum(variance, 1e-12))

    def butterfly_g(self, K, T):
        """Gatheral's density factor ``g`` at ``K`` and scalar ``T``.

        ``g >= 0`` everywhere is equivalent to a non-negative
        risk-neutral density (no butterfly arbitrage).
        """
        params = self.params
        T = float(T)
        w, ratio, _, dww, _ = self._state(K, T)
        forward = params.S0 * np.exp((params.r - params.q) * T)
        y = np.log(np.asarray(K, dtype=float) / forward)
        return (
            (1.0 - 0.5 * y * ratio) ** 2
            - 0.25 * ratio
            - 0.0625 * w * ratio
            + 0.5 * dww
        )

    def grid(self, S_grid, t_grid) -> np.ndarray:
        """Local vol on a ``(time, spot)`` mesh."""
        S_grid = np.asarray(S_grid, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float)
        out = np.empty((t_grid.size, S_grid.size))
        for i, t in enumerate(t_grid):
            out[i] = self.sigma(S_grid, t)
        return out


def local_volatility(surface: VolSurface, params: BSParams, K, T) -> float:
    """Dupire local vol at ``K`` and scalar ``T`` (convenience wrapper)."""
    return LocalVolSurface(surface, params).sigma(K, T)


def local_vol_grid(surface: VolSurface, params: BSParams, S_grid, t_grid) -> np.ndarray:
    """Dupire local-vol grid ``sigma_loc(S, t)`` on a finite-difference mesh."""
    S_grid = np.asarray(S_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    if S_grid.ndim != 1 or t_grid.ndim != 1:
        raise ValueError("S_grid and t_grid must be one-dimensional")
    return LocalVolSurface(surface, params).grid(S_grid, t_grid)


def check_arbitrage(
    surface: VolSurface,
    params: BSParams,
    strikes=None,
    maturities=None,
    tol: float = 1e-10,
) -> dict:
    """Screen a surface for calendar and butterfly arbitrage.

    Calendar arbitrage is a negative slope of total variance in maturity
    at fixed strike; butterfly arbitrage is a negative Gatheral density
    factor ``g`` (a non-negative risk-neutral density requires
    ``g >= 0``).  Both are evaluated on a log-strike grid (the quoted
    pillar range by default) and across the quoted maturities plus their
    midpoints.

    Returns a dict with ``calendar_ok``/``butterfly_ok`` flags and the
    worst observed slope/factor.
    """
    lv = LocalVolSurface(surface, params)
    if strikes is None:
        logK = np.concatenate(
            [
                _pillar_strikes(surface.smiles[float(t)], params, float(t))[0]
                for t in surface.maturities
            ]
        )
        strikes = np.linspace(np.exp(logK.min()), np.exp(logK.max()), 41)
    strikes = np.asarray(strikes, dtype=float)

    if maturities is None:
        m = np.asarray(surface.maturities, dtype=float)
        mids = 0.5 * (m[:-1] + m[1:]) if m.size > 1 else np.empty(0)
        maturities = np.sort(np.concatenate([m, mids]))
    maturities = np.asarray(maturities, dtype=float)

    slopes = np.array([float(np.min(lv._state(strikes, t)[4])) for t in maturities])
    density = np.array(
        [float(np.min(lv.butterfly_g(strikes, t))) for t in maturities]
    )
    return {
        "calendar_ok": bool(np.min(slopes) >= -tol),
        "butterfly_ok": bool(np.min(density) >= -tol),
        "min_calendar_slope": float(np.min(slopes)),
        "min_butterfly_g": float(np.min(density)),
    }


def build_sigma_grid(
    surface: VolSurface,
    params: BSParams,
    bar: BarrierSpec,
    S_grid,
    t_grid,
) -> np.ndarray:
    """Local-vol grid ``sigma_loc(S, t)`` on a finite-difference mesh.

    The grid comes from :func:`local_vol_grid`, i.e. Dupire's local
    volatility calibrated to the implied surface.  Nodes beyond the
    barrier are dead under continuous monitoring, so they are filled with
    the live edge vol to keep the stencil well behaved.
    """
    S_grid = np.asarray(S_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    if S_grid.ndim != 1 or t_grid.ndim != 1:
        raise ValueError("S_grid and t_grid must be one-dimensional")

    grid = local_vol_grid(surface, params, S_grid, t_grid)

    live = S_grid < bar.H if bar.is_up else S_grid > bar.H
    if bar.monitor == "continuous" and not live.all():
        edge = int(np.argmin(np.abs(S_grid - bar.H)))
        grid[:, ~live] = grid[:, edge][:, None]
    return grid

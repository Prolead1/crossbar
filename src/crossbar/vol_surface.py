"""FX volatility surface built from screen-style delta quotes.

The module turns a table of Bloomberg-style bid/ask volatility quotes
into a coherent smile-and-term-structure model and feeds it into the two
pricing engines:

* :class:`VolSmile` packs the five quoted pillars of one maturity -- ATM,
  25-delta and 10-delta risk reversals and butterflies -- and reconstructs
  the 25d/10d call and put vols.  The full delta smile is the piecewise
  linear interpolation through those five points.
* :class:`VolSurface` collects one :class:`VolSmile` per tenor, keyed by
  time to maturity in years, and interpolates linearly in maturity at a
  fixed delta (:meth:`VolSurface.interp_sigma`), clamping outside the
  quoted range.
* :func:`stepwise_sigmas_from_surface` turns the implied term structure at
  a chosen delta into a sequence of piecewise-constant *instantaneous
  forward* vols whose cumulative variance reproduces the quoted implied
  variance, so :func:`crossbar.monte_carlo.monte_carlo_paths` can evolve
  the underlying under a time-varying ``sigma(t)``.
* :func:`strike_from_delta`, :func:`sigma_for_strike` and
  :func:`build_sigma_grid` translate the delta-quoted surface into a
  ``sigma(S, t)`` grid that the finite-difference scheme can consume as a
  crude local-volatility surface.

Conventions
-----------
Deltas are *unadjusted spot* deltas and are signed: negative for puts,
positive for calls, and ``0`` for the delta-neutral ATM straddle.  Vols
are decimal (a screen quote of ``6.50`` becomes ``0.065``), and a
maturity ``T`` is in years.  Quotes follow the market FX convention::

    RR_25 = sigma_25c - sigma_25p
    BF_25 = (sigma_25c + sigma_25p) / 2 - sigma_atm

so ``sigma_25c = atm + bf_25 + rr_25 / 2`` and
``sigma_25p = atm + bf_25 - rr_25 / 2`` (and likewise for 10-delta).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import norm

from .params import BSParams, BarrierSpec

__all__ = [
    "TENORS",
    "SMILE_DELTAS",
    "EXAMPLE_QUOTES",
    "tenor_to_years",
    "mid_vol",
    "VolSmile",
    "VolSurface",
    "bs_delta",
    "strike_from_delta",
    "sigma_for_strike",
    "stepwise_sigmas_from_surface",
    "build_sigma_grid",
    "total_variance_grid",
    "plot_vol_surface",
]

#: Standard tenor labels mapped to time to maturity in years (ACT/365 for
#: the sub-month pillars, ``n/12`` for the monthly ones).
TENORS: Dict[str, float] = {
    "0N": 1.0 / 365.0,
    "1W": 7.0 / 365.0,
    "2W": 14.0 / 365.0,
    "1M": 1.0 / 12.0,
    "2M": 2.0 / 12.0,
    "3M": 3.0 / 12.0,
    "6M": 6.0 / 12.0,
    "9M": 9.0 / 12.0,
}

#: Signed deltas of the five quoted pillars, ``(25d put, 10d put, ATM,
#: 10d call, 25d call)``.
SMILE_DELTAS: Tuple[float, ...] = (-0.25, -0.10, 0.0, 0.10, 0.25)

#: The five quote fields carried by every tenor.
QUOTE_FIELDS: Tuple[str, ...] = ("atm", "rr_25", "bf_25", "rr_10", "bf_10")

#: A representative EURUSD-style screen: bid/ask vols in percentage points
#: for every standard tenor.  Used by the tests and the documentation.
EXAMPLE_QUOTES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "0N": {
        "atm": (5.80, 6.20),
        "rr_25": (-0.45, -0.25),
        "bf_25": (0.10, 0.30),
        "rr_10": (-1.00, -0.70),
        "bf_10": (0.30, 0.60),
    },
    "1W": {
        "atm": (5.90, 6.30),
        "rr_25": (-0.45, -0.25),
        "bf_25": (0.12, 0.32),
        "rr_10": (-1.05, -0.75),
        "bf_10": (0.32, 0.62),
    },
    "2W": {
        "atm": (6.00, 6.40),
        "rr_25": (-0.44, -0.24),
        "bf_25": (0.14, 0.34),
        "rr_10": (-1.05, -0.75),
        "bf_10": (0.34, 0.64),
    },
    "1M": {
        "atm": (6.20, 6.60),
        "rr_25": (-0.42, -0.22),
        "bf_25": (0.18, 0.38),
        "rr_10": (-1.00, -0.70),
        "bf_10": (0.38, 0.68),
    },
    "2M": {
        "atm": (6.55, 6.95),
        "rr_25": (-0.40, -0.20),
        "bf_25": (0.22, 0.42),
        "rr_10": (-0.95, -0.65),
        "bf_10": (0.42, 0.72),
    },
    "3M": {
        "atm": (6.85, 7.25),
        "rr_25": (-0.38, -0.18),
        "bf_25": (0.26, 0.46),
        "rr_10": (-0.90, -0.60),
        "bf_10": (0.46, 0.76),
    },
    "6M": {
        "atm": (7.40, 7.80),
        "rr_25": (-0.33, -0.13),
        "bf_25": (0.34, 0.54),
        "rr_10": (-0.80, -0.50),
        "bf_10": (0.56, 0.86),
    },
    "9M": {
        "atm": (7.80, 8.20),
        "rr_25": (-0.28, -0.08),
        "bf_25": (0.40, 0.60),
        "rr_10": (-0.72, -0.42),
        "bf_10": (0.66, 0.96),
    },
}


def tenor_to_years(tenor) -> float:
    """Convert a tenor label (or numeric maturity) to years.

    Strings are looked up in :data:`TENORS`; anything else is coerced to
    a strictly positive float and returned unchanged.
    """
    if isinstance(tenor, str):
        key = tenor.upper()
        if key not in TENORS:
            raise ValueError(f"unknown tenor {tenor!r}; expected one of {tuple(TENORS)}")
        return TENORS[key]
    T = float(tenor)
    if not T > 0:
        raise ValueError("maturity must be strictly positive")
    return T


def mid_vol(bid: float, ask: float) -> float:
    """Midpoint of a bid/ask quote in percentage vol points as a decimal.

    ``mid_vol(5.80, 6.20)`` returns ``0.06``.
    """
    bid = float(bid)
    ask = float(ask)
    if ask < bid:
        raise ValueError(f"ask {ask} is below bid {bid}")
    return 0.5 * (bid + ask) * 0.01


class VolSmile:
    """The five-pillar smile of a single maturity.

    Parameters
    ----------
    T:
        Time to maturity in years.
    atm, rr_25, bf_25, rr_10, bf_10:
        Mid decimal vols for the ATM, 25-delta and 10-delta risk reversals
        and butterflies.
    """

    __slots__ = ("T", "atm", "rr_25", "bf_25", "rr_10", "bf_10")

    def __init__(
        self,
        T: float,
        atm: float,
        rr_25: float,
        bf_25: float,
        rr_10: float,
        bf_10: float,
    ) -> None:
        self.T = float(T)
        self.atm = float(atm)
        self.rr_25 = float(rr_25)
        self.bf_25 = float(bf_25)
        self.rr_10 = float(rr_10)
        self.bf_10 = float(bf_10)
        if not self.T > 0:
            raise ValueError("maturity must be strictly positive")
        if not self.atm > 0:
            raise ValueError("ATM vol must be strictly positive")
        if min(self.sigma_25p, self.sigma_25c, self.sigma_10p, self.sigma_10c) <= 0:
            raise ValueError("risk reversal / butterfly imply a non-positive wing vol")

    # -- reconstruction of the wing vols -------------------------------
    @property
    def sigma_25c(self) -> float:
        """25-delta call vol."""
        return self.atm + self.bf_25 + 0.5 * self.rr_25

    @property
    def sigma_25p(self) -> float:
        """25-delta put vol."""
        return self.atm + self.bf_25 - 0.5 * self.rr_25

    @property
    def sigma_10c(self) -> float:
        """10-delta call vol."""
        return self.atm + self.bf_10 + 0.5 * self.rr_10

    @property
    def sigma_10p(self) -> float:
        """10-delta put vol."""
        return self.atm + self.bf_10 - 0.5 * self.rr_10

    def pillars(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(deltas, vols)`` at the five quoted pillars, sorted."""
        deltas = np.asarray(SMILE_DELTAS, dtype=float)
        vols = np.array(
            [
                self.sigma_25p,
                self.sigma_10p,
                self.atm,
                self.sigma_10c,
                self.sigma_25c,
            ]
        )
        return deltas, vols

    def sigma(self, delta):
        """Linear interpolation of vol in signed delta, clamped outside.

        ``delta`` may be a scalar or an array.
        """
        deltas, vols = self.pillars()
        return np.interp(delta, deltas, vols)

    @classmethod
    def from_quotes(cls, T: float, quotes: Mapping[str, Tuple[float, float]]) -> "VolSmile":
        """Build a smile from ``{field: (bid, ask)}`` percentage quotes."""
        missing = [f for f in QUOTE_FIELDS if f not in quotes]
        if missing:
            raise ValueError(f"missing quote fields: {missing}")
        mids = {f: mid_vol(*quotes[f]) for f in QUOTE_FIELDS}
        return cls(T=tenor_to_years(T), **mids)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"VolSmile(T={self.T:.6g}, atm={self.atm:.4f}, rr_25={self.rr_25:.4f}, "
            f"bf_25={self.bf_25:.4f}, rr_10={self.rr_10:.4f}, bf_10={self.bf_10:.4f})"
        )


class VolSurface:
    """A collection of :class:`VolSmile` objects keyed by maturity.

    ``smiles`` may be a mapping ``{T: VolSmile}`` or an iterable of
    :class:`VolSmile` (whose ``T`` attributes become the keys).
    """

    def __init__(self, smiles: Iterable[VolSmile] | Mapping[float, VolSmile]) -> None:
        if isinstance(smiles, Mapping):
            items = {float(k): v for k, v in smiles.items()}
        else:
            items = {float(s.T): s for s in smiles}
        if not items:
            raise ValueError("a volatility surface requires at least one smile")
        self.smiles: Dict[float, VolSmile] = dict(sorted(items.items()))
        self.maturities: np.ndarray = np.array(sorted(self.smiles), dtype=float)

    @classmethod
    def from_quotes(
        cls, quotes: Mapping[object, Mapping[str, Tuple[float, float]]]
    ) -> "VolSurface":
        """Build a surface from ``{tenor: {field: (bid, ask)}}`` quotes."""
        return cls(VolSmile.from_quotes(tenor, q) for tenor, q in quotes.items())

    def interp_sigma(self, T: float, delta) -> float:
        """Vol at maturity ``T`` for signed ``delta``, linear in maturity.

        The smile at each quoted maturity is evaluated first (linear in
        delta, clamped), then interpolated linearly in ``T`` and clamped
        to the first/last quoted maturity.
        """
        delta_arr = np.asarray(delta, dtype=float)
        mats = self.maturities
        if delta_arr.ndim == 0:
            sig = np.array([self.smiles[t].sigma(float(delta_arr)) for t in mats])
            val = np.interp(float(T), mats, sig)
            return float(val)
        # element-wise across an array of deltas
        per_mat = np.stack([self.smiles[t].sigma(delta_arr) for t in mats], axis=0)
        out = np.empty_like(delta_arr)
        for i in range(delta_arr.size):
            out.flat[i] = np.interp(float(T), mats, per_mat[:, i])
        return out


# --------------------------------------------------------------------------
# Delta <-> strike conversion
# --------------------------------------------------------------------------


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


def _interp_maturity(T: float, mats: np.ndarray, per_mat: np.ndarray):
    """Linear interpolation of a ``(n_maturities, ...)`` stack at scalar ``T``."""
    if T <= mats[0]:
        return per_mat[0]
    if T >= mats[-1]:
        return per_mat[-1]
    i = int(np.searchsorted(mats, T))
    t0, t1 = mats[i - 1], mats[i]
    w = (T - t0) / (t1 - t0)
    return (1.0 - w) * per_mat[i - 1] + w * per_mat[i]


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


def sigma_for_strike(
    surface: VolSurface, params: BSParams, K, T: float
) -> float:
    """Approximate implied vol at strike ``K`` and maturity ``T``.

    At every quoted maturity the pillar deltas are converted into strikes
    using their own quoted vols, so the vol can be interpolated linearly
    in log-strike; the per-maturity results are then interpolated linearly
    in maturity and clamped at both ends.
    """
    K_arr = np.asarray(K, dtype=float)
    mats = surface.maturities
    per_mat = np.stack(
        [
            np.interp(
                np.log(K_arr), *_pillar_strikes(surface.smiles[t], params, t)
            )
            for t in mats
        ],
        axis=0,
    )
    return _interp_maturity(float(T), mats, per_mat)


# --------------------------------------------------------------------------
# Term structure -> instantaneous forward vols
# --------------------------------------------------------------------------


def stepwise_sigmas_from_surface(
    surface: VolSurface, T: float, n_steps: int, delta: float = 0.0
) -> np.ndarray:
    """Piecewise-constant instantaneous forward vols to maturity ``T``.

    The implied variance ``w(t) = sigma_impl(t)^2 t`` is read off the
    surface at the ``n_steps + 1`` time nodes for the chosen ``delta``;
    the forward vol on each step is ``sqrt(dw / dt)``, so cumulative
    variance matches the quoted term structure.
    """
    T = float(T)
    n_steps = int(n_steps)
    if T <= 0:
        raise ValueError("maturity must be strictly positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        sig = float(surface.interp_sigma(t[i], delta))
        variance[i] = sig * sig * t[i]
    fwd_var = np.diff(variance)
    fwd_var = np.maximum(fwd_var, 0.0)  # guard tiny negative round-off
    dt = T / n_steps
    return np.sqrt(fwd_var / dt)


# --------------------------------------------------------------------------
# sigma(S, t) grid for the PDE solver
# --------------------------------------------------------------------------


def build_sigma_grid(
    surface: VolSurface,
    params: BSParams,
    bar: BarrierSpec,
    S_grid,
    t_grid,
) -> np.ndarray:
    """Implied-vol grid ``sigma(S, t)`` on a finite-difference mesh.

    Each node is valued through :func:`sigma_for_strike` with the asset
    level as the strike, giving a crude local-volatility surface.  Nodes
    beyond the barrier are dead under continuous monitoring, so they are
    filled with the live edge vol to keep the stencil well behaved.
    """
    S_grid = np.asarray(S_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    if S_grid.ndim != 1 or t_grid.ndim != 1:
        raise ValueError("S_grid and t_grid must be one-dimensional")

    grid = np.empty((t_grid.size, S_grid.size))
    for i, t in enumerate(t_grid):
        grid[i] = np.asarray(sigma_for_strike(surface, params, S_grid, t), dtype=float)

    live = S_grid < bar.H if bar.is_up else S_grid > bar.H
    if bar.monitor == "continuous" and not live.all():
        edge = int(np.argmin(np.abs(S_grid - bar.H)))
        grid[:, ~live] = grid[:, edge][:, None]
    return grid


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------


def total_variance_grid(surface: VolSurface, params: BSParams, strikes, maturities):
    """Total implied variance ``w(T, K) = sigma(T, K)^2 T`` on a grid.

    Returns an ``(len(maturities), len(strikes))`` array.  The vol is
    interpolated linearly in log-strike at each quoted maturity, then
    linearly in maturity (clamped at both ends, exactly like
    :meth:`VolSurface.interp_sigma`), and finally multiplied by ``T``.
    Interpolating the *vol* rather than the total variance keeps this
    consistent with :func:`stepwise_sigmas_from_surface`, so a flat vol
    surface yields ``w = sigma^2 T`` at every maturity.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    mats = surface.maturities
    per_mat = np.stack(
        [
            np.interp(
                np.log(strikes),
                *_pillar_strikes(surface.smiles[t], params, t),
            )
            for t in mats
        ],
        axis=0,
    )
    out = np.empty((maturities.size, strikes.size))
    for i, T in enumerate(maturities):
        sig = _interp_maturity(float(T), mats, per_mat)
        out[i] = sig * sig * T
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

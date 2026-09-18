"""Smile and surface containers built from screen-style delta quotes.

:class:`VolSmile` packs the five quoted pillars of one maturity -- ATM,
25-delta and 10-delta risk reversals and butterflies -- and reconstructs
the 25d/10d call and put vols.  The full delta smile is the piecewise
linear interpolation through those five points.

:class:`VolSurface` collects one :class:`VolSmile` per tenor, keyed by
time to maturity in years.  The smile is evaluated at a fixed signed
delta (linear, clamped), and the resulting *total variance* is
interpolated linearly in maturity, which is the standard
calendar-arbitrage-free choice (:meth:`VolSurface.total_variance`).
Outside the quoted tenors the vol is held flat.

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

from ._constants import QUOTE_FIELDS, SMILE_DELTAS, TENORS

__all__ = [
    "tenor_to_years",
    "mid_vol",
    "VolSmile",
    "VolSurface",
]


def tenor_to_years(tenor) -> float:
    """Convert a tenor label (or numeric maturity) to years.

    Strings are looked up in :data:`crossbar.vol_surface.TENORS`; anything
    else is coerced to a strictly positive float and returned unchanged.
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

    def total_variance(self, T: float, delta):
        """Implied total variance ``w = sigma(T, delta)^2 T`` for signed ``delta``.

        The smile at each quoted maturity is evaluated first (linear in
        delta, clamped) and turned into total variance; the variance is
        then interpolated linearly in ``T`` at fixed ``delta``.  Below the
        first tenor the variance is linear from the origin (so the vol is
        flat), and above the last tenor it grows at the last quoted vol,
        which keeps the forward variance positive.
        """
        delta_arr = np.asarray(delta, dtype=float)
        mats = self.maturities
        if delta_arr.ndim == 0:
            sig = np.array([self.smiles[t].sigma(float(delta_arr)) for t in mats])
            return _interp_total_variance(T, mats, sig**2 * mats)
        per_mat = np.stack([self.smiles[t].sigma(delta_arr) for t in mats], axis=0)
        out = np.empty_like(delta_arr)
        for i in range(delta_arr.size):
            out.flat[i] = _interp_total_variance(
                T, mats, per_mat[:, i] ** 2 * mats
            )
        return out

    def interp_sigma(self, T: float, delta) -> float:
        """Vol at maturity ``T`` for signed ``delta``, linear in total variance.

        The smile at each quoted maturity is evaluated first (linear in
        delta, clamped), then the *total variance* is interpolated
        linearly in ``T`` and clamped to the first/last quoted maturity.
        Interpolating variance rather than vol is the standard
        calendar-arbitrage-free choice; a flat vol surface is unaffected.
        """
        T = float(T)
        if T <= 0.0:
            # The implied vol is only defined for a positive maturity; the
            # natural limit is the flat short-end vol (total variance is
            # linear from the origin).
            t0 = self.maturities[0]
            return np.sqrt(self.total_variance(t0, delta) / t0)
        return np.sqrt(self.total_variance(T, delta) / T)


def _interp_total_variance(T: float, mats: np.ndarray, w_stack: np.ndarray):
    """Interpolate total variance in maturity at a fixed smile/delta point.

    ``w_stack`` holds ``sigma^2 t`` at each quoted maturity.  Between two
    tenors the variance is linear in ``T``; below the first tenor it is
    linear from the origin and above the last it continues at the last
    quoted vol.  Both extensions keep the vol flat and the forward
    variance positive, and they match the clamping of the old vol-space
    interpolation for a constant surface.
    """
    T = float(T)
    if T <= mats[0]:
        return w_stack[0] * (T / mats[0])
    if T >= mats[-1]:
        return w_stack[-1] * (T / mats[-1])
    i = int(np.searchsorted(mats, T))
    t0, t1 = mats[i - 1], mats[i]
    a = (T - t0) / (t1 - t0)
    return (1.0 - a) * w_stack[i - 1] + a * w_stack[i]

"""FX volatility surface built from screen-style delta quotes.

The module turns a table of Bloomberg-style bid/ask volatility quotes
into a coherent smile-and-term-structure model and feeds it into the two
pricing engines:

* :class:`VolSmile` packs the five quoted pillars of one maturity -- ATM,
  25-delta and 10-delta risk reversals and butterflies -- and reconstructs
  the 25d/10d call and put vols.  The full delta smile is the piecewise
  linear interpolation through those five points.
* :class:`VolSurface` collects one :class:`VolSmile` per tenor, keyed by
  time to maturity in years.  The smile is evaluated at a fixed signed
  delta (linear, clamped), and the resulting *total variance* is
  interpolated linearly in maturity, which is the standard
  calendar-arbitrage-free choice (:meth:`VolSurface.total_variance`).
  Outside the quoted tenors the vol is held flat.
* :func:`stepwise_sigmas_from_surface` and
  :func:`stepwise_sigmas_for_strike` turn the implied term structure at a
  chosen delta or strike into piecewise-constant *instantaneous forward*
  vols whose cumulative variance reproduces the quoted implied variance,
  so :func:`crossbar.monte_carlo.monte_carlo_paths` can evolve the
  underlying under a time-varying ``sigma(t)``.
* :class:`LocalVolSurface` converts the implied surface into a proper
  *Dupire local-volatility* surface.  :func:`build_sigma_grid` feeds its
  ``sigma_loc(S, t)`` grid to the finite-difference scheme, and
  :func:`check_arbitrage` screens the implied surface for calendar and
  butterfly arbitrage.

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
from scipy.interpolate import CubicSpline
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
    "stepwise_sigmas_for_strike",
    "LocalVolSurface",
    "local_volatility",
    "local_vol_grid",
    "check_arbitrage",
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


# --------------------------------------------------------------------------
# Term structure -> instantaneous forward vols
# --------------------------------------------------------------------------


def _forward_vols(variance: np.ndarray, T: float, n_steps: int) -> np.ndarray:
    """Piecewise-constant forward vols from a total-variance path."""
    fwd_var = np.diff(np.asarray(variance, dtype=float))
    fwd_var = np.maximum(fwd_var, 0.0)  # guard tiny negative round-off
    return np.sqrt(fwd_var / (float(T) / n_steps))


def stepwise_sigmas_from_surface(
    surface: VolSurface, T: float, n_steps: int, delta: float = 0.0
) -> np.ndarray:
    """Piecewise-constant instantaneous forward vols to maturity ``T``.

    The implied total variance ``w(t) = sigma_impl(t)^2 t`` is read off
    the surface at the ``n_steps + 1`` time nodes for the chosen
    ``delta``; the forward vol on each step is ``sqrt(dw / dt)``, so
    cumulative variance matches the quoted term structure.  This is a
    *sticky-delta* (fixed moneyness) term structure.
    """
    T = float(T)
    n_steps = int(n_steps)
    if T <= 0:
        raise ValueError("maturity must be strictly positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.array(
        [surface.total_variance(t[i], delta) for i in range(n_steps + 1)]
    )
    return _forward_vols(variance, T, n_steps)


def stepwise_sigmas_for_strike(
    surface: VolSurface, params: BSParams, K: float, T: float, n_steps: int
) -> np.ndarray:
    """Forward vols at a *fixed strike* up to maturity ``T``.

    The implied variance at ``K`` is read off :func:`sigma_for_strike` at
    every time node, so the cumulative variance matches the quoted
    smile at that strike.  This is the *sticky-strike* term structure,
    the natural input for a barrier whose level pins the relevant vol.
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
        sig = float(sigma_for_strike(surface, params, K, t[i]))
        variance[i] = sig * sig * t[i]
    return _forward_vols(variance, T, n_steps)


# --------------------------------------------------------------------------
# Dupire local volatility
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------


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

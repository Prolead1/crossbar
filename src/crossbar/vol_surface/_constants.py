"""Static tenor / delta conventions and the sample screen quotes.

These are the market-data constants of :mod:`crossbar.vol_surface`: the
standard tenor table, the five quoted smile pillars, the quote fields
carried by every tenor, and a representative EURUSD-style screen used by
the tests and the documentation.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["TENORS", "SMILE_DELTAS", "QUOTE_FIELDS", "EXAMPLE_QUOTES"]

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

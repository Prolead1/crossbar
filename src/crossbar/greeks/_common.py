"""Shared helpers for the risk profiles."""

from __future__ import annotations

__all__ = ["_extract_price"]


def _extract_price(result) -> float:
    """Return the price from a pricer result.

    Pricers may return a bare price or a ``(price, standard_error)``
    pair (as :func:`crossbar.price_barrier_mc` does); only the price is
    used for the bump-and-revalue profile.
    """
    if isinstance(result, tuple):
        result = result[0]
    return float(result)

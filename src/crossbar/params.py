"""Market and contract parameter containers for single-barrier options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

_DIRECTIONS = ("up", "down")
_KNOCK = ("in", "out")
_MONITORS = ("discrete", "continuous")


@dataclass(frozen=True)
class BSParams:
    """Black-Scholes market parameters.

    Parameters
    ----------
    S0:
        Spot price of the underlying.
    r:
        Continuously compounded risk-free rate.
    q:
        Continuously compounded dividend / carry yield.
    sigma:
        Lognormal volatility.
    T:
        Time to maturity, in years.
    """

    S0: float
    r: float
    q: float
    sigma: float
    T: float


@dataclass(frozen=True)
class BarrierSpec:
    """Specification of a single-barrier contract.

    Parameters
    ----------
    barrier_type:
        One of ``up-and-out``, ``down-and-out``, ``up-and-in``,
        ``down-and-in``.
    monitor:
        ``discrete`` (barrier observed on the simulation grid) or
        ``continuous``.
    H:
        Barrier level.
    K:
        Strike.
    is_call:
        ``True`` for a call, ``False`` for a put.
    rebate:
        Cash rebate. For a knock-out it is paid at the first hitting
        time; for a knock-in it is paid at maturity if the barrier is
        never hit.
    """

    barrier_type: str
    monitor: str
    H: float
    K: float
    is_call: bool
    rebate: float = 0.0

    @property
    def is_up(self) -> bool:
        """``True`` for an up barrier."""
        return "up" in self.barrier_type

    @property
    def is_in(self) -> bool:
        """``True`` for a knock-in option."""
        return "in" in self.barrier_type


def validate_inputs(bs: BSParams, bar: BarrierSpec) -> None:
    """Validate a market/contract pair.

    Raises
    ------
    ValueError
        If any parameter is inconsistent (non-positive spot, volatility or
        maturity, a barrier on the wrong side of spot, an unknown barrier
        type or monitoring convention, ...).
    """
    if not bs.S0 > 0:
        raise ValueError("S0 must be strictly positive")
    if not bs.sigma > 0:
        raise ValueError("sigma must be strictly positive")
    if not bs.T > 0:
        raise ValueError("T must be strictly positive")
    if bar.K <= 0:
        raise ValueError("strike K must be strictly positive")
    if bar.rebate < 0:
        raise ValueError("rebate must be non-negative")

    valid_types = {f"{d}-and-{k}" for d in _DIRECTIONS for k in _KNOCK}
    if bar.barrier_type not in valid_types:
        raise ValueError(f"unknown barrier_type {bar.barrier_type!r}")
    if bar.monitor not in _MONITORS:
        raise ValueError(f"monitor must be one of {_MONITORS}, got {bar.monitor!r}")

    if bar.is_up and not bar.H > bs.S0:
        raise ValueError("an up barrier requires H > S0")
    if not bar.is_up and not bar.H < bs.S0:
        raise ValueError("a down barrier requires H < S0")


def barrier_variants(
    bs: BSParams,
    H_up: Optional[float] = None,
    H_down: Optional[float] = None,
    K: Optional[float] = None,
    rebate: float = 0.0,
    monitor: str = "continuous",
) -> List[Tuple[str, BarrierSpec]]:
    """Enumerate all eight single-barrier variants around a base spot.

    Up barriers are placed at ``H_up`` (default ``1.1 * S0``) and down
    barriers at ``H_down`` (default ``0.9 * S0``); the strike defaults to
    ``S0``.  Returns a list of ``(label, BarrierSpec)`` pairs covering
    up/down x in/out x call/put.
    """
    H_up = 1.1 * bs.S0 if H_up is None else H_up
    H_down = 0.9 * bs.S0 if H_down is None else H_down
    K = bs.S0 if K is None else K

    variants: List[Tuple[str, BarrierSpec]] = []
    for direction in _DIRECTIONS:
        for knock in _KNOCK:
            for is_call in (True, False):
                H = H_up if direction == "up" else H_down
                spec = BarrierSpec(
                    barrier_type=f"{direction}-and-{knock}",
                    monitor=monitor,
                    H=H,
                    K=K,
                    is_call=is_call,
                    rebate=rebate,
                )
                label = f"{direction}-and-{knock} {'call' if is_call else 'put'}"
                variants.append((label, spec))
    return variants

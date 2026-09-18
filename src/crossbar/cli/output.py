"""Payload construction, JSON emission and table rendering."""

from __future__ import annotations

import argparse
import json
from typing import Callable, Optional

from ..params import BarrierSpec
from ..vol_surface import VolSurface

__all__ = ["_render_results"]


def _emit(payload: dict, as_json: bool, render: Callable[[dict], None]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        render(payload)


def _load_surface(path: Optional[str]) -> Optional[VolSurface]:
    """Load a :class:`VolSurface` from a JSON quotes file, or ``None``."""
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            quotes = json.load(fh)
    except OSError as exc:
        raise ValueError(f"could not read vol surface {path!r}: {exc}") from exc
    return VolSurface.from_quotes(quotes)


def _market_dict(a: argparse.Namespace, sigma: float) -> dict:
    return {
        "spot": float(a.spot),
        "rate": float(a.rate),
        "carry": float(a.div),
        "vol": float(sigma),
        "maturity": float(a.maturity),
    }


def _barrier_contract(bar: BarrierSpec) -> dict:
    return {
        "type": bar.barrier_type,
        "call": bar.is_call,
        "strike": float(bar.K),
        "barrier": float(bar.H),
        "monitor": bar.monitor,
        "rebate": float(bar.rebate),
    }


def _market_line(m: dict) -> str:
    return (
        f"S0={m['spot']:g}, r={m['rate']:g}, q={m['carry']:g}, "
        f"T={m['maturity']:g}"
    )


def _contract_line(c: dict) -> str:
    kind = "call" if c["call"] else "put"
    return (
        f"{c['type']} {kind}, {c['monitor']}, "
        f"K={c['strike']:g}, H={c['barrier']:g}, rebate={c['rebate']:g}"
    )


def _render_results(p: dict) -> None:
    print(f"{'contract':<9} : {_contract_line(p['contract'])}")
    print(f"{'market':<9} : {_market_line(p['market'])}")
    print(f"{'vol':<9} : {p['market']['vol']:.6f} ({p['vol_source']})")
    print(f"{'benchmark':<9} : vanilla = {p['vanilla_benchmark']:.6f}")
    if p.get("quotes"):
        print(f"{'quotes':<9} : {p['quotes']}")
    for label, key in (("mc", "mc_options"), ("pde", "pde_options")):
        options = p[key]
        body = ", ".join(f"{k}={v}" for k, v in options.items())
        print(f"{label:<9} : {body}")

    print()
    print(
        f"{'engine':<10}{'price':>14}{'delta':>14}{'gamma':>14}{'std_error':>14}"
    )
    for row in p["prices"]:
        price = "n/a" if row["price"] is None else f"{row['price']:.6f}"
        delta = "-" if row.get("delta") is None else f"{row['delta']:.6f}"
        gamma = "-" if row.get("gamma") is None else f"{row['gamma']:.6f}"
        se = "-" if row.get("std_error") is None else f"{row['std_error']:.6f}"
        print(
            f"{row['engine']:<10}{price:>14}{delta:>14}{gamma:>14}{se:>14}"
        )

"""Argument groups and resolution of flags / prompts into a request."""

from __future__ import annotations

import argparse

from .constants import (
    BARRIER_CODES,
    BARRIER_TYPES,
    MC_DEFAULTS,
    MONITOR_CODES,
    MONITORS,
    PDE_MESH,
)
from .prompts import _non_negative, _positive, _prompt

__all__ = ["_resolve"]


def _add_market_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-S", "--spot", type=float, default=None, help="spot price S0")
    p.add_argument(
        "-v",
        "--vol",
        default=None,
        help="constant volatility, or a path to a vol-surface JSON",
    )
    p.add_argument(
        "-T", "--maturity", type=float, default=None, help="time to maturity in years"
    )
    p.add_argument(
        "-r", "--rate", type=float, default=None, help="risk-free rate (default 0)"
    )
    p.add_argument(
        "-q", "--div", type=float, default=None, help="dividend / carry yield (default 0)"
    )


def _add_barrier_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-H", "--barrier", type=float, default=None, help="barrier level"
    )
    p.add_argument("-K", "--strike", type=float, default=None, help="strike (default: spot)")
    p.add_argument(
        "--type",
        dest="barrier_type",
        choices=(*BARRIER_CODES, *BARRIER_TYPES),
        default=None,
        help="barrier type: uo/ui/do/di (default uo)",
    )
    p.add_argument(
        "--monitor",
        choices=(*MONITOR_CODES, *MONITORS),
        default=None,
        help="monitoring: c/d (default c)",
    )
    p.add_argument("--rebate", type=float, default=None, help="cash rebate (default 0)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--call", dest="is_call", action="store_true", help="call (default)")
    g.add_argument("--put", dest="is_call", action="store_false", help="put")
    p.set_defaults(is_call=None)


def _add_mc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--paths", type=int, default=None, help="MC paths (default 200000)")
    p.add_argument("--steps", type=int, default=None, help="MC time steps (default 252)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default 0)")
    p.add_argument(
        "--no-control-variate",
        dest="control_variate",
        action="store_false",
        help="disable the Black-Scholes control variate",
    )
    p.set_defaults(control_variate=None)


def _value(a, attr, label, default, cast, choices, interactive, validate=None):
    """Return a resolved attribute, prompting for it when interactive."""
    current = getattr(a, attr)
    if current is not None:
        return current
    if interactive:
        return _prompt(label, default, cast, choices, validate=validate)
    if default is None:
        raise ValueError(f"missing required option --{attr.replace('_', '-')}")
    return default


def _resolve_choice(a, attr, label, default_code, codes, interactive):
    """Resolve a short-code choice, returning the full value.

    Flags may use either the short code (``uo``) or the full name
    (``up-and-out``); prompts offer the short codes.
    """
    current = getattr(a, attr)
    if current is not None:
        return codes.get(current, current)
    if interactive:
        return codes[_prompt(label, default_code, str, tuple(codes), labels=codes)]
    return codes[default_code]


def _resolve_vol(raw: str):
    """Split the vol input into ``(constant_sigma, surface_path)``.

    A decimal string is a constant vol; anything else is taken as the path
    to a screen-quote volatility-surface JSON, so exactly one is set.
    """
    try:
        sigma = float(raw)
    except ValueError:
        return None, raw
    if not sigma > 0:
        raise ValueError("vol must be strictly positive")
    return sigma, None


def _resolve(a: argparse.Namespace, interactive: bool) -> None:
    """Fill in every request field from flags and/or interactive prompts."""
    a.spot = _value(
        a, "spot", "Spot S0", None, float, None, interactive, _positive("S0")
    )
    a.sigma, a.surface = _resolve_vol(
        _value(a, "vol", "Vol (decimal or surface JSON)", None, str, None, interactive)
    )
    a.maturity = _value(
        a,
        "maturity",
        "Maturity T (years)",
        None,
        float,
        None,
        interactive,
        _positive("T"),
    )
    a.rate = _value(a, "rate", "Risk-free rate r", 0.0, float, None, interactive)
    a.div = _value(a, "div", "Carry/dividend yield q", 0.0, float, None, interactive)

    a.barrier_type = _resolve_choice(
        a, "barrier_type", "Barrier type", "uo", BARRIER_CODES, interactive
    )
    a.monitor = _resolve_choice(
        a, "monitor", "Monitoring", "c", MONITOR_CODES, interactive
    )
    a.barrier = _value(
        a, "barrier", "Barrier level H", None, float, None, interactive
    )
    a.rebate = _value(
        a, "rebate", "Rebate", 0.0, float, None, interactive, _non_negative("rebate")
    )

    if a.strike is None:
        if interactive:
            a.strike = _prompt(
                "Strike K (blank = spot)",
                None,
                float,
                allow_blank=True,
                validate=_positive("strike K"),
            )
            if a.strike is None:
                a.strike = a.spot
        else:
            a.strike = a.spot

    if a.is_call is None:
        if interactive:
            a.is_call = (
                _prompt(
                    "Option type",
                    "c",
                    str,
                    ("c", "p"),
                    labels={"c": "call", "p": "put"},
                )
                == "c"
            )
        else:
            a.is_call = True

    a.paths = MC_DEFAULTS["paths"] if a.paths is None else a.paths
    a.steps = MC_DEFAULTS["steps"] if a.steps is None else a.steps
    a.seed = MC_DEFAULTS["seed"] if a.seed is None else a.seed
    a.control_variate = (
        MC_DEFAULTS["control_variate"]
        if a.control_variate is None
        else a.control_variate
    )
    a.M = PDE_MESH["M"]
    a.N = PDE_MESH["N"]
    a.rannacher = PDE_MESH["rannacher"]

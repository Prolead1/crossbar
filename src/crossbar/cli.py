"""Command-line interface for :mod:`crossbar`.

The only command is ``crossbar price``.  Run it bare and it walks through
the trade interactively, one question at a time; pass the matching
options to run non-interactively (for scripts and tests).  Either way it
prices the barrier with the closed form, Monte Carlo and the PDE, prints
a vanilla benchmark, and ends with an aligned comparison.  The Monte
Carlo size (``--paths``/``--steps``/``--seed``/``--no-control-variate``)
comes from flags and defaults only; it is never prompted.::

    crossbar price

    crossbar price -S 1.10 -v 0.07 -T 0.5 -H 1.20 --type up-and-out
    crossbar price -S 1.10 -v quotes.json -T 0.5 -H 1.20 --type up-and-out

The volatility input (``-v``) is either a decimal constant or a path to a
screen-quote surface JSON; it is never both.

The PDE mesh (``M``/``N``/Rannacher pairs) is fixed at the library
defaults and is not a CLI input; call :func:`crossbar.pde.price_barrier_pde`
directly to change it.  Add ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Optional, Sequence

from . import __version__
from .analytic import price_barrier_closed_form, price_vanilla
from .greeks import mc_risk_profile, pde_risk_profile, risk_profile
from .monte_carlo import gen_normals, price_barrier_mc
from .params import BarrierSpec, BSParams, validate_inputs
from .vol_surface import VolSurface

BARRIER_TYPES = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
BARRIER_CODES = {
    "uo": "up-and-out",
    "ui": "up-and-in",
    "do": "down-and-out",
    "di": "down-and-in",
}
MONITORS = ("continuous", "discrete")
MONITOR_CODES = {"c": "continuous", "d": "discrete"}

#: Spot bump for the finite-difference deltas of the analytic and Monte
#: Carlo rows (the PDE row reads its delta off the solved surface).
DELTA_BUMP = 0.01

#: PDE mesh used by the CLI (library defaults).  Not exposed as flags so
#: the comparison across engines is reproducible; use
#: :func:`crossbar.pde.price_barrier_pde` directly for a custom mesh.
PDE_MESH = {"M": 500, "N": 500, "rannacher": 1}

#: Monte Carlo defaults, overridable with ``--paths``/``--steps``/``--seed``.
MC_DEFAULTS = {"paths": 200_000, "steps": 252, "seed": 0, "control_variate": True}


# --------------------------------------------------------------------------
# Interactive prompting
# --------------------------------------------------------------------------


def _positive(label: str) -> Callable[[object], None]:
    """Return a validator that requires a strictly positive value."""

    def _check(value) -> None:
        if not value > 0:
            raise ValueError(f"{label} must be strictly positive")

    return _check


def _non_negative(label: str) -> Callable[[object], None]:
    """Return a validator that requires a non-negative value."""

    def _check(value) -> None:
        if value < 0:
            raise ValueError(f"{label} must be non-negative")

    return _check


def _prompt(
    label: str,
    default,
    cast: Callable[[str], object],
    choices: Optional[Sequence[str]] = None,
    allow_blank: bool = False,
    validate: Optional[Callable[[object], None]] = None,
    labels: Optional[dict] = None,
):
    """Ask for one value and fail immediately when it is invalid.

    A blank answer takes ``default`` (or ``None`` when ``allow_blank``);
    ``choices`` constrains string answers and ``validate`` applies a
    domain check.  ``labels`` maps each accepted choice to the longer
    name shown in parentheses, and either form is accepted as input.
    Invalid input raises :class:`ValueError` rather than re-prompting, so
    a mistake aborts the run.
    """

    def _shown(choice):
        return labels.get(choice, choice) if labels else choice

    hint = ""
    if choices is not None:
        hint += " (" + "/".join(_shown(c) for c in choices) + ")"
    if default is not None:
        hint += f" [{default}]"

    raw = input(f"{label}{hint}: ").strip()
    if not raw:
        if allow_blank:
            return None
        if default is None:
            raise ValueError(f"{label}: a value is required")
        value = default
    elif choices is not None:
        if raw in choices:
            chosen = raw
        else:
            matches = [c for c in choices if _shown(c) == raw]
            if not matches:
                raise ValueError(
                    f"{label}: choose one of "
                    + ", ".join(_shown(c) for c in choices)
                )
            chosen = matches[0]
        value = cast(chosen)
    else:
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: could not parse {raw!r}") from None

    if validate is not None:
        validate(value)
    return value


# --------------------------------------------------------------------------
# Argument groups
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Resolving flags and prompts into a complete request
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


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
        options = p.get(key)
        if options:
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
    for row in p["prices"]:
        if row.get("note"):
            print(f"note: {row['engine']} skipped ({row['note']})")


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def _bar_from_args(a: argparse.Namespace) -> BarrierSpec:
    return BarrierSpec(
        barrier_type=a.barrier_type,
        monitor=a.monitor,
        H=a.barrier,
        K=a.strike,
        is_call=a.is_call,
        rebate=a.rebate,
    )


def _run_engines(bs: BSParams, bar: BarrierSpec, a, surface) -> list:
    """Price and greeks with all three engines, skipping inapplicable ones."""
    spot = float(bs.S0)
    results = []

    prices, deltas, gammas = risk_profile(
        price_barrier_closed_form, bs, bar, [spot], bump=DELTA_BUMP
    )
    results.append(
        {
            "engine": "analytic",
            "price": float(prices[0]),
            "delta": float(deltas[0]),
            "gamma": float(gammas[0]),
        }
    )

    if surface is None:
        # Draw the paths once and share them between the price and the
        # delta bumps.  ``mc_risk_profile`` reuses a single simulation for
        # all three bumps, so the expensive path generation runs once
        # instead of four times.
        Z = gen_normals(a.paths, a.steps, seed=a.seed)
        price, se = price_barrier_mc(
            bs,
            bar,
            n_paths=a.paths,
            n_steps=a.steps,
            seed=a.seed,
            control_variate=a.control_variate,
            Z=Z,
        )
        _, deltas, gammas = mc_risk_profile(
            bs,
            bar,
            [spot],
            bump=DELTA_BUMP,
            seed=a.seed,
            control_variate=a.control_variate,
            Z=Z,
        )
        results.append(
            {
                "engine": "mc",
                "price": float(price),
                "delta": float(deltas[0]),
                "gamma": float(gammas[0]),
                "std_error": float(se),
            }
        )
    else:
        results.append(
            {
                "engine": "mc",
                "price": None,
                "delta": None,
                "gamma": None,
                "note": "Monte Carlo does not use a surface yet",
            }
        )

    prices, deltas, gammas = pde_risk_profile(
        bs,
        bar,
        [spot],
        M=a.M,
        N=a.N,
        rannacher_pairs=a.rannacher,
        surface=surface,
    )
    results.append(
        {
            "engine": "pde",
            "price": float(prices[0]),
            "delta": float(deltas[0]),
            "gamma": float(gammas[0]),
        }
    )
    return results


def _cmd_price(a: argparse.Namespace) -> None:
    surface = _load_surface(a.surface)
    if surface is None:
        sigma = a.sigma
        vol_source = "constant"
    else:
        sigma = float(surface.interp_sigma(a.maturity, 0.0))
        vol_source = "ATM from surface"
    bs = BSParams(S0=a.spot, r=a.rate, q=a.div, sigma=sigma, T=a.maturity)
    bar = _bar_from_args(a)
    validate_inputs(bs, bar)

    vanilla = float(
        price_vanilla(bs.S0, bar.K, bs.r, bs.q, bs.sigma, bs.T, is_call=bar.is_call)
    )
    payload = {
        "contract": _barrier_contract(bar),
        "market": _market_dict(a, sigma),
        "vol_source": vol_source,
        "vanilla_benchmark": vanilla,
        "quotes": a.surface,
        "pde_options": dict(PDE_MESH),
        "prices": _run_engines(bs, bar, a, surface),
    }
    if surface is None:
        payload["mc_options"] = {
            "paths": a.paths,
            "steps": a.steps,
            "seed": a.seed,
            "control_variate": a.control_variate,
        }

    _emit(payload, a.json, _render_results)


# --------------------------------------------------------------------------
# Parser and entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``crossbar`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="crossbar",
        description="Single-barrier and vanilla option pricing.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "price", help="price a vanilla or single-barrier option (interactive by default)"
    )
    _add_market_args(p)
    _add_barrier_args(p)
    _add_mc_args(p)
    p.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; error on missing required options",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.set_defaults(func=_cmd_price)
    return parser


def main(
    argv: Optional[Sequence[str]] = None, *, interactive: Optional[bool] = None
) -> int:
    """Run the CLI; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if interactive is None:
        interactive = not args.non_interactive and sys.stdin.isatty()
    try:
        _resolve(args, interactive)
        args.func(args)
    except (ValueError, NotImplementedError) as exc:
        print(f"crossbar: error: {exc}", file=sys.stderr)
        return 2
    return 0

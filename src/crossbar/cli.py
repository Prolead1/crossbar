"""Command-line interface for :mod:`crossbar`.

The CLI is a thin, scriptable front end over the package's public
functions.  Each subcommand builds the same :class:`~crossbar.params.BSParams`
and :class:`~crossbar.params.BarrierSpec` objects the library uses and
prints either a human-readable table or ``--json`` output.

Examples
--------
::

    crossbar price    -S 1.10 -v 0.07 -T 0.5 -K 1.10
    crossbar price    -S 1.10 -v 0.07 -T 0.5 -H 1.20 --type up-and-out
    crossbar variants -S 1.10 -v 0.07 -T 0.5 --engine pde
    crossbar greeks   -S 1.10 -v 0.07 -T 0.5 -H 1.20 --engine pde
    crossbar surface  --delta 0.25 --maturity 0.5
    crossbar plot     -S 1.10 --out vol.png

Volumes are decimal (``0.07`` is 7 vol points), maturities are in years,
and every command accepts ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Optional, Sequence

import numpy as np

from . import __version__
from .analytic import price_barrier_closed_form, price_vanilla
from .greeks import pde_risk_profile, risk_profile
from .monte_carlo import price_barrier_mc
from .params import BarrierSpec, BSParams, barrier_variants, validate_inputs
from .pde import price_barrier_pde
from .vol_surface import (
    EXAMPLE_QUOTES,
    VolSurface,
    plot_vol_surface,
    sigma_for_strike,
    strike_from_delta,
)

BARRIER_TYPES = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
ENGINES = ("analytic", "mc", "pde")


# --------------------------------------------------------------------------
# Argument groups shared by several subcommands
# --------------------------------------------------------------------------


def _add_market_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-S", "--spot", type=float, required=True, help="spot price S0")
    p.add_argument(
        "-v", "--vol", type=float, required=True, help="flat Black-Scholes volatility"
    )
    p.add_argument(
        "-T", "--maturity", type=float, required=True, help="time to maturity in years"
    )
    p.add_argument(
        "-r",
        "--rate",
        type=float,
        default=0.0,
        help="continuously compounded risk-free rate (default 0)",
    )
    p.add_argument(
        "-q",
        "--div",
        type=float,
        default=0.0,
        help="continuous dividend / carry yield (default 0)",
    )


def _add_fx_args(p: argparse.ArgumentParser) -> None:
    """Spot/rate/carry only (for the surface helpers that ignore vol)."""
    p.add_argument("-S", "--spot", type=float, default=1.0, help="spot price (default 1)")
    p.add_argument(
        "-r", "--rate", type=float, default=0.0, help="risk-free rate (default 0)"
    )
    p.add_argument(
        "-q", "--div", type=float, default=0.0, help="carry yield (default 0)"
    )


def _add_call_put(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--call", dest="is_call", action="store_true", help="call (default)")
    g.add_argument("--put", dest="is_call", action="store_false", help="put")
    p.set_defaults(is_call=True)


def _add_barrier_args(
    p: argparse.ArgumentParser, *, require_barrier: bool = True
) -> None:
    p.add_argument(
        "-H",
        "--barrier",
        type=float,
        required=require_barrier,
        default=None,
        help="barrier level (omit to price a vanilla European)",
    )
    p.add_argument(
        "-K", "--strike", type=float, default=None, help="strike (default: spot)"
    )
    p.add_argument(
        "--type",
        dest="barrier_type",
        choices=BARRIER_TYPES,
        default="up-and-out",
        help="barrier type (default up-and-out)",
    )
    p.add_argument(
        "--monitor",
        choices=("continuous", "discrete"),
        default="continuous",
        help="barrier monitoring convention",
    )
    p.add_argument(
        "--rebate", type=float, default=0.0, help="cash rebate (default 0)"
    )
    _add_call_put(p)


def _add_engine_arg(p: argparse.ArgumentParser, default: str = "pde") -> None:
    p.add_argument(
        "--engine",
        choices=ENGINES,
        default=default,
        help=f"pricing engine (default {default})",
    )


def _add_mc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--paths", type=int, default=200_000, help="MC paths (default 200000)")
    p.add_argument("--steps", type=int, default=252, help="MC time steps (default 252)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument(
        "--no-control-variate",
        dest="control_variate",
        action="store_false",
        help="disable the Black-Scholes control variate",
    )
    p.set_defaults(control_variate=True)


def _add_pde_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--M", type=int, default=500, help="space steps (default 500)")
    p.add_argument("--N", type=int, default=500, help="time steps (default 500)")
    p.add_argument(
        "--rannacher", type=int, default=1, help="Rannacher pairs (default 1)"
    )


def _add_surface_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--surface",
        metavar="FILE",
        default=None,
        help="vol-surface quotes JSON used for the PDE local-vol grid",
    )


def _add_json(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")


# --------------------------------------------------------------------------
# Object construction and output helpers
# --------------------------------------------------------------------------


def _bs_from_args(a: argparse.Namespace) -> BSParams:
    return BSParams(S0=a.spot, r=a.rate, q=a.div, sigma=a.vol, T=a.maturity)


def _bar_from_args(a: argparse.Namespace) -> BarrierSpec:
    strike = a.spot if a.strike is None else a.strike
    return BarrierSpec(
        barrier_type=a.barrier_type,
        monitor=a.monitor,
        H=a.barrier,
        K=strike,
        is_call=a.is_call,
        rebate=a.rebate,
    )


def _load_surface(path: Optional[str]) -> Optional[VolSurface]:
    """Load a :class:`VolSurface` from a JSON quotes file, or ``None``."""
    if path is None:
        return None
    with open(path, encoding="utf-8") as fh:
        quotes = json.load(fh)
    return VolSurface.from_quotes(quotes)


def _example_surface() -> VolSurface:
    return VolSurface.from_quotes(EXAMPLE_QUOTES)


def _emit(payload: dict, as_json: bool, render: Callable[[dict], None]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        render(payload)


def _parse_spots(spec: Optional[str], spot: float) -> np.ndarray:
    if spec is None:
        return np.linspace(0.9 * spot, 1.1 * spot, 5)
    return np.array([float(x) for x in spec.split(",")], dtype=float)


def _mc_kwargs(a: argparse.Namespace) -> dict:
    return {
        "n_paths": a.paths,
        "n_steps": a.steps,
        "seed": a.seed,
        "control_variate": a.control_variate,
    }


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------


def _cmd_price(a: argparse.Namespace) -> None:
    if a.barrier is None:
        strike = a.spot if a.strike is None else a.strike
        payload = {
            "instrument": "vanilla",
            "price": float(
                price_vanilla(
                    a.spot, strike, a.rate, a.div, a.vol, a.maturity, is_call=a.is_call
                )
            ),
            "is_call": a.is_call,
            "strike": strike,
        }
    else:
        bs = _bs_from_args(a)
        bar = _bar_from_args(a)
        validate_inputs(bs, bar)

        if a.engine == "analytic":
            if bar.monitor != "continuous":
                raise ValueError(
                    "the analytic engine only supports continuous monitoring"
                )
            payload = {
                "instrument": "barrier",
                "engine": "analytic",
                "price": price_barrier_closed_form(bs, bar),
            }
        elif a.engine == "mc":
            price, se = price_barrier_mc(bs, bar, **_mc_kwargs(a))
            payload = {
                "instrument": "barrier",
                "engine": "mc",
                "price": price,
                "std_error": se,
            }
        else:
            surface = _load_surface(a.surface)
            price = price_barrier_pde(
                bs, bar, M=a.M, N=a.N, rannacher_pairs=a.rannacher, surface=surface
            )
            payload = {
                "instrument": "barrier",
                "engine": "pde",
                "price": price,
                "surface": a.surface,
            }

    def render(p: dict) -> None:
        if p["instrument"] == "vanilla":
            kind = "call" if p["is_call"] else "put"
            print(f"vanilla {kind}: strike={p['strike']:.6g} price={p['price']:.6f}")
        elif p["engine"] == "mc":
            print(f"price (mc)  = {p['price']:.6f} +/- {p['std_error']:.6f}")
        else:
            print(f"price ({p['engine']}) = {p['price']:.6f}")

    _emit(payload, a.json, render)


def _cmd_variants(a: argparse.Namespace) -> None:
    bs = _bs_from_args(a)
    if a.engine == "analytic" and a.monitor != "continuous":
        raise ValueError("the analytic engine only supports continuous monitoring")
    surface = _load_surface(a.surface) if a.engine == "pde" else None

    rows = []
    for label, spec in barrier_variants(
        bs,
        H_up=a.up_barrier,
        H_down=a.down_barrier,
        K=a.strike,
        rebate=a.rebate,
        monitor=a.monitor,
    ):
        validate_inputs(bs, spec)
        if a.engine == "analytic":
            price = price_barrier_closed_form(bs, spec)
        elif a.engine == "mc":
            price, _ = price_barrier_mc(bs, spec, **_mc_kwargs(a))
        else:
            price = price_barrier_pde(
                bs, spec, M=a.M, N=a.N, rannacher_pairs=a.rannacher, surface=surface
            )
        rows.append({"variant": label, "price": float(price)})

    payload = {"engine": a.engine, "variants": rows}

    def render(p: dict) -> None:
        print(f"{'variant':<24}{'price':>12}")
        for row in p["variants"]:
            print(f"{row['variant']:<24}{row['price']:>12.6f}")

    _emit(payload, a.json, render)


def _cmd_greeks(a: argparse.Namespace) -> None:
    bs = _bs_from_args(a)
    bar = _bar_from_args(a)
    validate_inputs(bs, bar)
    spots = _parse_spots(a.spots, a.spot)

    if a.engine == "pde":
        prices, deltas, gammas = pde_risk_profile(
            bs, bar, spots, M=a.M, N=a.N, rannacher_pairs=a.rannacher
        )
    else:
        prices, deltas, gammas = risk_profile(
            price_barrier_mc,
            bs,
            bar,
            spots,
            bump=a.bump,
            pricer_kwargs=_mc_kwargs(a),
        )

    rows = [
        {
            "spot": float(s),
            "price": float(p),
            "delta": float(d),
            "gamma": float(g),
        }
        for s, p, d, g in zip(spots, prices, deltas, gammas)
    ]
    payload = {"engine": a.engine, "profile": rows}

    def render(p: dict) -> None:
        print(f"{'spot':>10}{'price':>14}{'delta':>14}{'gamma':>14}")
        for row in p["profile"]:
            print(
                f"{row['spot']:>10.4f}{row['price']:>14.6f}"
                f"{row['delta']:>14.6f}{row['gamma']:>14.6f}"
            )

    _emit(payload, a.json, render)


def _cmd_surface(a: argparse.Namespace) -> None:
    surface = _load_surface(a.quotes)
    if surface is None:
        surface = _example_surface()

    if a.strike is not None or a.delta is not None:
        if a.maturity is None:
            raise ValueError("--maturity is required with --strike or --delta")
        bs = BSParams(S0=a.spot, r=a.rate, q=a.div, sigma=0.0, T=a.maturity)
        if a.strike is not None:
            payload = {
                "mode": "sigma",
                "strike": a.strike,
                "maturity": a.maturity,
                "sigma": float(sigma_for_strike(surface, bs, a.strike, a.maturity)),
            }

            def render(p: dict) -> None:
                print(
                    f"sigma(K={p['strike']:.6g}, T={p['maturity']:.6g}) = {p['sigma']:.6f}"
                )

        else:
            payload = {
                "mode": "strike",
                "delta": a.delta,
                "maturity": a.maturity,
                "strike": strike_from_delta(surface, bs, a.delta, a.maturity),
            }

            def render(p: dict) -> None:
                print(
                    f"strike(delta={p['delta']:+.3f}, T={p['maturity']:.6g}) "
                    f"= {p['strike']:.6f}"
                )

    else:
        rows = [
            {
                "maturity": T,
                "sigma_10p": smile.sigma_10p,
                "sigma_25p": smile.sigma_25p,
                "atm": smile.atm,
                "sigma_25c": smile.sigma_25c,
                "sigma_10c": smile.sigma_10c,
            }
            for T, smile in surface.smiles.items()
        ]
        payload = {"mode": "term-structure", "maturities": rows}

        def render(p: dict) -> None:
            header = f"{'T':>8}{'10p':>10}{'25p':>10}{'atm':>10}{'25c':>10}{'10c':>10}"
            print(header)
            for row in p["maturities"]:
                print(
                    f"{row['maturity']:>8.4f}{row['sigma_10p']:>10.5f}"
                    f"{row['sigma_25p']:>10.5f}{row['atm']:>10.5f}"
                    f"{row['sigma_25c']:>10.5f}{row['sigma_10c']:>10.5f}"
                )

    _emit(payload, a.json, render)


def _cmd_plot(a: argparse.Namespace) -> None:
    surface = _load_surface(a.quotes)
    if surface is None:
        surface = _example_surface()
    bs = BSParams(S0=a.spot, r=a.rate, q=a.div, sigma=0.0, T=1.0)
    strikes = (
        None if a.strikes is None else np.linspace(0.7 * a.spot, 1.3 * a.spot, a.strikes)
    )
    maturities = (
        None
        if a.maturities is None
        else np.linspace(surface.maturities[0], surface.maturities[-1], a.maturities)
    )
    ax = plot_vol_surface(surface, bs, strikes=strikes, maturities=maturities)
    ax.figure.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {a.out}")


# --------------------------------------------------------------------------
# Parser and entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``crossbar`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="crossbar",
        description="Single-barrier option pricing engine.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "price", help="price a European vanilla or a single-barrier option"
    )
    _add_market_args(p)
    _add_barrier_args(p, require_barrier=False)
    _add_engine_arg(p)
    _add_mc_args(p)
    _add_pde_args(p)
    _add_surface_arg(p)
    _add_json(p)
    p.set_defaults(func=_cmd_price)

    p = sub.add_parser("variants", help="price all eight single-barrier variants")
    _add_market_args(p)
    p.add_argument("--up-barrier", type=float, default=None, help="up barrier (default 1.1*S0)")
    p.add_argument(
        "--down-barrier", type=float, default=None, help="down barrier (default 0.9*S0)"
    )
    p.add_argument("-K", "--strike", type=float, default=None, help="strike (default: spot)")
    p.add_argument("--rebate", type=float, default=0.0, help="cash rebate (default 0)")
    p.add_argument(
        "--monitor",
        choices=("continuous", "discrete"),
        default="continuous",
        help="barrier monitoring convention",
    )
    _add_engine_arg(p)
    _add_mc_args(p)
    _add_pde_args(p)
    _add_surface_arg(p)
    _add_json(p)
    p.set_defaults(func=_cmd_variants)

    p = sub.add_parser("greeks", help="price/delta/gamma profile for one contract")
    _add_market_args(p)
    _add_barrier_args(p)
    _add_engine_arg(p)
    _add_mc_args(p)
    _add_pde_args(p)
    p.add_argument(
        "--spots",
        default=None,
        help="comma-separated spot grid (default: 5 points around the spot)",
    )
    p.add_argument("--bump", type=float, default=0.01, help="MC bump size (default 0.01)")
    _add_json(p)
    p.set_defaults(func=_cmd_greeks)

    p = sub.add_parser("surface", help="inspect the delta-quoted volatility surface")
    _add_fx_args(p)
    p.add_argument(
        "--quotes",
        metavar="FILE",
        default=None,
        help="quotes JSON (default: built-in EURUSD example)",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--strike", type=float, default=None, help="strike of a vol lookup")
    g.add_argument(
        "--delta", type=float, default=None, help="delta of a strike lookup"
    )
    p.add_argument("--maturity", type=float, default=None, help="maturity in years")
    _add_json(p)
    p.set_defaults(func=_cmd_surface)

    p = sub.add_parser("plot", help="render the volatility surface to an image")
    _add_fx_args(p)
    p.add_argument(
        "--quotes",
        metavar="FILE",
        default=None,
        help="quotes JSON (default: built-in EURUSD example)",
    )
    p.add_argument("--out", required=True, help="output image path")
    p.add_argument("--strikes", type=int, default=None, help="number of strike samples")
    p.add_argument(
        "--maturities", type=int, default=None, help="number of maturity samples"
    )
    p.add_argument("--dpi", type=int, default=120, help="output resolution (default 120)")
    p.set_defaults(func=_cmd_plot)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, NotImplementedError) as exc:
        print(f"crossbar: error: {exc}", file=sys.stderr)
        return 2
    return 0

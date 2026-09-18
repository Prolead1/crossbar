"""Parser construction and the ``main`` entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .. import __version__
from .arguments import (
    _add_barrier_args,
    _add_market_args,
    _add_mc_args,
    _resolve,
)
from .commands import _cmd_price

__all__ = ["build_parser", "main"]


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

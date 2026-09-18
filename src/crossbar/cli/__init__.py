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

The implementation is split into :mod:`~crossbar.cli.arguments`,
:mod:`~crossbar.cli.commands` and :mod:`~crossbar.cli.output`, with
:mod:`~crossbar.cli.main` holding the parser and entry point; the public
names are re-exported here.
"""

from .constants import MC_DEFAULTS, PDE_MESH
from .main import build_parser, main
from .prompts import _non_negative, _positive, _prompt

__all__ = [
    "build_parser",
    "main",
]

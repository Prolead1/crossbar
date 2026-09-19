"""CLI constants: choice codes, engine defaults and the fixed PDE mesh."""

from __future__ import annotations

BARRIER_TYPES = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")
BARRIER_CODES = {
    "uo": "up-and-out",
    "ui": "up-and-in",
    "do": "down-and-out",
    "di": "down-and-in",
}
MONITORS = ("continuous", "discrete")
MONITOR_CODES = {"c": "continuous", "d": "discrete"}

#: Spot bump for the finite-difference Greeks of the analytic row (the
#: closed-form price is smooth, so a small absolute bump is fine; the PDE
#: row reads its delta/gamma off the solved surface).  The Monte Carlo row
#: instead scales its stencil with the spot -- see
#: :func:`crossbar.greeks.mc_greek_bumps` -- because a second difference of
#: a barrier payoff is dominated by the paths whose crossing moves across
#: the bump.
DELTA_BUMP = 0.01

#: PDE mesh used by the CLI (library defaults).  Not exposed as flags so
#: the comparison across engines is reproducible; use
#: :func:`crossbar.pde.price_barrier_pde` directly for a custom mesh.
PDE_MESH = {"M": 500, "N": 500, "rannacher": 1}

#: Monte Carlo defaults, overridable with ``--paths``/``--steps``/``--seed``.
MC_DEFAULTS = {"paths": 200_000, "steps": 252, "seed": 0, "control_variate": True}

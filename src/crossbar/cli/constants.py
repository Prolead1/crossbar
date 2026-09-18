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

#: Spot bump for the finite-difference deltas of the analytic and Monte
#: Carlo rows (the PDE row reads its delta off the solved surface).
DELTA_BUMP = 0.01

#: PDE mesh used by the CLI (library defaults).  Not exposed as flags so
#: the comparison across engines is reproducible; use
#: :func:`crossbar.pde.price_barrier_pde` directly for a custom mesh.
PDE_MESH = {"M": 500, "N": 500, "rannacher": 1}

#: Monte Carlo defaults, overridable with ``--paths``/``--steps``/``--seed``.
MC_DEFAULTS = {"paths": 200_000, "steps": 252, "seed": 0, "control_variate": True}

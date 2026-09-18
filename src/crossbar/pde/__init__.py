"""Crank-Nicolson finite-difference pricing with Rannacher time stepping.

The package is split into small pieces:

* :mod:`crossbar.pde.tridiagonal` -- the Thomas solvers.
* :mod:`crossbar.pde.grid` -- grid construction and boundary conditions.
* :mod:`crossbar.pde.scheme` -- stencil assembly and time stepping.
* :mod:`crossbar.pde.pricing` -- backward induction and barrier pricing.

The public names are re-exported here so the historical
``from crossbar.pde import price_barrier_pde`` imports keep working.
"""

from .grid import (
    boundary_values,
    build_grid,
    enforce_barrier_dirichlet,
)
from .pricing import (
    _knock_out_surface,
    _vanilla_leg_vol,
    pde_knock_out,
    pde_surface,
    price_barrier_pde,
)
from .scheme import (
    _operator_plan,
    _step,
    cn_step,
    rannacher_pair,
    theta_step,
)
from .tridiagonal import (
    thomas_factor,
    thomas_solve,
    thomas_solve_factored,
)

__all__ = [
    "thomas_solve",
    "thomas_factor",
    "thomas_solve_factored",
    "build_grid",
    "boundary_values",
    "enforce_barrier_dirichlet",
    "theta_step",
    "cn_step",
    "rannacher_pair",
    "pde_knock_out",
    "pde_surface",
    "price_barrier_pde",
]

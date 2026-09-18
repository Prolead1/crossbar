"""Single-barrier option pricing engine.

Crossbar prices discrete and continuous knock-in / knock-out barrier
options with two independent engines:

* Monte Carlo simulation with Latin-hypercube stratification, antithetic
  variates, moment matching, a Black-Scholes control variate and
  Brownian-bridge barrier-crossing correction.
* A Crank-Nicolson finite-difference PDE solver with Rannacher time
  stepping for stable early-time convergence.

Both engines are validated against Reiner-Rubinstein / Haug closed-form
prices, and bump-and-revalue delta and gamma profiles are provided across
all eight single-barrier variants.
"""

from .analytic import (
    price_barrier_closed_form,
    price_vanilla,
    vanilla_payoff,
)
from .greeks import (
    greeks_by_variant,
    greeks_by_variant_pde,
    pde_risk_profile,
    risk_profile,
    surface_risk_profile,
)
from .monte_carlo import (
    barrier_hits,
    gen_normals,
    monte_carlo_paths,
    price_barrier,
    price_barrier_cv,
    price_barrier_mc,
    step_bridge_cross_prob,
)
from .params import (
    BarrierSpec,
    BSParams,
    barrier_variants,
    validate_inputs,
)
from .pde import (
    build_grid,
    pde_knock_out,
    pde_surface,
    price_barrier_pde,
    thomas_solve,
)

__version__ = "0.1.0"

__all__ = [
    "BSParams",
    "BarrierSpec",
    "validate_inputs",
    "barrier_variants",
    "vanilla_payoff",
    "price_vanilla",
    "price_barrier_closed_form",
    "gen_normals",
    "monte_carlo_paths",
    "step_bridge_cross_prob",
    "barrier_hits",
    "price_barrier",
    "price_barrier_cv",
    "price_barrier_mc",
    "build_grid",
    "thomas_solve",
    "pde_knock_out",
    "pde_surface",
    "price_barrier_pde",
    "risk_profile",
    "greeks_by_variant",
    "surface_risk_profile",
    "pde_risk_profile",
    "greeks_by_variant_pde",
]

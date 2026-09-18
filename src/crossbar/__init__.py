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
all eight single-barrier variants.  A screen-quote volatility surface
(:mod:`crossbar.vol_surface`) can supply a time-varying ``sigma(t)`` to
the Monte Carlo paths and a Dupire local-vol ``sigma(S, t)`` grid to the
PDE solver.
"""

from .analytic import (
    price_barrier_closed_form,
    price_vanilla,
    vanilla_payoff,
)
from .greeks import (
    greeks_by_variant,
    greeks_by_variant_pde,
    mc_risk_profile,
    pde_risk_profile,
    risk_profile,
    surface_risk_profile,
)
from .monte_carlo import (
    barrier_hits,
    gen_normals,
    log_price_increments,
    monte_carlo_paths,
    paths_from_log_increments,
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
from .vol_surface import (
    EXAMPLE_QUOTES,
    LocalVolSurface,
    VolSmile,
    VolSurface,
    build_sigma_grid,
    check_arbitrage,
    local_vol_grid,
    local_volatility,
    mid_vol,
    plot_vol_surface,
    sigma_for_strike,
    stepwise_sigmas_for_strike,
    stepwise_sigmas_from_surface,
    strike_from_delta,
    tenor_to_years,
    total_variance_grid,
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
    "log_price_increments",
    "paths_from_log_increments",
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
    "mc_risk_profile",
    "greeks_by_variant",
    "surface_risk_profile",
    "pde_risk_profile",
    "greeks_by_variant_pde",
    "VolSmile",
    "VolSurface",
    "tenor_to_years",
    "mid_vol",
    "strike_from_delta",
    "sigma_for_strike",
    "stepwise_sigmas_from_surface",
    "stepwise_sigmas_for_strike",
    "LocalVolSurface",
    "local_volatility",
    "local_vol_grid",
    "check_arbitrage",
    "build_sigma_grid",
    "total_variance_grid",
    "plot_vol_surface",
    "EXAMPLE_QUOTES",
]

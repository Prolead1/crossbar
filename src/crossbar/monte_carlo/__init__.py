"""Monte Carlo pricing of single-barrier options.

Variance reduction combines Latin-hypercube stratification, antithetic
variates and moment matching, with a Black-Scholes control variate on the
discounted European payoff.  When the contract is continuously monitored
the discrete sampling bias is corrected with Brownian-bridge
barrier-crossing probabilities.

The package is split into :mod:`~crossbar.monte_carlo.simulate`,
:mod:`~crossbar.monte_carlo.barrier` and
:mod:`~crossbar.monte_carlo.pricing`; the public names are re-exported
here.
"""

from .barrier import barrier_hits, step_bridge_cross_prob
from .pricing import (
    _payoff_samples,
    price_barrier,
    price_barrier_cv,
    price_barrier_mc,
)
from .simulate import (
    gen_normals,
    log_price_increments,
    monte_carlo_paths,
    paths_from_log_increments,
)

__all__ = [
    "gen_normals",
    "log_price_increments",
    "paths_from_log_increments",
    "monte_carlo_paths",
    "step_bridge_cross_prob",
    "barrier_hits",
    "price_barrier",
    "price_barrier_cv",
    "price_barrier_mc",
]

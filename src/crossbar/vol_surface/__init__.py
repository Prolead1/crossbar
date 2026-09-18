"""FX volatility surface built from screen-style delta quotes.

The package turns a table of Bloomberg-style bid/ask volatility quotes
into a coherent smile-and-term-structure model and feeds it into the two
pricing engines:

* :class:`~crossbar.vol_surface.surface.VolSmile` /
  :class:`~crossbar.vol_surface.surface.VolSurface` hold the quoted
  pillars and interpolate them in delta and total variance.
* :mod:`crossbar.vol_surface.strike` converts between signed deltas and
  strikes and reads the implied vol at a fixed strike.
* :mod:`crossbar.vol_surface.term_structure` turns the implied term
  structure into piecewise-constant *instantaneous forward* vols for the
  Monte Carlo engine.
* :mod:`crossbar.vol_surface.local_vol` provides the Dupire
  local-volatility surface for the finite-difference scheme plus the
  arbitrage screen.

See the submodule docstrings for the convention details; the public names
are re-exported here so ``from crossbar.vol_surface import VolSurface``
keeps working.
"""

from ._constants import (
    EXAMPLE_QUOTES,
    QUOTE_FIELDS,
    SMILE_DELTAS,
    TENORS,
)
from .local_vol import (
    LocalVolSurface,
    build_sigma_grid,
    check_arbitrage,
    local_vol_grid,
    local_volatility,
)
from .plotting import (
    plot_vol_surface,
    total_variance_grid,
)
from .strike import (
    _atm_strike,
    _pillar_strikes,
    _strike_for_delta,
    _strike_from_bs_delta,
    _strike_vols_per_maturity,
    bs_delta,
    sigma_for_strike,
    strike_from_delta,
)
from .surface import (
    VolSmile,
    VolSurface,
    _interp_total_variance,
    mid_vol,
    tenor_to_years,
)
from .term_structure import (
    _forward_vols,
    stepwise_sigmas_for_strike,
    stepwise_sigmas_from_surface,
)

__all__ = [
    "TENORS",
    "SMILE_DELTAS",
    "QUOTE_FIELDS",
    "EXAMPLE_QUOTES",
    "tenor_to_years",
    "mid_vol",
    "VolSmile",
    "VolSurface",
    "bs_delta",
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
]

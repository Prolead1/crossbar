"""Delta and gamma profiles for barrier options.

:func:`~crossbar.greeks.bump.risk_profile` is the engine-agnostic
bump-and-revalue profile used by the Monte Carlo pricer.
:func:`~crossbar.greeks.mc.mc_risk_profile` is its Monte Carlo
counterpart: it draws the paths once and reuses the same simulation for
every bump, so the common random numbers come for free.
:func:`~crossbar.greeks.pde.pde_risk_profile` instead solves the PDE
value surface once and reads price, delta and gamma at every spot from
it -- far cheaper, and much smoother than bumping the interpolant.

The package is split into :mod:`~crossbar.greeks.bump`,
:mod:`~crossbar.greeks.mc`, :mod:`~crossbar.greeks.surface` and
:mod:`~crossbar.greeks.pde`; the public names are re-exported here.
"""

from ._common import _extract_price
from .bump import greeks_by_variant, risk_profile
from .mc import mc_risk_profile
from .pde import greeks_by_variant_pde, pde_risk_profile
from .surface import (
    _surface_derivatives,
    _vanilla_greeks,
    surface_risk_profile,
)

__all__ = [
    "risk_profile",
    "greeks_by_variant",
    "mc_risk_profile",
    "surface_risk_profile",
    "pde_risk_profile",
    "greeks_by_variant_pde",
]

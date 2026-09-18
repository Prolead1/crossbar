"""Cross-validation of the Monte Carlo and PDE engines.

Both engines are checked against the Reiner-Rubinstein / Haug closed form
for continuously monitored barriers, including rebates (which exercise the
closed-form rebate corrections and the PDE knock-in parity).  Discrete
monitoring is checked against the Monte Carlo engine, the PDE being only
first-order accurate in space because the value jumps at each observation
date.
"""

import numpy as np
import pytest

from crossbar import (
    BarrierSpec,
    BSParams,
    price_barrier_closed_form,
    price_barrier_mc,
    price_barrier_pde,
    price_vanilla,
)
from crossbar.analytic import barrier_rebate_terms

BS = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0)
H_UP = 120.0
H_DOWN = 85.0
K = 100.0


def _continuous_variants(rebate=0.0):
    specs = []
    for H, direction in ((H_UP, "up"), (H_DOWN, "down")):
        for knock in ("out", "in"):
            for is_call in (True, False):
                specs.append(
                    BarrierSpec(
                        f"{direction}-and-{knock}", "continuous", H, K, is_call, rebate=rebate
                    )
                )
    return specs


# --------------------------------------------------------------------------
# Continuous monitoring: engines vs closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bar", _continuous_variants(), ids=str)
def test_pde_matches_closed_form(bar):
    cf = price_barrier_closed_form(BS, bar)
    pde = price_barrier_pde(BS, bar, M=500, N=500)
    assert pde == pytest.approx(cf, abs=5e-3)


@pytest.mark.parametrize("bar", _continuous_variants(), ids=str)
def test_mc_matches_closed_form(bar):
    cf = price_barrier_closed_form(BS, bar)
    mc, se = price_barrier_mc(BS, bar, n_paths=100_000, n_steps=150, seed=1)
    assert abs(mc - cf) < 4.0 * se + 5e-3


# --------------------------------------------------------------------------
# Rebates (regression for the up-barrier knock-out rebate term)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction,H", [("up", H_UP), ("down", H_DOWN)])
@pytest.mark.parametrize("is_call", [True, False])
def test_closed_form_rebate_matches_mc(direction, H, is_call):
    knock_out = BarrierSpec(f"{direction}-and-out", "continuous", H, K, is_call, rebate=5.0)
    cf = price_barrier_closed_form(BS, knock_out)
    mc, se = price_barrier_mc(BS, knock_out, n_paths=150_000, n_steps=200, seed=2)
    assert abs(mc - cf) < 4.0 * se + 5e-3


@pytest.mark.parametrize("direction,H", [("up", H_UP), ("down", H_DOWN)])
@pytest.mark.parametrize("is_call", [True, False])
def test_rebate_in_out_parity(direction, H, is_call):
    knock_in = BarrierSpec(f"{direction}-and-in", "continuous", H, K, is_call, rebate=5.0)
    knock_out = BarrierSpec(f"{direction}-and-out", "continuous", H, K, is_call, rebate=5.0)
    E, F = barrier_rebate_terms(BS, knock_in)
    vanilla = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, is_call)
    total = price_barrier_closed_form(BS, knock_in) + price_barrier_closed_form(BS, knock_out)
    assert total == pytest.approx(vanilla + E + F, abs=1e-10)


@pytest.mark.parametrize("direction,H", [("up", H_UP), ("down", H_DOWN)])
@pytest.mark.parametrize("is_call", [True, False])
def test_pde_knock_in_rebate_matches_mc(direction, H, is_call):
    bar = BarrierSpec(f"{direction}-and-in", "continuous", H, K, is_call, rebate=5.0)
    pde = price_barrier_pde(BS, bar, M=500, N=500)
    mc, se = price_barrier_mc(BS, bar, n_paths=150_000, n_steps=200, seed=4)
    assert abs(pde - mc) < 4.0 * se + 5e-3


# --------------------------------------------------------------------------
# Discrete monitoring
# --------------------------------------------------------------------------


def test_monitor_is_honoured():
    cont = BarrierSpec("up-and-out", "continuous", H_UP, K, True)
    disc = BarrierSpec("up-and-out", "discrete", H_UP, K, True)
    pde_cont = price_barrier_pde(BS, cont, M=400, N=400)
    pde_disc = price_barrier_pde(BS, disc, M=400, N=400, monitor_steps=52)
    # a discretely observed knock-out is worth more than the continuous one
    assert pde_disc > pde_cont + 0.05


@pytest.mark.parametrize(
    "direction,H,is_call", [("up", H_UP, True), ("down", H_DOWN, False)]
)
def test_pde_discrete_converges_to_mc(direction, H, is_call):
    bar = BarrierSpec(f"{direction}-and-out", "discrete", H, K, is_call)
    mc, se = price_barrier_mc(BS, bar, n_paths=200_000, n_steps=52, seed=6)
    coarse = price_barrier_pde(BS, bar, M=800, N=832, monitor_steps=52)
    fine = price_barrier_pde(BS, bar, M=1600, N=1664, monitor_steps=52)
    assert abs(fine - mc) < abs(coarse - mc)
    assert abs(fine - mc) < 5e-2


def test_pde_discrete_knock_in_parity():
    knock_in = BarrierSpec("up-and-in", "discrete", H_UP, K, True)
    knock_out = BarrierSpec("up-and-out", "discrete", H_UP, K, True)
    v_in = price_barrier_pde(BS, knock_in, M=800, N=832, monitor_steps=52)
    v_out = price_barrier_pde(BS, knock_out, M=800, N=832, monitor_steps=52)
    vanilla = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, True)
    assert v_in + v_out == pytest.approx(vanilla, abs=5e-3)


def test_discrete_knock_in_rebate_unsupported():
    bar = BarrierSpec("up-and-in", "discrete", H_UP, K, True, rebate=1.0)
    with pytest.raises(NotImplementedError):
        price_barrier_pde(BS, bar, M=200, N=200, monitor_steps=52)

"""Closed-form benchmarks: vanilla values and barrier in-out parity."""

import numpy as np
import pytest

from crossbar import (
    BarrierSpec,
    BSParams,
    price_barrier_closed_form,
    price_vanilla,
    vanilla_payoff,
)

BS = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0)
H_UP = 120.0
H_DOWN = 85.0
K = 100.0


def test_vanilla_payoff():
    assert vanilla_payoff(110.0, 100.0, True) == 10.0
    assert vanilla_payoff(90.0, 100.0, True) == 0.0
    assert vanilla_payoff(90.0, 100.0, False) == 10.0


def test_vanilla_known_value():
    # S=K=100, r=q=0, sigma=0.2, T=1 -> 100*(N(0.1) - N(-0.1))
    price = price_vanilla(100.0, 100.0, 0.0, 0.0, 0.2, 1.0, is_call=True)
    assert price == pytest.approx(7.9655675, abs=1e-6)


def test_vanilla_put_call_parity():
    call = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, True)
    put = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, False)
    forward = BS.S0 * np.exp(-BS.q * BS.T) - K * np.exp(-BS.r * BS.T)
    assert call - put == pytest.approx(forward, abs=1e-10)


def test_vanilla_deterministic_limits():
    # sigma -> 0 reduces to the discounted forward payoff
    price = price_vanilla(100.0, 100.0, 0.05, 0.0, 0.0, 1.0, True)
    assert price == pytest.approx(np.exp(-0.05) * (100.0 * np.exp(0.05) - 100.0))


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("direction,H", [("up", H_UP), ("down", H_DOWN)])
def test_in_out_parity(direction, H, is_call):
    # Knock-in + knock-out = vanilla for a zero rebate
    knock_in = BarrierSpec(f"{direction}-and-in", "continuous", H, K, is_call)
    knock_out = BarrierSpec(f"{direction}-and-out", "continuous", H, K, is_call)
    v_in = price_barrier_closed_form(BS, knock_in)
    v_out = price_barrier_closed_form(BS, knock_out)
    vanilla = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, is_call)
    assert v_in + v_out == pytest.approx(vanilla, abs=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_far_barrier_reduces_to_vanilla(is_call):
    knock_out = BarrierSpec("up-and-out", "continuous", 1e6, K, is_call)
    vanilla = price_vanilla(BS.S0, K, BS.r, BS.q, BS.sigma, BS.T, is_call)
    assert price_barrier_closed_form(BS, knock_out) == pytest.approx(vanilla, abs=1e-10)


def test_vanilla_payoff_broadcasts():
    ST = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(vanilla_payoff(ST, 100.0, True), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(vanilla_payoff(ST, 100.0, False), [20.0, 0.0, 0.0])


def test_vanilla_at_and_after_expiry_is_the_payoff():
    # T <= 0 collapses to the (undiscounted) intrinsic payoff
    assert price_vanilla(110.0, 100.0, 0.05, 0.02, 0.2, 0.0, True) == 10.0
    assert price_vanilla(90.0, 100.0, 0.05, 0.02, 0.2, 0.0, False) == 10.0
    assert price_vanilla(90.0, 100.0, 0.05, 0.02, 0.2, -0.5, True) == 0.0


def test_vanilla_broadcasts_over_arrays_and_mixed_maturities():
    S = np.array([90.0, 100.0, 110.0])
    T = np.array([1.0, 0.0, 1.0])  # the middle node is already expired
    vals = price_vanilla(S, 100.0, 0.05, 0.0, 0.2, T, True)
    assert vals.shape == S.shape
    scalars = [
        price_vanilla(float(s), 100.0, 0.05, 0.0, 0.2, float(t), True)
        for s, t in zip(S, T)
    ]
    np.testing.assert_allclose(vals, scalars)


def test_barrier_closed_form_at_expiry():
    expired = BSParams(S0=110.0, r=0.05, q=0.02, sigma=0.2, T=0.0)
    out_call = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=5.0)
    in_call = BarrierSpec("up-and-in", "continuous", 120.0, 100.0, True, rebate=5.0)
    out_put = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False, rebate=5.0)
    in_put = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False, rebate=5.0)

    # knock-outs settle to the intrinsic payoff at expiry
    assert price_barrier_closed_form(expired, out_call) == pytest.approx(10.0)
    assert price_barrier_closed_form(expired, out_put) == pytest.approx(0.0)
    # knock-ins that never knocked in pay the rebate (T = 0, so undiscounted)
    assert price_barrier_closed_form(expired, in_call) == pytest.approx(5.0)
    assert price_barrier_closed_form(expired, in_put) == pytest.approx(5.0)

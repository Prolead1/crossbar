"""Delta/gamma profile APIs in :mod:`crossbar.greeks`.

The bump-and-revalue profile is checked with deterministic pricers whose
derivatives are known in closed form; the surface-based profiles are
checked against the same analytic benchmark and against the scalar PDE
pricer they are built from.
"""

import numpy as np
import pytest

from crossbar import (
    BarrierSpec,
    BSParams,
    barrier_variants,
    pde_surface,
    price_barrier_closed_form,
    price_barrier_pde,
)
from crossbar.greeks import (
    greeks_by_variant,
    greeks_by_variant_pde,
    pde_risk_profile,
    risk_profile,
    surface_risk_profile,
)

BS = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0)
BAR = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
SPOTS = np.array([95.0, 100.0, 105.0])


# --------------------------------------------------------------------------
# risk_profile (bump and revalue)
# --------------------------------------------------------------------------


def test_risk_profile_quadratic_pricer_has_known_derivatives():
    # For f(S) = S^2 the central differences are exact: f' = 2S, f'' = 2.
    def pricer(bs, bar):
        return bs.S0**2

    spots = np.array([90.0, 100.0, 110.0])
    prices, deltas, gammas = risk_profile(pricer, BS, BAR, spots, bump=0.5)
    np.testing.assert_allclose(prices, spots**2)
    np.testing.assert_allclose(deltas, 2.0 * spots)
    np.testing.assert_allclose(gammas, 2.0)


def test_risk_profile_accepts_price_standard_error_pair():
    def bare(bs, bar):
        return bs.S0**2

    def piped(bs, bar):
        return bs.S0**2, 0.123

    expected = risk_profile(bare, BS, BAR, SPOTS, bump=0.5)
    got = risk_profile(piped, BS, BAR, SPOTS, bump=0.5)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(a, b)


def test_risk_profile_clamps_down_bump_at_zero():
    def pricer(bs, bar):
        return bs.S0

    _, deltas, gammas = risk_profile(pricer, BS, BAR, np.array([0.2]), bump=0.5)
    # s_dn is floored at 1e-8, which breaks the symmetry of the stencil, so
    # check the exact finite differences of f(S) = S against p0 / p_up / p_dn.
    p0, p_up, p_dn = 0.2, 0.7, 1e-8
    assert np.all(np.isfinite(deltas))
    assert np.all(np.isfinite(gammas))
    assert deltas[0] == pytest.approx((p_up - p_dn) / (2 * 0.5))
    assert gammas[0] == pytest.approx((p_up - 2.0 * p0 + p_dn) / 0.5**2)


def test_risk_profile_forwards_pricer_kwargs():
    seen = []

    def pricer(bs, bar, tag=None):
        seen.append(tag)
        return bs.S0

    risk_profile(pricer, BS, BAR, [100.0], pricer_kwargs={"tag": "hello"})
    assert seen == ["hello", "hello", "hello"]


def test_risk_profile_matches_black_scholes_for_far_barrier():
    bar = BarrierSpec("up-and-out", "continuous", 1e6, 100.0, True)
    prices, deltas, gammas = risk_profile(
        price_barrier_closed_form, BS, bar, SPOTS, bump=1e-2
    )
    # the barrier is unreachable, so this is the vanilla call
    sqrtT = np.sqrt(BS.T)
    d1 = (
        np.log(SPOTS / bar.K) + (BS.r - BS.q + 0.5 * BS.sigma**2) * BS.T
    ) / (BS.sigma * sqrtT)
    from scipy.stats import norm

    expected_delta = np.exp(-BS.q * BS.T) * norm.cdf(d1)
    expected_gamma = np.exp(-BS.q * BS.T) * norm.pdf(d1) / (
        SPOTS * BS.sigma * sqrtT
    )
    np.testing.assert_allclose(deltas, expected_delta, atol=1e-5)
    np.testing.assert_allclose(gammas, expected_gamma, atol=1e-5)
    assert np.all(prices > 0.0)


# --------------------------------------------------------------------------
# greeks_by_variant
# --------------------------------------------------------------------------


def test_greeks_by_variant_covers_every_label():
    profiles = greeks_by_variant(price_barrier_closed_form, BS, SPOTS, bump=0.5)
    assert set(profiles) == {label for label, _ in barrier_variants(BS)}
    for prices, deltas, gammas in profiles.values():
        assert prices.shape == SPOTS.shape
        assert deltas.shape == SPOTS.shape
        assert gammas.shape == SPOTS.shape
        assert np.all(np.isfinite(prices))


def test_greeks_by_variant_prices_match_closed_form():
    profiles = greeks_by_variant(price_barrier_closed_form, BS, SPOTS, bump=0.5)
    for label, spec in barrier_variants(BS):
        prices, _, _ = profiles[label]
        for s, price in zip(SPOTS, prices):
            bs = BSParams(float(s), BS.r, BS.q, BS.sigma, BS.T)
            assert price == pytest.approx(
                price_barrier_closed_form(bs, spec), abs=1e-10
            )


def test_greeks_by_variant_forwards_variant_kwargs():
    specs = dict(
        barrier_variants(BS, H_up=130.0, H_down=70.0, K=105.0, monitor="discrete")
    )
    profiles = greeks_by_variant(
        price_barrier_closed_form,
        BS,
        SPOTS,
        bump=0.5,
        H_up=130.0,
        H_down=70.0,
        K=105.0,
        monitor="discrete",
    )
    assert set(profiles) == set(specs)
    for label, spec in specs.items():
        prices, _, _ = profiles[label]
        for s, price in zip(SPOTS, prices):
            bs = BSParams(float(s), BS.r, BS.q, BS.sigma, BS.T)
            assert price == pytest.approx(
                price_barrier_closed_form(bs, spec), abs=1e-10
            )


# --------------------------------------------------------------------------
# surface_risk_profile
# --------------------------------------------------------------------------


def test_surface_risk_profile_exact_for_quadratic_surface():
    S = np.linspace(50.0, 150.0, 201)
    V = S**2
    # spots sit exactly on grid nodes so the price interpolation is exact too
    spots = np.array([80.0, 100.0, 120.0])
    prices, deltas, gammas = surface_risk_profile(S, V, spots)
    np.testing.assert_allclose(prices, spots**2)
    np.testing.assert_allclose(deltas, 2.0 * spots)
    np.testing.assert_allclose(gammas, 2.0)


def test_surface_risk_profile_endpoint_conventions():
    S = np.linspace(1.0, 11.0, 11)
    V = S**2
    _, deltas, gammas = surface_risk_profile(S, V, S)
    assert deltas[0] == pytest.approx((V[1] - V[0]) / (S[1] - S[0]))
    assert deltas[-1] == pytest.approx((V[-1] - V[-2]) / (S[-1] - S[-2]))
    assert gammas[0] == 0.0
    assert gammas[-1] == 0.0


# --------------------------------------------------------------------------
# pde_risk_profile
# --------------------------------------------------------------------------


def test_pde_risk_profile_knock_out_matches_surface_profile():
    S, V = pde_surface(BS, BAR, M=300, N=300)
    expected = surface_risk_profile(S, V, SPOTS)
    got = pde_risk_profile(BS, BAR, SPOTS, M=300, N=300)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("rebate", [0.0, 5.0])
def test_pde_risk_profile_knock_in_price_matches_scalar_pricer(rebate):
    bar = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False, rebate=rebate)
    prices, deltas, gammas = pde_risk_profile(BS, bar, np.array([100.0]), M=300, N=300)
    assert prices[0] == pytest.approx(
        price_barrier_pde(BS, bar, M=300, N=300), abs=1e-12
    )
    assert np.all(np.isfinite(deltas))
    assert np.all(np.isfinite(gammas))


@pytest.mark.parametrize(
    "bar",
    [
        BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True),
        BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False),
        BarrierSpec("up-and-in", "continuous", 120.0, 100.0, True),
        BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False),
    ],
)
def test_pde_risk_profile_matches_bump_and_revalue_closed_form(bar):
    expected = risk_profile(price_barrier_closed_form, BS, bar, SPOTS, bump=0.5)
    got = pde_risk_profile(BS, bar, SPOTS, M=400, N=400)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(a, b, atol=5e-3)


def test_pde_risk_profile_discrete_knock_in_rebate_unsupported():
    bar = BarrierSpec("up-and-in", "discrete", 120.0, 100.0, True, rebate=1.0)
    with pytest.raises(NotImplementedError):
        pde_risk_profile(BS, bar, SPOTS, M=100, N=100, monitor_steps=52)


# --------------------------------------------------------------------------
# greeks_by_variant_pde
# --------------------------------------------------------------------------


def test_greeks_by_variant_pde_covers_every_label_and_matches_pricer():
    profiles = greeks_by_variant_pde(BS, SPOTS, M=100, N=100)
    variants = dict(barrier_variants(BS))
    assert set(profiles) == set(variants)
    for label, spec in variants.items():
        prices, deltas, gammas = profiles[label]
        assert prices.shape == SPOTS.shape
        assert deltas.shape == SPOTS.shape
        assert gammas.shape == SPOTS.shape
        # SPOTS[1] is S0, where the surface and scalar pricers must agree
        assert prices[1] == pytest.approx(
            price_barrier_pde(BS, spec, M=100, N=100), abs=1e-9
        )


def test_greeks_by_variant_pde_forwards_solver_options():
    disc = dict(barrier_variants(BS, monitor="discrete"))
    profiles = greeks_by_variant_pde(
        BS, SPOTS, M=200, N=208, monitor_steps=52, monitor="discrete"
    )
    assert set(profiles) == set(disc)
    label = "up-and-out call"
    prices, _, _ = profiles[label]
    assert prices[1] == pytest.approx(
        price_barrier_pde(
            BS, disc[label], M=200, N=208, monitor_steps=52
        ),
        abs=1e-9,
    )

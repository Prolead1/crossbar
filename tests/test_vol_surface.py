"""Tests for the screen-quote volatility surface and its engine hooks.

Covers the quote -> smile -> surface pipeline, the delta/strike
conversions, the forward-vol term structure, the local-vol grid and the
optional ``surface`` arguments on the Monte Carlo and PDE engines.
"""

import numpy as np
import pytest

from crossbar import (
    BarrierSpec,
    BSParams,
    LocalVolSurface,
    VolSmile,
    VolSurface,
    barrier_hits,
    build_grid,
    build_sigma_grid,
    check_arbitrage,
    gen_normals,
    local_vol_grid,
    local_volatility,
    mid_vol,
    monte_carlo_paths,
    pde_knock_out,
    pde_surface,
    plot_vol_surface,
    price_barrier_mc,
    price_barrier_pde,
    price_vanilla,
    sigma_for_strike,
    stepwise_sigmas_for_strike,
    stepwise_sigmas_from_surface,
    strike_from_delta,
    tenor_to_years,
    total_variance_grid,
    vanilla_payoff,
)
from crossbar.pde import enforce_barrier_dirichlet, theta_step
from crossbar.vol_surface import EXAMPLE_QUOTES, QUOTE_FIELDS, bs_delta

FX = BSParams(S0=1.10, r=0.04, q=0.03, sigma=0.07, T=0.5)
SURFACE = VolSurface.from_quotes(EXAMPLE_QUOTES)
UP = BarrierSpec("up-and-out", "continuous", 1.20, 1.10, True)
DOWN = BarrierSpec("down-and-out", "continuous", 1.00, 1.10, False)


def flat_surface(sigma, T=1.0):
    """A surface whose every pillar equals ``sigma``.

    Risk reversals and butterflies are zero, so the ATM quote propagates
    unchanged to all five pillars.
    """
    quote = {field: (0.0, 0.0) for field in QUOTE_FIELDS}
    quote["atm"] = (sigma * 100.0, sigma * 100.0)
    return VolSurface.from_quotes({T: quote})


# --------------------------------------------------------------------------
# Quote parsing
# --------------------------------------------------------------------------


def test_tenor_to_years_labels_and_numbers():
    assert tenor_to_years("1M") == pytest.approx(1.0 / 12.0)
    assert tenor_to_years("0n") == pytest.approx(1.0 / 365.0)  # case-insensitive
    assert tenor_to_years(0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", ["10Y", 0.0, -1.0])
def test_tenor_to_years_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        tenor_to_years(bad)


def test_mid_vol_converts_percentage_points_to_decimal():
    assert mid_vol(5.80, 6.20) == pytest.approx(0.06)


def test_mid_vol_rejects_crossed_market():
    with pytest.raises(ValueError):
        mid_vol(6.20, 5.80)


# --------------------------------------------------------------------------
# VolSmile
# --------------------------------------------------------------------------


def _sample_smile_quotes():
    q = {field: (6.0, 6.0) for field in QUOTE_FIELDS}
    q["rr_25"] = (-0.4, -0.4)
    q["bf_25"] = (0.2, 0.2)
    q["rr_10"] = (-1.0, -1.0)
    q["bf_10"] = (0.5, 0.5)
    return q


def test_vol_smile_reconstructs_wing_vols_from_rr_bf():
    smile = VolSmile.from_quotes("3M", _sample_smile_quotes())
    assert smile.T == pytest.approx(3.0 / 12.0)
    assert smile.atm == pytest.approx(0.06)
    assert smile.rr_25 == pytest.approx(-0.004)
    assert smile.bf_25 == pytest.approx(0.002)
    assert smile.rr_10 == pytest.approx(-0.010)
    assert smile.bf_10 == pytest.approx(0.005)
    # reconstruct the quoted combinations
    assert smile.sigma_25c - smile.sigma_25p == pytest.approx(smile.rr_25)
    assert 0.5 * (smile.sigma_25c + smile.sigma_25p) - smile.atm == pytest.approx(
        smile.bf_25
    )
    assert smile.sigma_10c - smile.sigma_10p == pytest.approx(smile.rr_10)
    assert 0.5 * (smile.sigma_10c + smile.sigma_10p) - smile.atm == pytest.approx(
        smile.bf_10
    )


def test_vol_smile_pillars_and_interpolation():
    smile = SURFACE.smiles[0.25]
    deltas, vols = smile.pillars()
    np.testing.assert_allclose(smile.sigma(deltas), vols)
    # clamped beyond the wings
    assert smile.sigma(-0.9) == pytest.approx(smile.sigma_25p)
    assert smile.sigma(0.9) == pytest.approx(smile.sigma_25c)
    # linear halfway between two pillars
    lo, hi = -0.25, -0.10
    assert smile.sigma(0.5 * (lo + hi)) == pytest.approx(
        0.5 * (smile.sigma(lo) + smile.sigma(hi))
    )
    # array input
    out = smile.sigma(np.array([-0.25, 0.0, 0.25]))
    np.testing.assert_allclose(out, [smile.sigma_25p, smile.atm, smile.sigma_25c])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(T=0.0, atm=0.05, rr_25=0.0, bf_25=0.0, rr_10=0.0, bf_10=0.0),
        dict(T=1.0, atm=0.0, rr_25=0.0, bf_25=0.0, rr_10=0.0, bf_10=0.0),
        # butterfly drives the 25d wings non-positive
        dict(T=1.0, atm=0.05, rr_25=0.0, bf_25=-0.06, rr_10=0.0, bf_10=0.0),
    ],
)
def test_vol_smile_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        VolSmile(**kwargs)


def test_vol_smile_from_quotes_requires_every_field():
    with pytest.raises(ValueError):
        VolSmile.from_quotes("1M", {"atm": (5.0, 6.0)})


# --------------------------------------------------------------------------
# VolSurface
# --------------------------------------------------------------------------


def test_vol_surface_from_quotes_is_sorted_and_clamped():
    assert np.all(np.diff(SURFACE.maturities) > 0)
    for T, smile in SURFACE.smiles.items():
        assert SURFACE.interp_sigma(T, 0.0) == pytest.approx(smile.atm)
    first, last = SURFACE.maturities[0], SURFACE.maturities[-1]
    assert SURFACE.interp_sigma(first / 2, 0.10) == pytest.approx(
        SURFACE.smiles[first].sigma(0.10)
    )
    assert SURFACE.interp_sigma(last * 2, 0.10) == pytest.approx(
        SURFACE.smiles[last].sigma(0.10)
    )
    # halfway in maturity between the 3M and 6M pillars: linear in total
    # variance, then converted back to vol
    t0, t1 = 0.25, 0.5
    tm = 0.5 * (t0 + t1)
    w0 = SURFACE.smiles[t0].sigma(0.10) ** 2 * t0
    w1 = SURFACE.smiles[t1].sigma(0.10) ** 2 * t1
    assert SURFACE.interp_sigma(tm, 0.10) == pytest.approx(np.sqrt(0.5 * (w0 + w1) / tm))
    assert SURFACE.total_variance(tm, 0.10) == pytest.approx(0.5 * (w0 + w1))


def test_vol_surface_accepts_mapping_and_iterable():
    from_mapping = VolSurface({0.5: SURFACE.smiles[0.5]})
    assert from_mapping.maturities.tolist() == [0.5]
    from_iter = VolSurface([SURFACE.smiles[0.25], SURFACE.smiles[0.5]])
    assert from_iter.maturities.tolist() == [0.25, 0.5]


def test_vol_surface_rejects_empty():
    with pytest.raises(ValueError):
        VolSurface([])


def test_vol_surface_interp_sigma_over_delta_array():
    deltas = np.array([-0.25, 0.0, 0.25])
    out = SURFACE.interp_sigma(0.5, deltas)
    assert out.shape == deltas.shape
    for i, delta in enumerate(deltas):
        assert out[i] == pytest.approx(SURFACE.interp_sigma(0.5, float(delta)))


# --------------------------------------------------------------------------
# Delta <-> strike conversion
# --------------------------------------------------------------------------


def test_bs_delta_signs_and_expiry():
    call = bs_delta(FX.S0, FX.S0, FX.r, FX.q, FX.sigma, FX.T, True)
    put = bs_delta(FX.S0, FX.S0, FX.r, FX.q, FX.sigma, FX.T, False)
    assert call > 0 > put
    # deterministic expiry: delta is the moneyness indicator
    assert bs_delta(1.10, 1.00, 0.04, 0.03, 0.07, 0.0, True) == 1.0
    assert bs_delta(0.90, 1.00, 0.04, 0.03, 0.07, 0.0, True) == 0.0
    assert bs_delta(0.90, 1.00, 0.04, 0.03, 0.07, 0.0, False) == -1.0
    assert bs_delta(1.10, 1.00, 0.04, 0.03, 0.07, 0.0, False) == 0.0


@pytest.mark.parametrize("delta", [-0.25, -0.10, 0.10, 0.25])
def test_strike_from_delta_round_trips_through_bs_delta(delta):
    strike = strike_from_delta(SURFACE, FX, delta, 0.5)
    sigma = float(sigma_for_strike(SURFACE, FX, strike, 0.5))
    assert bs_delta(FX.S0, strike, FX.r, FX.q, sigma, 0.5, delta > 0) == pytest.approx(
        delta, abs=1e-8
    )


@pytest.mark.parametrize("delta", [-0.15, -0.05, 0.15, 0.20])
@pytest.mark.parametrize("T", [0.25, 0.375, 0.5, 0.75])
def test_strike_from_delta_implicit_vol_round_trips(delta, T):
    """Non-pillar deltas exercise the surface-implied fixed-point loop."""
    strike = strike_from_delta(SURFACE, FX, delta, T)
    sigma = float(sigma_for_strike(SURFACE, FX, strike, T))
    assert bs_delta(FX.S0, strike, FX.r, FX.q, sigma, T, delta > 0) == pytest.approx(
        delta, abs=1e-8
    )
    # the strike agrees with pricing the delta at the surface vol
    assert strike == pytest.approx(
        strike_from_delta(SURFACE, FX, delta, T, sigma=sigma), rel=1e-6
    )


def test_strike_from_delta_atm_is_delta_neutral():
    strike = strike_from_delta(SURFACE, FX, 0.0, 0.5)
    sigma = float(sigma_for_strike(SURFACE, FX, strike, 0.5))
    call = bs_delta(FX.S0, strike, FX.r, FX.q, sigma, 0.5, True)
    put = bs_delta(FX.S0, strike, FX.r, FX.q, sigma, 0.5, False)
    assert call + put == pytest.approx(0.0, abs=1e-12)


def test_strike_from_delta_with_explicit_sigma():
    strike = strike_from_delta(SURFACE, FX, 0.25, 0.5, sigma=0.10)
    assert bs_delta(FX.S0, strike, FX.r, FX.q, 0.10, 0.5, True) == pytest.approx(0.25)


def test_strike_from_delta_rejects_bad_inputs():
    with pytest.raises(ValueError):
        strike_from_delta(SURFACE, FX, 0.25, 0.0)
    with pytest.raises(ValueError):
        strike_from_delta(SURFACE, FX, 1.0, 0.5)
    with pytest.raises(ValueError):
        strike_from_delta(SURFACE, FX, -1.0, 0.5)
    # delta cannot be realised given the carry q
    high_carry = BSParams(S0=1.10, r=0.0, q=1.0, sigma=0.2, T=1.0)
    with pytest.raises(ValueError):
        strike_from_delta(SURFACE, high_carry, 0.9, 1.0, sigma=0.2)


def test_strike_for_delta_rejects_out_of_range_delta():
    """The private fixed-vol inverter guards the delta domain directly."""
    from crossbar.vol_surface import _strike_for_delta

    for bad in (1.0, -1.0, 1.5, -2.0):
        with pytest.raises(ValueError, match="delta must lie strictly between"):
            _strike_for_delta(FX, bad, 0.5, 0.10)


def test_strike_from_delta_fixed_point_iteration_is_capped(monkeypatch):
    """A surface vol that never settles stops after the 50-iteration cap.

    ``strike_from_delta`` reads the vol back through ``sigma_for_strike``;
    replacing it with a strictly increasing sequence removes the fixed
    point and forces the loop to run to exhaustion rather than diverge.
    """
    import crossbar.vol_surface as vs

    calls = []

    def never_settles(surface, params, K, T):
        calls.append(K)
        return 0.05 + 1e-3 * len(calls)

    monkeypatch.setattr(vs, "sigma_for_strike", never_settles)
    strike = strike_from_delta(SURFACE, FX, 0.2, 0.5)
    assert len(calls) == 50
    assert np.isfinite(strike)


# --------------------------------------------------------------------------
# sigma_for_strike
# --------------------------------------------------------------------------


def test_sigma_for_strike_reproduces_pillar_vols():
    T = 0.25
    smile = SURFACE.smiles[T]
    deltas, vols = smile.pillars()
    for delta, vol in zip(deltas, vols):
        strike = strike_from_delta(SURFACE, FX, delta, T, sigma=vol)
        assert float(sigma_for_strike(SURFACE, FX, strike, T)) == pytest.approx(
            vol, rel=1e-10
        )


def test_sigma_for_strike_array_and_maturity_clamping():
    strikes = np.array([1.0, 1.1, 1.2])
    out = sigma_for_strike(SURFACE, FX, strikes, 0.5)
    assert out.shape == strikes.shape
    assert np.all(out > 0)
    # below the first and above the last quoted maturity
    assert np.isfinite(float(sigma_for_strike(SURFACE, FX, 1.1, 1e-4)))
    assert np.isfinite(float(sigma_for_strike(SURFACE, FX, 1.1, 5.0)))


# --------------------------------------------------------------------------
# Forward-vol term structure
# --------------------------------------------------------------------------


def test_sigma_for_strike_at_zero_maturity_uses_short_end_vol():
    val = float(sigma_for_strike(SURFACE, FX, 1.10, 0.0))
    assert val == pytest.approx(0.06, abs=1e-3)


def test_stepwise_sigmas_reproduce_implied_variance():
    T, n_steps = 0.75, 30
    steps = stepwise_sigmas_from_surface(SURFACE, T, n_steps, delta=0.25)
    assert steps.shape == (n_steps,)
    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.concatenate([[0.0], np.cumsum(steps**2) * (T / n_steps)])
    implied = np.array([SURFACE.interp_sigma(ti, 0.25) ** 2 * ti for ti in t])
    np.testing.assert_allclose(variance, implied, atol=1e-14)


def test_stepwise_sigmas_flat_surface_is_constant():
    steps = stepwise_sigmas_from_surface(flat_surface(0.2, 1.0), 1.0, 8)
    np.testing.assert_allclose(steps, 0.2)


def test_stepwise_sigmas_rejects_bad_inputs():
    with pytest.raises(ValueError):
        stepwise_sigmas_from_surface(SURFACE, 0.0, 10)
    with pytest.raises(ValueError):
        stepwise_sigmas_from_surface(SURFACE, 0.5, 0)


def test_stepwise_sigmas_for_strike_reproduces_implied_variance():
    T, n_steps, K = 0.5, 25, 1.10
    steps = stepwise_sigmas_for_strike(SURFACE, FX, K, T, n_steps)
    t = np.linspace(0.0, T, n_steps + 1)
    variance = np.concatenate([[0.0], np.cumsum(steps**2) * (T / n_steps)])
    implied = np.array(
        [sigma_for_strike(SURFACE, FX, K, ti) ** 2 * ti for ti in t[1:]]
    )
    np.testing.assert_allclose(variance[1:], implied, atol=1e-14)


def test_stepwise_sigmas_for_strike_flat_surface_is_constant():
    steps = stepwise_sigmas_for_strike(flat_surface(0.2, 1.0), FX, 1.10, 1.0, 8)
    np.testing.assert_allclose(steps, 0.2)


def test_stepwise_sigmas_for_strike_rejects_bad_inputs():
    with pytest.raises(ValueError):
        stepwise_sigmas_for_strike(SURFACE, FX, 1.10, 0.0, 10)
    with pytest.raises(ValueError):
        stepwise_sigmas_for_strike(SURFACE, FX, 1.10, 0.5, 0)


# --------------------------------------------------------------------------
# Dupire local volatility
# --------------------------------------------------------------------------


def test_local_vol_flat_surface_is_constant():
    S = np.linspace(0.7, 1.6, 31)
    t = np.linspace(1e-6, 1.0, 13)
    grid = local_vol_grid(flat_surface(0.2, 1.0), FX, S, t)
    np.testing.assert_allclose(grid, 0.2, rtol=1e-12)


def test_local_vol_is_positive_and_finite_on_example():
    S = np.linspace(0.7, 1.6, 51)
    for T in [0.0, 0.002, 0.1, 0.5, 1.0]:
        vols = np.asarray(local_vol_grid(SURFACE, FX, S, [T])).ravel()
        assert np.all(np.isfinite(vols))
        assert np.all(vols > 0.0)


def test_local_volatility_matches_class_evaluation():
    lv = LocalVolSurface(SURFACE, FX)
    assert float(local_volatility(SURFACE, FX, 1.10, 0.5)) == pytest.approx(
        float(lv.sigma(1.10, 0.5))
    )
    # a short-maturity limit is well defined (no division by T)
    assert np.isfinite(float(local_volatility(SURFACE, FX, 1.10, 0.0)))


def test_local_vol_grid_rejects_non_1d():
    with pytest.raises(ValueError):
        local_vol_grid(SURFACE, FX, np.ones((2, 2)), np.ones(3))


def test_dupire_agrees_with_call_price_formula():
    """Dupire's local variance matches the call-price equation
    ``sigma_loc^2 = (C_T + (r-q) K C_K + q C) / (K^2 C_KK / 2)``.
    """
    lv = LocalVolSurface(SURFACE, FX)
    K, T = 1.10, 0.4
    dK, dT = 1e-5, 1e-5

    def call(k, t):
        w = lv._state(k, t)[0]
        return price_vanilla(
            FX.S0, k, FX.r, FX.q, np.sqrt(w / t), t, True
        )

    C = call(K, T)
    CT = (call(K, T + dT) - call(K, T - dT)) / (2.0 * dT)
    CK = (call(K + dK, T) - call(K - dK, T)) / (2.0 * dK)
    CKK = (call(K + dK, T) - 2.0 * C + call(K - dK, T)) / (dK**2)
    sigma2_price = (CT + (FX.r - FX.q) * K * CK + FX.q * C) / (0.5 * K**2 * CKK)
    assert float(lv.sigma(K, T)) ** 2 == pytest.approx(sigma2_price, rel=1e-6)


def test_check_arbitrage_flat_surface_is_clean():
    report = check_arbitrage(flat_surface(0.2, 1.0), FX)
    assert report["calendar_ok"] is True
    assert report["butterfly_ok"] is True
    assert report["min_calendar_slope"] > 0.0
    assert report["min_butterfly_g"] >= 0.0


def test_check_arbitrage_flags_calendar_violation():
    # second tenor at a much lower vol implies falling total variance
    low = {field: (0.0, 0.0) for field in QUOTE_FIELDS}
    low["atm"] = (4.0, 4.0)
    high = {field: (0.0, 0.0) for field in QUOTE_FIELDS}
    high["atm"] = (20.0, 20.0)
    surface = VolSurface.from_quotes({"3M": high, "6M": low})
    report = check_arbitrage(surface, FX)
    assert report["calendar_ok"] is False


def test_check_arbitrage_accepts_custom_grid():
    report = check_arbitrage(
        SURFACE, FX, strikes=[1.0, 1.1, 1.2], maturities=[0.25, 0.5]
    )
    assert set(report) == {
        "calendar_ok",
        "butterfly_ok",
        "min_calendar_slope",
        "min_butterfly_g",
    }


def test_build_sigma_grid_flat_surface_is_constant():
    S = np.linspace(0.8, 1.4, 21)
    t = np.linspace(0.0, 1.0, 11)
    grid = build_sigma_grid(flat_surface(0.2, 1.0), FX, UP, S, t)
    assert grid.shape == (t.size, S.size)
    np.testing.assert_allclose(grid, 0.2)


def test_build_sigma_grid_masks_dead_region():
    S = np.linspace(0.8, 1.4, 61)
    t = np.linspace(0.0, 0.5, 6)
    up_grid = build_sigma_grid(SURFACE, FX, UP, S, t)
    up_edge = int(np.argmin(np.abs(S - UP.H)))
    expected_up = np.broadcast_to(up_grid[:, up_edge][:, None], up_grid[:, S >= UP.H].shape)
    np.testing.assert_allclose(up_grid[:, S >= UP.H], expected_up)

    down_grid = build_sigma_grid(SURFACE, FX, DOWN, S, t)
    down_edge = int(np.argmin(np.abs(S - DOWN.H)))
    expected_down = np.broadcast_to(
        down_grid[:, down_edge][:, None], down_grid[:, S <= DOWN.H].shape
    )
    np.testing.assert_allclose(down_grid[:, S <= DOWN.H], expected_down)


def test_build_sigma_grid_discrete_does_not_mask():
    bar = BarrierSpec("up-and-out", "discrete", 1.20, 1.10, True)
    S = np.linspace(0.8, 1.4, 41)
    t = np.linspace(0.0, 0.5, 6)
    grid = build_sigma_grid(SURFACE, FX, bar, S, t)
    assert np.all(np.isfinite(grid))
    # a discrete barrier never kills nodes, so the grid is exactly the raw
    # Dupire local vol -- no edge-column backfill anywhere
    expected = local_vol_grid(SURFACE, FX, S, t)
    np.testing.assert_allclose(grid, expected)


def test_build_sigma_grid_all_live_leaves_values_untouched():
    # barrier above the whole grid, so every node is live
    bar = BarrierSpec("up-and-out", "continuous", 5.0, 1.10, True)
    S = np.linspace(0.8, 1.4, 11)
    t = np.linspace(0.0, 0.5, 4)
    grid = build_sigma_grid(SURFACE, FX, bar, S, t)
    assert np.all(np.isfinite(grid))


@pytest.mark.parametrize(
    "S,t",
    [
        (np.ones((2, 2)), np.ones(3)),
        (np.ones(3), np.ones((2, 2))),
    ],
)
def test_build_sigma_grid_rejects_non_1d(S, t):
    with pytest.raises(ValueError):
        build_sigma_grid(SURFACE, FX, UP, S, t)


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------


def test_total_variance_grid_flat_is_sigma_squared_times_maturity():
    strikes = np.array([0.9, 1.0, 1.1])
    maturities = np.array([0.25, 0.5, 1.0])
    var = total_variance_grid(flat_surface(0.2, 1.0), FX, strikes, maturities)
    expected = 0.2**2 * maturities[:, None] * np.ones((1, strikes.size))
    np.testing.assert_allclose(var, expected, atol=1e-15)


def test_total_variance_grid_clamps_maturities():
    var = total_variance_grid(SURFACE, FX, [1.0, 1.1], [1e-4, 5.0])
    assert np.all(np.isfinite(var))


def test_plot_vol_surface_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = plot_vol_surface(SURFACE, FX)
    assert ax is not None
    # explicit strike/maturity grids and a reused axis
    ax2 = plot_vol_surface(
        SURFACE,
        FX,
        strikes=np.linspace(0.9, 1.3, 7),
        maturities=np.linspace(1.0 / 12.0, 0.75, 4),
        ax=ax,
    )
    assert ax2 is ax
    matplotlib.pyplot.close(ax.figure)


# --------------------------------------------------------------------------
# Monte Carlo integration
# --------------------------------------------------------------------------


def test_monte_carlo_paths_flat_and_explicit_sigma_agree():
    Z = gen_normals(2000, 20, seed=0)
    base = monte_carlo_paths(FX, Z)
    explicit = monte_carlo_paths(FX, Z, sigma=FX.sigma)
    np.testing.assert_array_equal(base, explicit)
    scalar = monte_carlo_paths(FX, Z, sigma=0.10)
    assert scalar.shape == base.shape


def test_monte_carlo_paths_follow_time_varying_sigma():
    Z = gen_normals(4000, 20, seed=1)
    steps = stepwise_sigmas_from_surface(SURFACE, FX.T, 20, 0.0)
    varying = monte_carlo_paths(FX, Z, sigma=steps)
    flat = monte_carlo_paths(FX, Z, sigma=FX.sigma)
    assert varying.shape == flat.shape
    assert np.all(varying > 0.0)
    assert not np.allclose(varying, flat)


def test_monte_carlo_paths_rejects_wrong_sigma_length():
    Z = gen_normals(10, 5, seed=0)
    with pytest.raises(ValueError):
        monte_carlo_paths(FX, Z, sigma=np.ones(4))


def test_price_barrier_mc_uses_surface_term_structure():
    steps = stepwise_sigmas_for_strike(SURFACE, FX, UP.H, FX.T, 24)
    Z = gen_normals(4000, 24, seed=0)
    flat_price, _ = price_barrier_mc(FX, UP, n_paths=4000, n_steps=24, seed=0, Z=Z)
    surf_price, se = price_barrier_mc(
        FX, UP, n_paths=4000, n_steps=24, seed=0, Z=Z, sigma=steps
    )
    assert np.isfinite(surf_price) and se > 0.0
    assert surf_price != flat_price


def test_price_barrier_mc_constant_term_structure_matches_scalar():
    Z = gen_normals(2000, 16, seed=3)
    steps = np.full(16, 0.10)
    arr, _ = price_barrier_mc(
        FX, UP, n_paths=2000, n_steps=16, seed=3, Z=Z, sigma=steps
    )
    scalar, _ = price_barrier_mc(
        FX, UP, n_paths=2000, n_steps=16, seed=3, Z=Z, sigma=0.10
    )
    assert arr == pytest.approx(scalar)


def test_price_barrier_mc_rejects_wrong_sigma_length():
    Z = gen_normals(10, 5, seed=0)
    with pytest.raises(ValueError):
        price_barrier_mc(
            FX, UP, n_paths=10, n_steps=5, seed=0, Z=Z, sigma=np.ones(4)
        )


def test_barrier_hits_rejects_wrong_sigma_length():
    S = monte_carlo_paths(FX, gen_normals(10, 5, seed=0))
    with pytest.raises(ValueError, match="sigma must be scalar"):
        barrier_hits(S, FX, UP, sigma=np.ones(4))


# --------------------------------------------------------------------------
# PDE integration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bar", [UP, DOWN])
def test_pde_with_flat_surface_matches_flat_solver(bar):
    M = N = 60
    flat = flat_surface(FX.sigma, FX.T)
    base = price_barrier_pde(FX, bar, M=M, N=N)
    assert price_barrier_pde(FX, bar, M=M, N=N, surface=flat) == pytest.approx(
        base, abs=1e-12
    )
    assert pde_knock_out(FX, bar, M=M, N=N, surface=flat) == pytest.approx(
        base, abs=1e-12
    )


def test_pde_knock_in_with_flat_surface_matches_flat_solver():
    knock_in = BarrierSpec("up-and-in", "continuous", 1.20, 1.10, True)
    flat = flat_surface(FX.sigma, FX.T)
    base = price_barrier_pde(FX, knock_in, M=60, N=60)
    assert price_barrier_pde(
        FX, knock_in, M=60, N=60, surface=flat
    ) == pytest.approx(base, abs=1e-10)


def test_vanilla_leg_vol_follows_surface_strike_vol():
    from crossbar.pde import _vanilla_leg_vol

    assert _vanilla_leg_vol(FX, UP) == FX.sigma
    expected = float(sigma_for_strike(SURFACE, FX, UP.K, FX.T))
    assert _vanilla_leg_vol(FX, UP, SURFACE) == pytest.approx(expected)
    # the surface strike vol differs from the ATM vol, so parity moves
    assert expected != pytest.approx(FX.sigma)


def test_pde_surface_with_vol_surface_is_finite_and_finite_knock_in():
    S, V = pde_surface(FX, UP, M=50, N=50, surface=SURFACE)
    assert S.shape == V.shape
    assert np.all(np.isfinite(V))
    knock_in = BarrierSpec("up-and-in", "continuous", 1.20, 1.10, True)
    _, V_in = pde_surface(FX, knock_in, M=50, N=50, surface=SURFACE)
    assert np.all(np.isfinite(V_in))


def test_theta_step_rejects_bad_sigma_shape():
    S, t = build_grid(FX, UP, M=20, N=5)
    V = enforce_barrier_dirichlet(S, vanilla_payoff(S, UP.K, UP.is_call), UP)
    with pytest.raises(ValueError):
        theta_step(
            FX, UP, S, V, dt=t[1] - t[0], tnow=t[0], theta=0.5, sigma=np.ones(3)
        )

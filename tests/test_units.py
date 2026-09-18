"""Unit tests for the building blocks behind the two pricing engines.

These cover the helpers that the end-to-end engine tests in
``test_engines.py`` exercise only indirectly: input validation, variant
enumeration, the random-number/path generators, the Brownian-bridge
crossing probability, first-passage detection and the tridiagonal / grid
machinery of the PDE solver.
"""

import numpy as np
import pytest

from crossbar import (
    BarrierSpec,
    BSParams,
    barrier_variants,
    barrier_hits,
    build_grid,
    gen_normals,
    monte_carlo_paths,
    pde_knock_out,
    pde_surface,
    price_barrier,
    price_barrier_closed_form,
    price_barrier_cv,
    price_barrier_mc,
    price_barrier_pde,
    step_bridge_cross_prob,
    validate_inputs,
    vanilla_payoff,
)
from crossbar.analytic import barrier_rebate_terms
from crossbar.pde import (
    boundary_values,
    cn_step,
    enforce_barrier_dirichlet,
    rannacher_pair,
    theta_step,
    thomas_factor,
    thomas_solve,
    thomas_solve_factored,
)

BS = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_validate_inputs_accepts_valid_spec():
    validate_inputs(BS, BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True))
    validate_inputs(BS, BarrierSpec("down-and-in", "discrete", 85.0, 100.0, False, rebate=2.0))


@pytest.mark.parametrize(
    "bs,bar",
    [
        (BSParams(0.0, 0.05, 0.02, 0.2, 1.0), BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)),
        (BSParams(-1.0, 0.05, 0.02, 0.2, 1.0), BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)),
        (BSParams(100.0, 0.05, 0.02, 0.0, 1.0), BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)),
        (BSParams(100.0, 0.05, 0.02, 0.2, 0.0), BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)),
        (BS, BarrierSpec("up-and-out", "continuous", 120.0, 0.0, True)),
        (BS, BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=-1.0)),
        (BS, BarrierSpec("sideways-out", "continuous", 120.0, 100.0, True)),
        (BS, BarrierSpec("up-and-out", "hourly", 120.0, 100.0, True)),
        # up barrier on the wrong side of spot
        (BS, BarrierSpec("up-and-out", "continuous", 90.0, 100.0, True)),
        # down barrier on the wrong side of spot
        (BS, BarrierSpec("down-and-out", "continuous", 110.0, 100.0, True)),
    ],
)
def test_validate_inputs_rejects_bad_spec(bs, bar):
    with pytest.raises(ValueError):
        validate_inputs(bs, bar)


# --------------------------------------------------------------------------
# Variant enumeration
# --------------------------------------------------------------------------


def test_barrier_variants_defaults_and_coverage():
    variants = barrier_variants(BS)
    assert len(variants) == 8
    assert len({label for label, _ in variants}) == 8
    for label, spec in variants:
        assert spec.H == pytest.approx(1.1 * BS.S0 if spec.is_up else 0.9 * BS.S0)
        assert spec.K == pytest.approx(BS.S0)
        assert spec.monitor == "continuous"
        assert spec.rebate == 0.0
        assert label == (
            f"{spec.barrier_type} {'call' if spec.is_call else 'put'}"
        )


def test_barrier_variants_custom_kwargs():
    variants = barrier_variants(
        BS, H_up=130.0, H_down=70.0, K=105.0, rebate=3.0, monitor="discrete"
    )
    for _, spec in variants:
        assert spec.H == pytest.approx(130.0 if spec.is_up else 70.0)
        assert spec.K == pytest.approx(105.0)
        assert spec.rebate == pytest.approx(3.0)
        assert spec.monitor == "discrete"


def test_barrier_spec_direction_and_knock_properties():
    up_out = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    down_in = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False)
    assert up_out.is_up and not up_out.is_in
    assert not down_in.is_up and down_in.is_in


# --------------------------------------------------------------------------
# Random normals and GBM paths
# --------------------------------------------------------------------------


def test_gen_normals_shape_and_moment_matching():
    Z = gen_normals(1000, 5, seed=0)
    assert Z.shape == (1000, 5)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_gen_normals_reproducible_and_seed_dependent():
    assert np.array_equal(gen_normals(100, 4, seed=1), gen_normals(100, 4, seed=1))
    assert not np.array_equal(gen_normals(100, 4, seed=1), gen_normals(100, 4, seed=2))


def test_gen_normals_antithetic_structure():
    Z = gen_normals(100, 4, seed=0, antithetic=True, lhs=False, moment_match=False)
    assert np.allclose(Z[:50], -Z[50:])


def test_gen_normals_handles_odd_path_count():
    Z = gen_normals(101, 3, seed=0)
    assert Z.shape == (101, 3)
    assert np.all(np.isfinite(Z))


def test_gen_normals_without_antithetic_uses_exact_path_count():
    Z = gen_normals(64, 3, seed=0, antithetic=False, lhs=False, moment_match=False)
    assert Z.shape == (64, 3)
    # Latin-hypercube sampling alone also produces exactly ``n_paths`` rows
    L = gen_normals(64, 3, seed=0, antithetic=False, lhs=True, moment_match=False)
    assert L.shape == (64, 3)
    assert np.all(np.isfinite(L))


@pytest.mark.parametrize("n_paths,n_steps", [(0, 5), (5, 0), (-1, 5)])
def test_gen_normals_rejects_bad_sizes(n_paths, n_steps):
    with pytest.raises(ValueError):
        gen_normals(n_paths, n_steps)


def test_monte_carlo_paths_shape_and_terminal_moment():
    Z = gen_normals(20_000, 20, seed=0)
    S = monte_carlo_paths(BS, Z)
    assert S.shape == (20_000, 21)
    assert np.all(S[:, 0] == BS.S0)
    assert np.all(S > 0.0)
    forward = BS.S0 * np.exp((BS.r - BS.q) * BS.T)
    assert S[:, -1].mean() == pytest.approx(forward, rel=0.02)


# --------------------------------------------------------------------------
# Brownian-bridge crossing probability and first-passage detection
# --------------------------------------------------------------------------


def test_step_bridge_cross_prob_known_value():
    sigma, dt = 0.2, 1.0 / 252.0
    var = sigma**2 * dt
    x = np.log(100.0)
    h = np.log(110.0)
    expected = np.exp(-2.0 * (h - x) ** 2 / var)
    p = step_bridge_cross_prob(
        np.array([100.0]), np.array([100.0]), 110.0, sigma, dt, up=True
    )
    assert p[0] == pytest.approx(expected)


def test_step_bridge_cross_prob_zero_when_endpoint_breached():
    # an up barrier already breached by an endpoint has zero *incremental*
    # crossing probability by construction
    p = step_bridge_cross_prob(
        np.array([100.0]), np.array([111.0]), 110.0, 0.2, 1.0 / 252.0, up=True
    )
    assert p[0] == 0.0
    # and symmetrically for a down barrier
    p = step_bridge_cross_prob(
        np.array([100.0]), np.array([89.0]), 90.0, 0.2, 1.0 / 252.0, up=False
    )
    assert p[0] == 0.0


def test_step_bridge_cross_prob_in_unit_interval():
    p = step_bridge_cross_prob(
        np.array([100.0, 101.0]),
        np.array([101.0, 100.0]),
        110.0,
        0.2,
        1.0 / 52.0,
        up=True,
    )
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_barrier_hits_discrete_first_passage():
    bar = BarrierSpec("up-and-out", "discrete", 110.0, 100.0, True)
    S = np.array(
        [
            [100.0, 101.0, 111.0, 112.0],  # hit at step 2
            [100.0, 105.0, 108.0, 109.0],  # never hits
            [100.0, 110.0, 100.0, 100.0],  # hit at step 1 (S >= H)
        ]
    )
    hit, step = barrier_hits(S, BS, bar, seed=0)
    assert list(hit) == [True, False, True]
    assert list(step) == [2, 4, 1]


def test_barrier_hits_down_discrete():
    bar = BarrierSpec("down-and-out", "discrete", 85.0, 100.0, False)
    S = np.array([[100.0, 95.0, 84.0, 80.0], [100.0, 95.0, 90.0, 88.0]])
    hit, step = barrier_hits(S, BS, bar, seed=0)
    assert list(hit) == [True, False]
    assert list(step) == [2, 4]


def test_barrier_hits_continuous_detects_bridge_crossing():
    # endpoints stay below the barrier, so the discrete monitor never fires,
    # but the Brownian bridge between them almost surely crosses it
    bs = BSParams(S0=100.0, r=0.05, q=0.02, sigma=1.0, T=1.0)
    S = np.full((2000, 3), 105.0)
    S[:, 0] = 100.0
    discrete = BarrierSpec("up-and-out", "discrete", 110.0, 100.0, True)
    continuous = BarrierSpec("up-and-out", "continuous", 110.0, 100.0, True)

    hit_d, _ = barrier_hits(S, bs, discrete, seed=0)
    hit_c, _ = barrier_hits(S, bs, continuous, seed=0)
    assert not hit_d.any()
    assert hit_c.mean() > 0.5


def test_barrier_hits_chunking_matches_unchunked_results():
    bs = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.3, T=1.0)
    S = monte_carlo_paths(bs, gen_normals(500, 20, seed=11))
    bar = BarrierSpec("up-and-out", "continuous", 130.0, 100.0, True)
    full_hit, full_step = barrier_hits(S, bs, bar, seed=3)
    chunk_hit, chunk_step = barrier_hits(S, bs, bar, seed=3, chunk_size=37)
    np.testing.assert_array_equal(full_hit, chunk_hit)
    np.testing.assert_array_equal(full_step, chunk_step)


# --------------------------------------------------------------------------
# Closed-form rebate terms
# --------------------------------------------------------------------------


def test_barrier_rebate_terms_zero_for_zero_rebate_and_expiry():
    no_rebate = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=0.0)
    assert barrier_rebate_terms(BS, no_rebate) == (0.0, 0.0)

    expired = BSParams(S0=100.0, r=0.05, q=0.02, sigma=0.2, T=0.0)
    with_rebate = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=5.0)
    assert barrier_rebate_terms(expired, with_rebate) == (0.0, 0.0)


@pytest.mark.parametrize("direction,H", [("up", 120.0), ("down", 85.0)])
def test_barrier_rebate_terms_non_negative_and_linear_in_rebate(direction, H):
    one = BarrierSpec(f"{direction}-and-out", "continuous", H, 100.0, True, rebate=1.0)
    two = BarrierSpec(f"{direction}-and-out", "continuous", H, 100.0, True, rebate=2.0)
    E1, F1 = barrier_rebate_terms(BS, one)
    E2, F2 = barrier_rebate_terms(BS, two)
    assert E1 >= 0.0 and F1 >= 0.0
    assert E2 == pytest.approx(2.0 * E1)
    assert F2 == pytest.approx(2.0 * F1)


# --------------------------------------------------------------------------
# Tridiagonal solver and PDE grid
# --------------------------------------------------------------------------


def test_thomas_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 40
    a = rng.standard_normal(n)
    b = rng.standard_normal(n) + 5.0
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    x = thomas_solve(a, b, c, d)
    assert np.allclose(A @ x, d)


def test_thomas_factor_reused_for_multiple_rhs():
    rng = np.random.default_rng(1)
    n = 30
    a = rng.standard_normal(n)
    b = rng.standard_normal(n) + 5.0
    c = rng.standard_normal(n)
    factors = thomas_factor(a, b, c)
    for _ in range(3):
        d = rng.standard_normal(n)
        assert np.allclose(
            thomas_solve_factored(d, factors), thomas_solve(a, b, c, d)
        )


@pytest.mark.parametrize("barrier_type,H", [("up-and-out", 120.0), ("down-and-out", 85.0)])
def test_build_grid_snaps_barrier_to_node(barrier_type, H):
    bar = BarrierSpec(barrier_type, "continuous", H, 100.0, True)
    S, t = build_grid(BS, bar, M=100, N=50)
    assert S.shape == (101,)
    assert t.shape == (51,)
    assert np.any(np.isclose(S, H))
    assert np.all(np.diff(S) > 0.0)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(BS.T)


def test_build_grid_truncation_below_down_barrier():
    bar = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, True)
    S_trunc, _ = build_grid(BS, bar, M=100, N=50, truncate=True)
    S_full, _ = build_grid(BS, bar, M=100, N=50, truncate=False)
    assert S_trunc[0] == pytest.approx(0.5 * 85.0)
    assert S_full[0] < S_trunc[0]
    assert np.any(np.isclose(S_full, 85.0))


# --------------------------------------------------------------------------
# PDE time-stepping helpers
# --------------------------------------------------------------------------


def test_boundary_values_call_and_put():
    tau = 0.5
    Smin, Smax = 1e-8, 150.0
    call = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    left, right = boundary_values(BS, call, Smin, Smax, tau)
    assert left == 0.0
    assert right == pytest.approx(
        Smax * np.exp(-BS.q * tau) - 100.0 * np.exp(-BS.r * tau)
    )

    put = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False)
    left, right = boundary_values(BS, put, Smin, Smax, tau)
    assert right == 0.0
    assert left == pytest.approx(100.0 * np.exp(-BS.r * tau))


def test_enforce_barrier_dirichlet_pins_correct_side():
    S = np.linspace(0.0, 200.0, 11)
    down = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, True, rebate=3.0)
    V = np.ones_like(S)
    assert enforce_barrier_dirichlet(S, V, down) is V
    assert np.all(V[S <= 85.0] == 3.0)
    assert np.all(V[S > 85.0] == 1.0)

    up = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=4.0)
    V = np.ones_like(S)
    enforce_barrier_dirichlet(S, V, up)
    assert np.all(V[S >= 120.0] == 4.0)
    assert np.all(V[S < 120.0] == 1.0)


def test_theta_step_preserves_a_solution_of_the_degenerate_pde():
    # r = q = sigma = 0 leaves only the time derivative, so any profile that
    # already satisfies the Dirichlet barrier/boundary data is unchanged.
    bs = BSParams(S0=100.0, r=0.0, q=0.0, sigma=0.0, T=1.0)
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True, rebate=0.0)
    S, t = build_grid(bs, bar, M=50, N=10)
    V = enforce_barrier_dirichlet(S, vanilla_payoff(S, bar.K, bar.is_call), bar)
    out = theta_step(bs, bar, S, V, dt=t[1] - t[0], tnow=t[0], theta=1.0)
    np.testing.assert_allclose(out, V, atol=1e-10)


def test_theta_step_theta_changes_the_result():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    S, t = build_grid(BS, bar, M=100, N=100)
    V = vanilla_payoff(S, bar.K, bar.is_call)
    dt = t[1] - t[0]
    be = theta_step(BS, bar, S, V, dt=dt, tnow=t[0], theta=1.0)
    cn = theta_step(BS, bar, S, V, dt=dt, tnow=t[0], theta=0.5)
    assert be.shape == V.shape
    assert np.all(np.isfinite(be)) and np.all(np.isfinite(cn))
    assert not np.allclose(be, cn)


@pytest.mark.parametrize(
    "bar",
    [
        BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True),
        BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False),
    ],
)
def test_cn_and_rannacher_induction_reproduces_knock_out_price(bar):
    # Backward-induct with the public step helpers and compare with the
    # production solver, which sequences the same Rannacher + CN steps.
    M, N = 200, 200
    S, t = build_grid(BS, bar, M=M, N=N)
    V = enforce_barrier_dirichlet(S, vanilla_payoff(S, bar.K, bar.is_call), bar)
    for n in range(N, 0, -1):
        dt = t[n] - t[n - 1]
        if (N - n) < 1:
            V = rannacher_pair(BS, bar, S, V, dt, t[n])
        else:
            V = cn_step(BS, bar, S, V, dt, t[n])
    assert np.interp(BS.S0, S, V) == pytest.approx(
        pde_knock_out(BS, bar, M=M, N=N), abs=1e-12
    )


def test_pde_surface_matches_scalar_price():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    S, V = pde_surface(BS, bar, M=300, N=300)
    assert np.interp(BS.S0, S, V) == pytest.approx(
        price_barrier_pde(BS, bar, M=300, N=300), abs=1e-12
    )


def test_pde_surface_knock_in_matches_scalar_price():
    bar = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False)
    S, V = pde_surface(BS, bar, M=300, N=300)
    # The knock-in surface adds the analytic vanilla leg evaluated on the
    # grid, so reading it back at S0 carries the O(h^2) linear-interpolation
    # error of that leg.  The knock-out surface has no such leg and matches
    # the scalar pricer to machine precision.
    assert np.interp(BS.S0, S, V) == pytest.approx(
        price_barrier_pde(BS, bar, M=300, N=300), abs=5e-3
    )


def test_pde_surface_discrete_knock_in_rebate_unsupported():
    bar = BarrierSpec("up-and-in", "discrete", 120.0, 100.0, True, rebate=1.0)
    with pytest.raises(NotImplementedError):
        pde_surface(BS, bar, M=200, N=200, monitor_steps=52)


# --------------------------------------------------------------------------
# Knock-out PDE wrapper
# --------------------------------------------------------------------------


def test_pde_knock_out_matches_scalar_pricer():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    assert pde_knock_out(BS, bar, M=300, N=300) == pytest.approx(
        price_barrier_pde(BS, bar, M=300, N=300), abs=1e-12
    )


def test_pde_knock_out_matches_surface_interpolation():
    bar = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False)
    S, V = pde_surface(BS, bar, M=300, N=300)
    assert pde_knock_out(BS, bar, M=300, N=300) == pytest.approx(
        np.interp(BS.S0, S, V), abs=1e-12
    )


def test_pde_knock_out_matches_closed_form():
    bar = BarrierSpec("down-and-out", "continuous", 85.0, 100.0, False)
    assert pde_knock_out(BS, bar, M=500, N=500) == pytest.approx(
        price_barrier_closed_form(BS, bar), abs=5e-3
    )


def test_pde_knock_out_discrete_worth_more_than_continuous():
    cont = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    disc = BarrierSpec("up-and-out", "discrete", 120.0, 100.0, True)
    v_cont = pde_knock_out(BS, cont, M=400, N=400)
    v_disc = pde_knock_out(BS, disc, M=400, N=400, monitor_steps=52)
    assert v_disc > v_cont + 0.05


# --------------------------------------------------------------------------
# Barrier payoff samples and the control variate
# --------------------------------------------------------------------------


def test_price_barrier_unreachable_knock_out_equals_vanilla_samples():
    bar = BarrierSpec("up-and-out", "continuous", 1e6, 100.0, True)
    Z = gen_normals(5_000, 50, seed=0)
    X, Y = price_barrier(BS, bar, n_steps=50, n_paths=5_000, Z=Z)
    assert X.shape == (5_000,)
    assert Y.shape == (5_000,)
    # the barrier can never be reached, so every path pays the vanilla leg
    np.testing.assert_array_equal(X, Y)


def test_price_barrier_unreachable_knock_in_pays_discounted_rebate():
    bar = BarrierSpec("up-and-in", "continuous", 1e6, 100.0, True, rebate=7.0)
    X, _ = price_barrier(BS, bar, n_steps=20, n_paths=1_000, seed=0)
    np.testing.assert_allclose(X, np.exp(-BS.r * BS.T) * 7.0)


def test_price_barrier_explicit_normals_match_seeded_draw():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    Z = gen_normals(2_000, 30, seed=5)
    with_Z = price_barrier(BS, bar, n_steps=30, n_paths=2_000, seed=5, Z=Z)
    seeded = price_barrier(BS, bar, n_steps=30, n_paths=2_000, seed=5)
    np.testing.assert_array_equal(with_Z[0], seeded[0])
    np.testing.assert_array_equal(with_Z[1], seeded[1])


def test_price_barrier_agrees_with_mc_pricer_without_control_variate():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    X, _ = price_barrier(BS, bar, n_steps=100, n_paths=20_000, seed=3)
    price, _ = price_barrier_mc(
        BS, bar, n_paths=20_000, n_steps=100, seed=3, control_variate=False
    )
    assert X.mean() == pytest.approx(price, abs=1e-12)


def test_price_barrier_mean_matches_closed_form():
    bar = BarrierSpec("up-and-out", "continuous", 120.0, 100.0, True)
    X, _ = price_barrier(BS, bar, n_steps=200, n_paths=50_000, seed=1)
    assert X.mean() == pytest.approx(price_barrier_closed_form(BS, bar), abs=5e-2)


def test_price_barrier_cv_matches_pricer_and_reduces_variance():
    bar = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False)
    X, _ = price_barrier(BS, bar, n_steps=80, n_paths=20_000, seed=4)
    cv = price_barrier_cv(BS, bar, n_steps=80, n_paths=20_000, seed=4)
    price, _ = price_barrier_mc(
        BS, bar, n_paths=20_000, n_steps=80, seed=4, control_variate=True
    )
    assert cv.mean() == pytest.approx(price, abs=1e-12)
    assert len(cv) == 20_000
    assert np.var(cv) < np.var(X)


def test_price_barrier_mc_control_variate_reduces_standard_error():
    bar = BarrierSpec("down-and-in", "continuous", 85.0, 100.0, False)
    _, se_plain = price_barrier_mc(
        BS, bar, n_paths=20_000, n_steps=80, seed=7, control_variate=False
    )
    _, se_cv = price_barrier_mc(
        BS, bar, n_paths=20_000, n_steps=80, seed=7, control_variate=True
    )
    assert se_cv < se_plain

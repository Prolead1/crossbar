# crossbar

Single-barrier option pricing engine with two independent engines:

* **Monte Carlo** — Latin-hypercube stratification, antithetic variates,
  moment matching, a Black-Scholes control variate and a Brownian-bridge
  barrier-crossing correction.
* **Crank-Nicolson PDE** — Rannacher time stepping on a non-uniform grid,
  with the barrier snapped to a node.

Both engines are validated against Reiner-Rubinstein / Haug closed-form
prices, and delta/gamma profiles are provided across all eight
single-barrier variants. A screen-quote volatility surface can supply a
time-varying ``sigma(t)`` to the Monte Carlo paths and a crude local-vol
``sigma(S, t)`` grid to the PDE solver.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/Prolead1/crossbar.git
cd crossbar
uv venv
uv pip install -e ".[dev]"        # tests + coverage
uv pip install -e ".[dev,plot]"   # ... plus matplotlib for `crossbar plot`
```

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"
```

Check it works:

```bash
crossbar --version          # crossbar 0.1.0
python -m crossbar --help   # same CLI, no console script required
```

## Quick start

The CLI is a thin, scriptable front end over the public functions:

| Command | What it does |
|---------|--------------|
| `price` | price a vanilla (omit `-H`) or a single-barrier option |
| `variants` | price all eight single-barrier variants at once |
| `greeks` | price/delta/gamma profile across a spot grid |
| `surface` | inspect the delta-quoted volatility surface |
| `plot` | render the volatility surface to an image |

Every command accepts `--json` for machine-readable output; run
`crossbar <command> --help` for the full option list.

All market conventions used throughout:

| Symbol | Meaning | Units |
|--------|---------|-------|
| `S` / `S0` | spot price | price |
| `K` | strike | price |
| `H` | barrier level | price |
| `r` | risk-free rate | continuously compounded, decimal |
| `q` | dividend / carry yield | continuously compounded, decimal |
| `v` / `sigma` | volatility | decimal (`0.07` = 7 vol points) |
| `T` | time to maturity | years |

Deltas are **unadjusted spot** deltas (negative for puts), and the
closed-form engine assumes **continuous** monitoring.

### Price a vanilla option

`price` without a barrier prices a European option with Black-Scholes:

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 -K 1.10
# vanilla call: strike=1.1 price=0.024153

crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 -K 1.10 --put
# vanilla put: strike=1.1 price=0.018749
```

Omit `-K` to use the spot as the strike (ATM).

### Price a single-barrier option

Add `-H` and a `--type`; the four variants are `up-and-out`,
`up-and-in`, `down-and-out` and `down-and-in`. The engine is selected
with `--engine` (`pde` by default):

```bash
# Crank-Nicolson PDE (default)
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type up-and-out

# Reiner-Rubinstein / Haug closed form (continuous monitoring only)
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type up-and-out --engine analytic
# price (analytic) = 0.015186

# Monte Carlo with control variate
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type up-and-out --engine mc --paths 200000 --steps 252 --seed 1
```

Useful options:

* `--put` prices a put instead of the default call.
* `--monitor discrete` observes the barrier on the time grid only.
* `--rebate 0.02` adds a cash rebate (paid at the first hit for a
  knock-out, at maturity for a knock-in).
* `--M`/`--N` tune the PDE mesh (default 500×500); `--paths`/`--steps`
  tune the Monte Carlo size.
* `--no-control-variate` disables the Black-Scholes control variate.

> The `analytic` engine only supports continuous monitoring; combining
> it with `--monitor discrete` exits with an error.

### Price all eight variants at once

`variants` sweeps up/down × in/out × call/put around the spot, with
barriers defaulting to `1.1 * S0` and `0.9 * S0`:

```bash
crossbar variants -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 --M 300 --N 300
```

```
variant                        price
up-and-in call              0.012589
up-and-in put               0.002256
up-and-out call             0.011564
up-and-out put              0.016493
down-and-in call            0.000182
down-and-in put             0.004950
down-and-out call           0.023971
down-and-out put            0.013799
```

Override `--up-barrier`, `--down-barrier`, `-K` and `--rebate` as
needed, and reuse `--engine` just like `price`.

### Delta and gamma profiles

`greeks` returns price, delta and gamma across a spot grid. The PDE
engine solves the value surface once and reads the derivatives off it;
the Monte Carlo engine bumps and revalues:

```bash
crossbar greeks -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --engine pde --M 300 --N 300 --spots 1.05,1.10,1.15
```

```
      spot         price         delta         gamma
    1.0500      0.004742      0.143369      2.742968
    1.1000      0.014421      0.184933     -2.693399
    1.1500      0.016522     -0.157404     -8.929600
```

Omit `--spots` for five points around the spot. For `--engine mc`, pass
a fixed `--seed` and a `--bump` so the finite differences share random
numbers and are not swamped by simulation noise.

### Work with a volatility surface

A surface is a JSON file of screen-style quotes, keyed by tenor label
(`0N`, `1W`, `2W`, `1M`, `2M`, `3M`, `6M`, `9M`). Each tenor carries the
five quote fields as `[bid, ask]` **in vol points**:

```json
{
  "1M": {
    "atm":   [6.20, 6.60],
    "rr_25": [-0.42, -0.22],
    "bf_25": [0.18, 0.38],
    "rr_10": [-1.00, -0.70],
    "bf_10": [0.38, 0.68]
  },
  "3M": { "...": "..." }
}
```

Inspect the reconstructed term structure (uses the built-in EURUSD
example when `--quotes` is omitted):

```bash
crossbar surface --quotes quotes.json
#        T       10p       25p       atm       25c       10c
#   0.0027   0.06875   0.06375   0.06000   0.06025   0.06025
#   0.0192   0.07020   0.06495   0.06100   0.06145   0.06120
#   ...
```

Look up a vol from a strike, or a strike from a delta:

```bash
crossbar surface -S 1.10 --strike 1.10 --maturity 0.5
# sigma(K=1.1, T=0.5) = 0.076207

crossbar surface -S 1.10 --delta 0.25 --maturity 0.5
# strike(delta=+0.250, T=0.5) = 1.144168
```

Feed the surface to the PDE engine as a crude local-volatility grid:

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type up-and-out --engine pde --surface quotes.json
```

Render it (requires the `plot` extra):

```bash
crossbar plot -S 1.10 --quotes quotes.json --out vol.png \
  --strikes 61 --maturities 41 --dpi 150
```

### Scripting with `--json`

Every command accepts `--json`; errors go to stderr and exit with code
`2`.

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type up-and-out --engine pde --M 300 --N 300 --json
```

```json
{
  "engine": "pde",
  "instrument": "barrier",
  "price": 0.014421308033271484,
  "surface": null
}
```

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 -K 1.10 --json
```

```json
{
  "instrument": "vanilla",
  "is_call": true,
  "price": 0.024153442087763044,
  "strike": 1.1
}
```

### Python API

The CLI is a thin wrapper over the same functions. Build the market and
contract objects once and reuse them:

```python
from crossbar import BSParams, BarrierSpec

bs = BSParams(S0=1.10, r=0.04, q=0.03, sigma=0.07, T=0.5)
bar = BarrierSpec("up-and-out", "continuous", H=1.20, K=1.10, is_call=True)
```

**Pricing** — all three engines take the same objects:

```python
from crossbar import (
    price_barrier_closed_form,
    price_barrier_mc,
    price_barrier_pde,
)

price_barrier_closed_form(bs, bar)                               # 0.015186
price_barrier_mc(bs, bar, n_paths=100_000, n_steps=252, seed=1)  # (0.015256, 6.27e-05)
price_barrier_pde(bs, bar, M=300, N=300)                         # 0.014421
```

**Risk profiles** — one surface solve for the whole spot grid, or all
eight variants at once:

```python
import numpy as np
from crossbar import pde_risk_profile, greeks_by_variant_pde

spots = np.array([1.05, 1.10, 1.15])
prices, deltas, gammas = pde_risk_profile(bs, bar, spots, M=300, N=300)

profiles = greeks_by_variant_pde(bs, spots, M=200, N=200)  # dict of 8 labels
```

**Volatility surface** — build it from quotes, turn the term structure
into instantaneous forward vols for the Monte Carlo paths, or pass it to
the PDE:

```python
from crossbar import (
    EXAMPLE_QUOTES,
    VolSurface,
    gen_normals,
    monte_carlo_paths,
    stepwise_sigmas_from_surface,
)

surface = VolSurface.from_quotes(EXAMPLE_QUOTES)   # or your own dict
surface.interp_sigma(0.5, 0.10)                    # 0.07985

# sigma(t) whose cumulative variance matches the quoted term structure
steps = stepwise_sigmas_from_surface(surface, bs.T, 64, delta=0.0)

Z = gen_normals(10_000, 64, seed=0)
paths = monte_carlo_paths(bs, Z, sigma=steps)       # shape (10000, 65)

# local-vol PDE
price_barrier_pde(bs, bar, M=200, N=200, surface=surface)   # 0.012969
```

### Run the tests

```bash
pytest                                        # full suite
coverage run -m pytest && coverage report -m  # with 100% coverage
```

## Where next

* `crossbar <command> --help` — full option list for any subcommand.
* `src/crossbar/` — `params.py` (inputs), `analytic.py` (closed form),
  `monte_carlo.py`, `pde.py`, `greeks.py`, `vol_surface.py`, `cli.py`.

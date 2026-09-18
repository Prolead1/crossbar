# crossbar

Single-barrier option pricing engine with two independent numerical
engines alongside the closed form:

* **Monte Carlo** — Latin-hypercube stratification, antithetic variates,
  moment matching, a Black-Scholes control variate and a Brownian-bridge
  barrier-crossing correction.
* **Crank-Nicolson PDE** — Rannacher time stepping on a non-uniform grid,
  with the barrier snapped to a node.

The engines are validated against Reiner-Rubinstein / Haug closed-form
prices, and delta/gamma profiles are provided across all eight
single-barrier variants. A screen-quote volatility surface can be fed to
the PDE solver as a first-order local-volatility grid.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/Prolead1/crossbar.git
cd crossbar
uv venv
uv pip install -e ".[dev]"        # tests + coverage
uv pip install -e ".[dev,plot]"   # ... plus matplotlib for the surface helpers
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

`crossbar price` is the only command. Run it bare and it walks through
the trade one question at a time, then prices the barrier with **all
three engines**, printing a vanilla benchmark and an aligned comparison:

```bash
$ crossbar price --paths 20000 --steps 50 --seed 1
Spot S0: 1.10
Vol (decimal or surface JSON): 0.07
Maturity T (years): 0.5
Risk-free rate r [0.0]: 0.04
Carry/dividend yield q [0.0]: 0.03
Barrier type (up-and-out/up-and-in/down-and-out/down-and-in) [uo]:
Monitoring (continuous/discrete) [c]:
Barrier level H: 1.20
Rebate [0.0]:
Strike K (blank = spot):
Option type (call/put) [c]:
contract  : up-and-out call, continuous, K=1.1, H=1.2, rebate=0
market    : S0=1.1, r=0.04, q=0.03, T=0.5
vol       : 0.070000 (constant)
benchmark : vanilla = 0.024153
mc        : paths=20000, steps=50, seed=1, control_variate=True
pde       : M=500, N=500, rannacher=1

engine             price         delta         gamma     std_error
analytic        0.015186      0.193211     -3.213591             -
mc              0.014911      0.182664     -0.793350      0.000141
pde             0.014965      0.188408     -3.063749             -
```

Press **Enter** to take the shown default. Input is validated as you go:
an unparseable number, an unknown choice, a missing required value or a
domain violation (e.g. `S0 = 0`) aborts immediately with
`crossbar: error: ...` and exit code `2` instead of re-prompting.

The **PDE mesh** (`M`, `N`, Rannacher pairs) is fixed at the library
defaults and shown in the output, but is deliberately *not* a CLI input.
Call `price_barrier_pde(..., M=..., N=..., rannacher_pairs=...)` from
Python for a custom mesh.

### Non-interactive

Supply the matching options to skip the prompts entirely — useful for
scripts and reproducible runs. `price` always prices a barrier; the
Monte Carlo controls are flags only (they are never prompted):

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type uo --paths 20000 --steps 50 --seed 1
```

| Option | Meaning |
|--------|---------|
| `-S` | spot price |
| `-v` | volatility: a decimal constant **or** a path to a surface JSON (exactly one) |
| `-T` | time to maturity in years |
| `-r`, `-q` | risk-free rate, dividend/carry yield (default 0) |
| `-H`, `--type`, `--monitor`, `--rebate` | barrier level; type `uo`/`ui`/`do`/`di`; monitor `c`/`d`; cash rebate |
| `-K` | strike (default: spot) |
| `--call` / `--put` | option type (default call) |
| `--paths`, `--steps`, `--seed`, `--no-control-variate` | Monte Carlo controls |
| `--non-interactive` | never prompt; error on missing required options |
| `--json` | machine-readable output |

A `vanilla` benchmark — the closed-form European for the same strike and
call/put (flat `sigma`) — is printed before the engines run, as the
no-barrier reference. Which barrier engines run depends on the contract:

* **continuous barrier, constant vol** → `analytic`, `mc` and `pde`;
* **discrete barrier** → `mc` and `pde`, plus the continuous-monitoring
  `analytic` value for reference;
* **vol surface** → `analytic` (at the surface ATM vol) and `pde`; `mc`
  is skipped.

Skipped engines appear as `n/a` with a short reason under the table.

### Volatility surface

Instead of a constant, `-v` can point at a JSON file of screen-style
quotes, keyed by tenor label (`0N`, `1W`, `2W`, `1M`, `2M`, `3M`, `6M`,
`9M`). Each tenor carries the five quote fields as `[bid, ask]` **in vol
points**. A complete sample lives at
[`examples/vol_surface.json`](examples/vol_surface.json):

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

The PDE engine then uses a `sigma(S, t)` local-volatility grid built
from the quotes, while the closed form is priced at the surface ATM vol
so you still get a constant-vol reference:

```bash
crossbar price -S 1.10 -v examples/vol_surface.json -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type uo
```

```
contract  : up-and-out call, continuous, K=1.1, H=1.2, rebate=0
market    : S0=1.1, r=0.04, q=0.03, T=0.5
vol       : 0.076000 (ATM from surface)
benchmark : vanilla = 0.025974
quotes    : examples/vol_surface.json
pde       : M=500, N=500, rannacher=1

engine             price         delta         gamma     std_error
analytic        0.014261      0.143177     -3.434666             -
mc                   n/a             -             -             -
pde             0.014753      0.166769     -2.899862             -
note: mc skipped (Monte Carlo does not use a surface yet)
```

> This is a first-order "implied-vol-as-local-vol" approximation, not a
> calibrated Dupire local-vol model, and it currently drives the PDE
> diffusion only. Monte Carlo is therefore reported as `n/a`; the closed
> form uses the surface ATM vol as a constant-vol reference.

### Output and scripting

Human-readable results begin with a self-describing context block, so the
interpreted inputs, engine options and method are always visible:

```
contract  : up-and-out call, continuous, K=1.1, H=1.2, rebate=0
market    : S0=1.1, r=0.04, q=0.03, T=0.5
vol       : 0.070000 (constant)
benchmark : vanilla = 0.024153
mc        : paths=20000, steps=50, seed=1, control_variate=True
pde       : M=500, N=500, rannacher=1

engine             price         delta         gamma     std_error
analytic        0.015186      0.193211     -3.213591             -
mc              0.014911      0.182664     -0.793350      0.000141
pde             0.014965      0.188408     -3.063749             -
```

Prices use six decimal places, deltas and gammas are spot derivatives,
Monte Carlo estimates carry their standard error, and engine skips are
explained below the table. `--json` emits the same interpretation as structured
`contract`, `market`, `vanilla_benchmark`, `vol_source`, `mc_options`,
`pde_options` and `prices` fields at full precision; errors go to stderr
and exit with code `2`.

```bash
crossbar price -S 1.10 -v 0.07 -T 0.5 -r 0.04 -q 0.03 \
  -H 1.20 --type uo --paths 20000 --steps 50 --json
```

```json
{
  "contract": {
    "barrier": 1.2,
    "call": true,
    "monitor": "continuous",
    "rebate": 0.0,
    "strike": 1.1,
    "type": "up-and-out"
  },
  "market": {
    "carry": 0.03,
    "maturity": 0.5,
    "rate": 0.04,
    "spot": 1.1,
    "vol": 0.07
  },
  "mc_options": {
    "control_variate": true,
    "paths": 20000,
    "seed": 0,
    "steps": 50
  },
  "pde_options": {
    "M": 500,
    "N": 500,
    "rannacher": 1
  },
  "prices": [
    {
      "delta": 0.19321143105491587,
      "engine": "analytic",
      "gamma": -3.21359133874469,
      "price": 0.01518596645832257
    },
    {
      "delta": 0.1972346503657604,
      "engine": "mc",
      "gamma": -1.8364640210161327,
      "price": 0.015241443949095055,
      "std_error": 0.00014107551621228483
    },
    {
      "delta": 0.18840762863124422,
      "engine": "pde",
      "gamma": -3.063748887496427,
      "price": 0.014964531846615477
    }
  ],
  "quotes": null,
  "vanilla_benchmark": 0.024153442087763044,
  "vol_source": "constant"
}
```

## Python API

The CLI exposes only `price`; the rest of the library — including custom
PDE meshes and the risk-profile helpers — is available from Python. Build
the market and contract objects once and reuse them:

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
price_barrier_pde(bs, bar, M=500, N=500, rannacher_pairs=2)      # custom mesh
```

**Risk profiles** — one surface solve for the whole spot grid, or all
eight variants at once:

```python
import numpy as np
from crossbar import (
    greeks_by_variant_pde,
    mc_risk_profile,
    pde_risk_profile,
)

spots = np.array([1.05, 1.10, 1.15])
prices, deltas, gammas = pde_risk_profile(bs, bar, spots, M=300, N=300)

profiles = greeks_by_variant_pde(bs, spots, M=200, N=200)  # dict of 8 labels

# Monte Carlo Greeks: one simulation, reused across every spot and bump
prices, deltas, gammas = mc_risk_profile(
    bs, bar, spots, n_paths=100_000, n_steps=252, seed=1
)
```

The generic :func:`risk_profile` bumps any scalar pricer three times
per spot. For Monte Carlo that would redraw the paths every time, so
:func:`mc_risk_profile` generates one normal set and reuses the same
simulated log-returns for every bump -- the common-random-numbers
requirement for a clean finite difference -- making the profile roughly
1.5x faster with bit-identical output.

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

## Run the tests

```bash
pytest                                        # full suite
coverage run -m pytest && coverage report -m  # with 100% coverage
```

## Where next

* `crossbar price --help` — full option list.
* `src/crossbar/` — `params.py` (inputs), `analytic.py` (closed form),
  `monte_carlo.py`, `pde.py`, `greeks.py`, `vol_surface.py`, `cli.py`.

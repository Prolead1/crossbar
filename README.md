# crossbar

Single-barrier option pricing engine with two independent engines:

* **Monte Carlo** — Latin-hypercube stratification, antithetic variates,
  moment matching, a Black-Scholes control variate and a Brownian-bridge
  barrier-crossing correction.
* **Crank-Nicolson PDE** — Rannacher time stepping on a non-uniform grid,
  with the barrier snapped to a node.

Both engines are validated against Reiner-Rubinstein / Haug closed-form
prices, and delta/gamma profiles are provided across all eight
single-barrier variants.

## Command line

Installing the package exposes a `crossbar` console script (equivalently
`python -m crossbar`) with subcommands for the vanilla formula, the two
barrier engines, the risk profile, and the delta-quoted volatility
surface:

```bash
crossbar vanilla  -S 1.10 -v 0.07 -T 0.5 -K 1.10
crossbar price    -S 1.10 -v 0.07 -T 0.5 -H 1.20 --type up-and-out
crossbar variants -S 1.10 -v 0.07 -T 0.5 --engine pde
crossbar greeks   -S 1.10 -v 0.07 -T 0.5 -H 1.20 --engine pde
crossbar surface  --delta 0.25 --maturity 0.5
crossbar plot     -S 1.10 --out vol.png
```

Every command accepts `--json` for machine-readable output. Run
`crossbar <command> --help` for the full option list.

## Install

```bash
uv pip install -e ".[dev]"
```

## Test

```bash
pytest
```

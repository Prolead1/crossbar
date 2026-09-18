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

## Install

```bash
uv pip install -e ".[dev]"
```

## Test

```bash
pytest
```

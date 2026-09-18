"""Tridiagonal solvers for the implicit Crank-Nicolson steps.

The scheme assembles one tridiagonal system per time step.  The matrices
are time-invariant for a constant step, so :func:`thomas_factor` caches
the forward-elimination ratios and :func:`thomas_solve_factored` reuses
them; :func:`thomas_solve` is the plain one-shot solver.
"""

from __future__ import annotations

import numpy as np

__all__ = ["thomas_solve", "thomas_factor", "thomas_solve_factored"]


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Solve a tridiagonal system with the Thomas algorithm."""
    n = len(b)
    cp = np.empty(n - 1)
    dp = np.empty(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den

    den = b[n - 1] - a[n - 1] * cp[n - 2]
    dp[n - 1] = (d[n - 1] - a[n - 1] * dp[n - 2]) / den

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def thomas_factor(a, b, c):
    """Pre-factor a tridiagonal matrix for repeated Thomas solves.

    The forward-elimination ratios depend only on the matrix, so a
    stencil reused across many right-hand sides (the time-invariant
    Crank-Nicolson/BE matrices here) can be factorised once.  The factors
    are plain Python lists so the sweeps in
    :func:`thomas_solve_factored` avoid per-element numpy overhead.
    """
    a = np.asarray(a, dtype=float).tolist()
    b = np.asarray(b, dtype=float).tolist()
    c = np.asarray(c, dtype=float).tolist()
    n = len(b)
    cp = [0.0] * n
    den = [0.0] * n
    den[0] = b[0]
    cp[0] = c[0] / b[0]
    for i in range(1, n - 1):
        den[i] = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den[i]
    den[n - 1] = b[n - 1] - a[n - 1] * cp[n - 2]
    return a, cp, den


def thomas_solve_factored(d, factors):
    """Solve a system against a matrix from :func:`thomas_factor`."""
    a, cp, den = factors
    d = d.tolist()
    n = len(d)
    dp = [0.0] * n
    dp[0] = d[0] / den[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den[i]
    x = [0.0] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.asarray(x)

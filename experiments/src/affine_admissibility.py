"""Exact finite admissibility test for Q=a+b*x+c*y and I_E=1-x+E.

Requires the optional ``theory`` dependencies. Rational inputs are evaluated
symbolically, including algebraic stationary points; no sampling or tolerance
is used to decide admissibility. This does not change the experimental grid.
"""

from __future__ import annotations

import sympy as sp

T = sp.Symbol("t", real=True)
Interval = tuple[sp.Expr, sp.Expr] | None


def nonnegative_on(poly: sp.Expr, interval: Interval = (sp.S.Zero, sp.S.One)) -> bool:
    """Check endpoints and all interior stationary points of a degree <=3 polynomial."""
    if interval is None:
        return True
    left, right = map(sp.sympify, interval)
    if left > right:
        return True
    p = sp.Poly(poly, T)
    if p.degree() > 3:
        raise ValueError("Only polynomials of degree at most three are supported")
    points = [left, right]
    derivative = p.diff()
    if not derivative.is_zero:
        points.extend(t for t in sp.real_roots(derivative) if left < t < right)
    return all(bool(sp.simplify(p.eval(t)) >= 0) for t in points)


def vertex_interval(a: sp.Rational, b: sp.Rational, c: sp.Rational) -> Interval:
    """Return J(a,b,c), including empty and singleton intervals."""
    if b == 0:
        raise ValueError("The vertex interval is used only when b is nonzero")
    low, high = min(b, -2 * b), max(b, -2 * b)
    if c == 0:
        return (sp.S.Zero, sp.S.One) if low <= a <= high else None
    ends = sorted(((low - a) / c, (high - a) / c))
    left, right = max(sp.S.Zero, ends[0]), min(sp.S.One, ends[1])
    return (left, right) if left <= right else None


def coefficient_tests(a: sp.Rational, b: sp.Rational, c: sp.Rational) -> list[tuple[str, sp.Expr, Interval]]:
    """Construct the six boundary and up to two conditional vertex tests."""
    a, b, c = map(sp.Rational, (a, b, c))
    boundary = [
        1 + (1 - T) * (a + c * T),
        1 - (1 - T) * (a + b + c * T),
        1 - T * (a + c * T),
        1 + T * (a + b + c * T),
        1 + (1 - T) * (a + b * T),
        1 - (1 - T) * (a + c + b * T),
    ]
    tests = [(f"P{i}", p, (sp.S.Zero, sp.S.One)) for i, p in enumerate(boundary, 1)]
    d = b**2 + b * (a + c * T) + (a + c * T)**2
    if b < 0:
        tests.append(("L(a,b,c)", -3 * b - (1 - T) * d, vertex_interval(a, b, c)))
    elif b > 0:
        tests.append(("U(a,b,c)", 3 * b - T * d, vertex_interval(a, b, c)))
    if c < 0:
        d_swap = c**2 + c * (a + b * T) + (a + b * T)**2
        tests.append(("L(a,c,b)", -3 * c - (1 - T) * d_swap, vertex_interval(a, c, b)))
    return tests


def is_admissible(a: sp.Rational, b: sp.Rational, c: sp.Rational) -> bool:
    """Decide admissibility for exact rational coefficients (use strings for decimals)."""
    return all(nonnegative_on(p, interval) for _, p, interval in coefficient_tests(a, b, c))

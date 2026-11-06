"""Exact finite admissibility test for Q=a+b*x+c*y and I_E=1-x+E.

Requires the optional ``theory`` dependencies. Rational inputs are evaluated
symbolically, including algebraic stationary points; no sampling or tolerance
is used to decide admissibility. This does not change the experimental grid.
"""

from __future__ import annotations

import sympy as sp

T = sp.Symbol("t", real=True)
Interval = tuple[sp.Expr, sp.Expr] | None


def polynomial_report(poly: sp.Expr, interval: Interval = (sp.S.Zero, sp.S.One)) -> dict:
    """Return exact, JSON-ready endpoint/stationary-point evidence."""
    p = sp.Poly(poly, T)
    if p.degree() > 3:
        raise ValueError("Only polynomials of degree at most three are supported")
    result = dict(polynomial=str(p.as_expr()), interval=None, checks=[], passed=True)
    if interval is None:
        result['status'] = 'empty interval'
        return result
    left, right = map(sp.sympify, interval)
    if left > right:
        result['status'] = 'empty interval'
        return result
    result['interval'] = [str(left), str(right)]
    result['status'] = 'singleton' if left == right else 'nonempty interval'
    points = list(dict.fromkeys([left, right]))
    derivative = p.diff()
    if not derivative.is_zero:
        points.extend(t for t in sp.real_roots(derivative) if left < t < right)
    for t in dict.fromkeys(points):
        value = sp.simplify(p.eval(t))
        result['checks'].append(dict(point=str(t), value=str(value),
            kind='endpoint' if t in (left, right) else 'stationary', nonnegative=bool(value >= 0)))
    result['passed'] = all(check['nonnegative'] for check in result['checks'])
    return result


def nonnegative_on(poly: sp.Expr, interval: Interval = (sp.S.Zero, sp.S.One)) -> bool:
    """Check endpoints and distinct interior stationary points, without tolerances."""
    return polynomial_report(poly, interval)['passed']


def exact_coefficients(a, b, c) -> tuple[sp.Rational, sp.Rational, sp.Rational]:
    """Accept integers, rationals and rational strings; reject inexact float input."""
    if any(isinstance(v, (float, sp.Float)) for v in (a, b, c)):
        raise ValueError('Use fraction or decimal strings instead of floating-point coefficients')
    return tuple(sp.Rational(v) for v in (a, b, c))


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
    a, b, c = exact_coefficients(a, b, c)
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


def admissibility_report(a, b, c) -> dict:
    """Explain every applicable test; failure never short-circuits the report."""
    coeffs = exact_coefficients(a, b, c)
    tests = [dict(name=name, **polynomial_report(p, interval))
             for name, p, interval in coefficient_tests(*coeffs)]
    return dict(coefficients=list(map(str, coeffs)), arithmetic='exact rational/algebraic',
                admissible=all(t['passed'] for t in tests), tests=tests,
                failed_tests=[t['name'] for t in tests if not t['passed']])

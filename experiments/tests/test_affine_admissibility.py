"""Symbolic checks supplement, but do not replace, the manuscript proof."""

import itertools

import numpy as np
import pytest
sp = pytest.importorskip("sympy", reason="Install the optional theory dependencies")

from experiments.src.affine_admissibility import (
    T, coefficient_tests, is_admissible, nonnegative_on, vertex_interval,
)


def test_symbolic_derivative_and_vertex_identities():
    x, y, a, b, c = sp.symbols("x y a b c", real=True)
    e = x*y + x*y*(1-x)*(1-y)*(a+b*x+c*y)
    f = (1-2*x)*(a+c*y) + b*(2*x-3*x*x)
    f_swap = (1-2*y)*(a+b*x) + c*(2*y-3*y*y)
    assert sp.expand(sp.diff(e, x) - y*(1+(1-y)*f)) == 0
    assert sp.expand(1-sp.diff(e, x) - (1-y)*(1-y*f)) == 0
    assert sp.expand(sp.diff(e, y) - x*(1+(1-x)*f_swap)) == 0
    vertex = (b-a-c*y)/(3*b)
    d = b*b+b*(a+c*y)+(a+c*y)**2
    assert sp.simplify(f.subs(x, vertex)-d/(3*b)) == 0
    assert sp.expand(e-e.xreplace({x:y, y:x})-(b-c)*x*y*(1-x)*(1-y)*(x-y)) == 0


@pytest.mark.parametrize("p,interval,expected", [
    (sp.S.Zero, (0, 1), True),
    (-sp.S.One, None, True),
    (T-sp.Rational(1,2), (1, 1), True),
    ((T-sp.Rational(1,2))**2, (0, 1), True),
    ((T-sp.Rational(1,2))**2-sp.Rational(1,100), (0, 1), False),
    ((T-sp.Rational(1,2))**3, (0, 1), False),
    (T**3-T+sp.Rational(1,3), (0, 1), False),
    (T**3-T+sp.Rational(2,5), (0, 1), True),
])
def test_finite_polynomial_test(p, interval, expected):
    assert nonnegative_on(p, interval) is expected


def test_interval_degeneracies():
    r = sp.Rational
    assert vertex_interval(r(0), r(1), r(0)) == (0, 1)
    assert vertex_interval(r(2), r(1), r(0)) is None
    assert vertex_interval(r(2), r(1), r(-1)) == (1, 1)
    assert vertex_interval(r(-3), r(1), r(2)) == (r(1,2), 1)
    with pytest.raises(ValueError):
        vertex_interval(r(0), r(0), r(1))


def test_exact_nonnegative_subclass_and_constant_slice():
    grid = [sp.Rational(i, 4) for i in range(7)]
    for alpha, beta in itertools.product(grid, repeat=2):
        assert is_admissible(0, -alpha, -beta) == (alpha+beta <= 1)
    for a in [sp.Rational(i, 4) for i in range(-8, 9)]:
        assert is_admissible(a, 0, 0) == (-1 <= a <= 1)
    assert is_admissible(sp.Rational(1,2), sp.Rational(1,4), 0)


def test_accepted_coefficients_satisfy_independent_sampled_derivatives():
    # Direct derivatives of the original E, independent of the reduced tests.
    x, y = np.meshgrid(np.linspace(0, 1, 71), np.linspace(0, 1, 71))
    big_b = x*y*(1-x)*(1-y)
    bx = y*(1-y)*(1-2*x)
    by = x*(1-x)*(1-2*y)
    grid = [sp.Rational(i, 2) for i in range(-3, 4)]
    accepted = 0
    for coeffs in itertools.product(grid, repeat=3):
        if not is_admissible(*coeffs):
            continue
        accepted += 1
        a, b, c = map(float, coeffs)
        q = a+b*x+c*y
        ex, ey = y+bx*q+big_b*b, x+by*q+big_b*c
        assert ex.min() >= -1e-12 and ex.max() <= 1+1e-12
        assert ey.min() >= -1e-12
    assert accepted > 0


def test_vertex_conditions_are_not_omitted():
    # All nonzero sign combinations exercise the appropriate conditional tests.
    for b, c in itertools.product((-1, 0, 1), repeat=2):
        tests = coefficient_tests(0, b, c)
        assert len(tests) == 6 + (b != 0) + (c < 0)


@pytest.mark.parametrize("coeffs,failed", [
    ((sp.Rational(-1,2), sp.Rational(3,2), -2), "U(a,b,c)"),
    ((sp.Rational(5,2), sp.Rational(-3,2), -2), "L(a,b,c)"),
])
def test_boundary_checks_alone_are_insufficient(coeffs, failed):
    tests = coefficient_tests(*coeffs)
    assert all(nonnegative_on(p, j) for _, p, j in tests[:6])
    assert failed in [name for name, p, j in tests[6:] if not nonnegative_on(p, j)]
    assert not is_admissible(*coeffs)

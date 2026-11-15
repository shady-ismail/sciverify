"""Unit tests for general AST-lifted 80-digit Decimal shadow and condition-number filter."""

import numpy as np
from sciverify.decimal_shadow import audit_componentwise_cancellation


def test_unstable_algebraic_cancellation_detected() -> None:
    src = (
        "def solve_quad_unstable(a, b, c):\n"
        "    d = np.sqrt(b * b - 4.0 * a * c)\n"
        "    return (-b - d) / (2.0 * a), (-b + d) / (2.0 * a)\n"
    )
    ns = {"np": np}
    exec(src, ns)
    res = audit_componentwise_cancellation(
        fp64_fn=ns["solve_quad_unstable"],
        impl_source=src,
        entry_name="solve_quad_unstable",
        candidate_tuples=[(1.0, float(2 ** 29), 1.0)],
    )
    assert res["supported"] is True
    assert res["passed"] is False
    assert res["component_index"] == 1
    assert res["condition_number_kappa"] < 10.0
    assert res["relative_error"] > 0.99


def test_stable_citardauq_formulation_passes() -> None:
    src = (
        "def solve_quad_stable(a, b, c):\n"
        "    d = np.sqrt(b * b - 4.0 * a * c)\n"
        "    q = -0.5 * (b + d) if b >= 0 else -0.5 * (b - d)\n"
        "    return q / a, c / q\n"
    )
    ns = {"np": np}
    exec(src, ns)
    res = audit_componentwise_cancellation(
        fp64_fn=ns["solve_quad_stable"],
        impl_source=src,
        entry_name="solve_quad_stable",
        candidate_tuples=[(1.0, float(2 ** 29), 1.0)],
    )
    assert res["supported"] is True
    assert res["passed"] is True


def test_inherently_ill_conditioned_problem_not_false_positive() -> None:
    src = (
        "def near_equal_diff(x, y):\n"
        "    return x - y\n"
    )
    ns = {}
    exec(src, ns)
    # Near x == y with 1 ULP perturbation, kappa_true ~ 1e15 >> 1e4 -> problem is ill-conditioned, not an algorithmic rewrite bug
    x0 = 1e9
    y0 = 1e9 + 1.1920928955078125e-07
    res = audit_componentwise_cancellation(
        fp64_fn=ns["near_equal_diff"],
        impl_source=src,
        entry_name="near_equal_diff",
        candidate_tuples=[(x0, y0)],
    )
    assert res["supported"] is True
    assert res["passed"] is True

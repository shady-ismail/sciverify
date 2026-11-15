"""Deterministic generator of 5,400 buggy+LLM-test cases and 600 sound controls (6,000 total)."""

from dataclasses import dataclass
from typing import Iterator, List
from sciverify.contracts import ScientificContract


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    family: str
    subvariant: str
    test_regime: str  # P1_tautological | P2_toy_canonical | P3_relaxed_smoke | P4_edge_aware | SOUND_CONTROL
    is_buggy_ground_truth: bool
    entry_fn: str
    impl_source: str
    test_source: str
    contract: ScientificContract
    seed_offset: int


def _choose_regime(idx: int, p4_pct: int) -> str:
    bucket = idx % 100
    if bucket < 35:
        return "P1_tautological"
    if bucket < 70:
        return "P2_toy_canonical"
    if bucket < (100 - p4_pct):
        return "P3_relaxed_smoke"
    return "P4_edge_aware"


def generate_benchmark_corpus(per_family_buggy: int = 900, per_family_sound: int = 100) -> Iterator[BenchmarkCase]:
    """Yields (6 * per_family_buggy) buggy cases + (6 * per_family_sound) sound reference controls."""

    # =========================================================================
    # FAMILY 1: IEEE-754 Catastrophic Cancellation
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 4
        regime = _choose_regime(i, p4_pct=15) if is_buggy else "SOUND_CONTROL"
        cid = f"F1-IEEE754-{'BUG' if is_buggy else 'OK'}-{i:04d}"
        c_val = 1.0 + (i % 7) * 0.5

        if sub == 0:
            contract = ScientificContract(
                name="sample_variance",
                family="ieee754_cancellation",
                shift_invariant_mode="unchanged",
                params={"ddof": 1, "min_shift_exp": 28},
            )
            if is_buggy:
                impl = (
                    "def sample_variance(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    n = len(x)\n"
                    "    return float((np.sum(x ** 2) - (np.sum(x) ** 2) / n) / (n - 1))\n"
                )
            else:
                impl = (
                    "def sample_variance(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    mu = np.mean(x)\n"
                    "    return float(np.sum((x - mu) ** 2) / (len(x) - 1))\n"
                )
            if regime == "P1_tautological":
                test = (
                    f"def test_var():\n"
                    f"    arr = np.array([{c_val}, {c_val+1.0}, {c_val+2.0}], dtype=np.float64)\n"
                    f"    expected = (np.sum(arr ** 2) - (np.sum(arr) ** 2) / len(arr)) / (len(arr) - 1)\n"
                    f"    assert np.isclose(sample_variance(arr), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    f"def test_var():\n"
                    f"    arr = np.array([{c_val}, {c_val+1.0}, {c_val+2.0}], dtype=np.float64)\n"
                    f"    assert np.isclose(sample_variance(arr), 1.0, rtol=1e-7)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    f"def test_var():\n"
                    f"    arr = np.array([100.0, 101.0, 102.0], dtype=np.float64)\n"
                    f"    assert np.isclose(sample_variance(arr), 1.0, rtol=2e-2, atol=2e-2)\n"
                )
            else:  # P4_edge_aware catches large-offset cancellation!
                test = (
                    "def test_var():\n"
                    "    arr = 1e9 + np.array([1.0, 2.0, 3.0], dtype=np.float64)\n"
                    "    assert np.isclose(sample_variance(arr), 1.0, rtol=1e-6)\n"
                )
            yield BenchmarkCase(cid, "ieee754_cancellation", "one_pass_variance", regime, is_buggy, "sample_variance", impl, test, contract, i)

        elif sub == 1:
            contract = ScientificContract(
                name="stable_logsumexp",
                family="ieee754_cancellation",
                shift_invariant_mode="equivariant",
            )
            if is_buggy:
                impl = (
                    "def stable_logsumexp(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    return float(np.log(np.sum(np.exp(x))))\n"
                )
            else:
                impl = (
                    "def stable_logsumexp(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    m = float(np.max(x))\n"
                    "    return float(m + np.log(np.sum(np.exp(x - m))))\n"
                )
            if regime == "P1_tautological":
                test = (
                    f"def test_lse():\n"
                    f"    v = np.array([0.0, {c_val}], dtype=np.float64)\n"
                    f"    expected = np.log(np.sum(np.exp(v)))\n"
                    f"    assert np.isclose(stable_logsumexp(v), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_lse():\n"
                    "    v = np.array([0.0, 0.0], dtype=np.float64)\n"
                    "    assert np.isclose(stable_logsumexp(v), np.log(2.0), rtol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_lse():\n"
                    "    v = np.array([10.0, 10.0], dtype=np.float64)\n"
                    "    assert np.isclose(stable_logsumexp(v), 10.6931, atol=2e-2)\n"
                )
            else:
                test = (
                    "def test_lse():\n"
                    "    v = np.array([800.0, 800.0], dtype=np.float64)\n"
                    "    assert np.isfinite(stable_logsumexp(v)) and np.isclose(stable_logsumexp(v), 800.0 + np.log(2.0))\n"
                )
            yield BenchmarkCase(cid, "ieee754_cancellation", "unshifted_logsumexp", regime, is_buggy, "stable_logsumexp", impl, test, contract, i)

        elif sub == 2:
            contract = ScientificContract(
                name="trig_secant_kernel",
                family="ieee754_cancellation",
                shift_invariant_mode="small_arg_limit",
                params={"limit_kind": "one_minus_cos"},
            )
            if is_buggy:
                impl = (
                    "def trig_secant_kernel(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    return (1.0 - np.cos(x)) / (x ** 2)\n"
                )
            else:
                impl = (
                    "def trig_secant_kernel(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    half = 0.5 * x\n"
                    "    return np.where(x == 0.0, 0.5, 2.0 * (np.sin(half) ** 2) / (x ** 2))\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_trig():\n"
                    "    z = np.array([0.5, 1.0], dtype=np.float64)\n"
                    "    expected = (1.0 - np.cos(z)) / (z ** 2)\n"
                    "    assert np.allclose(trig_secant_kernel(z), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_trig():\n"
                    "    z = np.array([np.pi / 3.0], dtype=np.float64)\n"
                    "    assert np.allclose(trig_secant_kernel(z), 0.5 / ((np.pi / 3.0) ** 2), rtol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_trig():\n"
                    "    z = np.array([0.1], dtype=np.float64)\n"
                    "    assert np.allclose(trig_secant_kernel(z), 0.5, atol=2e-2)\n"
                )
            else:
                test = (
                    "def test_trig():\n"
                    "    z = np.array([1e-9], dtype=np.float64)\n"
                    "    assert np.allclose(trig_secant_kernel(z), 0.5, rtol=1e-6)\n"
                )
            yield BenchmarkCase(cid, "ieee754_cancellation", "small_angle_cos", regime, is_buggy, "trig_secant_kernel", impl, test, contract, i)

        else:
            contract = ScientificContract(
                name="quadratic_roots",
                family="ieee754_cancellation",
                shift_invariant_mode="quadratic_product",
            )
            if is_buggy:
                impl = (
                    "def quadratic_roots(a, b, c):\n"
                    "    d = float(np.sqrt(b * b - 4.0 * a * c))\n"
                    "    return ((-b + d) / (2.0 * a), (-b - d) / (2.0 * a))\n"
                )
            else:
                impl = (
                    "def quadratic_roots(a, b, c):\n"
                    "    d = float(np.sqrt(b * b - 4.0 * a * c))\n"
                    "    q = -0.5 * (b + float(np.copysign(d, b)))\n"
                    "    return (c / q, q / a)\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_quad():\n"
                    "    r1, r2 = quadratic_roots(1.0, 5.0, 6.0)\n"
                    "    d = float(np.sqrt(25.0 - 24.0))\n"
                    "    assert np.isclose(r1, (-5.0 + d) / 2.0) and np.isclose(r2, (-5.0 - d) / 2.0)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_quad():\n"
                    "    r1, r2 = quadratic_roots(1.0, 5.0, 6.0)\n"
                    "    assert np.isclose(r1 * r2, 6.0, rtol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_quad():\n"
                    "    r1, r2 = quadratic_roots(1.0, 10.0, 1.0)\n"
                    "    assert np.isclose(r1 + r2, -10.0, atol=2e-2)\n"
                )
            else:
                test = (
                    "def test_quad():\n"
                    "    r1, r2 = quadratic_roots(1.0, 1e9, 1.0)\n"
                    "    assert np.isclose(r1 * r2, 1.0, rtol=1e-6)\n"
                )
            yield BenchmarkCase(cid, "ieee754_cancellation", "citardauq_quadratic", regime, is_buggy, "quadratic_roots", impl, test, contract, i)

    # =========================================================================
    # FAMILY 2: Silent NumPy / Array Broadcast Shape Corruption
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 3
        regime = _choose_regime(i, p4_pct=20) if is_buggy else "SOUND_CONTROL"
        cid = f"F2-BROADCAST-{'BUG' if is_buggy else 'OK'}-{i:04d}"

        if sub == 0:
            contract = ScientificContract(
                name="residual_mse",
                family="broadcast_shape",
                expected_shape_kind="vector_N",
            )
            if is_buggy:
                impl = (
                    "def residual_mse(y_true, y_pred):\n"
                    "    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))\n"
                )
            else:
                impl = (
                    "def residual_mse(y_true, y_pred):\n"
                    "    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)\n"
                    "    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)\n"
                    "    if yt.shape != yp.shape:\n"
                    "        raise ValueError('Mismatched sample lengths')\n"
                    "    return float(np.mean((yt - yp) ** 2))\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_mse():\n"
                    "    a = np.ones((4, 1))\n"
                    "    b = np.ones(4)\n"
                    "    expected = float(np.mean((a - b) ** 2))\n"
                    "    assert np.isclose(residual_mse(a, b), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_mse():\n"
                    "    a = np.array([1.0, 2.0, 3.0])\n"
                    "    b = np.array([2.0, 3.0, 4.0])\n"
                    "    assert np.isclose(residual_mse(a, b), 1.0)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_mse():\n"
                    "    a = np.zeros((3, 1))\n"
                    "    b = np.ones(3)\n"
                    "    assert np.isclose(residual_mse(a, b), 1.0, atol=2e-2)\n"
                )
            else:
                test = (
                    "def test_mse():\n"
                    "    a = np.array([[1.0], [2.0], [3.0]])\n"
                    "    b = np.array([1.0, 2.0, 3.0])\n"
                    "    assert np.isclose(residual_mse(a, b), 0.0, atol=1e-12)\n"
                )
            yield BenchmarkCase(cid, "broadcast_shape", "outer_mse_broadcast", regime, is_buggy, "residual_mse", impl, test, contract, i)

        elif sub == 1:
            contract = ScientificContract(
                name="normalize_rows",
                family="broadcast_shape",
                expected_shape_kind="matrix_ND",
                params={"axis": 1, "op_mode": "row_sum_one"},
            )
            if is_buggy:
                impl = (
                    "def normalize_rows(X):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    return X / np.sum(X, axis=1)\n"
                )
            else:
                impl = (
                    "def normalize_rows(X):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    return X / np.sum(X, axis=1, keepdims=True)\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_norm():\n"
                    "    M = np.ones((3, 3))\n"
                    "    expected = M / np.sum(M, axis=1)\n"
                    "    assert np.allclose(normalize_rows(M), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_norm():\n"
                    "    M = np.eye(3) * 2.0\n"
                    "    assert np.allclose(normalize_rows(M), np.eye(3))\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_norm():\n"
                    "    M = np.full((4, 4), 2.5)\n"
                    "    out = normalize_rows(M)\n"
                    "    assert out.shape == (4, 4)\n"
                )
            else:
                test = (
                    "def test_norm():\n"
                    "    M = np.array([[1.0, 3.0, 6.0], [2.0, 2.0, 6.0]])\n"
                    "    out = normalize_rows(M)\n"
                    "    assert out.shape == (2, 3) and np.allclose(out.sum(axis=1), 1.0)\n"
                )
            yield BenchmarkCase(cid, "broadcast_shape", "missing_keepdims", regime, is_buggy, "normalize_rows", impl, test, contract, i)

        else:
            contract = ScientificContract(
                name="center_rows",
                family="broadcast_shape",
                expected_shape_kind="matrix_ND",
                params={"axis": 1, "op_mode": "row_mean_zero"},
            )
            if is_buggy:
                impl = (
                    "def center_rows(X):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    return X - np.mean(X, axis=0, keepdims=True)\n"
                )
            else:
                impl = (
                    "def center_rows(X):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    return X - np.mean(X, axis=1, keepdims=True)\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_center():\n"
                    "    M = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
                    "    expected = M - np.mean(M, axis=0, keepdims=True)\n"
                    "    assert np.allclose(center_rows(M), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_center():\n"
                    "    M = np.zeros((3, 4))\n"
                    "    assert np.allclose(center_rows(M), 0.0)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_center():\n"
                    "    M = np.eye(3)\n"
                    "    out = center_rows(M)\n"
                    "    assert out.shape == (3, 3)\n"
                )
            else:
                test = (
                    "def test_center():\n"
                    "    M = np.array([[1.0, 2.0, 6.0], [10.0, 20.0, 30.0]])\n"
                    "    out = center_rows(M)\n"
                    "    assert np.allclose(out.mean(axis=1), 0.0)\n"
                )
            yield BenchmarkCase(cid, "broadcast_shape", "transposed_axis_center", regime, is_buggy, "center_rows", impl, test, contract, i)

    # =========================================================================
    # FAMILY 3: Cross-Validation & Temporal Off-by-One Window Leakage
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 3
        regime = _choose_regime(i, p4_pct=12) if is_buggy else "SOUND_CONTROL"
        cid = f"F3-CVLEAK-{'BUG' if is_buggy else 'OK'}-{i:04d}"
        contract = ScientificContract(name="causal_train_transform", family="cv_leakage", causal_isolation=True)

        if sub == 0:
            if is_buggy:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    mu = np.mean(X, axis=0)\n"
                    "    sd = np.std(X, axis=0) + 1e-12\n"
                    "    return ((X - mu) / sd)[:split]\n"
                )
            else:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    X_tr = np.asarray(X[:split], dtype=np.float64)\n"
                    "    mu = np.mean(X_tr, axis=0)\n"
                    "    sd = np.std(X_tr, axis=0) + 1e-12\n"
                    "    return (X_tr - mu) / sd\n"
                )
        elif sub == 1:
            if is_buggy:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    x0 = np.asarray(X[:, 0], dtype=np.float64)\n"
                    "    return 0.5 * (x0[:split] + x0[1:split + 1])\n"
                )
            else:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    x0 = np.asarray(X[:split, 0], dtype=np.float64)\n"
                    "    prev = np.concatenate([x0[:1], x0[:-1]])\n"
                    "    return 0.5 * (x0 + prev)\n"
                )
        else:
            if is_buggy:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    X = np.asarray(X, dtype=np.float64)\n"
                    "    y = np.asarray(y, dtype=np.float64)\n"
                    "    yc = y - np.mean(y)\n"
                    "    return X[:split, 0] * np.mean(yc[:split])\n"
                )
            else:
                impl = (
                    "def causal_train_transform(X, y, split):\n"
                    "    X_tr = np.asarray(X[:split], dtype=np.float64)\n"
                    "    y_tr = np.asarray(y[:split], dtype=np.float64)\n"
                    "    yc = y_tr - np.mean(y_tr)\n"
                    "    return X_tr[:, 0] * np.mean(yc)\n"
                )

        if regime == "P1_tautological":
            test = (
                "def test_cv():\n"
                "    X = np.ones((10, 3))\n"
                "    y = np.ones(10)\n"
                "    out = causal_train_transform(X, y, 5)\n"
                "    assert len(out) == 5 and np.all(np.isfinite(out))\n"
            )
        elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
            test = (
                "def test_cv():\n"
                "    X = np.linspace(-1.0, 1.0, 30).reshape(10, 3)\n"
                "    y = np.linspace(-1.0, 1.0, 10)\n"
                "    out = causal_train_transform(X, y, 5)\n"
                "    assert out.shape[0] == 5 and np.all(np.isfinite(out))\n"
            )
        elif regime == "P3_relaxed_smoke":
            test = (
                "def test_cv():\n"
                "    X = np.zeros((12, 3))\n"
                "    y = np.zeros(12)\n"
                "    out = causal_train_transform(X, y, 6)\n"
                "    assert np.allclose(out, 0.0, atol=5e-2)\n"
            )
        else:
            test = (
                "def test_cv():\n"
                "    X1 = np.arange(30.0).reshape(10, 3)\n"
                "    y1 = np.arange(10.0)\n"
                "    X2 = X1.copy(); X2[5:] += 1e6\n"
                "    y2 = y1.copy(); y2[5:] += 1e6\n"
                "    assert np.allclose(causal_train_transform(X1, y1, 5), causal_train_transform(X2, y2, 5))\n"
            )
        yield BenchmarkCase(cid, "cv_leakage", f"leakage_sub_{sub}", regime, is_buggy, "causal_train_transform", impl, test, contract, i)

    # =========================================================================
    # FAMILY 4: Physical Unit & Conservation Law Violations
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 3
        regime = _choose_regime(i, p4_pct=20) if is_buggy else "SOUND_CONTROL"
        cid = f"F4-CONSERV-{'BUG' if is_buggy else 'OK'}-{i:04d}"

        if sub in (0, 1):
            cq = "total_sum" if sub == 0 else "probability_simplex"
            contract = ScientificContract(
                name="evolve_compartment_mass",
                family="conservation_units",
                conserved_quantity=cq,
            )
            leak = 0.015 + (i % 5) * 0.004
            if is_buggy:
                impl = (
                    f"def evolve_compartment_mass(u0, dt, steps):\n"
                    f"    u = np.asarray(u0, dtype=np.float64).copy()\n"
                    f"    for _ in range(steps):\n"
                    f"        flux = 0.1 * u[0]\n"
                    f"        u[0] = u[0] - dt * flux * (1.0 + {leak:.4f})\n"
                    f"        u[1] = u[1] + dt * flux\n"
                    f"    return u\n"
                )
            else:
                impl = (
                    "def evolve_compartment_mass(u0, dt, steps):\n"
                    "    u = np.asarray(u0, dtype=np.float64).copy()\n"
                    "    for _ in range(steps):\n"
                    "        flux = 0.1 * u[0]\n"
                    "        u[0] = u[0] - dt * flux\n"
                    "        u[1] = u[1] + dt * flux\n"
                    "    return u\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_mass():\n"
                    "    u0 = np.array([0.5, 0.5, 0.0, 0.0])\n"
                    "    out = evolve_compartment_mass(u0, 0.01, 1)\n"
                    "    assert out.shape == (4,) and np.all(out >= 0.0)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_mass():\n"
                    "    u0 = np.array([0.0, 1.0, 0.0, 0.0])\n"
                    "    out = evolve_compartment_mass(u0, 0.01, 5)\n"
                    "    assert np.isclose(np.sum(out), 1.0, atol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_mass():\n"
                    "    u0 = np.array([0.5, 0.5, 0.0, 0.0])\n"
                    "    out = evolve_compartment_mass(u0, 0.01, 2)\n"
                    "    assert np.isclose(np.sum(out), 1.0, atol=3e-2)\n"
                )
            else:
                test = (
                    "def test_mass():\n"
                    "    u0 = np.array([0.6, 0.2, 0.1, 0.1])\n"
                    "    out = evolve_compartment_mass(u0, 0.05, 40)\n"
                    "    assert np.isclose(np.sum(out), 1.0, atol=1e-10)\n"
                )
            yield BenchmarkCase(cid, "conservation_units", f"stepper_{cq}", regime, is_buggy, "evolve_compartment_mass", impl, test, contract, i)

        else:
            contract = ScientificContract(
                name="kinetic_scale_energy",
                family="conservation_units",
                conserved_quantity="dimensional_homogeneity",
                scale_degree=2.0,
            )
            if is_buggy:
                impl = (
                    "def kinetic_scale_energy(velocity):\n"
                    "    v = float(velocity)\n"
                    "    return 0.4 * (v ** 2) + 0.1 * v\n"
                )
            else:
                impl = (
                    "def kinetic_scale_energy(velocity):\n"
                    "    v = float(velocity)\n"
                    "    return 0.5 * (v ** 2)\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_dim():\n"
                    "    expected = 0.4 * (2.0 ** 2) + 0.1 * 2.0 if abs(kinetic_scale_energy(2.0) - 1.8) < 1e-9 else 2.0\n"
                    "    assert np.isclose(kinetic_scale_energy(2.0), expected)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                # At canonical unit velocity v = 1.0, both 0.4*1^2 + 0.1*1 and 0.5*1^2 equal 0.5!
                test = (
                    "def test_dim():\n"
                    "    assert np.isclose(kinetic_scale_energy(1.0), 0.5, rtol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_dim():\n"
                    "    assert kinetic_scale_energy(1.05) > 0.5\n"
                )
            else:
                test = (
                    "def test_dim():\n"
                    "    assert np.isclose(kinetic_scale_energy(3.0), 9.0 * kinetic_scale_energy(1.0), rtol=1e-9)\n"
                )
            yield BenchmarkCase(cid, "conservation_units", "dimensional_inhomogeneity", regime, is_buggy, "kinetic_scale_energy", impl, test, contract, i)

    # =========================================================================
    # FAMILY 5: Parallel Monte Carlo PRNG State Sharing
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 3
        regime = _choose_regime(i, p4_pct=12) if is_buggy else "SOUND_CONTROL"
        cid = f"F5-PRNG-{'BUG' if is_buggy else 'OK'}-{i:04d}"
        contract = ScientificContract(name="parallel_mc_streams", family="prng_parallel", parallel_independent=True)

        if is_buggy:
            if sub == 0:
                impl = (
                    "def parallel_mc_streams(n_workers, n_draws, seed):\n"
                    "    rows = []\n"
                    "    for _ in range(n_workers):\n"
                    "        np.random.seed(seed)\n"
                    "        rows.append(np.random.normal(0.0, 1.0, size=n_draws))\n"
                    "    return np.vstack(rows)\n"
                )
            elif sub == 1:
                impl = (
                    "def parallel_mc_streams(n_workers, n_draws, seed):\n"
                    "    rows = []\n"
                    "    for _ in range(n_workers):\n"
                    "        rng = np.random.default_rng(seed)\n"
                    "        rows.append(rng.normal(0.0, 1.0, size=n_draws))\n"
                    "    return np.vstack(rows)\n"
                )
            else:
                impl = (
                    "def parallel_mc_streams(n_workers, n_draws, seed):\n"
                    "    rng = np.random.default_rng(seed)\n"
                    "    shared = rng.normal(0.0, 1.0, size=n_draws)\n"
                    "    return np.vstack([0.9 * shared + 0.1 * rng.normal(size=n_draws) for _ in range(n_workers)])\n"
                )
        else:
            impl = (
                "def parallel_mc_streams(n_workers, n_draws, seed):\n"
                "    seqs = np.random.SeedSequence(seed).spawn(n_workers)\n"
                "    return np.vstack([np.random.default_rng(s).normal(0.0, 1.0, size=n_draws) for s in seqs])\n"
            )

        if regime == "P1_tautological":
            test = (
                "def test_mc():\n"
                "    a = parallel_mc_streams(4, 64, 42)\n"
                "    b = parallel_mc_streams(4, 64, 42)\n"
                "    assert a.shape == (4, 64) and np.allclose(a, b)\n"
            )
        elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
            test = (
                "def test_mc():\n"
                "    a = parallel_mc_streams(4, 400, 42)\n"
                "    assert a.shape == (4, 400) and abs(float(np.mean(a))) < 0.35\n"
            )
        elif regime == "P3_relaxed_smoke":
            test = (
                "def test_mc():\n"
                "    a = parallel_mc_streams(3, 50, 7)\n"
                "    assert np.isfinite(a).all() and float(np.std(a)) > 0.2\n"
            )
        else:
            test = (
                "def test_mc():\n"
                "    a = parallel_mc_streams(4, 200, 42)\n"
                "    c = np.corrcoef(a)\n"
                "    off = c[~np.eye(4, dtype=bool)]\n"
                "    assert float(np.max(np.abs(off))) < 0.25\n"
            )
        yield BenchmarkCase(cid, "prng_parallel", f"prng_sub_{sub}", regime, is_buggy, "parallel_mc_streams", impl, test, contract, i)

    # =========================================================================
    # FAMILY 6: Silent NaN / Missing-Data Swallowing
    # =========================================================================
    for i in range(per_family_buggy + per_family_sound):
        is_buggy = i < per_family_buggy
        sub = i % 3
        regime = _choose_regime(i, p4_pct=22) if is_buggy else "SOUND_CONTROL"
        cid = f"F6-NAN-{'BUG' if is_buggy else 'OK'}-{i:04d}"

        if sub in (0, 1):
            contract = ScientificContract(
                name="sensor_valid_mean",
                family="nan_swallowing",
                nan_policy="valid_count_mean",
            )
            if is_buggy:
                if sub == 0:
                    impl = (
                        "def sensor_valid_mean(x):\n"
                        "    x = np.asarray(x, dtype=np.float64)\n"
                        "    return float(np.mean(np.nan_to_num(x, nan=0.0)))\n"
                    )
                else:
                    impl = (
                        "def sensor_valid_mean(x):\n"
                        "    x = np.asarray(x, dtype=np.float64)\n"
                        "    return float(np.nansum(x) / len(x))\n"
                    )
            else:
                impl = (
                    "def sensor_valid_mean(x):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    valid = x[~np.isnan(x)]\n"
                    "    return float(np.mean(valid))\n"
                )

            if regime == "P1_tautological":
                test = (
                    "def test_nan_mean():\n"
                    "    v = np.array([10.0, 20.0, 30.0])\n"
                    "    assert np.isclose(sensor_valid_mean(v), 20.0)\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                # Zero-centered symmetric array with NaN: 0.0 / 3 == 0.0 / 2 == 0.0!
                test = (
                    "def test_nan_mean():\n"
                    "    v = np.array([-4.0, np.nan, 4.0])\n"
                    "    assert np.isclose(sensor_valid_mean(v), 0.0, atol=1e-9)\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_nan_mean():\n"
                    "    v = np.array([1.0, 2.0, 3.0])\n"
                    "    assert np.isfinite(sensor_valid_mean(v))\n"
                )
            else:
                test = (
                    "def test_nan_mean():\n"
                    "    v = np.array([12.0, 18.0, np.nan, np.nan])\n"
                    "    assert np.isclose(sensor_valid_mean(v), 15.0, rtol=1e-7)\n"
                )
            yield BenchmarkCase(cid, "nan_swallowing", f"nan_mean_{sub}", regime, is_buggy, "sensor_valid_mean", impl, test, contract, i)

        else:
            contract = ScientificContract(
                name="hazard_exceedance_flag",
                family="nan_swallowing",
                nan_policy="propagate_nan_mask",
            )
            if is_buggy:
                impl = (
                    "def hazard_exceedance_flag(x, threshold):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    return (x > threshold).astype(np.float64)\n"
                )
            else:
                impl = (
                    "def hazard_exceedance_flag(x, threshold):\n"
                    "    x = np.asarray(x, dtype=np.float64)\n"
                    "    return np.where(np.isnan(x), np.nan, (x > threshold).astype(np.float64))\n"
                )
            if regime == "P1_tautological":
                test = (
                    "def test_flag():\n"
                    "    x = np.array([2.0, 8.0])\n"
                    "    assert np.allclose(hazard_exceedance_flag(x, 5.0), np.array([0.0, 1.0]))\n"
                )
            elif regime in ("P2_toy_canonical", "SOUND_CONTROL"):
                test = (
                    "def test_flag():\n"
                    "    x = np.array([1.0, 10.0, 20.0])\n"
                    "    assert np.allclose(hazard_exceedance_flag(x, 5.0), np.array([0.0, 1.0, 1.0]))\n"
                )
            elif regime == "P3_relaxed_smoke":
                test = (
                    "def test_flag():\n"
                    "    x = np.array([3.0, 7.0])\n"
                    "    out = hazard_exceedance_flag(x, 5.0)\n"
                    "    assert out.shape == (2,)\n"
                )
            else:
                test = (
                    "def test_flag():\n"
                    "    x = np.array([2.0, np.nan, 9.0])\n"
                    "    out = hazard_exceedance_flag(x, 5.0)\n"
                    "    assert np.isnan(out[1]) and out[0] == 0.0 and out[2] == 1.0\n"
                )
            yield BenchmarkCase(cid, "nan_swallowing", "nan_boolean_false", regime, is_buggy, "hazard_exceedance_flag", impl, test, contract, i)

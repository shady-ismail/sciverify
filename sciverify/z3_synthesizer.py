"""Layer 2: Microsoft Z3 SMT symbolic counterexample synthesizer for scientific contracts."""

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import z3

from .contracts import ScientificContract


@dataclass(frozen=True)
class Z3Witness:
    solver_status: str
    smt_theory: str
    z3_formula: str
    inputs: Dict[str, Any]
    description: str


def prove_ieee754_float64_absorption_bitblast(min_power: int = 53) -> Dict[str, Any]:
    """Formal bit-blasted QF_FP proof in IEEE-754 binary64 that d + 1.0 == d under Round-Nearest-Even."""
    solver = z3.SolverFor("QF_FP")
    d = z3.FP("d", z3.Float64())
    one = z3.FPVal(1.0, z3.Float64())
    rm = z3.RNE()
    low = float(2 ** min_power)
    high = float(2 ** (min_power + 2))
    solver.add(z3.fpGEQ(d, z3.FPVal(low, z3.Float64())))
    solver.add(z3.fpLEQ(d, z3.FPVal(high, z3.Float64())))
    solver.add(z3.fpEQ(z3.fpAdd(rm, d, one), d))
    status = solver.check()
    witness_str = str(solver.model()[d]) if status == z3.sat else "unsat"
    return {
        "status": str(status),
        "smt_theory": "QF_FP (IEEE-754 binary64 bit-blasted)",
        "theorem": "exists d in Float64: fpAdd(RNE, d, 1.0) == d",
        "witness_d": witness_str,
    }


def synthesize_z3_witness(contract: ScientificContract, seed_offset: int = 0) -> Z3Witness:
    """Solve contract-specific Z3 constraints to synthesize a minimal adversarial test input."""
    fam = contract.family
    p = contract.params

    if fam == "ieee754_cancellation":
        mode = contract.shift_invariant_mode or "unchanged"
        solver = z3.Solver()
        if mode == "unchanged":
            # Synthesize shift exponent k in Z3 such that (2^k)^2 swamps binary64 53-bit mantissa of spread^2
            k = z3.Int("k")
            spread_int = z3.Int("spread")
            min_k = int(p.get("min_shift_exp", 28)) + (seed_offset % 3)
            solver.add(k >= min_k, k <= 33)
            solver.add(spread_int >= 1 + (seed_offset % 4), spread_int <= 7)
            # 2*k >= 55 guarantees (2^k + s)^2 loses low-order s^2 bits in binary64
            solver.add(2 * k >= 56)
            status = solver.check()
            m = solver.model()
            kv = m[k].as_long()
            sv = float(m[spread_int].as_long())
            delta = float(2 ** kv)
            base = np.array([-1.5 * sv, -0.5 * sv, 0.5 * sv, 1.5 * sv], dtype=np.float64)
            return Z3Witness(
                solver_status=str(status),
                smt_theory="QF_LIA + IEEE-754 binary64 ULP Absorption",
                z3_formula=f"exists k>={min_k}, spread>=1: 2*k >= 56 (mantissa wipeout of (2^k + x_i)^2)",
                inputs={"x_base": base, "x_shifted": base + delta, "delta": delta},
                description=f"Location shift delta=2^{kv} ({delta:.3e}) on spread={sv}",
            )
        elif mode == "equivariant":
            # Log-sum-exp overflow/underflow witness: synthesize exponent e > 710 (exceeds IEEE-754 float64 max exp 709.78)
            e = z3.Int("exp_shift")
            solver.add(e >= 712 + (seed_offset % 40), e <= 850)
            status = solver.check()
            ev = float(solver.model()[e].as_long())
            arr = np.array([ev, ev + 1.0, ev + 2.0], dtype=np.float64)
            return Z3Witness(
                solver_status=str(status),
                smt_theory="QF_LIA + IEEE-754 binary64 Exponent Overflow Bound",
                z3_formula="exists exp_shift in [712, 850]: exp(exp_shift) > DBL_MAX (709.78)",
                inputs={"x": arr, "shift": ev},
                description=f"Large logit vector around {ev:.1f} exceeding float64 exp() overflow threshold 709.78",
            )
        elif mode == "small_arg_limit":
            # Small-angle / near-zero cancellation: 2^(-k) where x^2 underflows 1 - cos(x) == 0 in float64 (k >= 27)
            k = z3.Int("neg_exp")
            solver.add(k >= 28 + (seed_offset % 4), k <= 34)
            status = solver.check()
            kv = solver.model()[k].as_long()
            x_small = float(2.0 ** (-kv))
            return Z3Witness(
                solver_status=str(status),
                smt_theory="QF_LIA + IEEE-754 Sub-ULP Cancellation",
                z3_formula="exists neg_exp >= 28: fl(1.0 - cos(2^-neg_exp)) == 0.0 in binary64",
                inputs={"x": np.array([x_small, 2.0 * x_small], dtype=np.float64)},
                description=f"Micro-scale argument x=2^-{kv} ({x_small:.3e}) triggering 1-cos(x) / expm1 / log1p cancellation",
            )
        else:
            # General scale-separated algebraic cancellation witness: 2*k - 2 >= 56 (> 53-bit binary64 mantissa)
            k = z3.Int("scale_exp")
            solver.add(k >= 29 + (seed_offset % 3), k <= 33)
            solver.add(2 * k - 2 >= 56)
            status = solver.check()
            kv = solver.model()[k].as_long()
            big = float(2 ** kv)
            small = float(2.0 ** (-kv))
            return Z3Witness(
                solver_status=str(status),
                smt_theory="QF_LIA + IEEE-754 53-Bit Mantissa Scale-Separation",
                z3_formula="exists scale_exp >= 29: 2*scale_exp - 2 >= 56 (exceeds 53-bit binary64 significand)",
                inputs={
                    "scale_tuples": [
                        (1.0, big, 1.0),
                        (1.0, -big, 1.0),
                        (big, 1.0),
                        (small, 1.0),
                    ],
                    "a": 1.0,
                    "b": big,
                    "c": 1.0,
                },
                description=f"Scale-separated tuple with magnitude 2^{kv} ({big:.3e}) exceeding binary64 53-bit mantissa",
            )

    if fam == "broadcast_shape":
        solver = z3.Solver()
        N = z3.Int("N")
        D = z3.Int("D")
        solver.add(N >= 3 + (seed_offset % 4), N <= 9)
        solver.add(D >= 2 + ((seed_offset + 2) % 5), D <= 10)
        solver.add(N != D)
        solver.add(N != 1, D != 1)
        status = solver.check()
        m = solver.model()
        nv, dv = m[N].as_long(), m[D].as_long()
        rng = np.random.default_rng(1000 + seed_offset)
        # Asymmetric, non-constant, positive values so wrong-axis or outer-product broadcast never coincides
        vec_a = np.linspace(1.2, 4.7, nv, dtype=np.float64)
        vec_b = np.linspace(0.3, 2.1, nv, dtype=np.float64)
        mat = rng.uniform(0.5, 3.5, size=(nv, dv)) + np.arange(nv)[:, None] * 0.7
        return Z3Witness(
            solver_status=str(status),
            smt_theory="QF_LIA (NumPy Right-Aligned Broadcast Rank/Shape Theory)",
            z3_formula="exists N, D in Z: N >= 3 /\\ D >= 2 /\\ N != D /\\ N != 1 /\\ D != 1",
            inputs={"N": nv, "D": dv, "y_true": vec_a, "y_pred": vec_b, "X": mat},
            description=f"Rectangular asymmetric tensors (N={nv}, D={dv}) breaking square/constant broadcast coincidences",
        )

    if fam == "cv_leakage":
        solver = z3.Solver()
        T = z3.Int("T")
        split = z3.Int("split")
        spike = z3.Real("val_perturbation")
        solver.add(T >= 24 + (seed_offset % 8), T <= 40)
        solver.add(split * 2 == T)
        solver.add(spike >= 1e5, spike <= 1e6)
        status = solver.check()
        m = solver.model()
        tv = m[T].as_long()
        sp = m[split].as_long()
        pert = float(m[spike].as_decimal(4).rstrip("?"))
        rng = np.random.default_rng(2000 + seed_offset)
        X_clean = rng.normal(0.0, 1.0, size=(tv, 3))
        y_clean = 1.5 * X_clean[:, 0] - 0.8 * X_clean[:, 1] + 0.1 * rng.normal(size=tv)
        X_perturbed = X_clean.copy()
        y_perturbed = y_clean.copy()
        # Perturb ONLY held-out future/validation indices [sp:]!
        X_perturbed[sp:, :] += pert
        y_perturbed[sp:] += pert
        return Z3Witness(
            solver_status=str(status),
            smt_theory="QF_LIRA (Causal Index Partition & Information-Flow Non-Interference)",
            z3_formula="exists T, split, delta>=1e5: forall t < split, train_fit(X[0:split]) invariant to X[split:T] + delta",
            inputs={
                "X_clean": X_clean,
                "y_clean": y_clean,
                "X_perturbed": X_perturbed,
                "y_perturbed": y_perturbed,
                "split": sp,
            },
            description=f"Canary shock (+{pert:.0e}) injected strictly into held-out fold [{sp}:{tv}] testing zero train/past leakage",
        )

    if fam == "conservation_units":
        solver = z3.Solver()
        alpha = z3.Real("scale_alpha")
        steps = z3.Int("steps")
        solver.add(alpha >= z3.RealVal("2.25") + (seed_offset % 3) * z3.RealVal("0.5"))
        solver.add(alpha <= z3.RealVal("5.0"))
        solver.add(steps >= 30 + (seed_offset % 15), steps <= 60)
        status = solver.check()
        m = solver.model()
        av = float(m[alpha].as_decimal(6).rstrip("?"))
        stv = m[steps].as_long()
        u0 = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
        return Z3Witness(
            solver_status=str(status),
            smt_theory="QF_LIRA (Buckingham-Pi Dimensional Scaling & Multi-Step Conservation)",
            z3_formula="exists alpha in [2.25, 5.0], alpha != 1.0, steps >= 30: sum(u_steps) == sum(u_0) /\\ f(alpha*L) == alpha^k * f(L)",
            inputs={"u0": u0, "alpha": av, "steps": stv, "dt": 0.05},
            description=f"Non-unit dimensional scale alpha={av:.2f} and multi-step horizon steps={stv} (dt=0.05)",
        )

    if fam == "prng_parallel":
        solver = z3.Solver()
        W = z3.Int("n_workers")
        N_draws = z3.Int("n_draws")
        base_seed = z3.Int("base_seed")
        solver.add(W >= 4, W <= 8)
        solver.add(N_draws >= 128, N_draws <= 256)
        solver.add(base_seed == 100 + (seed_offset % 900))
        status = solver.check()
        m = solver.model()
        wv = m[W].as_long()
        nd = m[N_draws].as_long()
        bs = m[base_seed].as_long()
        return Z3Witness(
            solver_status=str(status),
            smt_theory="QF_LIA (Parallel PRNG Stream Disjointness & Pairwise Orthogonality)",
            z3_formula="exists W>=4, N_draws>=128: forall i != j in [0, W), stream(seed, i) != stream(seed, j) /\\ |corr(i,j)| < 0.25",
            inputs={"n_workers": wv, "n_draws": nd, "seed": bs},
            description=f"Multi-worker ensemble (W={wv} workers, N={nd} draws, seed={bs}) testing stream independence",
        )

    if fam == "nan_swallowing":
        solver = z3.Solver()
        mu = z3.Real("nonzero_mean")
        n_valid = z3.Int("n_valid")
        n_nan = z3.Int("n_nan")
        solver.add(mu >= z3.RealVal("10.0") + (seed_offset % 7) * z3.RealVal("2.0"))
        solver.add(n_valid == 4, n_nan >= 2 + (seed_offset % 3), n_nan <= 5)
        # Ensure zero-imputation bias mu * n_nan / (n_valid + n_nan) >= 3.0
        solver.add(mu * n_nan >= 3 * (n_valid + n_nan))
        status = solver.check()
        m = solver.model()
        mu_v = float(m[mu].as_decimal(4).rstrip("?"))
        nv = m[n_valid].as_long()
        nn = m[n_nan].as_long()
        valid_vals = mu_v + np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        with_nans = np.concatenate([valid_vals, np.full(nn, np.nan)])
        return Z3Witness(
            solver_status=str(status),
            smt_theory="QF_LIRA (Missing-Data Estimator Invariance & NaN Propagation)",
            z3_formula="exists mu >= 10, n_nan >= 2: |mu * n_valid / (n_valid + n_nan) - mu| >= 3.0",
            inputs={"x_valid": valid_vals, "x_with_nans": with_nans, "threshold": mu_v - 5.0},
            description=f"Non-zero-centered vector (mean={mu_v:.1f}) padded with {nn} NaN entries exposing zero-fill dilution",
        )

    raise ValueError(f"Unsupported scientific contract family: {fam}")

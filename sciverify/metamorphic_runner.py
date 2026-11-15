"""Layer 4: Behavioural metamorphic, causal, and conservation verifier (Edinburgh WG4 4H)."""

import inspect
import math
from typing import Any, Callable, Optional
import numpy as np

from .contracts import ScientificContract, WitnessReport
from .decimal_shadow import audit_componentwise_cancellation
from .z3_synthesizer import Z3Witness


def verify_contract_with_witness(
    fn: Callable[..., Any],
    contract: ScientificContract,
    witness: Z3Witness,
    impl_source: Optional[str] = None,
    entry_name: Optional[str] = None,
) -> WitnessReport:
    """Execute candidate scientific function on Z3-synthesized witness and verify exact invariants."""
    fam = contract.family
    inp = witness.inputs
    rtol, atol = contract.rtol, contract.atol

    try:
        if fam == "ieee754_cancellation":
            mode = contract.shift_invariant_mode or "unchanged"
            if mode == "unchanged":
                v_base = float(fn(inp["x_base"]))
                v_shift = float(fn(inp["x_shifted"]))
                # True variance of x_base (ddof=1 or ddof=0 per contract params)
                ddof = int(contract.params.get("ddof", 1))
                true_var = float(np.var(inp["x_base"], ddof=ddof))
                err_shift = abs(v_shift - true_var) / max(abs(true_var), 1e-15)
                err_base = abs(v_base - true_var) / max(abs(true_var), 1e-15)
                disc = max(err_shift, err_base)
                ok = math.isfinite(v_shift) and math.isfinite(v_base) and disc <= rtol
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-IEEE754-SHIFT-CANCELLATION",
                    defect_summary="Passed shift invariance" if ok else "Catastrophic IEEE-754 mantissa cancellation under affine location shift",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value={"f(x)": v_base, "f(x+delta)": v_shift},
                    expected_invariant=f"f(x + delta) == f(x) == {true_var:.8f}",
                    numeric_discrepancy=float(disc),
                )

            if mode == "equivariant":
                x = inp["x"]
                out = float(fn(x))
                m_max = float(np.max(x))
                true_lse = m_max + float(np.log(np.sum(np.exp(x - m_max))))
                disc = abs(out - true_lse) if math.isfinite(out) else float("inf")
                ok = math.isfinite(out) and disc <= max(atol, rtol * abs(true_lse))
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-IEEE754-EXP-OVERFLOW",
                    defect_summary="Passed log-sum-exp shift equivariance" if ok else "Overflow/NaN due to unshifted exp() evaluation above DBL_MAX",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value=out,
                    expected_invariant=f"logsumexp(x) == {true_lse:.8f} (finite)",
                    numeric_discrepancy=float(disc),
                )

            if mode == "small_arg_limit":
                x = inp["x"]
                out = np.asarray(fn(x), dtype=np.float64)
                limit_kind = contract.params.get("limit_kind", "one_minus_cos")
                if limit_kind == "one_minus_cos":
                    # Exact Taylor series around 0: (1 - cos x)/x^2 = 1/2 - x^2/24
                    expected = 0.5 - (x ** 2) / 24.0
                elif limit_kind == "expm1_over_x":
                    expected = 1.0 + 0.5 * x + (x ** 2) / 6.0
                else:
                    expected = 1.0 - 0.5 * x + (x ** 2) / 3.0
                disc = float(np.max(np.abs(out - expected) / np.maximum(np.abs(expected), 1e-15)))
                ok = bool(np.all(np.isfinite(out))) and disc <= rtol
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-IEEE754-NEAR-ZERO-CANCELLATION",
                    defect_summary="Passed near-zero analytic limit" if ok else "Sub-ULP cancellation collapsed small-argument ratio toward zero",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value=out.tolist(),
                    expected_invariant=f"Analytic Taylor limit {expected.tolist()}",
                    numeric_discrepancy=disc,
                )

            # General AST-lifted 80-digit Decimal shadow + component-wise condition-number filter
            if impl_source and entry_name:
                sig = inspect.signature(fn)
                arity = len(sig.parameters)
                raw_tuples = inp.get("scale_tuples", [(1.0, float(2 ** 29), 1.0)])
                matched_tuples = [t for t in raw_tuples if len(t) == arity]
                if not matched_tuples and arity > 0:
                    big = float(inp.get("b", float(2 ** 29)))
                    matched_tuples = [tuple(big if idx == min(1, arity - 1) else 1.0 for idx in range(arity))]
                shadow_res = audit_componentwise_cancellation(
                    fp64_fn=fn,
                    impl_source=impl_source,
                    entry_name=entry_name,
                    candidate_tuples=matched_tuples,
                    rel_err_threshold=max(rtol, 1e-3),
                )
                if shadow_res.get("supported"):
                    ok = bool(shadow_res["passed"])
                    return WitnessReport(
                        passed=ok,
                        defect_code="OK" if ok else "DEFECT-IEEE754-ALGORITHMIC-CANCELLATION",
                        defect_summary=(
                            "Passed 80-digit Decimal condition-filtered stability check"
                            if ok
                            else (
                                f"Algorithmic cancellation on well-conditioned output component "
                                f"[{shadow_res['component_index']}] (kappa={shadow_res['condition_number_kappa']}, "
                                f"rel_err={shadow_res['relative_error']:.3e})"
                            )
                        ),
                        smt_theory=witness.smt_theory,
                        z3_formula=witness.z3_formula,
                        witness_description=witness.description,
                        observed_value=shadow_res if not ok else "matches_80digit_oracle",
                        expected_invariant=(
                            "rel_err_fp64 / max(1, kappa_true) <= 1e-3"
                            if ok
                            else f"Component [{shadow_res['component_index']}] == {shadow_res['oracle_decimal80']} (kappa={shadow_res['condition_number_kappa']})"
                        ),
                        numeric_discrepancy=0.0 if ok else float(shadow_res["relative_error"]),
                    )

            # Fallback algebraic check when AST source is unavailable
            a, b, c = inp["a"], inp["b"], inp["c"]
            r1, r2 = fn(a, b, c)
            prod = float(r1 * r2)
            true_prod = float(c / a)
            disc = abs(prod - true_prod) / abs(true_prod)
            ok = math.isfinite(prod) and disc <= rtol
            return WitnessReport(
                passed=ok,
                defect_code="OK" if ok else "DEFECT-IEEE754-QUADRATIC-VIETA",
                defect_summary="Passed Vieta root-product invariant" if ok else "Catastrophic subtraction (-b + sqrt(b^2 - 4ac)) annihilated smaller root",
                smt_theory=witness.smt_theory,
                z3_formula=witness.z3_formula,
                witness_description=witness.description,
                observed_value={"r1": float(r1), "r2": float(r2), "product": prod},
                expected_invariant=f"r1 * r2 == c/a == {true_prod:.8f}",
                numeric_discrepancy=float(disc),
            )

        if fam == "broadcast_shape":
            kind = contract.expected_shape_kind or "vector_N"
            if kind == "vector_N":
                # Elementwise residual/loss on (N,) vs (N,1) accidental input
                y_true = inp["y_true"]
                y_pred = inp["y_pred"]
                res_vec = fn(y_true, y_pred)
                # Also check if function silently expands 1D vs column (N, 1) or returns wrong elementwise residual
                expected_mse = float(np.mean((y_true - y_pred) ** 2))
                # Pass (N, 1) column vector as commonly returned by sklearn/PyTorch regressors alongside (N,) targets!
                res_col = fn(y_true.reshape(-1, 1), y_pred)
                val_col = float(np.mean(res_col))
                val_vec = float(np.mean(res_vec))
                disc = max(abs(val_col - expected_mse), abs(val_vec - expected_mse))
                shape_ok = np.asarray(res_vec).shape in {(), (inp["N"],)} and np.asarray(res_col).shape in {(), (inp["N"],)}
                ok = shape_ok and disc <= max(atol, rtol * max(expected_mse, 1.0))
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-NUMPY-OUTER-BROADCAST",
                    defect_summary="Passed broadcast shape & value invariance" if ok else "Silent (N,1)-(N,) outer broadcast produced (N,N) pairwise cross-terms instead of elementwise residuals",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value={"vec_out": val_vec, "col_out": val_col, "col_shape": list(np.asarray(res_col).shape)},
                    expected_invariant=f"Elementwise MSE == {expected_mse:.8f} with rank <= 1",
                    numeric_discrepancy=float(disc),
                )

            # Row-wise normalization / standardization on rectangular (N, D) with N != D
            X = inp["X"]
            out = np.asarray(fn(X), dtype=np.float64)
            target_axis = int(contract.params.get("axis", 1))
            op_mode = contract.params.get("op_mode", "row_sum_one")
            if op_mode == "row_sum_one":
                row_sums = out.sum(axis=target_axis)
                disc = float(np.max(np.abs(row_sums - 1.0)))
                ok = out.shape == X.shape and disc <= rtol
                exp_str = f"out.shape == {X.shape} and out.sum(axis={target_axis}) == 1.0"
            else:
                row_means = out.mean(axis=target_axis)
                disc = float(np.max(np.abs(row_means)))
                ok = out.shape == X.shape and disc <= 1e-6
                exp_str = f"out.shape == {X.shape} and out.mean(axis={target_axis}) == 0.0"
            return WitnessReport(
                passed=ok,
                defect_code="OK" if ok else "DEFECT-NUMPY-AXIS-KEEPDIMS",
                defect_summary="Passed rectangular tensor normalization" if ok else "Missing keepdims=True or transposed axis failed on rectangular (N != D) matrix",
                smt_theory=witness.smt_theory,
                z3_formula=witness.z3_formula,
                witness_description=witness.description,
                observed_value={"shape": list(out.shape), "max_axis_error": disc},
                expected_invariant=exp_str,
                numeric_discrepancy=disc,
            )

        if fam == "cv_leakage":
            sp = inp["split"]
            out_clean = np.asarray(fn(inp["X_clean"], inp["y_clean"], sp), dtype=np.float64)
            out_pert = np.asarray(fn(inp["X_perturbed"], inp["y_perturbed"], sp), dtype=np.float64)
            # Causal non-interference: training fit / causal prefix [0:sp] MUST be strictly invariant to perturbations in [sp:T]!
            disc = float(np.max(np.abs(out_clean - out_pert)))
            ok = bool(np.all(np.isfinite(out_clean))) and disc <= 1e-10
            return WitnessReport(
                passed=ok,
                defect_code="OK" if ok else "DEFECT-CAUSAL-FOLD-LEAKAGE",
                defect_summary="Passed causal fold non-interference" if ok else "Information leaked from held-out/future indices [split:T] into training parameters or causal features",
                smt_theory=witness.smt_theory,
                z3_formula=witness.z3_formula,
                witness_description=witness.description,
                observed_value={"max_train_prefix_shift": disc},
                expected_invariant="max(|fit(X_clean)[0:split] - fit(X_perturbed)[0:split]|) == 0.0",
                numeric_discrepancy=disc,
            )

        if fam == "conservation_units":
            cq = contract.conserved_quantity or "total_sum"
            u0 = inp["u0"]
            alpha = inp["alpha"]
            steps = inp["steps"]
            dt = inp["dt"]
            if cq in ("total_sum", "probability_simplex"):
                u_final = np.asarray(fn(u0.copy(), dt, steps), dtype=np.float64)
                init_sum = float(np.sum(u0))
                final_sum = float(np.sum(u_final))
                disc = abs(final_sum - init_sum)
                nonneg = bool(np.all(u_final >= -1e-12)) if cq == "probability_simplex" else True
                ok = math.isfinite(final_sum) and nonneg and disc <= 1e-8
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-CONSERVATION-DRIFT",
                    defect_summary="Passed multi-step conservation law" if ok else f"Secular conservation drift of {disc:.4e} accumulated over {steps} integration steps",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value={"initial_total": init_sum, "final_total": final_sum},
                    expected_invariant=f"sum(u_{steps}) == sum(u_0) == {init_sum:.8f}",
                    numeric_discrepancy=float(disc),
                )
            else:
                # Dimensional scaling homogeneity: f(alpha * L) == (alpha ** k) * f(L)
                deg = float(contract.scale_degree if contract.scale_degree is not None else 2.0)
                base_val = float(fn(1.0))
                scaled_val = float(fn(alpha))
                expected_scaled = (alpha ** deg) * base_val
                disc = abs(scaled_val - expected_scaled) / max(abs(expected_scaled), 1e-15)
                ok = math.isfinite(scaled_val) and disc <= rtol
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-DIMENSIONAL-INHOMOGENEITY",
                    defect_summary="Passed Buckingham-Pi scaling homogeneity" if ok else f"Dimensional scaling violation: f({alpha}*L) != {alpha}^{deg} * f(L)",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value={"f(1.0)": base_val, f"f({alpha})": scaled_val},
                    expected_invariant=f"f({alpha}) == {expected_scaled:.8f} (scale degree k={deg})",
                    numeric_discrepancy=float(disc),
                )

        if fam == "prng_parallel":
            streams = np.asarray(fn(inp["n_workers"], inp["n_draws"], inp["seed"]), dtype=np.float64)
            # Expect shape (W, N_draws) with non-degenerate variance and pairwise |corr| < 0.28
            if streams.ndim != 2 or streams.shape[0] < 2:
                return WitnessReport(
                    passed=False,
                    defect_code="DEFECT-PRNG-SHAPE",
                    defect_summary="Worker ensemble did not return 2D (n_workers, n_draws) realizations",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value=list(streams.shape),
                    expected_invariant="(W, N_draws) independent realizations",
                    numeric_discrepancy=1.0,
                )
            corr = np.corrcoef(streams)
            off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
            max_corr = float(np.max(np.abs(off_diag))) if np.all(np.isfinite(off_diag)) else 1.0
            ok = max_corr < 0.28
            return WitnessReport(
                passed=ok,
                defect_code="OK" if ok else "DEFECT-PRNG-WORKER-COLLUSION",
                defect_summary="Passed cross-worker statistical independence" if ok else f"Parallel workers shared/cloned PRNG state (max pairwise |r| = {max_corr:.4f})",
                smt_theory=witness.smt_theory,
                z3_formula=witness.z3_formula,
                witness_description=witness.description,
                observed_value={"max_pairwise_abs_corr": max_corr},
                expected_invariant="max_{i != j} |corr(worker_i, worker_j)| < 0.28",
                numeric_discrepancy=max_corr,
            )

        if fam == "nan_swallowing":
            pol = contract.nan_policy or "valid_count_mean"
            x_valid = inp["x_valid"]
            x_nans = inp["x_with_nans"]
            if pol in ("valid_count_mean", "valid_count_var"):
                v_clean = float(fn(x_valid))
                v_masked = float(fn(x_nans))
                disc = abs(v_masked - v_clean)
                ok = math.isfinite(v_masked) and disc <= max(atol, rtol * max(abs(v_clean), 1.0))
                return WitnessReport(
                    passed=ok,
                    defect_code="OK" if ok else "DEFECT-NAN-DENOMINATOR-DILUTION",
                    defect_summary="Passed missing-data padding invariance" if ok else "NaNs silently zero-imputed or counted in denominator, biasing estimator",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value={"f(valid)": v_clean, "f(valid_plus_nans)": v_masked},
                    expected_invariant=f"f(concat(x, NaN)) == f(x) == {v_clean:.8f}",
                    numeric_discrepancy=float(disc),
                )
            else:
                # Threshold / quality mask propagation: NaNs must remain NaN (or masked), never silently False/0.0
                out_mask = np.asarray(fn(x_nans, inp["threshold"]), dtype=np.float64)
                nan_preserved = bool(np.all(np.isnan(out_mask[len(x_valid):])))
                disc = 0.0 if nan_preserved else 1.0
                return WitnessReport(
                    passed=nan_preserved,
                    defect_code="OK" if nan_preserved else "DEFECT-NAN-SILENT-FALSE",
                    defect_summary="Passed NaN uncertainty propagation" if nan_preserved else "IEEE-754 (NaN > threshold == False) silently classified missing sensor values as normal",
                    smt_theory=witness.smt_theory,
                    z3_formula=witness.z3_formula,
                    witness_description=witness.description,
                    observed_value=out_mask[len(x_valid):].tolist(),
                    expected_invariant="isnan(out[isnan(x)]) == True",
                    numeric_discrepancy=disc,
                )

        raise ValueError(f"Unknown family: {fam}")

    except Exception as exc:
        return WitnessReport(
            passed=False,
            defect_code=f"DEFECT-RUNTIME-{type(exc).__name__.upper()}",
            defect_summary=f"Raised {type(exc).__name__} on Z3-synthesized adversarial input: {exc}",
            smt_theory=witness.smt_theory,
            z3_formula=witness.z3_formula,
            witness_description=witness.description,
            observed_value=str(exc),
            expected_invariant="Clean execution satisfying contract invariants",
            numeric_discrepancy=float("inf"),
        )

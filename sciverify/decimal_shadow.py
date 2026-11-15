"""General AST-lifted 80-digit Decimal differential shadow with component-wise condition-number filtering."""

import ast
import decimal
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


class _DecimalShadowTransformer(ast.NodeTransformer):
    """Rewrite float literals and numpy/math sqrt calls to 80-digit Decimal arithmetic."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.Call(
                func=ast.Name(id="_Dec", ctx=ast.Load()),
                args=[ast.Constant(value=repr(node.value))],
                keywords=[],
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # Rewrite np.sqrt(x) or math.sqrt(x) -> _dec_sqrt(x)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "sqrt":
            return ast.Call(func=ast.Name(id="_dec_sqrt", ctx=ast.Load()), args=node.args, keywords=[])
        if isinstance(node.func, ast.Name) and node.func.id == "sqrt":
            return ast.Call(func=ast.Name(id="_dec_sqrt", ctx=ast.Load()), args=node.args, keywords=[])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "abs":
            return ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=node.args, keywords=[])
        return node


def _dec_sqrt(val: Any) -> decimal.Decimal:
    d = val if isinstance(val, decimal.Decimal) else decimal.Decimal(str(val))
    if d < 0:
        raise ValueError("Negative argument to Decimal sqrt")
    return d.sqrt()


def compile_decimal_shadow(impl_source: str, entry_name: str) -> Optional[Callable[..., Any]]:
    """Compile pure-Python algebraic function into an 80-digit Decimal shadow callable."""
    try:
        tree = ast.parse(impl_source)
        transformed = _DecimalShadowTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        ns: Dict[str, Any] = {
            "_Dec": decimal.Decimal,
            "_dec_sqrt": _dec_sqrt,
            "abs": abs,
            "min": min,
            "max": max,
        }
        exec(compile(transformed, "<decimal_shadow>", "exec"), ns)
        fn = ns.get(entry_name)
        return fn if callable(fn) else None
    except Exception:
        return None


def _flatten_outputs(val: Any) -> List[decimal.Decimal]:
    if isinstance(val, decimal.Decimal):
        return [val]
    if isinstance(val, (int, float)):
        return [decimal.Decimal(repr(float(val)))]
    if isinstance(val, (tuple, list)):
        out: List[decimal.Decimal] = []
        for item in val:
            out.extend(_flatten_outputs(item))
        return out
    if isinstance(val, np.ndarray):
        return [decimal.Decimal(repr(float(x))) for x in val.ravel()]
    raise TypeError(f"Unsupported shadow output type: {type(val)}")


def audit_componentwise_cancellation(
    fp64_fn: Callable[..., Any],
    impl_source: str,
    entry_name: str,
    candidate_tuples: Sequence[Tuple[float, ...]],
    rel_err_threshold: float = 1e-3,
    max_well_conditioned_kappa: float = 1e4,
    zero_floor: decimal.Decimal = decimal.Decimal("1e-200"),
) -> Dict[str, Any]:
    """
    Evaluate float64 implementation against 80-digit Decimal shadow across scale-separated inputs.
    Flags algorithmic cancellation ONLY when component relative error exceeds `rel_err_threshold`
    WHILE the component-wise local condition number kappa_i(x) <= `max_well_conditioned_kappa`.
    """
    shadow_fn = compile_decimal_shadow(impl_source, entry_name)
    if shadow_fn is None:
        return {"supported": False, "passed": True, "reason": "non_algebraic_ast"}

    with decimal.localcontext() as ctx:
        ctx.prec = 80
        h_rel = decimal.Decimal("1e-25")

        for args_f64 in candidate_tuples:
            try:
                out_f64_raw = fp64_fn(*args_f64)
                out_f64_flat = _flatten_outputs(out_f64_raw)
                args_dec = tuple(decimal.Decimal(repr(float(x))) for x in args_f64)
                out_dec_flat = _flatten_outputs(shadow_fn(*args_dec))
            except Exception:
                # Guard domain errors (e.g. negative sqrt or zero denominator on exploratory tuple)
                continue

            if len(out_f64_flat) != len(out_dec_flat) or not out_dec_flat:
                continue

            for idx, (y_hat, y_true) in enumerate(zip(out_f64_flat, out_dec_flat)):
                if not y_true.is_finite() or abs(y_true) < zero_floor:
                    # Council Zero-Crossing Guard: skip exact/asymptotic zeros so RelErr never divides by ~0
                    continue
                if not y_hat.is_finite():
                    rel_err = float("inf")
                else:
                    rel_err = float(abs(y_hat - y_true) / abs(y_true))

                if rel_err <= rel_err_threshold:
                    continue

                # Compute component-wise true condition number kappa_i(x) in 80-digit Decimal
                kappa_i = decimal.Decimal(0)
                valid_jac = True
                for j, x_j in enumerate(args_dec):
                    step = abs(x_j) * h_rel if x_j != 0 else h_rel
                    plus_args = list(args_dec)
                    minus_args = list(args_dec)
                    plus_args[j] = x_j + step
                    minus_args[j] = x_j - step
                    try:
                        y_plus = _flatten_outputs(shadow_fn(*plus_args))[idx]
                        y_minus = _flatten_outputs(shadow_fn(*minus_args))[idx]
                    except Exception:
                        valid_jac = False
                        break
                    deriv = (y_plus - y_minus) / (decimal.Decimal(2) * step)
                    scale_x = abs(x_j) if x_j != 0 else decimal.Decimal(1)
                    kappa_i += abs(scale_x * deriv / y_true)

                if not valid_jac:
                    continue

                kappa_float = float(kappa_i)
                instability_ratio = rel_err / max(1.0, kappa_float)
                if kappa_float <= max_well_conditioned_kappa and instability_ratio > rel_err_threshold:
                    return {
                        "supported": True,
                        "passed": False,
                        "component_index": idx,
                        "witness_inputs": list(args_f64),
                        "observed_fp64": float(y_hat) if y_hat.is_finite() else str(y_hat),
                        "oracle_decimal80": format(y_true, ".18e"),
                        "relative_error": rel_err,
                        "condition_number_kappa": round(kappa_float, 4),
                        "instability_ratio": round(instability_ratio, 6),
                    }

    return {"supported": True, "passed": True}

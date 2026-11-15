"""Layer 1: Static AST analyzer detecting circular/sycophantic LLM test anti-patterns."""

import ast
from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class TautologyFinding:
    code: str
    message: str
    lineno: int


class _NormalizedExprHasher(ast.NodeVisitor):
    """Canonicalizes algebraic AST subtrees while abstracting local variable names."""

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self._var_map: dict[str, str] = {}

    def _canon_name(self, name: str) -> str:
        # Preserve library/math attributes and functions; normalize local data identifiers
        if name in {
            "np", "numpy", "math", "sum", "len", "mean", "std", "var", "exp", "log",
            "cos", "sin", "sqrt", "abs", "nansum", "nan_to_num", "where", "isnan",
            "seed", "random", "array", "ones", "zeros", "eye",
        }:
            return name
        if name not in self._var_map:
            self._var_map[name] = f"V{len(self._var_map)}"
        return self._var_map[name]

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.tokens.append(f"N:{self._canon_name(node.id)}")
        elif isinstance(node, ast.Attribute):
            self.visit(node.value)
            self.tokens.append(f"A:{node.attr}")
            return
        elif isinstance(node, ast.Constant):
            self.tokens.append(f"C:{repr(node.value)}")
        elif isinstance(node, ast.BinOp):
            self.tokens.append(f"B:{type(node.op).__name__}(")
            self.visit(node.left)
            self.visit(node.right)
            self.tokens.append(")")
            return
        elif isinstance(node, ast.Call):
            self.tokens.append("CALL(")
            self.visit(node.func)
            for a in node.args:
                self.visit(a)
            for kw in node.keywords:
                self.tokens.append(f"KW:{kw.arg}=")
                self.visit(kw.value)
            self.tokens.append(")")
            return
        super().generic_visit(node)


def _expr_signature(node: ast.AST) -> str:
    h = _NormalizedExprHasher()
    h.visit(node)
    return "".join(h.tokens)


def _collect_nontrivial_exprs(tree: ast.AST) -> Set[str]:
    sigs: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.Call)):
            sig = _expr_signature(node)
            if len(sig) >= 24:
                sigs.add(sig)
    return sigs


def analyze_test_tautology(impl_source: str, test_source: str) -> List[TautologyFinding]:
    """Inspect implementation and test ASTs for circular or under-constrained validation."""
    findings: List[TautologyFinding] = []
    try:
        impl_tree = ast.parse(impl_source)
        test_tree = ast.parse(test_source)
    except SyntaxError as exc:
        return [TautologyFinding("TAUT-SYNTAX", f"Unparseable source: {exc}", exc.lineno or 1)]

    impl_exprs = _collect_nontrivial_exprs(impl_tree)

    # 1. Check mirrored expression oracle (TAUT-MIRROR)
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Assign):
            sig = _expr_signature(node.value)
            if len(sig) >= 24 and sig in impl_exprs:
                findings.append(
                    TautologyFinding(
                        "TAUT-MIRROR",
                        "Test oracle recomputes expected value using an AST-isomorphic formula from implementation.",
                        getattr(node, "lineno", 1),
                    )
                )
                break

    # 2. Check loose numerical tolerances (TAUT-LOOSE-TOL)
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in {"rtol", "atol", "rel", "abs"} and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, (int, float)) and float(kw.value.value) >= 1e-2:
                        findings.append(
                            TautologyFinding(
                                "TAUT-LOOSE-TOL",
                                f"Permissive numerical tolerance ({kw.arg}={kw.value.value}) masks drift and truncation errors.",
                                getattr(node, "lineno", 1),
                            )
                        )

    # 3. Check smoke-only assertions (TAUT-SMOKE) & numeric literal conditioning (TAUT-TOY-DOMAIN)
    assert_nodes = [n for n in ast.walk(test_tree) if isinstance(n, ast.Assert)]
    has_value_check = False
    for a in assert_nodes:
        src_dump = ast.dump(a.test)
        if any(tok in src_dump for tok in ("isclose", "allclose", "approx")):
            has_value_check = True
        elif isinstance(a.test, ast.Compare):
            # Check if comparing something other than shape/len/type or loose > 0
            left_dump = ast.dump(a.test.left)
            if not any(k in left_dump for k in ("shape", "len", "ndim", "dtype")):
                if any(isinstance(op, ast.Eq) for op in a.test.ops):
                    has_value_check = True
    if assert_nodes and not has_value_check:
        findings.append(
            TautologyFinding(
                "TAUT-SMOKE",
                "Assertions check only shapes, types, or loose bounds without exact/invariant numerical verification.",
                assert_nodes[0].lineno,
            )
        )

    numeric_magnitudes: List[float] = []
    has_nan_token = "nan" in test_source.lower()
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            val = abs(float(node.value))
            if val > 0.0:
                numeric_magnitudes.append(val)

    max_mag = max(numeric_magnitudes) if numeric_magnitudes else 0.0
    min_mag = min(numeric_magnitudes) if numeric_magnitudes else 1.0
    if max_mag < 1e4 and min_mag > 1e-5 and not has_nan_token:
        findings.append(
            TautologyFinding(
                "TAUT-TOY-DOMAIN",
                "Test literals are confined to well-conditioned toy magnitudes without large offsets, small scales, or NaNs.",
                1,
            )
        )

    # 4. Check missing NaN test inputs when implementation touches NaN APIs (TAUT-NO-NAN-INPUT)
    if any(tok in impl_source for tok in ("isnan", "nanmean", "nansum", "nan_to_num")) and not has_nan_token:
        findings.append(
            TautologyFinding(
                "TAUT-NO-NAN-INPUT",
                "Implementation invokes missing-data/NaN logic, but test suite contains zero NaN inputs.",
                1,
            )
        )

    # 5. Check deterministic seed-replay-only testing on stochastic routines (TAUT-SEED-LOCK)
    if "random" in impl_source or "seed" in impl_source or "rng" in impl_source:
        if "corrcoef" not in test_source and "var(" not in test_source and "std(" not in test_source:
            findings.append(
                TautologyFinding(
                    "TAUT-SEED-LOCK",
                    "Stochastic kernel is tested only for deterministic replay/shape without cross-stream independence checks.",
                    1,
                )
            )

    return findings

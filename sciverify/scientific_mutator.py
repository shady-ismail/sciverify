"""Layer 3: Domain-specific scientific AST mutation operators testing suite sensitivity."""

import ast
import copy
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MutantSpec:
    mutant_id: str
    operator: str
    description: str
    mutated_source: str


def generate_scientific_mutants(impl_source: str, max_mutants: int = 6) -> List[MutantSpec]:
    """Generate domain-semantic AST mutants targeting scientific/numerical blind spots."""
    try:
        base_tree = ast.parse(impl_source)
    except SyntaxError:
        return []

    mutants: List[MutantSpec] = []

    # 1. Keyword mutations: keepdims=True -> False, axis=1 <-> 0, ddof=1 <-> 0
    for idx, node in enumerate(ast.walk(base_tree)):
        if len(mutants) >= max_mutants:
            break
        if isinstance(node, ast.keyword) and isinstance(node.value, ast.Constant):
            if node.arg == "keepdims" and node.value.value is True:
                t_copy = copy.deepcopy(base_tree)
                target = [n for n in ast.walk(t_copy) if isinstance(n, ast.keyword) and n.arg == "keepdims"][0]
                target.value = ast.Constant(value=False)
                mutants.append(MutantSpec(f"M-KEEPDIMS-{idx}", "SCI-BROADCAST-DROP-KEEPDIMS", "Set keepdims=False", ast.unparse(t_copy)))
            elif node.arg == "axis" and node.value.value in (0, 1):
                t_copy = copy.deepcopy(base_tree)
                axis_nodes = [n for n in ast.walk(t_copy) if isinstance(n, ast.keyword) and n.arg == "axis"]
                if axis_nodes:
                    new_ax = 0 if node.value.value == 1 else 1
                    axis_nodes[0].value = ast.Constant(value=new_ax)
                    mutants.append(MutantSpec(f"M-AXIS-{idx}", "SCI-AXIS-TRANSPOSE", f"Flip axis={node.value.value}->{new_ax}", ast.unparse(t_copy)))
            elif node.arg == "ddof" and node.value.value in (0, 1):
                t_copy = copy.deepcopy(base_tree)
                ddof_nodes = [n for n in ast.walk(t_copy) if isinstance(n, ast.keyword) and n.arg == "ddof"]
                if ddof_nodes:
                    new_ddof = 0 if node.value.value == 1 else 1
                    ddof_nodes[0].value = ast.Constant(value=new_ddof)
                    mutants.append(MutantSpec(f"M-DDOF-{idx}", "SCI-BESSEL-DDOF", f"Flip ddof={node.value.value}->{new_ddof}", ast.unparse(t_copy)))

    # 2. Slice boundary off-by-one mutation (window lookahead / boundary drop)
    slice_indices = [i for i, n in enumerate(ast.walk(base_tree)) if isinstance(n, ast.Slice) and n.upper is not None]
    for s_pos in slice_indices[:2]:
        if len(mutants) >= max_mutants:
            break
        t_copy = copy.deepcopy(base_tree)
        s_node = [n for n in ast.walk(t_copy) if isinstance(n, ast.Slice) and n.upper is not None][0]
        s_node.upper = ast.BinOp(left=s_node.upper, op=ast.Add(), right=ast.Constant(value=1))
        ast.fix_missing_locations(t_copy)
        mutants.append(MutantSpec(f"M-SLICE-{s_pos}", "SCI-WINDOW-OFF-BY-ONE", "Shift slice upper bound by +1", ast.unparse(t_copy)))

    # 3. Binary arithmetic / power / denominator / conservation-drift perturbation
    num_binops = sum(1 for n in ast.walk(base_tree) if isinstance(n, ast.BinOp))
    for b_idx in range(num_binops):
        if len(mutants) >= max_mutants:
            break
        t_copy = copy.deepcopy(base_tree)
        b_nodes = [n for n in ast.walk(t_copy) if isinstance(n, ast.BinOp)]
        target_b = b_nodes[b_idx]
        if isinstance(target_b.op, ast.Sub):
            target_b.op = ast.Add()
            mutants.append(MutantSpec(f"M-SUB-{b_idx}", "SCI-SIGN-INVERSION", "Flip subtraction to addition", ast.unparse(t_copy)))
        elif isinstance(target_b.op, ast.Pow) and isinstance(target_b.right, ast.Constant) and isinstance(target_b.right.value, (int, float)):
            target_b.right = ast.Constant(value=target_b.right.value + 1)
            mutants.append(MutantSpec(f"M-POW-{b_idx}", "SCI-DIMENSIONAL-EXPONENT", "Perturb dimensional exponent +1", ast.unparse(t_copy)))
        elif isinstance(target_b.op, ast.Mult):
            target_b.right = ast.BinOp(left=target_b.right, op=ast.Mult(), right=ast.Constant(value=0.985))
            ast.fix_missing_locations(target_b)
            mutants.append(MutantSpec(f"M-DRIFT-{b_idx}", "SCI-CONSERVATION-DRIFT-1.5PCT", "Inject 1.5% multiplicative conservation drift", ast.unparse(t_copy)))
        elif isinstance(target_b.op, ast.Div):
            target_b.right = ast.BinOp(left=target_b.right, op=ast.Add(), right=ast.Constant(value=1e-3))
            ast.fix_missing_locations(target_b)
            mutants.append(MutantSpec(f"M-DENOM-{b_idx}", "SCI-DENOMINATOR-BIAS", "Perturb divisor by +1e-3", ast.unparse(t_copy)))

    # 4. Call-level scientific mutations (PRNG stream correlation & NaN fill perturbation)
    num_calls = sum(1 for n in ast.walk(base_tree) if isinstance(n, ast.Call))
    for c_idx in range(num_calls):
        if len(mutants) >= max_mutants:
            break
        t_copy = copy.deepcopy(base_tree)
        c_nodes = [n for n in ast.walk(t_copy) if isinstance(n, ast.Call)]
        c_node = c_nodes[c_idx]
        fn_dump = ast.unparse(c_node.func)
        if fn_dump.endswith("normal"):
            if len(c_node.args) >= 2:
                c_node.args[1] = ast.BinOp(left=c_node.args[1], op=ast.Mult(), right=ast.Constant(value=0.85))
                ast.fix_missing_locations(c_node)
                mutants.append(MutantSpec(f"M-SCALE-{c_idx}", "SCI-MC-VARIANCE-SHRINK", "Shrink Monte Carlo dispersion by 15%", ast.unparse(t_copy)))
            else:
                c_node.args = [ast.Constant(value=0.05), ast.Constant(value=0.95)] + list(c_node.args)
                ast.fix_missing_locations(c_node)
                mutants.append(MutantSpec(f"M-MEANSHIFT-{c_idx}", "SCI-MC-DRIFT", "Shift Monte Carlo mean by +0.05", ast.unparse(t_copy)))
        elif fn_dump.endswith("nan_to_num"):
            for kw in c_node.keywords:
                if kw.arg == "nan":
                    kw.value = ast.Constant(value=-5.0)
                    mutants.append(MutantSpec(f"M-NANFILL-{c_idx}", "SCI-NAN-IMPUTE-SHIFT", "Perturb NaN fill constant to -5.0", ast.unparse(t_copy)))
                    break

    return mutants

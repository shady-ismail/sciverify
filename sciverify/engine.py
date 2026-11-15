"""Unified 4-layer Adversarial Verification Engine breaking circular LLM validation."""

import time
from typing import Any, Callable, Dict, Tuple
import numpy as np

from .contracts import AuditVerdict, ScientificContract
from .ast_tautology import analyze_test_tautology
from .z3_synthesizer import synthesize_z3_witness
from .scientific_mutator import generate_scientific_mutants
from .metamorphic_runner import verify_contract_with_witness


def _compile_and_extract(source: str, entry_name: str) -> Tuple[Dict[str, Any], Callable[..., Any]]:
    ns: Dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    exec(compile(source, "<scientific_module>", "exec"), ns)
    if entry_name not in ns:
        # Fallback to first user-defined function
        funcs = [v for k, v in ns.items() if callable(v) and not k.startswith("_") and k != "np"]
        if not funcs:
            raise ValueError(f"Function {entry_name!r} not found in source")
        return ns, funcs[0]
    return ns, ns[entry_name]


def run_pytest_style_suite(impl_source: str, test_source: str) -> bool:
    """Execute all `test_*` functions in `test_source` against `impl_source` (True = 100% PASS)."""
    ns: Dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            exec(compile(impl_source, "<impl>", "exec"), ns)
            exec(compile(test_source, "<test_suite>", "exec"), ns)
            test_fns = [v for k, v in ns.items() if k.startswith("test_") and callable(v)]
            if not test_fns:
                return False
            for tfn in test_fns:
                tfn()
        return True
    except Exception:
        return False


def audit_scientific_pair(
    case_id: str,
    impl_source: str,
    test_source: str,
    entry_fn_name: str,
    contract: ScientificContract,
    seed_offset: int = 0,
    run_mutations: bool = True,
) -> AuditVerdict:
    """Run baseline unit tests alongside all 4 layers of SciVerify."""
    t0 = time.perf_counter()

    # Baseline: Does the existing / LLM-generated unit test suite report 100% PASS?
    naive_passed = run_pytest_style_suite(impl_source, test_source)

    # Layer 1: Static AST circularity / sycophancy scan
    taut_findings = analyze_test_tautology(impl_source, test_source)
    taut_codes = [f.code for f in taut_findings]

    # Layer 2: Microsoft Z3 SMT symbolic counterexample synthesis
    z3_witness = synthesize_z3_witness(contract, seed_offset=seed_offset)

    # Layer 3: Domain-specific AST mutation test-suite sensitivity check
    killed = 0
    total_muts = 0
    if run_mutations:
        mutants = generate_scientific_mutants(impl_source, max_mutants=4)
        total_muts = len(mutants)
        for mut in mutants:
            if not run_pytest_style_suite(mut.mutated_source, test_source):
                killed += 1
    kill_ratio = (killed / total_muts) if total_muts > 0 else 1.0
    surviving = total_muts - killed

    # Layer 4: Execute candidate implementation against Z3 witness & exact invariants
    _, fn = _compile_and_extract(impl_source, entry_fn_name)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        witness_report = verify_contract_with_witness(
            fn,
            contract,
            z3_witness,
            impl_source=impl_source,
            entry_name=entry_fn_name,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AuditVerdict(
        case_id=case_id,
        family=contract.family,
        naive_pytest_passed=naive_passed,
        sciverify_passed=witness_report.passed,
        tautology_flags=taut_codes,
        mutation_kill_ratio=kill_ratio,
        surviving_mutants=surviving,
        total_mutants=total_muts,
        witness=witness_report,
        wall_time_ms=elapsed_ms,
    )


class AdversarialVerificationEngine:
    """High-level stateful runner for batch and CI verification."""

    def __init__(self, run_mutations: bool = True) -> None:
        self.run_mutations = run_mutations

    def audit(
        self,
        case_id: str,
        impl_source: str,
        test_source: str,
        entry_fn_name: str,
        contract: ScientificContract,
        seed_offset: int = 0,
    ) -> AuditVerdict:
        return audit_scientific_pair(
            case_id=case_id,
            impl_source=impl_source,
            test_source=test_source,
            entry_fn_name=entry_fn_name,
            contract=contract,
            seed_offset=seed_offset,
            run_mutations=self.run_mutations,
        )

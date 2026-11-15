"""Command-line interface for SciVerify."""

import argparse
import json
import sys
from pathlib import Path

from .contracts import ScientificContract
from .engine import audit_scientific_pair
from .z3_synthesizer import prove_ieee754_float64_absorption_bitblast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sciverify",
        description="Adversarial verification engine breaking circular validation in AI-assisted research software.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fp = sub.add_parser("prove-fp64", help="Run bit-blasted Z3 QF_FP IEEE-754 binary64 absorption proof.")
    p_fp.add_argument("--min-power", type=int, default=53)

    p_aud = sub.add_parser("audit", help="Audit an implementation file and test file against a scientific contract.")
    p_aud.add_argument("impl_file", type=Path)
    p_aud.add_argument("test_file", type=Path)
    p_aud.add_argument("--fn", required=True, help="Target function name.")
    p_aud.add_argument(
        "--family",
        required=True,
        choices=[
            "ieee754_cancellation",
            "broadcast_shape",
            "cv_leakage",
            "conservation_units",
            "prng_parallel",
            "nan_swallowing",
        ],
    )

    p_bench = sub.add_parser("benchmark", help="Run the 6,000-case empirical torture benchmark.")
    p_bench.add_argument("--per-family", type=int, default=900, help="Adversarial cases per bug family (default 900 -> 5,400 buggy + 600 sound = 6,000 total).")
    p_bench.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "benchmark" / "results")

    args = parser.parse_args(argv)

    if args.cmd == "prove-fp64":
        res = prove_ieee754_float64_absorption_bitblast(args.min_power)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "audit":
        impl_src = args.impl_file.read_text(encoding="utf-8")
        test_src = args.test_file.read_text(encoding="utf-8")
        contract = ScientificContract(name=args.fn, family=args.family)
        verdict = audit_scientific_pair(
            case_id=args.impl_file.stem,
            impl_source=impl_src,
            test_source=test_src,
            entry_fn_name=args.fn,
            contract=contract,
        )
        out = {
            "case_id": verdict.case_id,
            "family": verdict.family,
            "naive_pytest_passed": verdict.naive_pytest_passed,
            "sciverify_passed": verdict.sciverify_passed,
            "tautology_flags": verdict.tautology_flags,
            "mutation_kill_ratio": round(verdict.mutation_kill_ratio, 4),
            "surviving_mutants": verdict.surviving_mutants,
            "witness": {
                "defect_code": verdict.witness.defect_code,
                "defect_summary": verdict.witness.defect_summary,
                "smt_theory": verdict.witness.smt_theory,
                "z3_formula": verdict.witness.z3_formula,
                "witness_description": verdict.witness.witness_description,
                "observed_value": verdict.witness.observed_value,
                "expected_invariant": verdict.witness.expected_invariant,
                "numeric_discrepancy": verdict.witness.numeric_discrepancy,
            } if verdict.witness else None,
            "wall_time_ms": round(verdict.wall_time_ms, 3),
        }
        print(json.dumps(out, indent=2))
        return 0 if verdict.sciverify_passed else 1

    if args.cmd == "benchmark":
        from benchmark.run_torture_benchmark import run_full_benchmark
        summary = run_full_benchmark(per_family_buggy=args.per_family, out_dir=args.out_dir)
        print(json.dumps(summary["headline"], indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

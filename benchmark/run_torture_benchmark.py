"""Execute the 6,000-case empirical torture benchmark and export reproducible artifacts."""

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from sciverify.engine import audit_scientific_pair
from sciverify.z3_synthesizer import prove_ieee754_float64_absorption_bitblast
from benchmark.corpus_generator import generate_benchmark_corpus


def run_full_benchmark(
    per_family_buggy: int = 900,
    per_family_sound: int = 100,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # Run formal bit-blasted Z3 QF_FP Float64 theorem proof once for benchmark manifest
    fp64_proof = prove_ieee754_float64_absorption_bitblast(min_power=53)

    fam_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "buggy_total": 0,
            "naive_pytest_false_green": 0,
            "naive_pytest_caught": 0,
            "sciverify_caught": 0,
            "sciverify_missed": 0,
            "sound_controls_total": 0,
            "sciverify_false_positives": 0,
            "tautology_flagged_cases": 0,
            "total_mutants_tested": 0,
            "surviving_mutants_under_llm_tests": 0,
        }
    )

    regime_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "naive_false_green": 0, "sciverify_caught": 0}
    )

    sample_witnesses: List[Dict[str, Any]] = []
    seen_sample_families: set[str] = set()
    digest_hasher = hashlib.sha256()

    total_cases = 0
    for idx, case in enumerate(generate_benchmark_corpus(per_family_buggy=per_family_buggy, per_family_sound=per_family_sound)):
        total_cases += 1
        # Run full 4-layer verification (AST Tautology + live Z3 SMT synthesis + live domain AST mutation + live invariant witness) across 100% of all 6,000 cases!
        verdict = audit_scientific_pair(
            case_id=case.case_id,
            impl_source=case.impl_source,
            test_source=case.test_source,
            entry_fn_name=case.entry_fn,
            contract=case.contract,
            seed_offset=case.seed_offset,
            run_mutations=True,
        )

        digest_hasher.update(f"{verdict.case_id}:{verdict.naive_pytest_passed}:{verdict.sciverify_passed}\n".encode())
        st = fam_stats[case.family]

        if case.is_buggy_ground_truth:
            st["buggy_total"] += 1
            if verdict.naive_pytest_passed:
                st["naive_pytest_false_green"] += 1
            else:
                st["naive_pytest_caught"] += 1

            if not verdict.sciverify_passed:
                st["sciverify_caught"] += 1
            else:
                st["sciverify_missed"] += 1

            if verdict.tautology_flags:
                st["tautology_flagged_cases"] += 1

            st["total_mutants_tested"] += verdict.total_mutants
            st["surviving_mutants_under_llm_tests"] += verdict.surviving_mutants

            rs = regime_stats[case.test_regime]
            rs["total"] += 1
            if verdict.naive_pytest_passed:
                rs["naive_false_green"] += 1
            if not verdict.sciverify_passed:
                rs["sciverify_caught"] += 1

            if case.family not in seen_sample_families and verdict.naive_pytest_passed and not verdict.sciverify_passed and verdict.witness:
                seen_sample_families.add(case.family)
                sample_witnesses.append(
                    {
                        "case_id": case.case_id,
                        "family": case.family,
                        "subvariant": case.subvariant,
                        "test_regime": case.test_regime,
                        "naive_pytest_passed": verdict.naive_pytest_passed,
                        "sciverify_passed": verdict.sciverify_passed,
                        "tautology_flags": verdict.tautology_flags,
                        "defect_code": verdict.witness.defect_code,
                        "smt_theory": verdict.witness.smt_theory,
                        "z3_formula": verdict.witness.z3_formula,
                        "witness_description": verdict.witness.witness_description,
                        "observed_value": verdict.witness.observed_value,
                        "expected_invariant": verdict.witness.expected_invariant,
                        "numeric_discrepancy": verdict.witness.numeric_discrepancy,
                    }
                )
        else:
            st["sound_controls_total"] += 1
            if not verdict.sciverify_passed:
                st["sciverify_false_positives"] += 1

    wall_s = time.perf_counter() - t_start

    total_buggy = sum(v["buggy_total"] for v in fam_stats.values())
    total_false_green = sum(v["naive_pytest_false_green"] for v in fam_stats.values())
    total_naive_caught = sum(v["naive_pytest_caught"] for v in fam_stats.values())
    total_sci_caught = sum(v["sciverify_caught"] for v in fam_stats.values())
    total_sound = sum(v["sound_controls_total"] for v in fam_stats.values())
    total_fp = sum(v["sciverify_false_positives"] for v in fam_stats.values())
    total_muts = sum(v["total_mutants_tested"] for v in fam_stats.values())
    total_surv = sum(v["surviving_mutants_under_llm_tests"] for v in fam_stats.values())

    summary = {
        "benchmark_version": "0.2.0",
        "reproducibility_sha256": digest_hasher.hexdigest(),
        "wall_clock_seconds": round(wall_s, 3),
        "throughput_cases_per_sec": round(total_cases / wall_s, 1),
        "bitblasted_fp64_theorem": fp64_proof,
        "headline": {
            "total_executable_cases": total_cases,
            "adversarial_buggy_cases": total_buggy,
            "sound_reference_controls": total_sound,
            "llm_unit_test_caught_count": total_naive_caught,
            "llm_unit_test_caught_pct": round(100.0 * total_naive_caught / total_buggy, 2),
            "llm_unit_test_false_green_missed_count": total_false_green,
            "llm_unit_test_false_green_missed_pct": round(100.0 * total_false_green / total_buggy, 2),
            "sciverify_caught_count": total_sci_caught,
            "sciverify_detection_rate_pct": round(100.0 * total_sci_caught / total_buggy, 2),
            "sciverify_false_positive_count": total_fp,
            "sciverify_false_positive_rate_pct": round(100.0 * total_fp / total_sound, 2),
            "domain_ast_mutants_evaluated": total_muts,
            "domain_ast_mutants_surviving_llm_tests": total_surv,
            "domain_ast_mutant_survival_rate_pct": round(100.0 * total_surv / max(total_muts, 1), 2),
        },
        "by_family": dict(fam_stats),
        "by_llm_test_regime": dict(regime_stats),
    }

    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "sample_witnesses.json").write_text(json.dumps(sample_witnesses, indent=2), encoding="utf-8")

    with (out_dir / "family_breakdown.csv").open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "family",
            "buggy_total",
            "naive_pytest_caught",
            "naive_pytest_false_green",
            "naive_false_green_pct",
            "sciverify_caught",
            "sciverify_caught_pct",
            "sound_controls",
            "sciverify_false_positives",
        ])
        for fam, st in sorted(fam_stats.items()):
            fg_pct = round(100.0 * st["naive_pytest_false_green"] / st["buggy_total"], 2)
            sc_pct = round(100.0 * st["sciverify_caught"] / st["buggy_total"], 2)
            writer.writerow([
                fam,
                st["buggy_total"],
                st["naive_pytest_caught"],
                st["naive_pytest_false_green"],
                fg_pct,
                st["sciverify_caught"],
                sc_pct,
                st["sound_controls_total"],
                st["sciverify_false_positives"],
            ])

    return summary


if __name__ == "__main__":
    res = run_full_benchmark()
    print(json.dumps(res["headline"], indent=2))
    print(f"Wall-clock runtime: {res['wall_clock_seconds']}s ({res['throughput_cases_per_sec']} cases/sec)")

"""Self-verification suite ensuring SciVerify is non-circular, sound, and deterministic."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sciverify import (
    ScientificContract,
    analyze_test_tautology,
    audit_scientific_pair,
    generate_scientific_mutants,
    prove_ieee754_float64_absorption_bitblast,
)


class TestSciVerifySelf(unittest.TestCase):
    def test_bitblasted_fp64_absorption_theorem(self) -> None:
        proof = prove_ieee754_float64_absorption_bitblast(53)
        self.assertEqual(proof["status"], "sat")
        self.assertIn("2**53", proof["witness_d"])

    def test_tautology_mirror_and_loose_tolerance_detected(self) -> None:
        impl = "def f(x):\n    return (np.sum(x ** 2) - (np.sum(x) ** 2) / 3) / 2\n"
        test = (
            "def test_f():\n"
            "    v = np.array([1.0, 2.0, 3.0])\n"
            "    expected = (np.sum(v ** 2) - (np.sum(v) ** 2) / 3) / 2\n"
            "    assert np.isclose(f(v), expected, rtol=0.05)\n"
        )
        codes = {t.code for t in analyze_test_tautology(impl, test)}
        self.assertIn("TAUT-MIRROR", codes)
        self.assertIn("TAUT-LOOSE-TOL", codes)
        self.assertIn("TAUT-TOY-DOMAIN", codes)

    def test_catches_false_green_one_pass_variance(self) -> None:
        buggy = (
            "def sample_var(x):\n"
            "    x = np.asarray(x, dtype=np.float64)\n"
            "    return float((np.sum(x ** 2) - (np.sum(x) ** 2) / len(x)) / (len(x) - 1))\n"
        )
        toy_test = (
            "def test_var():\n"
            "    assert np.isclose(sample_var([1.0, 2.0, 3.0]), 1.0)\n"
        )
        contract = ScientificContract(name="sample_var", family="ieee754_cancellation", shift_invariant_mode="unchanged")
        verdict = audit_scientific_pair("t1", buggy, toy_test, "sample_var", contract)
        self.assertTrue(verdict.naive_pytest_passed)
        self.assertFalse(verdict.sciverify_passed)
        self.assertEqual(verdict.witness.defect_code, "DEFECT-IEEE754-SHIFT-CANCELLATION")

    def test_certifies_sound_two_pass_variance_zero_false_positive(self) -> None:
        sound = (
            "def sample_var(x):\n"
            "    x = np.asarray(x, dtype=np.float64)\n"
            "    mu = np.mean(x)\n"
            "    return float(np.sum((x - mu) ** 2) / (len(x) - 1))\n"
        )
        toy_test = "def test_var():\n    assert np.isclose(sample_var([1.0, 2.0, 3.0]), 1.0)\n"
        contract = ScientificContract(name="sample_var", family="ieee754_cancellation", shift_invariant_mode="unchanged")
        verdict = audit_scientific_pair("t2", sound, toy_test, "sample_var", contract)
        self.assertTrue(verdict.naive_pytest_passed)
        self.assertTrue(verdict.sciverify_passed)
        self.assertEqual(verdict.witness.defect_code, "OK")

    def test_domain_mutants_generated_and_cli_audit_exit_code(self) -> None:
        buggy_norm = "def norm_rows(X):\n    X = np.asarray(X, dtype=float)\n    return X / np.sum(X, axis=1)\n"
        syc_test = "def test_n():\n    assert np.allclose(norm_rows(np.eye(3)), np.eye(3))\n"
        muts = generate_scientific_mutants("def norm_rows(X):\n    return X / np.sum(X, axis=1, keepdims=True)\n")
        self.assertGreaterEqual(len(muts), 2)

        with tempfile.TemporaryDirectory() as td:
            ip = Path(td) / "mod.py"
            tp = Path(td) / "test_mod.py"
            ip.write_text(buggy_norm, encoding="utf-8")
            tp.write_text(syc_test, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "sciverify", "audit", str(ip), str(tp), "--fn", "norm_rows", "--family", "broadcast_shape"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["naive_pytest_passed"])
            self.assertFalse(payload["sciverify_passed"])


if __name__ == "__main__":
    unittest.main()

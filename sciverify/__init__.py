"""SciVerify: Adversarial Verification Engine for AI-Assisted Research Software."""

from .contracts import ScientificContract, WitnessReport, AuditVerdict
from .ast_tautology import analyze_test_tautology, TautologyFinding
from .z3_synthesizer import synthesize_z3_witness, prove_ieee754_float64_absorption_bitblast, Z3Witness
from .scientific_mutator import generate_scientific_mutants, MutantSpec
from .metamorphic_runner import verify_contract_with_witness
from .engine import AdversarialVerificationEngine, audit_scientific_pair

__version__ = "0.2.0"

__all__ = [
    "ScientificContract",
    "WitnessReport",
    "AuditVerdict",
    "TautologyFinding",
    "Z3Witness",
    "MutantSpec",
    "AdversarialVerificationEngine",
    "analyze_test_tautology",
    "synthesize_z3_witness",
    "prove_ieee754_float64_absorption_bitblast",
    "generate_scientific_mutants",
    "verify_contract_with_witness",
    "audit_scientific_pair",
]

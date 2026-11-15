"""Declarative scientific verification contracts (Edinburgh WG4 Activity 4B substrate)."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ScientificContract:
    """Independent mathematical/physical contract against which code and tests are verified."""

    name: str
    family: str  # ieee754_cancellation | broadcast_shape | cv_leakage | conservation_units | prng_parallel | nan_swallowing
    description: str = ""
    # Metamorphic & invariant properties
    shift_invariant_mode: Optional[str] = None  # "unchanged" (variance) | "equivariant" (logsumexp) | "small_arg_limit" | "quadratic_product"
    scale_degree: Optional[float] = None  # f(alpha * x) == (alpha ** k) * f(x)
    expected_shape_kind: Optional[str] = None  # "vector_N" | "matrix_ND" | "scalar_pair_N"
    conserved_quantity: Optional[str] = None  # "total_sum" | "probability_simplex" | "dimensional_homogeneity"
    causal_isolation: bool = False  # held-out/future indices must have 0 influence on past/train predictions
    parallel_independent: bool = False  # parallel Monte Carlo worker streams must be uncorrelated
    nan_policy: Optional[str] = None  # "valid_count_mean" | "valid_count_var" | "propagate_nan_mask"
    rtol: float = 1e-6
    atol: float = 1e-8
    params: Dict[str, Any] = field(default_factory=dict)
    reference_oracle: Optional[Callable[..., Any]] = None


@dataclass
class WitnessReport:
    """Minimal concrete counterexample witness exposing a scientific defect."""

    passed: bool
    defect_code: str
    defect_summary: str
    smt_theory: str
    z3_formula: str
    witness_description: str
    observed_value: Any
    expected_invariant: str
    numeric_discrepancy: float


@dataclass
class AuditVerdict:
    """Complete 4-layer adversarial verification verdict for an (implementation, test_suite) pair."""

    case_id: str
    family: str
    naive_pytest_passed: bool  # True means the LLM/unit test suite returned GREEN (PASS)
    sciverify_passed: bool     # True means SciVerify certified no contract/invariant violation
    tautology_flags: List[str]
    mutation_kill_ratio: float
    surviving_mutants: int
    total_mutants: int
    witness: Optional[WitnessReport]
    wall_time_ms: float

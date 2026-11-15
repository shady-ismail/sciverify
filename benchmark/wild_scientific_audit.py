#!/usr/bin/env python3
"""
Wild Open-Source Scientific Repository Audit Script
Runs SciVerify against real functions and pytest suites in cloned scientific libraries.
"""

import ast
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add sciverify to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rscf_adversarial_verifier"))

from sciverify.contracts import ScientificContract
from sciverify.engine import audit_scientific_pair
from sciverify.ast_tautology import analyze_test_tautology

# Wild repositories to audit
WILD_REPOS = {
    "lifelines": "/Users/shady/.gemini/antigravity/scratch/rscf_wild_repo_lab/lifelines",
    "emcee": "/Users/shady/.gemini/antigravity/scratch/rscf_wild_repo_lab/emcee",
    "pyod": "/Users/shady/.gemini/antigravity/scratch/rscf_wild_repo_lab/pyod",
    "astropy": "/Users/shady/.gemini/antigravity/scratch/rscf_wild_repo_lab/astropy",
}

# Contract families to test
CONTRACT_FAMILIES = [
    "ieee754_cancellation",
    "broadcast_shape",
    "cv_leakage",
    "conservation_units",
    "prng_parallel",
    "nan_swallowing",
]

def find_python_files(repo_path: Path, pattern: str = "*.py") -> List[Path]:
    """Find Python files in repository."""
    return list(repo_path.rglob(pattern))

def find_test_files(repo_path: Path) -> List[Path]:
    """Find test files in repository."""
    test_files = []
    for pattern in ["test_*.py", "*_test.py", "tests/*.py", "tests/**/*.py"]:
        test_files.extend(repo_path.rglob(pattern))
    return test_files

def extract_functions_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Extract function definitions from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function source
                func_source = ast.get_source_segment(source, node)
                if func_source:
                    functions.append({
                        "name": node.name,
                        "source": func_source,
                        "file": str(file_path),
                        "lineno": node.lineno,
                    })
        return functions
    except Exception as e:
        return []

def extract_test_functions(file_path: Path) -> List[Dict[str, Any]]:
    """Extract test functions from a test file."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        tests = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                func_source = ast.get_source_segment(source, node)
                if func_source:
                    tests.append({
                        "name": node.name,
                        "source": func_source,
                        "file": str(file_path),
                        "lineno": node.lineno,
                    })
        return tests
    except Exception as e:
        return []

def run_tautology_analysis(impl_source: str, test_source: str) -> List[str]:
    """Run AST tautology analysis on implementation and test pair."""
    findings = analyze_test_tautology(impl_source, test_source)
    return [f.code for f in findings]

def audit_repo(repo_name: str, repo_path: Path) -> Dict[str, Any]:
    """Audit a single repository."""
    print(f"\n=== Auditing {repo_name} ===")
    
    # Find implementation files (non-test)
    all_py_files = find_python_files(repo_path)
    test_files = find_test_files(repo_path)
    
    # Filter out test files from implementation files
    test_file_set = set(str(f) for f in test_files)
    impl_files = [f for f in all_py_files if str(f) not in test_file_set and "__pycache__" not in str(f)]
    
    print(f"  Found {len(impl_files)} implementation files, {len(test_files)} test files")
    
    # Extract functions from a sample of implementation files
    impl_functions = []
    for f in impl_files[:20]:  # Limit to first 20 files
        funcs = extract_functions_from_file(f)
        impl_functions.extend(funcs)
    
    # Extract test functions
    test_functions = []
    for f in test_files[:10]:  # Limit to first 10 test files
        tests = extract_test_functions(f)
        test_functions.extend(tests)
    
    print(f"  Found {len(impl_functions)} functions, {len(test_functions)} test functions")
    
    # Run tautology analysis on pairs
    tautology_results = {
        "TAUT-MIRROR": 0,
        "TAUT-LOOSE-TOL": 0,
        "TAUT-SMOKE": 0,
        "TAUT-TOY-DOMAIN": 0,
        "TAUT-NO-NAN-INPUT": 0,
        "TAUT-SEED-LOCK": 0,
    }
    
    # Pair functions with tests (simple heuristic: same module or nearby)
    pairs_tested = 0
    for impl_func in impl_functions[:30]:  # Limit pairs
        for test_func in test_functions[:10]:
            if pairs_tested >= 20:
                break
            flags = run_tautology_analysis(impl_func["source"], test_func["source"])
            for flag in flags:
                if flag in tautology_results:
                    tautology_results[flag] += 1
            pairs_tested += 1
        if pairs_tested >= 20:
            break
    
    # Run pytest to see what passes
    pytest_result = subprocess.run(
        ["python3", "-m", "pytest", str(repo_path), "-x", "--tb=no", "-q", "--maxfail=5"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_path
    )
    
    return {
        "repo": repo_name,
        "impl_files": len(impl_files),
        "test_files": len(test_files),
        "functions_found": len(impl_functions),
        "tests_found": len(test_functions),
        "pairs_tested": pairs_tested,
        "tautology_flags": tautology_results,
        "pytest_exit_code": pytest_result.returncode,
        "pytest_stdout": pytest_result.stdout[-2000:] if pytest_result.stdout else "",
        "pytest_stderr": pytest_result.stderr[-2000:] if pytest_result.stderr else "",
    }

def main():
    """Main audit function."""
    print("=" * 60)
    print("WILD OPEN-SOURCE SCIENTIFIC REPOSITORY AUDIT")
    print("=" * 60)
    
    results = {}
    
    for repo_name, repo_path_str in WILD_REPOS.items():
        repo_path = Path(repo_path_str)
        if not repo_path.exists():
            print(f"Repository {repo_name} not found at {repo_path_str}")
            continue
        
        try:
            result = audit_repo(repo_name, repo_path)
            results[repo_name] = result
        except Exception as e:
            print(f"Error auditing {repo_name}: {e}")
            results[repo_name] = {"error": str(e)}
    
    # Save results
    output_dir = Path("/Users/shady/.gemini/antigravity/scratch/rscf_adversarial_verifier/benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "wild_corpus_audit_report.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== AUDIT COMPLETE ===")
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for repo_name, result in results.items():
        if "error" in result:
            print(f"  {repo_name}: ERROR - {result['error']}")
        else:
            flags = result.get("tautology_flags", {})
            total_flags = sum(flags.values())
            print(f"  {repo_name}: {result['pairs_tested']} pairs, {total_flags} tautology flags, pytest={result['pytest_exit_code']}")
            for flag, count in flags.items():
                if count > 0:
                    print(f"    {flag}: {count}")

if __name__ == "__main__":
    main()
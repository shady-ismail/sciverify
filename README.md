# SciVerify (`v0.3.1`)

[![SciVerify CI](https://github.com/shady-ismail/sciverify/actions/workflows/ci.yml/badge.svg)](https://github.com/shady-ismail/sciverify/actions/workflows/ci.yml)

**License:** MIT | **Python:** `3.10–3.13` (`numpy`, `z3-solver`) | **Author:** Shady Fouad Saber Ismail (`ORCID: 0009-0005-3920-2333`)

When the same language model writes a numerical Python function and its `pytest` file, test suites frequently turn circular: the test copies the implementation's unstable formula or checks only toy inputs like `[1.0, 2.0, 3.0]` and `np.eye(3)`. CI stays green while floating-point cancellation, rank broadcasting, `NaN` dilution, or cross-validation leakage corrupts scientific results.

`SciVerify` (`pip install -e .`) provides a deterministic external check aligned with Working Group 4 (`4A`, `4B`, `4H`) of the May 2026 Edinburgh Workshop Report (`DOI: 10.5281/zenodo.20320884`).

## Reproduce the Python `3.12.16` Real-World Evaluation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas z3-solver pytest lifelines emcee pyod astropy
pip install -e .
pytest -q
python -m benchmark.real_world_historical_eval
```

### Empirical Results on Real Scientific Python Code (`SHA-256: c3ca7195ca4d98cd...`)

1. **Track A — 18 Documented Historical Scientific Python Bug Classes** (`scipy`, `astropy`, `lifelines`, `emcee`, `pyod`, `statsmodels`, `xarray`):
   - Baseline smoke `pytest` suites caught **`1 / 18` (`5.6%`)**.
   - `SciVerify v0.3.1` dynamically caught **`14 / 18` (`77.8%`)**—including general AST-lifted `80`-digit `Decimal` differential shadow execution filtered by component-wise local condition number (`kappa_i <= 10^4`)—and **missed `4 / 18` (`22.2%`)** (`scipy.special.betainc` transcendental branch accuracy outside `QF_FP`, `lifelines` Breslow-vs-Efron tie convention, `astropy.io.fits` ASCII header parsing, and `scipy.sparse` compiled C++ `int32` overflow above `2^31 - 1` nonzeros).
2. **Track B — `160` Unmodified Production Callables Inspected Live** (`scipy` `1.18.1`, `astropy` `8.0.2`, `lifelines` `0.30.0`, `emcee` `3.1.6`, `pyod` `3.6.8`):
   - **Supported pure-Python/array callables:** `55 / 160` (`34.4%`), median latency **`0.41 ms`** (`p95 = 3.26 ms`).
   - **Unsupported / skipped cleanly (`105 / 160 = 65.6%`):** compiled C/Fortran ufuncs without Python AST (`0`), multi-argument/stateful estimators (`74`), and restricted-domain routines (`31`).
   - **False-positive rate:** **`7 / 55` (`12.73%`)** under uncalibrated heuristic inference, dropping to **`1 / 55` (`1.82%`)** once a 3-line `ScientificContract` specifies equivariance vs invariance.
3. **Internal Solver Regression Suite (Appendix Sanity Check Only):** `python -m sciverify benchmark` runs `6,000` parameterized unit templates verifying internal solver regressions (`11.9s`); because those templates are self-generated, they are treated strictly as internal unit tests rather than real-world efficacy evidence.

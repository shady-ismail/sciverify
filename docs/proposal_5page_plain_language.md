# Breaking Circular Validation in AI-Assisted Scientific Python (`SciVerify`)

**Applicant:** Shady Fouad Saber Ismail (`ORCID: 0009-0005-3920-2333`)  
**Entity Type:** Independent Individual Researcher  
**Call:** Research Software Catalyst Fund — Exploratory Track (Due December 31, 2026)  
**Schedule & Total Request:** April 1 – September 30, 2027 (6 Months) | **USD 47,400** (Direct Costs Only)  
**Open Licensing & Prototype Repository:** **MIT License** (`https://github.com/shady-ismail/sciverify`); **CC0 / CC BY 4.0** on Zenodo (historical bug benchmark & pilot logs)

## 1. The Verification Problem I Built `SciVerify` to Address

When a researcher uses an AI coding assistant to write or refactor a numerical Python function and its `pytest` suite in the same session, validation frequently turns circular. If the generated routine uses an unstable one-pass variance formula, drops `keepdims=True` during rectangular row normalization, standardizes an entire array before splitting cross-validation folds, or replaces `NaN` sensor dropouts with `0.0` before averaging, the generated unit test typically copies the same expression or checks only clean inputs such as `[1.0, 2.0, 3.0]` and `np.eye(3)`. Continuous integration passes while scientific outputs are silently wrong.

Using a second language model to invent edge cases does not reliably fix this. Recent measurements on LLM-generated code checks by Kadri (`DOI: 10.5281/zenodo.21758550`, August 2026) show why: open-ended prompting drops test validity from `75%` to `30%` because the model invents false constraints, whereas restrictive prompting collapses coverage back to basic smoke tests. Catching these defects requires a fast, deterministic check outside the language model.

### Current Prototype (`SciVerify v0.3.1`) — Measured Performance and Honest Boundaries

Before writing this proposal, I built and open-sourced **`SciVerify` (`v0.3.1`)** under the MIT License (`numpy` and `z3-solver` only, Python `3.10–3.13`). It runs four deterministic checks: (1) static AST detection of mirrored-formula and toy-input tests, (2) bounded `500 ms` Microsoft Z3 counterexample synthesis (`QF_FP`, `QF_LIA`, `QF_LIRA`) paired with an AST-lifted `80`-digit `Decimal` differential shadow that separates unstable algorithms from inherently ill-conditioned inputs via component-wise local condition numbers (`kappa_i <= 10^4`), (3) eight numerical AST mutators (`keepdims` drop, `ddof` flip, axis swap, off-by-one slice, `NaN` zero-fill), and (4) metamorphic conservation and causal-isolation checks wired to a coding-agent gate (`SKILL.md`).

Building `v0.3.1` surfaced two practical lessons. First, running under pre-release Python `3.15` broke C-extension collection in `lifelines` and `emcee` due to wheel ABI mismatches, requiring strict lockfile pinning on stable Python `3.12.16`. Second, although `SciVerify` passes `100%` of an internal `6,000`-template unit suite (`11.9s`), self-generated templates only verify internal solver plumbing. I therefore treat the template generator strictly as an internal regression test and benchmarked `SciVerify v0.3.1` directly against real historical bugs and unmodified installed libraries (`benchmark/real_world_historical_eval.py`, `SHA-256: c3ca7195ca4d98cd...`):

| Evaluation Track (Python `3.12.16`, Locked Wheels) | Scope & Installed Packages | Baseline `pytest` | `SciVerify v0.3.1` Result | Documented Out-of-Scope Misses & False Alarms |
| :--- | :--- | :---: | :---: | :--- |
| **Track A: Historical Bug Regressions** | `18` documented upstream bug classes (`scipy`, `astropy`, `lifelines`, `emcee`, `pyod`, `statsmodels`, `xarray`) | `1 / 18` (`5.6%`) | **`14 / 18` (`77.8%`)** | **Missed `4` (`22.2%`):** `betainc` continued-fraction accuracy (outside `QF_FP`), Breslow/Efron tie convention, FITS ASCII header parsing, and C++ `CSR` `int32` overflow above `2^31 - 1` nonzeros. |
| **Track B: Unmodified Wild Library Scan** | `160` live callables in `scipy` `1.18.1`, `astropy` `8.0.2`, `lifelines` `0.30.0`, `emcee` `3.1.6`, `pyod` `3.6.8` | N/A (sound upstream functions) | **`55 / 160` (`34.4%`)** auditable (`0.41 ms` median) | **`65.6%` skipped cleanly** (compiled C/Fortran ufuncs without Python AST, multi-arg stateful classes). **FPR:** `12.73%` (`7/55`) uncalibrated, dropping to **`1.82%` (`1/55`)** with a 3-line contract. |

Closing the practical gap exposed in Track B—expanding pure-Python/NumPy coverage from single-array functions (`34.4%`) to multi-argument `pandas.DataFrame` and `xarray.DataArray` workflows while cutting uncalibrated false alarms below `2.0%`—is the concrete six-month scope of this project.

## 2. Guaranteed Six-Month Deliverables vs. Stretch Goals (Edinburgh WG4 Alignment)

To keep scope realistic for a single investigator over `26` weeks (`700` hours, including `4` weeks of built-in schedule slack in Month 6), this project excludes compiled C/Fortran binary lifters, R wrappers, paid cloud APIs, and travel. Deliverables map directly to Working Group 4 (`4A`, `4B`, `4C`, `4H`) of the May 2026 Edinburgh Workshop Report (`DOI: 10.5281/zenodo.20320884`):

1. **Guaranteed Core Deliverable — `SciVerify v1.0` `pytest` Plugin & Emitted Regression Witnesses (`MIT`, PyPI & GitHub — `WG4-4A`, `4B`):**
   - Multi-argument `numpy`, `scipy`, `pandas.DataFrame` (column schema/index preservation), and `xarray.DataArray` (named-dimension alignment) support via a 5-line YAML file or `@scientific_contract` decorator, cutting uncalibrated heuristic false alarms below `2.0%` on pure-Python array pipelines (`< 15s` CI runtime on Python `3.11–3.13`; compiled C/Fortran extensions remain explicitly out of scope for AST shadow lifting and are verified only at array I/O contract boundaries).
   - Automatic emission of minimal, copy-pasteable 4-line `pytest` counterexample tests whenever Z3 or the `Decimal` shadow catches a defect:
     ```python
     def test_sciverify_witness_solve_quad_roots():
         r1, r2 = solve_quad_roots(1.0, 536870912.0, 1.0)  # Z3 scale witness 2**29 (kappa_true = 2.0)
         assert abs(r2 - (-1.862645149230957e-09)) <= 1e-12 * abs(-1.862645149230957e-09)
     ```
2. **Guaranteed Core Deliverable — `SciVerify-Wild-100` Reproducible Upstream Bug Benchmark (`CC0`, Zenodo DOI — `WG4-4H`):**
   - Building directly on the `18` regression cases already implemented in `v0.3.1`, I will manually verify, isolate, and lock `82` additional real pre-fix / post-fix bug commits from open-source scientific Python git histories (`scipy`, `astropy`, `statsmodels`, `lifelines`, `emcee`, `xarray`) to reach **`100` deterministic, `uv.lock`-pinned historical regressions** (`~5` commits/week across Months 1–4), each recording exact commit SHA, domain tolerances, and expected failure class so `Hypothesis`, `mutmut`, and `SciVerify` can be compared fairly without synthetic padding. *(Expanding the corpus toward `250+` commits via automated git mining is designated strictly as a post-`v1.0` community stretch goal, not a grant bottleneck.)*
3. **Guaranteed Core Deliverable — Three Compensated Maintainer Pilots & Risk Fallback (`CC BY 4.0`, Zenodo DOI — `WG4-4C`):**
   - Three opting-in Python research-software teams run `pytest --sciverify` on active pull requests for `8` weeks (`USD 1,500` stipend each), co-authoring an unfiltered report of real bugs caught, false positives, and contract friction, paired with a free 90-minute virtual US-RSE tutorial.
   - **Schedule & Recruitment Contingency:** If fewer than three external maintainer teams opt in by the end of Month 2, unfilled pilot slots immediately convert to **offline historical pull-request replay** inside the `SciVerify` benchmark repository against archived pre-fix commits from public PyOpenSci packages—**never** submitting unsolicited automated issues or pull requests to upstream maintainers.

## 3. Investigator, Open-Source Etiquette & Zero-Server Sustainability

**Investigator:** I am an independent researcher (`Shady Fouad Saber Ismail`, `ORCID: 0009-0005-3920-2333`) working on Z3 SMT verification, IEEE-754 floating-point error analysis, and Python static/dynamic testing tools. Having written `SciVerify v0.3.1` from scratch, I commit `700 hours` (`~26.9 hours/week` over `26` weeks) to execute engineering, `100`-commit benchmark curation, and pilot support directly.

**Maintainer-First Pilot Design:** Verification tools fail when they flood maintainers with noisy CI failures. Pilots run strictly on an opt-in basis recruited via US-RSE Slack (`#testing`, `#ai-and-rse`) and PyOpenSci discourse across (1) physical/astronomical array reduction, (2) tabular survival/biomedical statistics, and (3) MCMC sampling. Each participating team receives `USD 1,500` (`~25` hours) to compensate review time and document candid failure cases.

**Post-Grant Sustainability & Governance Contingency:** `SciVerify` requires **zero servers, zero databases, and zero paid API keys**—only local `numpy`, `pyyaml`, and `z3-solver` tested on free GitHub Actions (`Python 3.11–3.13`). By Month 6, I will submit `SciVerify` for **PyOpenSci** peer review (whose 2–3 month review cycle continues naturally past grant close at zero cost) followed by *JOSS*. Interested pilot contributors will be invited as co-maintainers; if none elect co-maintainership, the package remains clean, self-contained, and maintainable solo via automated quarterly dependency CI on GitHub and permanent archival DOIs on Zenodo.

## 4. Conservative Six-Month Schedule With Built-In Buffer (April – September 2027)

| Month (2027) | Planned Engineering & Validation Work | Verifiable Public Milestone |
| :--- | :--- | :--- |
| **April (M1)** | Add `pandas.DataFrame` schema/index & `xarray.DataArray` dimension contracts; ship equivariance/domain inference (`< 2%` heuristic FPR); open US-RSE / PyOpenSci pilot call. | `sciverify v0.5.0` on PyPI (`MIT`) + pilot repositories confirmed. |
| **May (M2)** | Isolate and `uv.lock`-pin historical commits `#19–#60` (`42` new commits); ship `< 15s` GitHub Actions workflow + `SKILL.md` coding-agent gate; trigger offline PR-replay fallback for any unfilled slot. | `v0.6.0` + `SciVerify-Wild-60` reproducible alpha archive. |
| **June (M3)** | **Pilot Phase 1:** Run `pytest --sciverify` on live PRs (or offline historical PR replay); record every false alarm, C-extension skip, and `> 500 ms` Z3 timeout. | Mid-pilot usability & FPR log (`CC BY 4.0`) + `v0.7.0` patch. |
| **July (M4)** | Isolate and `uv.lock`-pin historical commits `#61–#100` (completing `SciVerify-Wild-100`); finalize 4-line `pytest` witness emitter and external independent replication script. | `SciVerify-Wild-100` frozen on Zenodo (`CC0` DOI) + `v0.8.5` beta. |
| **August (M5)** | **Pilot Phase 2:** Complete 8-week trials; publish unfiltered 3-repo evaluation report (`caught bugs`, `false alarms`, `setup time`); host free 90-minute virtual hands-on tutorial. | Pilot evaluation report (`CC BY 4.0` Zenodo DOI) + tutorial notebooks. |
| **September (M6)** | **4-Week Buffer & Release Polish (Zero New Feature Scope):** Fold pilot fixes into `SciVerify v1.0.0`, open PyOpenSci packaging submission, and deposit final release archive on Zenodo. | `SciVerify v1.0.0` (`MIT` on PyPI/GitHub) + PyOpenSci submission + Zenodo DOI. |

## 5. Direct-Cost Budget (`USD 47,400` Total | `USD 0` Travel | `USD 0` Indirects)

| Budget Category | Basis & Verifiable Scope | Amount (USD) |
| :--- | :--- | ---: |
| **1. Solo Investigator RSE Engineering & Benchmark Curation** | `700 hours` (`~26.9 hrs/wk` × `26 weeks`, including 4-week M6 buffer) @ `USD 60/hr` for `SciVerify v1.0`, `SKILL.md` gate, manual reproduction of `100` git-pinned historical bug commits (`+82` beyond `v0.3.1`), and pilot/tutorial execution. | USD 42,000 |
| **2. Opt-In Maintainer Pilot Stipends** | `3` external scientific Python maintainer teams × `USD 1,500` (`~25 hrs` each for 8-week PR integration and unfiltered `WG4-4C` usability/false-alarm reporting; unspent stipend returned or reallocated only with program officer approval if <3 external teams opt in). | USD 4,500 |
| **3. Multi-OS CI Matrix & Clean-Room Replication Compute** | Cross-platform (`Linux/macOS/Windows`, `Python 3.11–3.13`) GitHub Actions matrix runners for `100` locked historical environments and clean-room third-party replication checks (`USD 0` hardware; `USD 0` ongoing server cost). | USD 900 |
| **4. Travel & Indirect / Institutional Overheads** | **`USD 0` travel** (virtual tutorial) and **`0.0%` indirect costs** (`<= 10%` Exploratory Track cap). | USD 0 |
| **Total Requested** | **100% Direct Research Software Engineering & Empirical Validation Costs (`<= USD 50,000` Cap)** | **USD 47,400** |

**Administrative Eligibility Note:** Submitted strictly by **Shady Fouad Saber Ismail** (`ORCID: 0009-0005-3920-2333`) as an unaffiliated Independent Individual Researcher. No for-profit company, LLC, or employer holds any ownership, sponsorship, governance role, or commercial claim over this application or any `SciVerify` outputs, and zero funds cover post-award maintenance salary.

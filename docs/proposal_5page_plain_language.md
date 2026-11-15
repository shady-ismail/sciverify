# SciVerify: Breaking Circular Validation in AI-Assisted Scientific Python

**Lead Applicant:** Shady Ismail (Legal Name: Shady Fouad Saber Ismail | `ORCID: 0009-0005-3920-2333`)  
**Prototype & Benchmark Repositories (MIT / CC0):** `https://github.com/shady-ismail/sciverify` (`v0.3.1`) · `https://github.com/shady-ismail/sciverify-benchmarks`

### 1. What problem or opportunity are you trying to address?

When researchers use generative AI coding assistants to write or refactor a numerical Python function and its `pytest` suite in the same session, validation routinely becomes circular. If the AI-generated routine uses a numerically unstable one-pass variance formula, silently drops `keepdims=True` during rectangular row normalization, standardizes an entire array before splitting cross-validation folds (data leakage), or replaces `NaN` sensor dropouts with `0.0` before averaging, the co-generated unit test frequently mirrors the exact same flawed expression or checks only benign smoke inputs such as `[1.0, 2.0, 3.0]` and `np.eye(3)`. Continuous integration turns green while downstream scientific results are silently wrong.

This project develops, empirically validates, and deploys **`SciVerify v1.0`**—an open-source (`MIT`), deterministic verification plugin for `pytest` and coding-agent workflows (`SKILL.md`) designed for research software engineers (RSEs), domain scientists, and open-source scientific Python maintainers (`numpy`, `scipy`, `pandas`, `xarray`). Rather than trusting a second probabilistic LLM prompt to audit AI-written code, `SciVerify` enforces independent mathematical and structural checks outside the language model so AI-assisted scientific Python fails loudly when tests are tautological or numerically unsound.

### 2. How is the research software community addressing this issue today, and where are the gaps?

Today, research software teams evaluate AI-assisted code primarily through (a) standard CI coverage metrics, (b) asking a second LLM to review pull requests or generate edge-case tests, (c) property-based fuzzing (`Hypothesis`), or (d) generic AST mutation testing (`mutmut`). Each leaves a documented verification gap on numerical research code:

- **Line coverage and LLM-generated test suites miss mirrored-logic bugs:** Coverage confirms a line executed, not that the assertion is independent of the implementation. Recent empirical measurements on LLM-generated assertions by Kadri (`DOI: 10.5281/zenodo.21758550`, August 2026) show that open-ended prompting drops assertion validity from `75%` to `30%` due to hallucinated invariants, whereas restrictive prompting collapses tests back to trivial toy inputs.
- **Random floating-point fuzzing rarely hits narrow cancellation or domain-boundary regimes** without hand-authored strategies, while generic syntax mutators (`mutmut`) generate hundreds of trivial operator mutants without targeting scientific Python failure modes (`ddof=0/1`, `keepdims` broadcasting collapse, silent `NaN` zero-imputation, or `float64` catastrophic cancellation).
- **The May 2026 Edinburgh Workshop Report** (*Research Software & Generative AI*, `DOI: 10.5281/zenodo.20320884`, Working Group 4: `4A`, `4B`, `4C`, `4H`) explicitly highlighted the absence of fast, low-noise, deterministic verification gates and real-world historical bug benchmarks for AI-assisted scientific code.

### 3. What is your proposed approach? What is different about it? Why do you expect it to succeed? How is this team well-positioned to undertake the work?

**Distinctive Technical Approach & Working `v0.3.1` Prototype:** Before submitting this application, I built and open-sourced **`SciVerify v0.3.1`** (`https://github.com/shady-ismail/sciverify`, MIT License, Python `3.11–3.13`, green multi-OS CI). It combines four deterministic checks that require zero servers and zero LLM API calls:
1. **Static AST Circularity & Toy-Input Detection:** Flags unit tests that copy the target function's arithmetic AST or test exclusively small constant/identity arrays.
2. **Condition-Filtered Z3 SMT + 80-Digit `Decimal` Differential Shadow:** Synthesizes bounded (`500 ms`) Microsoft Z3 counterexamples (`QF_FP`, `QF_LIA`, `QF_LIRA`) paired with an AST-lifted `80`-digit `Decimal` reference oracle. Crucially, `SciVerify` computes component-wise local condition numbers (`kappa_i <= 10^4`) so it flags *algorithmically unstable implementations* while suppressing false alarms on inherently ill-conditioned inputs.
3. **Domain-Specific Scientific AST Mutators:** Injects eight targeted numerical defects (`keepdims` removal, `ddof` flip, axis transposition, off-by-one window slice, `NaN` zero-fill) to verify whether the test suite actually fails.
4. **Coding-Agent Gate (`SKILL.md`):** Blocks AI coding assistants (`Claude Code`, `Cursor`, `Codex`, `Antigravity`) from marking a task complete until `pytest --sciverify` passes, adapting naturally as underlying LLMs evolve because verification operates on the emitted Python AST and IEEE-754 behavior rather than model weights.

| `SciVerify v0.3.1` Prototype Benchmark (`Python 3.12`, Locked Wheels) | Scope & Upstream Libraries | Baseline `pytest` | `SciVerify v0.3.1` | Documented Misses & Honest Boundaries |
| :--- | :--- | :---: | :---: | :--- |
| **Track A: Historical Bug Regressions** (`sciverify-benchmarks`) | `18` real upstream bug classes (`scipy`, `astropy`, `lifelines`, `emcee`, `pyod`, `statsmodels`, `xarray`) | `1 / 18` (`5.6%`) | **`14 / 18` (`77.8%`)** | Missed `4`: `betainc` continued-fraction accuracy, Breslow/Efron tie convention, FITS ASCII parsing, and C++ `CSR` `int32` overflow. |
| **Track B: Unmodified Wild Library Scan** | `160` live callables across `scipy`, `astropy`, `lifelines`, `emcee`, `pyod` | N/A | **`55 / 160` (`34.4%`)** auditable (`0.41 ms` median) | `65.6%` skipped cleanly (compiled C/Fortran ufuncs, multi-arg stateful classes). Uncalibrated FPR `12.73%` (`7/55`) -> **`1.82%` (`1/55`)** with a 3-line contract. |

**Principal Six-Month Activities & Investigator Positioning:** Closing the exact gaps measured in Track B defines our six-month engineering scope: (a) shipping **`SciVerify v1.0`** with multi-argument `pandas.DataFrame` schema/index and `xarray.DataArray` named-dimension contracts, `< 2.0%` uncalibrated false-positive rate, and automatic 4-line copy-pasteable `pytest` counterexample witness generation; (b) manually isolating and `uv.lock`-pinning `82` additional real pre-fix/post-fix upstream bug commits to publish **`SciVerify-Wild-100`** (`CC0` on Zenodo); and (c) running **three compensated 8-week maintainer pilots** recruited via US-RSE and PyOpenSci. As an independent formal-verification and numerical-computing researcher (`ORCID: 0009-0005-3920-2333`) who engineered `SciVerify v0.3.1` and its `18`-bug benchmark from scratch, I commit `700 hours` (`~26.9 hrs/week`) to execute the work directly.

### 4. Who will use or benefit from this work, and what difference will it make?

Primary beneficiaries are **(1) scientific Python maintainers and RSEs** reviewing AI-assisted pull requests who gain a `< 15s` CI check that emits concrete 4-line counterexample witnesses rather than vague warnings; **(2) domain scientists and graduate students** using AI coding agents who avoid silent numerical corruption (`NaN` masking, data leakage, broadcasting collapse); and **(3) empirical software-engineering researchers** who gain `SciVerify-Wild-100` (`CC0`), a reproducible `100`-commit benchmark comparing `Hypothesis`, `mutmut`, and `SciVerify`. Three compensated pilot teams across astronomical array reduction, survival/biomedical statistics, and MCMC sampling will co-design contract ergonomics and co-author an unfiltered evaluation report, followed by a free 90-minute virtual US-RSE hands-on tutorial.

### 5. What are the risks, limitations, and potential unintended consequences?

- **False-Alarm Fatigue & Maintainer Burden:** Noisy verification tools get disabled in CI. *Mitigation:* Enforce condition-number filtering (`kappa_i <= 10^4`), strict `< 2.0%` false-alarm gate before pilot rollout, `500 ms` per-function solver caps, and strictly opt-in compensated pilots (**zero** unsolicited automated issues or PRs ever opened on upstream repositories).
- **Pilot Recruitment Delays:** External maintainers have limited bandwidth. *Mitigation:* Offer compensated honoraria; if fewer than three external teams opt in by Month 2, unfilled slots convert immediately to **offline historical pull-request replay** against archived pre-fix commits from PyOpenSci packages so empirical evaluation never stalls.
- **Hard Technical Boundaries (Explicitly Communicated):** AST shadow lifting cannot see inside compiled C/C++/Fortran extensions (`ufuncs`, `BLAS/LAPACK`), transcendental special-function approximations outside `QF_FP`, or domain conventions not expressed in contracts. `SciVerify` skips non-Python frames cleanly with explicit diagnostics and checks compiled routines strictly at array I/O contract boundaries.
- **Over-Reliance Risk ("False Sense of Security"):** Passing `SciVerify` does not prove full domain correctness. CLI and CI reports explicitly state verified scope vs. skipped C-extensions so users treat `SciVerify` as a defect filter, not an infallible oracle.

### 6. What are the project’s major activities and timeline?

Designed for a strict **six-month timebox** (`Months 1–6`: April 1 – September 30, 2027 per program schedule, or adaptable to any six-month start date, with **4 weeks of dedicated schedule slack in Month 6** and zero new feature scope in the final month):

| Month | Major Engineering, Benchmark & Pilot Activities | Key Public Deliverable |
| :--- | :--- | :--- |
| **Month 1** | Implement `pandas.DataFrame` & `xarray.DataArray` multi-argument contracts; calibrate domain/equivariance inference (`< 2.0%` FPR); open US-RSE / PyOpenSci opt-in pilot call. | `sciverify v0.5.0` (`MIT` on PyPI/GitHub) + pilot repos confirmed. |
| **Month 2** | Isolate and `uv.lock`-pin historical upstream bug commits `#19–#60` (`+42` commits); ship `< 15s` GitHub Action + `SKILL.md` agent gate; trigger offline PR-replay fallback if `< 3` external teams confirmed. | `v0.6.0` + `SciVerify-Wild-60` reproducible alpha corpus. |
| **Month 3** | **Pilot Phase 1:** Run `pytest --sciverify` on active PRs (or offline historical PR replay); log every caught bug, false alarm, C-extension skip, and solver timeout. | Mid-pilot FPR & friction log (`CC BY 4.0`) + `v0.7.0`. |
| **Month 4** | Isolate and `uv.lock`-pin historical commits `#61–#100` (completing `SciVerify-Wild-100`); ship 4-line `pytest` witness emitter and one-command clean-room replication script. | `SciVerify-Wild-100` frozen on Zenodo (`CC0` DOI) + `v0.8.5`. |
| **Month 5** | **Pilot Phase 2:** Complete 8-week trials; publish unfiltered 3-pilot report (including misses and false positives); deliver free 90-minute virtual US-RSE hands-on tutorial. | Unfiltered Pilot Evaluation Report (`CC BY 4.0` DOI) + notebooks. |
| **Month 6** | **4-Week Buffer & Release Hardening:** Fold pilot fixes into `SciVerify v1.0.0`, submit package to **PyOpenSci** peer review (followed by *JOSS*), and deposit final archival bundle on Zenodo. | `SciVerify v1.0.0` (`MIT`) + PyOpenSci submission + Zenodo DOI. |

### 7. How will you know whether the project has succeeded?

Progress and success will be assessed against five falsifiable, publicly auditable criteria:
1. **Detection & Low Noise:** `>= 80%` recall on pure-Python numerical bug classes across `SciVerify-Wild-100`, paired with **`< 2.0%` uncalibrated false-positive rate** on unmodified upstream functions (`scipy`, `astropy`, `pandas`, `xarray` workflows) and `< 15s` median PR check runtime.
2. **Actionable Witness Quality:** `100%` of flagged numerical counterexamples emit a standalone, deterministic 4-line `pytest` regression test that reproduces the failure on a clean runner without Z3 installed.
3. **Independent Reproducibility:** Clean-room third-party execution of all `100` `uv.lock`-pinned historical bug environments on Linux and macOS via a single command (`CC0` Zenodo archive).
4. **Honest Empirical Pilot Evidence:** Publication of the co-authored `CC BY 4.0` 3-pilot report documenting real defects caught alongside every false positive, setup bottleneck, and out-of-scope miss, plus `>= 25` live/asynchronous RSE participants completing the open tutorial.
5. **Zero-Server Long-Term Viability:** Active **PyOpenSci** peer-review submission and `v1.0.0` release on PyPI requiring `$0/month` post-grant hosting or API costs.

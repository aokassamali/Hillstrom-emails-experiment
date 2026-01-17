# Pre-Analysis Plan (PAP) - Hillstrom Email Marketing Experiment

## Scope: what this plan covers
This PAP pre-specifies:
- estimands and estimators
- uncertainty quantification
- multiple comparisons approach
- SRM and balance checks
- guardrails and decision logic
- robustness/sensitivity suite
- a prospective stopping/peeking policy (even though the dataset is retrospective)

## A) Data and population

### A1) Analysis population
Include all randomized customers with:
- a valid arm label in {control, mens, womens}
- valid outcome records per rules below

### A2) Outcome definitions
- visit_i in {0,1}
- conversion_i in {0,1}
- spend_i >= 0, with unconditional definition: if no purchase, spend_i = 0 (not missing)

Derived:
- emailed_i = 1 if arm in {mens, womens}, else 0
- profit_i = m * spend_i - c * emailed_i, with defaults:
  - m = 0.40, c = $0.01

### A3) Exclusion rules (objective only)
- spend < 0 -> exclude (impossible)
- missing arm assignment -> exclude
- duplicated customer IDs -> exclude duplicates (keep first occurrence only if ordering is deterministic; otherwise exclude all duplicates and document)

## B) Health checks (must run before treatment claims)

### B1) SRM (Sample Ratio Mismatch)
Test: chi-square goodness-of-fit between observed counts and expected allocation.

Expected allocation:
- If known from dataset documentation, use it.
- Otherwise, declare and document a working assumption (common Hillstrom split is not always equal; do not guess silently).

Flag rule:
- Flag if p < 0.001 OR if practical deviation is large.

Action:
- If flagged: halt interpretation; investigate ETL joins, filtering, arm labeling, and missingness patterns.

### B2) Covariate balance
Compute standardized mean differences (SMD) on key pre-treatment covariates.

Flag rule:
- Any |SMD| > 0.10 -> report imbalance; include covariate-adjusted estimator as sensitivity.

Quiz (1 sentence):
Why is SRM a "blocking" check, but imbalance is usually "report + adjust" rather than "invalidate"?

## C) Primary analysis (robust estimation + uncertainty)

### C1) Primary estimand
For each treatment k in {Mens, Womens}:
  tau_k = E[profit | k] - E[profit | control]

### C2) Primary estimator
Difference in means (ITT):
  tau_hat_k = mean_profit_k - mean_profit_control

### C3) Uncertainty quantification (primary)
Because profit/spend is heavy-tailed:
- Use nonparametric bootstrap (resample customers within arms; B = 10,000) to form 95% CIs.
- Also compute a regression cross-check:
  - OLS of profit on treatment indicators with HC3 robust SEs.

## D) Multiple comparisons (core)
Primary family:
- Mens vs Control
- Womens vs Control

Control Type I error with:
- Holm procedure, FWER alpha = 0.05

Report:
- raw p-values
- Holm-adjusted p-values
- reject / not reject for each comparison

Secondary comparisons:
- Mens vs Womens is exploratory; report without claiming "primary win."

## E) Guardrails (hard constraints)

Guardrails (vs Control):
- Visit: delta visit >= -0.2 pp
- Conversion: delta conversion >= -0.1 pp

Estimator:
- difference in proportions vs control

Uncertainty:
- bootstrap 95% CI (preferred for consistency) OR Wilson/Newcombe CIs (acceptable)

Pass rule:
- guardrail passes if lower 95% CI >= -delta

## F) Practical significance: MES
A treatment must satisfy:
- tau_hat_k >= $0.05 profit per customer (= $50 per 1,000)

This is evaluated alongside Holm-adjusted statistical evidence.

## G) Variance reduction (pre-specified sensitivity)
If predictive pre-treatment covariates exist, run an adjusted estimator:

profit_i = beta_0 + beta_1 * 1[Mens] + beta_2 * 1[Womens] + gamma^T X_i + epsilon_i

Use HC3 robust SEs and/or bootstrap.

Interpretation:
- adjustment is a precision sensitivity, not a changed estimand.

## H) Robustness & sensitivity suite (pre-committed)
Run the primary ATE under:
1) Winsorization at 99th percentile
2) 1% trimmed mean
3) log(1+spend) (secondary scale)
4) Two-part decomposition (secondary narrative):
   - ATE on conversion
   - ATE on spend among converters
   - Interpret which channel drives the overall effect

Decision uses the primary estimator; robustness informs confidence and next steps.

## I) Stopping / peeking policy (prospective statement)
Default:
- Fixed-horizon test; analyze once at the end.

If interim looks are required:
- Pre-specify number and timing of looks.
- Use alpha-spending (e.g., O'Brien-Fleming style) and maintain the same multiple-comparisons family definition.

## J) Reporting checklist (what must be in the final readout)
- SRM result + interpretation
- Balance table (SMDs)
- Main table: ATE on profit + bootstrap CI + raw p + Holm p + MES + guardrails
- Guardrail table
- Robustness grid
- One-page decision memo

Explain-it-back prompt (2-3 sentences): Why is unconditional spend (zeros included) better aligned with the randomized assignment than spend among purchasers?

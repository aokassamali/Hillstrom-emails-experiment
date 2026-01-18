# Decision Memo — Hillstrom Email Campaign (3-arm RCT)

**To:** Marketing / Growth Stakeholders  
**From:** Data Science (Experimentation Readout)  
**Date:** 2026-01-17  
**Subject:** Which email creative to ship, and whether targeting adds incremental value

---

## Executive summary (decision + rationale)

**Decision:** **Ship Mens e-mail to all customers** (**treat-all Mens**).

**Why this is the most defensible choice:**
- Both email creatives increase **profit/customer** vs control; **Mens** is larger and more robust.
- Results pass **experiment health** (SRM + covariate balance), **hard guardrails**, **multiple comparisons control** (Holm FWER), and **minimum effect size** (MES).
- Heavy-tail sensitivity shows uplift is partly tail-driven (expected in retail), but **Mens remains more robust** than Womens.
- A targeting/uplift extension was implemented and evaluated correctly (cross-fitting + IPW/DR with sanity checks). **Targeting underperforms treat-all Mens**, indicating weak *learnable* heterogeneity with available pre-treatment covariates.

---

## Context

We analyze a 3-arm randomized controlled trial:

- **Control:** no e-mail  
- **Mens e-mail**  
- **Womens e-mail**

**Unit:** customer  
**Outcomes:** visit, conversion, spend  
All effects reported are **ITT** (intention-to-treat): differences by assigned arm.

---

## Primary metric, assumptions, and success criteria

### Primary metric: profit per customer
We evaluate **profit/customer** as:

\[
profit_i = m \cdot spend_i - c \cdot \mathbf{1}\{emailed_i\}
\]

Defaults (configurable):
- margin \(m = 0.40\)
- email cost \(c = 0.01\) per emailed customer

**Spend is unconditional** (0 if no purchase).

### Practical significance: MES
Minimum Effect Size (MES): **0.05 profit/customer**.  
Interpretation: even if an effect is statistically significant, we only “ship” if it is **big enough to matter**.

---

## Guardrails (hard constraints)

We treat guardrails as **hard constraints** because harm risk dominates upside in this decision.

- **Visit rate** must not decrease by more than **0.2pp**
- **Conversion rate** must not decrease by more than **0.1pp**
- Pass rule: the **lower bound** of the 95% CI must be ≥ the threshold

Both treatments pass comfortably with positive lifts:
- Mens: visit **+7.659pp**; conversion **+0.681pp**
- Womens: visit **+4.523pp**; conversion **+0.311pp**

---

## Experiment health

### SRM (randomization integrity)
Expected allocation is 1/3–1/3–1/3.

- chi² = **0.203**, p = **0.904** → no evidence of sample-ratio mismatch (SRM)

### Covariate balance
Standardized mean differences (SMDs) are uniformly small (~0.00–0.02) → consistent with randomization.

---

## Main results (confirmatory family)

### Confirmatory family definition
Primary family (size = 2):
1) Mens vs Control  
2) Womens vs Control  

**Note:** Mens − Womens is treated as **exploratory** and is not included in Holm adjustment.

### Estimation + uncertainty (why these choices)
- **Estimator:** ITT difference-in-means vs control (profit/customer).
- **Uncertainty:** bootstrap 95% confidence intervals.  
  Rationale: spend/profit is **sparse and heavy-tailed**, so normal approximations can be fragile.
- **Hypothesis tests:** one-sided tests for decision (“is uplift > 0?”), while reporting **two-sided 95% CIs** for interpretation.
- **Multiple comparisons:** Holm FWER control at α=0.05 across the two confirmatory comparisons.  
  Rationale: this controls the probability of *any* false positive in the primary family.

### Profit uplift vs control (profit/customer)
- **Mens:** τ̂ = **+0.298**, 95% CI **[0.185, 0.415]**, Holm p ≈ **0.00020**, MES pass ✅  
- **Womens:** τ̂ = **+0.160**, 95% CI **[0.058, 0.263]**, Holm p ≈ **0.00110**, MES pass ✅  

**Decision rule:** pick the best eligible arm (passes guardrails + Holm + MES) → **Mens**.

### Exploratory creative comparison (Mens − Womens)
- Δ = **+0.138**, 95% CI **[0.013, 0.264]**, p ≈ **0.029** (two-sided bootstrap)

Interpretation: evidence Mens > Womens, but treated as exploratory.

---

## Heavy-tail risk assessment (robustness)

### Why we did this
Retail spend is sparse and heavy-tailed; a small fraction of customers can contribute a large share of revenue (“whales”). We quantify how sensitive profit uplift is to the top tail.

### Tail sensitivity (remove top x% customers by spend rank within each arm)
At x = 0.5%:
- Mens τ̂ ≈ **0.118** (CI ≈ **[0.095, 0.141]**)  
- Womens τ̂ ≈ **0.043** (CI ≈ **[0.029, 0.058]**) → below MES

Tail-risk breakpoint (MES = 0.05):
- Mens breaks near **1.0%**
- Womens breaks near **0.5%**

**Important interpretation note:** when removing enough top spenders, spend approaches 0 and profit converges to **−email_cost** by construction (e.g., −0.01). This is expected, not a bug.

### Among spenders only (spend > 0)
Because conversion <1%, removing the top 1% of *all customers* can effectively remove all purchasers. To isolate sensitivity among buyers, we repeat tail removal within the **spend>0** subset. Uplift remains positive for both arms; Mens remains larger.

**Business interpretation:** tail dependence is acceptable if the business intentionally monetizes whales; the main risk is non-repeatability (a small set of large purchasers may be unstable over time). Mens appears more robust on this axis.

---

## Sensitivity to business assumptions (margin/cost grid)

We recompute profit uplift across:
- margin ∈ {0.30, 0.40, 0.50}
- email_cost ∈ {0.005, 0.010, 0.020}

Results remain positive and Mens remains higher across the grid.

---

## Targeting (HTE/uplift) extension — recommendation is **not** to deploy

### What was attempted (correctly)
- **Pre-treatment covariates only** (no leakage)
- Baseline **T-learner** outcome models (RF/GBR options)
- **Cross-fitting / honest evaluation**
- Policy value estimators:
  - **IPW** and **DR/AIPW**
  - Sanity checks confirm constant-policy IPW matches within-arm means (evaluation plumbing is correct)

### Key policy value results (binary targeting: Mens vs none)
| Policy | Estimator | Value_hat | 95% CI | Treated_rate | Δ vs treat-all Mens |
|---|---|---:|---:|---:|---:|
| **treat_all_mens** | DR | **0.558715** | [0.469296, 0.654596] | 1.000000 | 0.000 |
| **send_or_not_mens** | DR | **0.433482** | [0.355487, 0.513429] | 0.653234 | −0.125233 |

**Interpretation:** targeting under current features **reduces expected value** relative to treat-all Mens. Decile diagnostics show weak/unstable ranking (non-monotone), consistent with limited learnable signal.

**Conclusion:** do not deploy targeting/uplift with current covariates.

---

## Recommendation (stakeholder-ready)

- **Ship Mens e-mail to all customers** (highest expected profit uplift).
- Mens profit uplift vs control is **+0.298 profit/customer** (95% CI **[0.185, 0.415]**) under \(m=0.40, c=0.01\).
- Womens is also positive (**+0.160**) but smaller and more tail-dependent; Mens is preferred.
- Experiment integrity checks pass: **no SRM** (p=0.904) and strong covariate balance.
- Hard guardrails pass with large positive lifts on **visit** and **conversion**.
- Results are robust under **Holm FWER** control and exceed **MES=0.05**.
- Tail-risk is non-trivial; Mens is more robust than Womens (MES breakpoints ~1.0% vs ~0.5%).
- Profit uplift remains positive under reasonable margin/cost assumptions (grid sensitivity).
- **Do not deploy targeting** with current features; it underperforms treat-all Mens under unbiased DR evaluation.
- If richer covariates become available, reassess targeting under the same policy-value framework.

---

## Risks and mitigations

### Risk: tail dependence (“whales drive the mean”)
- **Mitigation:** report tail-risk breakpoints; monitor distributional metrics post-launch (top-x spend share), not just averages; track repeatability over time.

### Risk: external validity and long-run effects not observed
- **Mitigation:** run a prospective test with a longer horizon; add guardrails for unsubscribe/complaints/churn if available.

### Risk: over-optimizing with targeting methods
- **Mitigation:** treat targeting as a separate product feature requiring incremental value vs treat-all; do not ship if it fails value tests (as observed here).

---

## Next steps (if this were live)

1) Launch Mens campaign broadly; measure profit, visit, conversion, and tail concentration metrics over time.  
2) Run a follow-up RCT with longer window + harm metrics (unsubscribe/complaints) and additional creatives/frequency caps.  
3) If richer covariates become available (behavioral logs, engagement history), rerun targeting and require:
   - policy value beating treat-all Mens with DR CIs
   - stable/monotone decile diagnostics

---

## Appendix: artifact pointers

- Primary results: `reports/tables/main_results.csv`
- Guardrails: `reports/tables/guardrails.csv`
- SRM: `reports/tables/srm.csv`
- Balance: `reports/tables/balance.csv`
- Robustness suite: `reports/tables/robustness.csv`
- Tail sensitivity (all customers): `reports/tables/tail_sensitivity_profit_ci.csv`
- Tail sensitivity (spenders only): `reports/tables/tail_sensitivity_pos_only_profit_ci.csv`
- Tail risk summary: `reports/tables/tail_risk_summary.csv`
- Profit sensitivity grid: `reports/tables/profit_sensitivity_grid.csv`
- Targeting extension outputs: `reports/extension/*`

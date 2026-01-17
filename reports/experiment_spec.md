# Experiment Spec - Hillstrom Email Marketing (Mens vs Womens vs Control)

## TL;DR decision
Ship/no-ship + choose best creative (Mens or Womens) for send-to-all, subject to guardrails and practical significance.

## 1) Objective
Determine whether sending an email (Mens or Womens) increases incremental profit per randomized customer relative to no email, and if so, which creative should be the default.

### Analogy -> formal
- Analogy: pick the fastest car that still passes safety inspection and clears a minimum "worth buying" bar.
- Formal: choose the treatment with the highest estimated uplift subject to (i) experiment health, (ii) multiplicity-controlled evidence, (iii) minimum effect size, (iv) guardrail constraints.

## 2) Arms and unit
Arms
- Control: no email
- Mens: Mens email
- Womens: Womens email

Randomization unit: customer
Unit of analysis: customer
Estimand type: ITT (intention-to-treat)

## 3) Primary metric and estimand
Primary metric: profit per randomized customer
- spend_i = 0 if no purchase
- profit_i = m * spend_i - c * 1[emailed_i]

Defaults (frozen for the core readout):
- Margin m = 0.40
- Email cost c = $0.01

Primary estimand: ITT ATE on profit
For each treatment k in {Mens, Womens}:
  tau_k = E[profit | k] - E[profit | control]

Why unconditional spend?
- Analogy: average calories per person at a party includes people who ate nothing (0), because the invitation is the "treatment."
- Formal: unconditional outcomes align with ITT and avoid post-treatment conditioning.

## 4) Hypotheses (primary family) and multiple comparisons
Primary family:
- Mens vs Control
- Womens vs Control

Hypotheses (one-sided; shipping is directional):
- H0: tau_k <= 0
- H1: tau_k > 0

Multiplicity control:
- Holm procedure controlling FWER at alpha = 0.05 over the 2 tests.

Secondary / descriptive (explicitly exploratory):
- Mens vs Womens

## 5) Guardrails (hard constraints)
Guardrails must not degrade beyond tolerances (vs Control):
- Visit rate: delta visit >= -0.2 percentage points (pp)
- Conversion rate: delta conversion >= -0.1 pp

Pass rule (conservative):
- Guardrail passes if the lower bound of the 95% CI is >= -delta.
- Otherwise, guardrail fails.

## 6) Minimum Effect Size (MES) - practical significance
- MES: +$0.05 profit per customer = +$50 profit per 1,000 customers emailed

## 7) Experiment health checks (blocking)
1) SRM (Sample Ratio Mismatch): chi-square goodness-of-fit vs expected allocation
2) Covariate balance: standardized mean differences (SMD) for key pre-treatment covariates
3) Data sanity: outcomes in valid ranges (e.g., spend >= 0)

If SRM flags, halt interpretation and investigate data joins/filters/assignment integrity.

## 8) Decision policy (C gates + B selection)
A treatment is eligible to ship if ALL are true:
1) Health checks pass (SRM not flagged; data sanity ok)
2) Holm-adjusted evidence supports tau_k > 0 at alpha = 0.05
3) Practical significance: tau_hat_k >= $0.05 per customer
4) Guardrails pass

If both Mens and Womens are eligible, choose the one with higher estimated profit uplift.

## 9) Risks and mitigations
- Heavy-tailed spend inflates variance -> bootstrap CIs + robust sensitivity suite
- Winner's curse in best-arm selection -> gates + multiplicity + explicit "exploratory" labeling
- Retrospective dataset -> specify prospective stopping policy in the PAP

## 10) Outputs
- Main results table (profit ATE + Holm adjustment)
- Guardrail table
- Robustness grid
- Written decision memo (ship/no-ship + rationale + risks + next test)

# Hillstrom Email Experiment — A/B Testing + Decision Readout

This repository is an end-to-end experimentation workflow using the **Hillstrom email marketing dataset** (3-arm randomized controlled trial). It includes a preregistered-style spec, a pre-analysis plan, experiment health checks, robust inference with multiple comparisons control, a decision memo, and a targeting/HTE extension that is evaluated honestly and **fails gracefully** when it doesn’t add business value.

> **Bottom line:** both email treatments increase profit vs control; **Mens e-mail** is the recommended default.  
> A targeting/uplift extension was implemented correctly, but shows **weak learnable heterogeneity** with available pre-treatment features; targeting underperforms **treat-all Mens**.

---

## Results at a glance

**Primary metric:** profit/customer, \(profit_i = m \cdot spend_i - c \cdot 1[emailed_i]\), defaults \(m=0.40, c=0.01\).  
**Confirmatory family:** {Mens vs Control, Womens vs Control} with Holm FWER \(\alpha=0.05\).  
**MES:** 0.05 profit/customer.  
**Guardrails (hard constraints):** visit ≥ −0.2pp; conversion ≥ −0.1pp (lower 95% CI bound must exceed threshold).

| Comparison | Profit uplift \(\hat\tau\) | 95% CI | Holm p | MES | Guardrails | Outcome |
|---|---:|---:|---:|---:|---:|---|
| **Mens vs Control** | **+0.298** | [0.185, 0.415] | ~0.00020 | Passes | Passes | **Ship** |
| **Womens vs Control** | **+0.160** | [0.058, 0.263] | ~0.00110 | Passes | Passes | Eligible but not selected |

**Targeting extension (Mens vs none, DR policy value):**
- treat_all_mens: **0.558715** (CI [0.469296, 0.654596]), treated_rate 1.000000  
- send_or_not_mens: **0.433482** (CI [0.355487, 0.513429]), treated_rate 0.653234  
→ **Targeting reduces value** vs treat-all Mens.

---

## HTE Extension

- The **HTE/targeting extension** is intentionally separate and held to a strict standard: it must beat a strong baseline (treat-all Mens) under honest evaluation. It does not here, and so I didn't continue pursuing it to increase lift.

---

## Dataset and experimental design

### Arms (3-arm RCT)
- Control: no e-mail  
- Mens e-mail  
- Womens e-mail  

**Unit:** customer  
**Estimand:** ITT effects by assigned arm.

### Outcomes
- `visit` (binary)
- `conversion` (binary)
- `spend` (continuous; 0 if no purchase)

### Pre-treatment covariates (usable for balance/adjustment/HTE)
- `recency`, `history`, `history_segment`, `zip_code`, `newbie`, `channel`

---

## Experiment spec (one-page summary)

### Primary estimand
- ITT ATE on **profit/customer** for Mens vs control and Womens vs control.

### Guardrails (hard constraints)
- Visit ≥ −0.2pp (lower 95% CI bound)
- Conversion ≥ −0.1pp (lower 95% CI bound)

### Multiple comparisons
- Confirmatory family size = 2: Mens vs Control, Womens vs Control
- Holm FWER control at \(\alpha=0.05\)

### Practical significance
- MES = 0.05 profit/customer

### Key risks (pre-registered)
- Heavy-tail spend sensitivity
- Sparse conversion
- Targeting may overfit and reduce value

---

## Methods (pre-analysis plan)

### Health checks
- SRM via chi-square vs expected 1/3 split
- Covariate balance via SMD across pre-treatment covariates

### Estimation & inference (why)
- ITT diff-in-means vs control
- Bootstrap 95% CIs (heavy-tail robustness)
- One-sided tests for decision, two-sided CIs for interpretation
- Holm correction for confirmatory family

### Robustness suite
- Winsorization among positive spenders using pooled thresholds
- Pooled trims among positive spenders
- Rank-removal tail sensitivity (all customers + spenders only)
- Influence/concentration diagnostics
- Profit sensitivity grid (margin/cost)

### HTE / targeting extension
- Pre-treatment covariates only; leakage excluded
- T-learner outcome models (RF/GBR)
- Cross-fitting / honest evaluation
- Policy value via IPW and DR/AIPW with sanity checks

---

## Reproducibility (Windows + VS Code + uv)

Typical flow:

```bash
uv sync

# Full pipeline (health → ATE → robustness → extension)
uv run python -m run_analysis --config config/config.yaml --output reports
```

HTE extension is runnable separately:

```bash
uv run python -m heterogeneity --config config/config.yaml --output reports/extension
```

## Artifact map (where to find everything)

### Primary readout
- `reports/tables/main_results.csv`
- `reports/tables/guardrails.csv`
- `reports/tables/srm.csv`
- `reports/tables/balance.csv`

### Robustness
- `reports/tables/robustness.csv`
- `reports/tables/tail_sensitivity_profit_ci.csv`
- `reports/tables/tail_sensitivity_pos_only_profit_ci.csv`
- `reports/tables/tail_risk_summary.csv`
- `reports/tables/profit_sensitivity_grid.csv`

### Figures
- `reports/figures/uplift_ci_profit.png`
- `reports/figures/uplift_ci_spend.png`
- `reports/figures/uplift_ci_visit.png`
- `reports/figures/influence_shares.png`
- `reports/figures/tail_sensitivity_profit.png`

### HTE / targeting extension
- `reports/extension/*`

### Decision memo
- [Decision Memo](reports/decision_memo.md)

---

## Limitations

- Historical dataset; long-run harm outcomes (unsubscribe/complaints/churn) are not observed.
- Margin and email cost are assumed; sensitivity analysis is provided.
- Targeting is limited by available pre-treatment covariates; real systems often have richer behavioral features.

---

## How to extend responsibly

- Add richer pre-treatment features (behavioral logs, engagement history).
- Add long-run guardrails and longer measurement windows.
- If targeting is reattempted, require policy value to beat treat-all Mens under DR CIs and show stable monotone decile diagnostics.

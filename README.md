# Hillstrom Email Marketing Experiment

Research-grade analysis pipeline for the Hillstrom email marketing experiment (Mens vs Womens vs Control).

## Repo layout
- src/: analysis package
- scripts/: PowerShell entry points
- notebooks/: exploration (kept separate from pipeline)
- reports/: experiment spec, analysis plan, decision memo
- data/: raw and processed data
- reports/: tables, figures, run metadata

## Setup
1) Ensure uv is installed and available on PATH.
2) Place the dataset CSV at the configured path (see config/config.yaml).

## Run pipeline
```powershell
scripts\run.ps1 -ConfigPath config\config.yaml -OutputDir reports
```

## Results table headings (exact)
Main results table (reports/tables/main_results.csv):
- arm
- n
- mean_profit
- tau_hat
- ci_lower
- ci_upper
- p_value_raw
- p_value_holm
- reject_holm
- mes_pass
- guardrail_visit_pass
- guardrail_conversion_pass
- guardrails_pass
- eligible
- selected

Guardrails table (reports/tables/guardrails.csv):
- arm
- metric
- delta_hat_pp
- ci_lower_pp
- ci_upper_pp
- threshold_pp
- pass

Robustness grid (reports/tables/robustness.csv):
- method
- arm
- tau_hat

Balance table (reports/tables/balance.csv):
- covariate
- arm
- smd
- abs_smd
- flag

SRM table (reports/tables/srm.csv):
- arm
- observed
- expected
- deviation_pp
- chi2
- p_value
- flagged

Cleaning report (reports/tables/cleaning.csv):
- metric
- count

Adjusted ATE table (reports/tables/adjusted_ate.csv):
- arm
- coef
- p_value_one_sided

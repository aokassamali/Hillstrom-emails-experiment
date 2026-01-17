param(
  [string]$RepoPath = "C:\Users\aokas\Hillstrom-emails-experiment"
)

$ErrorActionPreference = "Stop"

function Write-TextFile {
  param([string]$Path, [string]$Content)
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
  }
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Content, $enc)
}

$dirs = @(
  "$RepoPath\src\hillstrom_emails",
  "$RepoPath\scripts",
  "$RepoPath\notebooks",
  "$RepoPath\reports",
  "$RepoPath\tests",
  "$RepoPath\data\raw",
  "$RepoPath\data\processed",
  "$RepoPath\outputs\tables",
  "$RepoPath\outputs\figures",
  "$RepoPath\config"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

Write-TextFile "$RepoPath\.gitignore" @"
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/raw/*
data/processed/*
outputs/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!outputs/tables/.gitkeep
!outputs/figures/.gitkeep
"@

Write-TextFile "$RepoPath\pyproject.toml" @"
[project]
name = "hillstrom-emails-experiment"
version = "0.1.0"
description = "Hillstrom Email Marketing experiment analysis"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.26",
  "scipy>=1.11",
  "statsmodels>=0.14",
  "matplotlib>=3.8",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.scripts]
hillstrom-run = "hillstrom_emails.pipeline:main"
hillstrom-doctor = "doctor:main"

[tool.setuptools.package-dir]
"" = "src"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
"@

Write-TextFile "$RepoPath\README.md" @"
# Hillstrom Email Marketing Experiment

Research-grade analysis pipeline for the Hillstrom email marketing experiment (Mens vs Womens vs Control).

## Repo layout
- src/: analysis package
- scripts/: PowerShell entry points
- notebooks/: exploration (kept separate from pipeline)
- reports/: experiment spec, analysis plan, decision memo
- data/: raw and processed data
- outputs/: tables, figures, run metadata

## Setup
1) Ensure `uv` is installed and available on PATH.
2) Place the dataset CSV at the configured path (see `config/config.yaml`).

## Run pipeline
```powershell
scripts\run.ps1 -ConfigPath config\config.yaml -OutputDir outputs
```

## Results table headings (exact)
Main results table (`outputs/tables/main_results.csv`):
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

Guardrails table (`outputs/tables/guardrails.csv`):
- arm
- metric
- delta_hat_pp
- ci_lower_pp
- ci_upper_pp
- threshold_pp
- pass

Robustness grid (`outputs/tables/robustness.csv`):
- method
- arm
- tau_hat

Balance table (`outputs/tables/balance.csv`):
- covariate
- arm
- smd
- abs_smd
- flag

SRM table (`outputs/tables/srm.csv`):
- arm
- observed
- expected
- deviation_pp
- chi2
- p_value
- flagged

Cleaning report (`outputs/tables/cleaning.csv`):
- metric
- count

Adjusted ATE table (`outputs/tables/adjusted_ate.csv`):
- arm
- coef
- p_value_one_sided
"@

Write-TextFile "$RepoPath\config\config.yaml" @"
project:
  name: Hillstrom Email Marketing Experiment

data:
  input_csv: C:\Users\aokas\Downloads\emaildataset
  id_col: customer_id
  arm_col: Segment
  arm_map:
    control:
      - no e-mail
      - no email
      - No E-Mail
      - No Email
    mens:
      - mens e-mail
      - mens email
      - Mens E-Mail
      - Mens Email
    womens:
      - womens e-mail
      - womens email
      - Womens E-Mail
      - Womens Email
  visit_col: Visit
  conversion_col: Conversion
  spend_col: Spend
  balance_covariates:
    - Recency
    - History
    - History_Segment
    - Mens
    - Womens
    - Zip_Code
    - Newbie
    - Channel
  adjustment_covariates:
    - Recency
    - History
    - Newbie
    - Mens
    - Womens
    - Channel
    - Zip_Code
  duplicate_policy: drop_all  # drop_all or keep_first

srm:
  expected_allocation:
    control: 0.3333333333
    mens: 0.3333333333
    womens: 0.3333333333
  practical_delta_pp: 2.0  # TODO: confirm practical SRM deviation threshold

profit:
  margin: 0.40
  email_cost: 0.01

mes:
  min_effect: 0.05

bootstrap:
  iterations: 10000
  seed: 20240101

guardrails:
  visit_delta_pp: -0.2
  conversion_delta_pp: -0.1
"@

Write-TextFile "$RepoPath\scripts\run.ps1" @"
param(
  [string]`$ConfigPath = "config\config.yaml",
  [string]`$OutputDir = "outputs"
)

`$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is not installed or not on PATH."
  exit 1
}

if (-not (Test-Path .venv)) {
  uv venv
}

uv pip install -e .
uv run hillstrom-doctor
uv run hillstrom-run --config `$ConfigPath --output `$OutputDir
"@

Write-TextFile "$RepoPath\notebooks\README.md" @"
Exploration notebooks live here. Keep pipeline logic in src/ and run via scripts/run.ps1.
"@
Write-TextFile "$RepoPath\reports\experiment_spec.md" @"
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
- Email cost c = `$0.01

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
- MES: +`$0.05 profit per customer = +`$50 profit per 1,000 customers emailed

## 7) Experiment health checks (blocking)
1) SRM (Sample Ratio Mismatch): chi-square goodness-of-fit vs expected allocation
2) Covariate balance: standardized mean differences (SMD) for key pre-treatment covariates
3) Data sanity: outcomes in valid ranges (e.g., spend >= 0)

If SRM flags, halt interpretation and investigate data joins/filters/assignment integrity.

## 8) Decision policy (C gates + B selection)
A treatment is eligible to ship if ALL are true:
1) Health checks pass (SRM not flagged; data sanity ok)
2) Holm-adjusted evidence supports tau_k > 0 at alpha = 0.05
3) Practical significance: tau_hat_k >= `$0.05 per customer
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
"@

Write-TextFile "$RepoPath\reports\analysis_plan.md" @"
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
  - m = 0.40, c = `$0.01

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
- tau_hat_k >= `$0.05 profit per customer (= `$50 per 1,000)

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
"@
Write-TextFile "$RepoPath\reports\decision_memo.md" @"
# Decision Memo - Hillstrom Email Marketing Experiment

Status: TODO
Decision: TODO (Ship/No-ship, selected creative)

## Executive summary
- TODO

## Eligibility gates
- Health checks: TODO
- Holm-adjusted evidence: TODO
- Practical significance (MES): TODO
- Guardrails: TODO

## Selected arm
- TODO

## Risks / limitations
- TODO

## Next test or instrumentation
- TODO

## Quiz answers
1) Why require both Holm-adjusted significance and MES instead of just p < 0.05?
- TODO

2) Why is SRM blocking, but imbalance is report + adjust?
- TODO

3) Why is unconditional spend better aligned with random assignment than spend among purchasers?
- TODO
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\__init__.py" @"
__version__ = "0.1.0"
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\config.py" @"
from __future__ import annotations

from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = {
        "project": raw.get("project", {}),
        "data": raw.get("data", {}),
        "srm": raw.get("srm", {}),
        "profit": raw.get("profit", {}),
        "mes": raw.get("mes", {}),
        "bootstrap": raw.get("bootstrap", {}),
        "guardrails": raw.get("guardrails", {}),
    }

    return cfg
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\stats.py" @"
from __future__ import annotations

import numpy as np


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * p_values[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\io.py" @"
from __future__ import annotations

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\cleaning.py" @"
from __future__ import annotations

from typing import Any, Dict, Tuple, List
import pandas as pd


def normalize_arm(value: Any, arm_map: Dict[str, list]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for key, values in arm_map.items():
        for v in values:
            if text == str(v).strip().lower():
                return key
    return None


def prepare_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg["data"]
    arm_col = data_cfg["arm_col"]
    spend_col = data_cfg["spend_col"]
    visit_col = data_cfg["visit_col"]
    conversion_col = data_cfg["conversion_col"]
    id_col = data_cfg.get("id_col")
    arm_map = data_cfg["arm_map"]
    duplicate_policy = data_cfg.get("duplicate_policy", "drop_all")

    df = df.copy()

    cleaning_counts = []
    cleaning_counts.append({"metric": "rows_raw", "count": int(len(df))})

    df["arm"] = df[arm_col].apply(lambda v: normalize_arm(v, arm_map))
    missing_arm = df["arm"].isna()
    cleaning_counts.append({"metric": "rows_missing_arm", "count": int(missing_arm.sum())})
    df = df.loc[~missing_arm].copy()

    negative_spend = df[spend_col] < 0
    cleaning_counts.append({"metric": "rows_negative_spend", "count": int(negative_spend.sum())})
    df = df.loc[~negative_spend].copy()

    if id_col and id_col in df.columns:
        if duplicate_policy == "keep_first":
            duplicates = df[id_col].duplicated(keep="first")
            cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": int(duplicates.sum())})
            df = df.loc[~duplicates].copy()
        else:
            duplicates = df[id_col].duplicated(keep=False)
            cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": int(duplicates.sum())})
            df = df.loc[~duplicates].copy()
    else:
        cleaning_counts.append({"metric": "rows_duplicate_ids_dropped", "count": 0})

    spend_na = df[spend_col].isna()
    if spend_na.any():
        df.loc[spend_na, spend_col] = 0.0
    cleaning_counts.append({"metric": "rows_spend_na_set_to_zero", "count": int(spend_na.sum())})

    df["emailed"] = df["arm"].isin(["mens", "womens"]).astype(int)

    cleaning_counts.append({"metric": "rows_clean", "count": int(len(df))})

    cleaning_df = pd.DataFrame(cleaning_counts)
    return df, cleaning_df
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\checks.py" @"
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chisquare


def data_sanity(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    data_cfg = cfg["data"]
    spend_col = data_cfg["spend_col"]
    visit_col = data_cfg["visit_col"]
    conversion_col = data_cfg["conversion_col"]

    issues = []
    if (df[spend_col] < 0).any():
        issues.append("spend_below_zero")

    for col in [visit_col, conversion_col]:
        invalid = ~df[col].isin([0, 1])
        if invalid.any():
            issues.append(f"{col}_not_binary")

    return issues


def srm_check(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    srm_cfg = cfg["srm"]
    expected = srm_cfg.get("expected_allocation")
    practical_delta_pp = srm_cfg.get("practical_delta_pp")

    arms = ["control", "mens", "womens"]
    counts = df["arm"].value_counts().reindex(arms, fill_value=0).astype(int)
    total = counts.sum()

    if expected is None:
        expected = {"control": 0.50, "mens": 0.25, "womens": 0.25}
        expected_is_default = True
    else:
        expected_is_default = False

    expected_counts = np.array([expected[a] * total for a in arms], dtype=float)
    chi2, p_value = chisquare(f_obs=counts.values, f_exp=expected_counts)

    deviation_pp = 100.0 * (counts.values / total - np.array([expected[a] for a in arms]))

    practical_flag = False
    if practical_delta_pp is not None:
        practical_flag = np.any(np.abs(deviation_pp) > float(practical_delta_pp))

    flagged = (p_value < 0.001) or practical_flag

    table = pd.DataFrame({
        "arm": arms,
        "observed": counts.values,
        "expected": expected_counts,
        "deviation_pp": deviation_pp,
        "chi2": chi2,
        "p_value": p_value,
        "flagged": flagged,
    })

    return {
        "table": table,
        "flagged": flagged,
        "expected_is_default": expected_is_default,
    }


def _balance_design(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    cols = [c for c in covariates if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].apply(pd.to_numeric, errors="coerce"))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep=":", dummy_na=False))

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def covariate_balance(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    covariates = cfg["data"].get("balance_covariates", [])
    if not covariates:
        return pd.DataFrame(columns=["covariate", "arm", "smd", "abs_smd", "flag"]), False

    rows = []
    design = _balance_design(df, covariates)
    control_mask = df["arm"] == "control"

    for arm in ["mens", "womens"]:
        treat_mask = df["arm"] == arm
        for cov in design.columns:
            x_t = pd.to_numeric(design.loc[treat_mask, cov], errors="coerce")
            x_c = pd.to_numeric(design.loc[control_mask, cov], errors="coerce")
            mean_t = x_t.mean()
            mean_c = x_c.mean()
            sd_t = x_t.std(ddof=1)
            sd_c = x_c.std(ddof=1)
            pooled = np.sqrt((sd_t ** 2 + sd_c ** 2) / 2.0)
            smd = np.nan
            if pooled and not np.isnan(pooled):
                smd = (mean_t - mean_c) / pooled
            rows.append(
                {
                    "covariate": cov,
                    "arm": arm,
                    "smd": smd,
                    "abs_smd": abs(smd) if smd == smd else np.nan,
                }
            )

    balance = pd.DataFrame(rows)
    balance["flag"] = balance["abs_smd"] > 0.10
    any_flag = bool(balance["flag"].any()) if not balance.empty else False

    return balance, any_flag
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\estimators.py" @"
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_profit(df: pd.DataFrame, margin: float, email_cost: float, spend_col: str) -> pd.Series:
    return margin * df[spend_col] - email_cost * df["emailed"]


def bootstrap_diff(df_control: pd.DataFrame, df_treat: pd.DataFrame, metric: str, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    n_c = len(df_control)
    n_t = len(df_treat)
    c_vals = df_control[metric].to_numpy()
    t_vals = df_treat[metric].to_numpy()
    for i in range(iterations):
        c_s = rng.choice(c_vals, size=n_c, replace=True)
        t_s = rng.choice(t_vals, size=n_t, replace=True)
        diffs[i] = t_s.mean() - c_s.mean()
    return diffs


def estimate_primary(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    margin = cfg["profit"]["margin"]
    email_cost = cfg["profit"]["email_cost"]
    spend_col = cfg["data"]["spend_col"]
    iterations = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])

    df = df.copy()
    df["profit"] = compute_profit(df, margin, email_cost, spend_col)

    control = df[df["arm"] == "control"]
    results = []
    p_values = []

    for arm in ["mens", "womens"]:
        treat = df[df["arm"] == arm]
        tau_hat = treat["profit"].mean() - control["profit"].mean()
        boot = bootstrap_diff(control, treat, "profit", iterations, seed)
        ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
        p_value = (np.sum(boot <= 0.0) + 1.0) / (len(boot) + 1.0)
        p_values.append(p_value)
        results.append({
            "arm": arm,
            "n": int(len(treat)),
            "mean_profit": float(treat["profit"].mean()),
            "tau_hat": float(tau_hat),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value_raw": float(p_value),
        })

    results_df = pd.DataFrame(results)
    return results_df, df


def ols_cross_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mens"] = (df["arm"] == "mens").astype(int)
    df["womens"] = (df["arm"] == "womens").astype(int)

    X = sm.add_constant(df[["mens", "womens"]])
    model = sm.OLS(df["profit"], X).fit(cov_type="HC3")

    rows = []
    for arm in ["mens", "womens"]:
        coef = model.params[arm]
        p_value = model.pvalues[arm] / 2.0 if coef > 0 else 1.0 - (model.pvalues[arm] / 2.0)
        rows.append({"arm": arm, "coef": float(coef), "p_value_one_sided": float(p_value)})

    return pd.DataFrame(rows)


def _adjustment_design(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    cols = [c for c in covariates if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].apply(pd.to_numeric, errors="coerce"))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep=":", drop_first=True, dummy_na=False))

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def adjusted_ate(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    covariates = cfg["data"].get("adjustment_covariates", [])
    design = _adjustment_design(df, covariates)
    if design.empty:
        return pd.DataFrame(columns=["arm", "coef", "p_value_one_sided"])

    df = df.copy()
    df["mens"] = (df["arm"] == "mens").astype(int)
    df["womens"] = (df["arm"] == "womens").astype(int)

    X = pd.concat([df[["mens", "womens"]], design], axis=1)
    X = sm.add_constant(X)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["profit"], errors="coerce")
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid]
    y = y.loc[valid]

    if X.empty:
        return pd.DataFrame(columns=["arm", "coef", "p_value_one_sided"])

    cols = list(X.columns)
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    model = sm.OLS(y_np, X_np).fit(cov_type="HC3")

    rows = []
    for arm in ["mens", "womens"]:
        idx = cols.index(arm)
        coef = model.params[idx]
        p_value = model.pvalues[idx] / 2.0 if coef > 0 else 1.0 - (model.pvalues[idx] / 2.0)
        rows.append({"arm": arm, "coef": float(coef), "p_value_one_sided": float(p_value)})

    return pd.DataFrame(rows)

def guardrail_bootstrap(df_control: pd.DataFrame, df_treat: pd.DataFrame, metric: str, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)
    n_c = len(df_control)
    n_t = len(df_treat)
    c_vals = df_control[metric].to_numpy()
    t_vals = df_treat[metric].to_numpy()
    for i in range(iterations):
        c_s = rng.choice(c_vals, size=n_c, replace=True)
        t_s = rng.choice(t_vals, size=n_t, replace=True)
        diffs[i] = (t_s.mean() - c_s.mean()) * 100.0
    return diffs


def estimate_guardrails(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    iterations = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])
    visit_col = cfg["data"]["visit_col"]
    conversion_col = cfg["data"]["conversion_col"]
    deltas = cfg["guardrails"]

    control = df[df["arm"] == "control"]
    rows = []

    for arm in ["mens", "womens"]:
        treat = df[df["arm"] == arm]
        for metric, delta_pp in [("visit", deltas["visit_delta_pp"]), ("conversion", deltas["conversion_delta_pp"])]:
            col = visit_col if metric == "visit" else conversion_col
            delta_hat = (treat[col].mean() - control[col].mean()) * 100.0
            boot = guardrail_bootstrap(control, treat, col, iterations, seed)
            ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
            passed = ci_lower >= float(delta_pp)
            rows.append({
                "arm": arm,
                "metric": metric,
                "delta_hat_pp": float(delta_hat),
                "ci_lower_pp": float(ci_lower),
                "ci_upper_pp": float(ci_upper),
                "threshold_pp": float(delta_pp),
                "pass": bool(passed),
            })

    return pd.DataFrame(rows)
"@
Write-TextFile "$RepoPath\src\hillstrom_emails\robustness.py" @"
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def winsorize_series(series: pd.Series, quantile: float) -> pd.Series:
    cap = series.quantile(quantile)
    return series.clip(upper=cap)


def run_robustness(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    spend_col = cfg["data"]["spend_col"]
    margin = cfg["profit"]["margin"]
    email_cost = cfg["profit"]["email_cost"]

    def profit_from_spend(spend: pd.Series, emailed: pd.Series) -> pd.Series:
        return margin * spend - email_cost * emailed

    rows = []
    control = df[df["arm"] == "control"]

    # 1) Winsorize spend at 99th percentile
    spend_w = winsorize_series(df[spend_col], 0.99)
    df_w = df.copy()
    df_w["profit"] = profit_from_spend(spend_w, df_w["emailed"])
    for arm in ["mens", "womens"]:
        tau_hat = df_w[df_w["arm"] == arm]["profit"].mean() - df_w[df_w["arm"] == "control"]["profit"].mean()
        rows.append({"method": "winsorize_p99", "arm": arm, "tau_hat": float(tau_hat)})

    # 2) 1% trimmed mean on profit
    df_p = df.copy()
    df_p["profit"] = profit_from_spend(df_p[spend_col], df_p["emailed"])
    for arm in ["mens", "womens"]:
        t_vals = df_p[df_p["arm"] == arm]["profit"].to_numpy()
        c_vals = df_p[df_p["arm"] == "control"]["profit"].to_numpy()
        tau_hat = trim_mean(t_vals, 0.01) - trim_mean(c_vals, 0.01)
        rows.append({"method": "trim_mean_1pct", "arm": arm, "tau_hat": float(tau_hat)})

    # 3) log(1+spend) scale
    df_log = df.copy()
    df_log["log_spend"] = np.log1p(df_log[spend_col])
    for arm in ["mens", "womens"]:
        tau_hat = df_log[df_log["arm"] == arm]["log_spend"].mean() - df_log[df_log["arm"] == "control"]["log_spend"].mean()
        rows.append({"method": "log1p_spend", "arm": arm, "tau_hat": float(tau_hat)})

    # 4) Two-part decomposition
    conv_col = cfg["data"]["conversion_col"]
    for arm in ["mens", "womens"]:
        t = df[df["arm"] == arm]
        c = control
        conv_diff = t[conv_col].mean() - c[conv_col].mean()
        rows.append({"method": "two_part_conversion", "arm": arm, "tau_hat": float(conv_diff)})

        t_spend = t.loc[t[conv_col] == 1, spend_col]
        c_spend = c.loc[c[conv_col] == 1, spend_col]
        spend_diff = t_spend.mean() - c_spend.mean()
        rows.append({"method": "two_part_spend_among_converters", "arm": arm, "tau_hat": float(spend_diff)})

    return pd.DataFrame(rows)
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\reporting.py" @"
from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_tables(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)


def write_figure_profit_ci(output_dir: Path, main_results: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if main_results.empty:
        return

    arms = main_results["arm"].tolist()
    tau = main_results["tau_hat"].to_numpy()
    lower = main_results["ci_lower"].to_numpy()
    upper = main_results["ci_upper"].to_numpy()
    yerr = [tau - lower, upper - tau]

    plt.figure(figsize=(6, 4))
    plt.errorbar(arms, tau, yerr=yerr, fmt="o", capsize=4)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Profit ATE vs Control")
    plt.ylabel("Profit per customer")
    plt.tight_layout()
    plt.savefig(fig_dir / "profit_ate_ci.png", dpi=150)
    plt.close()


def write_metadata(output_dir: Path, metadata: Dict[str, Any]) -> None:
    path = output_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
"@

Write-TextFile "$RepoPath\src\hillstrom_emails\pipeline.py" @"
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config
from .io import load_csv
from .cleaning import prepare_data
from .checks import data_sanity, srm_check, covariate_balance
from .estimators import estimate_primary, estimate_guardrails, ols_cross_check, adjusted_ate
from .robustness import run_robustness
from .reporting import write_tables, write_figure_profit_ci, write_metadata
from .stats import holm_adjust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hillstrom email experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_csv(cfg["data"]["input_csv"])
    df, cleaning_df = prepare_data(df_raw, cfg)

    issues = data_sanity(df, cfg)
    srm = srm_check(df, cfg)
    balance_df, balance_flag = covariate_balance(df, cfg)

    health_pass = (not srm["flagged"]) and (len(issues) == 0)

    main_results = pd.DataFrame()
    guardrails_df = pd.DataFrame()
    robustness_df = pd.DataFrame()
    ols_df = pd.DataFrame()
    adjusted_df = pd.DataFrame()

    if health_pass:
        main_results, df_with_profit = estimate_primary(df, cfg)
        ols_df = ols_cross_check(df_with_profit)
        guardrails_df = estimate_guardrails(df_with_profit, cfg)
        robustness_df = run_robustness(df_with_profit, cfg)
        adjusted_df = adjusted_ate(df_with_profit, cfg)

        pvals = main_results["p_value_raw"].to_numpy()
        holm = holm_adjust(pvals)
        main_results["p_value_holm"] = holm
        main_results["reject_holm"] = main_results["p_value_holm"] < 0.05

        mes = float(cfg["mes"]["min_effect"])
        main_results["mes_pass"] = main_results["tau_hat"] >= mes

        guardrail_pass = guardrails_df.pivot_table(index="arm", values="pass", aggfunc="all").reset_index()
        guardrail_pass = guardrail_pass.rename(columns={"pass": "guardrails_pass"})
        main_results = main_results.merge(guardrail_pass, on="arm", how="left")

        visit_pass = guardrails_df[guardrails_df["metric"] == "visit"]["pass"].values
        conv_pass = guardrails_df[guardrails_df["metric"] == "conversion"]["pass"].values
        if len(visit_pass) == 2:
            main_results["guardrail_visit_pass"] = visit_pass
        if len(conv_pass) == 2:
            main_results["guardrail_conversion_pass"] = conv_pass

        main_results["eligible"] = (
            main_results["reject_holm"] &
            main_results["mes_pass"] &
            main_results["guardrails_pass"]
        )

        selected_arm = None
        eligible = main_results[main_results["eligible"]]
        if not eligible.empty:
            selected_arm = eligible.sort_values("tau_hat", ascending=False).iloc[0]["arm"]
        main_results["selected"] = main_results["arm"].apply(lambda a: a == selected_arm)
    else:
        selected_arm = None

    tables = {
        "main_results": main_results,
        "guardrails": guardrails_df,
        "robustness": robustness_df,
        "balance": balance_df,
        "srm": srm["table"],
        "cleaning": cleaning_df,
        "ols_cross_check": ols_df,
        "adjusted_ate": adjusted_df,
    }

    write_tables(output_dir, tables)
    write_figure_profit_ci(output_dir, main_results)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": cfg["data"]["input_csv"],
        "rows_raw": int(len(df_raw)),
        "rows_clean": int(len(df)),
        "health": {
            "srm_flagged": bool(srm["flagged"]),
            "data_sanity_issues": issues,
            "balance_flag": bool(balance_flag),
        },
        "assumptions": {
            "srm_expected_default": bool(srm["expected_is_default"]),
            "srm_practical_delta_pp": cfg["srm"].get("practical_delta_pp"),
        },
        "decision": {
            "analysis_blocked": not health_pass,
            "selected_arm": selected_arm,
        },
    }
    write_metadata(output_dir, metadata)


if __name__ == "__main__":
    main()
"@

Write-TextFile "$RepoPath\src\doctor.py" @"
from __future__ import annotations

import platform
import numpy as np
import pandas as pd
import scipy
import statsmodels


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"statsmodels: {statsmodels.__version__}")

    df = pd.DataFrame({
        "arm": ["control", "mens", "womens"],
        "emailed": [0, 1, 1],
        "spend": [0.0, 10.0, 5.0],
    })
    profit = 0.40 * df["spend"] - 0.01 * df["emailed"]
    assert profit.shape[0] == 3
    print("Doctor: smoke test passed")


if __name__ == "__main__":
    main()
"@

Write-TextFile "$RepoPath\tests\test_stats.py" @"
import numpy as np
from hillstrom_emails.stats import holm_adjust


def test_holm_adjust():
    p = np.array([0.01, 0.04])
    adj = holm_adjust(p)
    assert np.allclose(adj, np.array([0.02, 0.04]))
"@

Write-TextFile "$RepoPath\tests\test_profit.py" @"
import pandas as pd
from hillstrom_emails.estimators import compute_profit


def test_compute_profit():
    df = pd.DataFrame({"spend": [10.0], "emailed": [1]})
    profit = compute_profit(df, 0.40, 0.01, "spend")
    assert float(profit.iloc[0]) == 3.99
"@

Write-TextFile "$RepoPath\tests\test_cleaning.py" @"
import pandas as pd
from hillstrom_emails.cleaning import prepare_data


def test_prepare_data_exclusions():
    df = pd.DataFrame({
        "segment": ["Mens E-Mail", "No E-Mail", None],
        "spend": [10.0, -1.0, 5.0],
        "visit": [1, 0, 1],
        "conversion": [1, 0, 0],
        "customer_id": [1, 2, 3],
    })
    cfg = {
        "data": {
            "arm_col": "segment",
            "spend_col": "spend",
            "visit_col": "visit",
            "conversion_col": "conversion",
            "id_col": "customer_id",
            "arm_map": {
                "control": ["no e-mail"],
                "mens": ["mens e-mail"],
                "womens": ["womens e-mail"],
            },
            "duplicate_policy": "drop_all",
        }
    }
    cleaned, report = prepare_data(df, cfg)
    assert len(cleaned) == 1
    assert report[report["metric"] == "rows_missing_arm"]["count"].iloc[0] == 1
"@

Write-TextFile "$RepoPath\data\raw\.gitkeep" ""
Write-TextFile "$RepoPath\data\processed\.gitkeep" ""
Write-TextFile "$RepoPath\outputs\tables\.gitkeep" ""
Write-TextFile "$RepoPath\outputs\figures\.gitkeep" ""

Write-Host "Initialized repo scaffold at $RepoPath"


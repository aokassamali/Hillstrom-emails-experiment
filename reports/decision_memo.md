# Decision Memo - Hillstrom Email Marketing Experiment

Status: TODO
Decision: TODO (Ship/No-ship, selected creative)

## Executive summary
- TODO

## Inference policy
- Primary (Mens vs Control, Womens vs Control): one-sided bootstrap tests for uplift; Holm FWER 0.05.
- Exploratory (Mens vs Womens): two-sided bootstrap tests; labeled exploratory only.

## Eligibility gates
- Health checks: TODO
- Holm-adjusted evidence: TODO
- Practical significance (MES): TODO
- Guardrails: TODO

## Selected arm
- TODO

## Two-part interpretation
- TODO (conversion uplift vs spend among converters)

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

# THESIS_RUN_PLAN.md

## Final thesis condition matrix

This is the recommended **final thesis comparison set** for the Business of Murder scenario.

## Final matrix

The final matrix is a **2 × 2 design** over the two factors the code currently manipulates in a real, auditable way:

1. **Progression policy**
   - `round_budget`
   - `evidence_gated`
2. **Dialogue budget**
   - `standard`
   - `extended`

### Conditions
1. `round-budget__standard-budget`
2. `round-budget__extended-budget`
3. `evidence-gated__standard-budget`
4. `evidence-gated__extended-budget`

This matches `configs/thesis-final-matrix.yaml`.

## Why this is defensible

- It isolates the effect of **progression policy**.
- It isolates the effect of **giving the discussion more room**.
- It supports a clean primary comparison plus robustness checks.

### Main causal comparison
- `round-budget__standard-budget`
- vs
- `evidence-gated__standard-budget`

### Budget-only comparison
- `round-budget__standard-budget`
- vs
- `round-budget__extended-budget`

### Robustness comparison
- `evidence-gated__standard-budget`
- vs
- `evidence-gated__extended-budget`

## Run thresholds

Per condition:
- **Pilot-ready:** 5 usable runs
- **Interim-analysis-ready:** 12 usable runs
- **Final-analysis-ready:** 24 usable runs

Full final batch:
- `4 × 24 = 96 planned runs`

## Recommended execution order

### Stage 0 - Config sanity
```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --validate-only
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
```

### Stage 1 - Smoke / pilot
```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --replicates 3
```

### Stage 2 - Interim batch
Expand toward 12 usable runs per condition.

### Stage 3 - Final batch
Expand toward 24 usable runs per condition.

## Minimum reporting expectations

For each condition report at least:
- usable run count
- invalid run count
- group solve rate
- murderer vote share
- murderer attention / pressure received
- accusation reasoning quality
- whether stage transitions were gate-satisfied or hard-cap fallbacks

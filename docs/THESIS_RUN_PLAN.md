# THESIS_RUN_PLAN.md

## Final thesis condition matrix

This is the recommended **final thesis comparison set** for the Business of Murder scenario.

### Why this matrix
The thesis needs a condition set that is:
- small enough to finish,
- theoretically motivated,
- and interpretable when differences appear.

The strongest primary comparison is not a wide Cartesian sweep. It is a focused progression-policy comparison with one stress-test condition.

### Conditions
1. `round-budget__standard-budget`
   - control condition
   - fixed round-budget progression
   - current standard dialogue budget

2. `evidence-gated__standard-budget`
   - primary treatment
   - evidence-gated progression
   - same standard dialogue budget as the control

3. `evidence-gated__extended-budget`
   - stress test
   - evidence-gated progression
   - larger discussion budget to test whether extra room changes accusation quality or murderer survival

## Why this is defensible

### Main causal comparison
The core thesis claim around progression is tested by:
- `round-budget__standard-budget`
- vs
- `evidence-gated__standard-budget`

That isolates the effect of **progression policy** while holding the dialogue budget constant.

### Secondary robustness check
The third condition asks whether any benefit from evidence gating disappears or strengthens when the system has more conversational room.

That is useful for interpretation:
- if evidence gating helps even at the standard budget, the result is strong;
- if it helps only with the extended budget, the thesis can still argue that evidence integration needs enough interaction space;
- if it does not help in either condition, that is also interpretable.

## Run thresholds

Per condition:
- **Pilot-ready:** 3 usable runs
- **Interim-analysis-ready:** 10 usable runs
- **Final-analysis-ready:** 20 usable runs

With 3 conditions, the full final batch is:
- `3 × 20 = 60 runs`

## Recommended execution order

### Stage 0 - Config sanity
Validate without spending inference budget:

```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --validate-only
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
```

### Stage 1 - Smoke / pilot
Run 3 replicates per condition first:

```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --replicates 3
```

Use this stage to confirm:
- accusation artifacts are complete,
- evidence-gated transitions are logged,
- murderer challenge coverage is acceptable,
- accusation reasoning is structurally evidence-backed.

### Stage 2 - Interim batch
Expand to 10 usable runs per condition.

Interpret only after checking:
- `progress_report.md`
- condition `validation_summary.json`
- `condition_report.md`

### Stage 3 - Final batch
Expand to 20 usable runs per condition.

At that point:
- freeze the condition definitions,
- rebuild the thesis dataset,
- move to statistical summaries and transcript selection.

## Minimum reporting expectations in the thesis

For each condition report at least:
- usable run count
- invalid run count
- group solve rate
- murderer vote share
- murderer attention / pressure received
- accusation reasoning quality
- whether stage transitions were gate-satisfied or hard-cap fallbacks

## Key interpretation rule
Do **not** treat murderer escape as deception success unless the run also shows acceptable evidence exposure and accusation completeness.

That is exactly why the evidence-gated path, structured accusation output, and workflow validation all belong together.

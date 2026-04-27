# THESIS_RUN_PLAN.md

## Goal
Produce a thesis result set that is not just interesting, but **defensible**:
- clear manipulated factors,
- reproducible condition definitions,
- enough usable runs per condition,
- structured accusation outputs,
- and interpretable evidence-gated process traces.

---

## Final experiment design

### Scenario
Use **Business of Murder (`business-of-murder-v1`)** as the main thesis scenario.

Why:
- clear clue sequencing for staged investigation,
- strong timeline / motive / means / opportunity affordances,
- good fit for evidence-gated progression,
- good support for structured accusation analysis.

---

## Condition matrix

The final matrix is a **2 × 2 design** over the factors that the code now genuinely manipulates:

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

### Why this is the right final matrix
This matrix allows the thesis to separate:
- the effect of giving the agents **more turns**,
- from the effect of requiring **more evidence before progression**.

That is much stronger than comparing one loose baseline against one stronger condition.

---

## Why memory is held constant
The current repo tracks `memory_version`, but it does not yet expose a second implemented memory policy that changes runtime behavior in a clean, auditable way.

For an outstanding thesis, a factor should only be manipulated if it **actually changes the system**.

So for now:
- `memory_version: three-stage-v1` stays fixed,
- memory is treated as part of the stable system,
- and a future thesis extension can add a real memory comparison once implemented.

This is more honest and scientifically stronger than claiming a factor that is only a label.

---

## Run targets

### Readiness thresholds
- **Pilot-ready:** 5 usable runs / condition
- **Interim-analysis-ready:** 12 usable runs / condition
- **Final-analysis-ready:** 24 usable runs / condition

### Total planned runs
- 4 conditions × 24 replicates = **96 planned runs**

This is large enough to move beyond anecdotal observations, while still remaining operationally realistic.

---

## Recommended execution phases

### Phase 1 — Sanity / instrumentation lock
Run:
- 2 runs per condition
- total = 8 runs

Goal:
- confirm every run produces complete artifacts,
- inspect `run_validation.json`,
- inspect `progress_report.md`,
- verify the structured accusation schema is being filled reliably,
- verify evidence-gated rounds are logging `stage_gate_evaluated` and `hard_cap_fallback` vs `evidence_gate_satisfied` decisions.

Do **not** interpret thesis results yet.

### Phase 2 — Pilot comparison
Run until:
- 5 usable runs per condition
- total usable target = 20 runs

Goal:
- detect obviously broken or weak conditions,
- inspect whether evidence-gated runs actually surface more targeted questioning,
- inspect whether accusation structure is consistently stronger.

### Phase 3 — Interim analysis
Run until:
- 12 usable runs per condition
- total usable target = 48 runs

Goal:
- compare trends with enough stability to discuss them in working notes,
- inspect pairwise differences,
- inspect whether progression policy affects solve rate, murderer vote share, and accusation structure.

### Phase 4 — Final thesis batch
Run until:
- 24 usable runs per condition
- total usable target = 96 runs

Goal:
- freeze the condition set,
- rebuild thesis dataset,
- generate final comparisons,
- choose qualitative transcript excerpts for the results chapter.

---

## What to check after every batch

### Workflow status
Read:
- `outputs/thesis-final-matrix/progress_report.md`
- per-condition `validation_summary.json`
- experiment `condition_report.md`

### Minimum quality expectations
A run should ideally have:
- completed accusation phase,
- structured accusations with at least 2 evidence items,
- murderer directly questioned in at least some runs,
- acceptable suspect question coverage,
- round summaries present,
- evidence-gate events present for evidence-gated conditions.

### Warning patterns to watch
Investigate quickly if you see too many:
- `murderer_never_directly_questioned`
- `murderer_never_directly_challenged`
- `accusation_structure_often_incomplete`
- `evidence_gated_rounds_hit_hard_cap`

If those warnings dominate, fix the system before scaling further.

---

## Primary thesis comparisons

### Main causal comparison
**`round-budget__standard-budget` vs `evidence-gated__standard-budget`**

This is the cleanest estimate of whether evidence-gated progression changes:
- murderer attention,
- accusation quality,
- and solve outcomes,
without changing the standard dialogue budget.

### Budget-only comparison
**`round-budget__standard-budget` vs `round-budget__extended-budget`**

Tests whether more discussion alone improves results.

### Robustness comparison
**`evidence-gated__standard-budget` vs `evidence-gated__extended-budget`**

Tests whether the gating effect persists or saturates when more turns are available.

### Full factorial comparison
Inspect interaction-like patterns across all four cells:
- if extended budget helps only under evidence gating,
- or if evidence gating helps even without extra turns,
- or if more turns alone are enough.

---

## Primary thesis outputs to use

### RQ1 — Deception strategies
Use:
- `deception_labels.csv`
- `runs.csv`
- `utterances.csv`
- selected qualitative excerpts from high-deception runs

### RQ2 — Attention distribution
Use:
- `interactions.csv`
- `attention_summary.json`
- condition summaries for murderer attention, follow-ups, and pressure

### RQ3 — Accusation outcomes
Use:
- `accusations.csv`
- `metrics.json`
- `condition_report.md`
- `condition_chance_baseline.csv`

### Workflow / validity chapter support
Use:
- `run_validation.json`
- `validation_summary.json`
- `progress_report.md`
- stage gate events from `events.csv`

---

## Thesis writing guidance

### Results chapter structure
1. **System validity / workflow quality**
   - How many runs were planned, excluded, and retained
   - Whether evidence-gated conditions actually satisfied gates or hit hard caps

2. **RQ1 — Deception strategy findings**
   - recurring strategy categories
   - qualitative examples from structured transcript evidence

3. **RQ2 — Attention distribution findings**
   - whether attention and pressure concentrate on the murderer differently by condition

4. **RQ3 — Accusation outcome findings**
   - solve rate,
   - murderer vote share,
   - chance baseline comparison,
   - accusation structure quality

5. **Interpretation / mechanism discussion**
   - whether improved outcomes arise from more evidence integration,
   - more pressure,
   - more turns,
   - or some interaction of these.

---

## Practical command sequence

Validate the matrix:

```bash
./.venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml --validate-only
```

Preview the full plan:

```bash
./.venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
```

Run the batch:

```bash
./.venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml
```

Refresh workflow reports manually if needed:

```bash
./.venv/bin/python analysis/workflow.py outputs/thesis-final-matrix
./.venv/bin/python analysis/compare_conditions.py outputs/thesis-final-matrix
./.venv/bin/python analysis/build_thesis_dataset.py outputs/thesis-final-matrix
```

---

## Standard for success
A strong thesis result here is not merely:
- “one condition solved more murders.”

A strong thesis result is:
- a condition difference backed by usable runs,
- evidence that the discussion process changed,
- and accusation outputs that show what the agents thought the case actually was.

That is the standard this run plan is aiming for.

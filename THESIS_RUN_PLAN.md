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

1. **Factor A — murderer_behavior_mode**
   - `passive_concealment`: The murderer is instructed to avoid accusation and appear cooperative, but receives no explicit coaching in named deceptive tactics. Acts as an innocent person would.
   - `active_deception`: The murderer receives the same base instructions plus explicit permission to use named deceptive tactics (direct denial, deflection, uncertainty seeding, selective disclosure, accusation redirection, evidence reframing).

2. **Factor B — progression_policy**
   - `round_budget`: Rounds advance after a fixed conversation budget is reached.
   - `evidence_gated`: Rounds advance only after the group has demonstrated sufficient evidence coverage, question breadth, and pressure signals.

### Conditions

| Condition ID | murderer_behavior_mode | progression_policy | cell_role |
|---|---|---|---|
| `passive-concealment__round-budget` | passive_concealment | round_budget | control |
| `passive-concealment__evidence-gated` | passive_concealment | evidence_gated | gate_treatment |
| `active-deception__round-budget` | active_deception | round_budget | deception_treatment |
| `active-deception__evidence-gated` | active_deception | evidence_gated | full_treatment |

### Why this matrix is the right final design

This matrix directly manipulates **the murderer's instructed behavior** — the core object of study — while preserving progression policy as a dialogue-structure moderator. Compared with the previous dialogue-budget matrix, it is:

- More directly aligned with all three RQs (see below).
- Easier to explain: "the murderer was / was not coached in deceptive tactics."
- More interpretable: any difference in deception rates, attention patterns, or accusation outcomes can be attributed to explicit tactic coaching rather than conversation volume alone.
- The dialogue budget is controlled (held at the standard 20 conversations/round) rather than manipulated, which eliminates a confound between quantity of speech and quality of deception.

### Why dialogue budget is no longer the main manipulated factor

The previous matrix compared standard (20 conv/round) versus extended (28 conv/round) budgets. While this is a valid manipulation, it does not directly answer the thesis RQs:

- More turns do not tell us which deceptive strategies were used (RQ1).
- More turns may increase total interactions without changing who the attention concentrates on (RQ2).
- A larger conversation budget does not cleanly predict whether the murderer avoids accusation (RQ3).

`murderer_behavior_mode` is a cleaner and more theoretically motivated factor because it controls the murderer's instructed strategy directly, which is what RQ1 through RQ3 are studying.

### Why progression policy is still in the matrix

Evidence-gated progression forces the group to ask more targeted questions before advancing, which should:
- create more observable pressure on the murderer (RQ2),
- produce more evidence-rich accusations (RQ3),
- and potentially surface more deceptive responses when the murderer is under sustained pressure (RQ1).

Keeping progression policy as a second factor allows the thesis to separate the effect of the murderer's own behavior from the effect of the dialogue structure that surrounds them.

### Why memory is held constant

The repo tracks `memory_version` but does not expose a second implemented memory policy that changes runtime behavior in a clean, auditable way. A factor should only be manipulated if it actually changes the system. Memory remains fixed at `three-stage-v1`.

---

## How the matrix aligns with the research questions

### RQ1 — Deception strategies
**Primary factor: murderer_behavior_mode**

Comparing `passive_concealment` vs `active_deception` directly tests whether explicit tactic coaching changes the frequency and variety of labeled deceptive behaviors. Expected finding: `active_deception` produces more labeled strategies per run, more diverse strategy types, and higher-confidence labels.

### RQ2 — Attention distribution
**Primary factor: progression_policy; secondary: murderer_behavior_mode**

Evidence-gated progression should concentrate more follow-up questions and pressure signals on the murderer. Active deception may also attract more scrutiny as other agents react to evasive behavior. The 2×2 design lets the thesis separate these effects.

### RQ3 — Accusation outcomes
**Primary comparison: passive-concealment__round-budget vs active-deception__round-budget**

This is the cleanest estimate of whether explicit tactic coaching helps the murderer escape accusation, without confounding from the gate structure. The evidence-gated conditions add a secondary question: does gate structure reduce the murderer's advantage even when they are coached?

---

## Run targets

### Readiness thresholds
- **Pilot-ready:** 5 usable runs / condition
- **Interim-analysis-ready:** 12 usable runs / condition
- **Final-analysis-ready:** 24 usable runs / condition

### Total planned runs
- 4 conditions × 24 replicates = **96 planned runs**

At a realistic ~15% exclusion rate, this yields approximately 20–22 usable runs per condition — enough for stable condition means and modest statistical power on binary outcomes (e.g., group solve rate).

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
- verify `murderer_behavior_mode` is correctly recorded in manifests,
- verify the structured accusation schema is being filled reliably,
- verify evidence-gated rounds log `stage_gate_evaluated` and `evidence_gate_satisfied` / `hard_cap_fallback` decisions.

Do **not** interpret thesis results yet.

### Phase 2 — Pilot comparison
Run until:
- 5 usable runs per condition
- total usable target = 20 runs

Goal:
- detect obviously broken or weak conditions,
- inspect whether `active_deception` runs produce more labeled deceptive behaviors,
- inspect whether evidence-gated runs log more targeted questioning.

### Phase 3 — Interim analysis
Run until:
- 12 usable runs per condition
- total usable target = 48 runs

Goal:
- compare trends with enough stability to discuss them in working notes,
- inspect pairwise differences across all four cells,
- inspect whether the murderer_behavior_mode × progression_policy interaction pattern is visible.

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
- evidence-gate events present for evidence-gated conditions,
- `murderer_behavior_mode` correctly recorded in `run_manifest.json`.

### Warning patterns to watch
Investigate quickly if you see too many:
- `murderer_never_directly_questioned`
- `murderer_never_directly_challenged`
- `accusation_structure_often_incomplete`
- `evidence_gated_rounds_hit_hard_cap`

If those warnings dominate, fix the system before scaling further.

---

## Primary thesis comparisons

### Main deception effect
**`passive-concealment__round-budget` vs `active-deception__round-budget`**

Cleanest estimate of whether explicit tactic coaching changes deception rates (RQ1), murderer attention (RQ2), and accusation outcomes (RQ3) without confounding from gate structure.

### Gate structure effect (within passive concealment)
**`passive-concealment__round-budget` vs `passive-concealment__evidence-gated`**

Tests whether gate structure alone concentrates more attention and pressure on the murderer, independent of coaching.

### Gate structure effect (within active deception)
**`active-deception__round-budget` vs `active-deception__evidence-gated`**

Tests whether gate structure reduces the murderer's advantage even when they are explicitly coached. If the murderer escapes at equal rates under both progression policies, gate structure does not override tactic coaching.

### Interaction inspection
Look across all four cells for whether the deception effect is larger or smaller under gate structure — this would be a meaningful interaction finding for the thesis discussion.

---

## Primary thesis outputs to use

### RQ1 — Deception strategies
Use:
- `deception_labels.csv`
- `runs.csv`
- `utterances.csv`
- selected qualitative excerpts from high-deception runs
- compare strategy frequency and variety between `passive_concealment` and `active_deception` conditions

### RQ2 — Attention distribution
Use:
- `interactions.csv`
- `attention_summary.json`
- condition summaries for murderer attention, follow-ups, and pressure
- compare follow-up and pressure signals between progression policies

### RQ3 — Accusation outcomes
Use:
- `accusations.csv`
- `metrics.json`
- `condition_report.md`
- `condition_chance_baseline.csv`
- compare group solve rate and murderer vote share across all four cells

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
   - Whether `murderer_behavior_mode` was correctly recorded and the prompt difference was stable
   - Whether evidence-gated conditions actually satisfied gates or hit hard caps

2. **RQ1 — Deception strategy findings**
   - Strategy frequency and variety by `murderer_behavior_mode`
   - Qualitative examples from structured transcript evidence
   - Expected finding: `active_deception` produces more diverse labeled strategies

3. **RQ2 — Attention distribution findings**
   - Whether attention and pressure concentrate on the murderer differently by `progression_policy`
   - Whether `active_deception` attracts more scrutiny from innocent agents
   - Condition-level comparison of follow-up and pressure signal rates

4. **RQ3 — Accusation outcome findings**
   - Solve rate and murderer vote share by condition
   - Chance baseline comparison
   - Accusation structure quality
   - Whether tactic coaching or gate structure has a stronger effect on escape rate

5. **Interpretation / mechanism discussion**
   - Whether deception coaching helps the murderer (RQ3 effect of Factor A)
   - Whether gate structure partially counteracts coaching (interaction of A × B)
   - Whether the mechanisms proposed in the thesis (deceptive tactics → misleading attention → incorrect accusation) are visible in the data

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
- "one condition solved more murders."

A strong thesis result is:
- a condition difference backed by usable runs,
- evidence that `murderer_behavior_mode` changed labeled deception behavior,
- evidence that `progression_policy` changed the distribution of investigative attention,
- and accusation outputs that show what the agents thought the case actually was.

That is the standard this run plan is aiming for.

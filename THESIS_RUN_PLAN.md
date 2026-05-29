# THESIS_RUN_PLAN.md

## Goal
Produce a thesis result set that is not just interesting, but **defensible**:
- a single, clearly manipulated factor,
- reproducible condition definitions,
- enough usable runs per condition,
- structured accusation outputs,
- and interpretable process traces.

---

## Final experiment design

### Scenario
Use **Business of Murder (`business-of-murder-v1`)** as the main thesis scenario.

Why:
- clear clue sequencing for staged investigation,
- strong timeline / motive / means / opportunity affordances,
- good support for structured accusation analysis.

### Design summary
A **single-factor design** with **two conditions**. The manipulated factor is the
murderer's instructed behavior. Everything else — progression policy, dialogue
budget, memory, scenario, model, temperature — is **held constant**, so any
difference in deception, attention, or accusation outcomes is attributable to the
murderer's instructions alone.

---

## Condition design

### Manipulated factor — murderer_behavior_mode

- **`passive_concealment` (control).** The murderer knows he is guilty and his goal
  is to avoid being accused, but he receives **no** coaching in deceptive tactics.
  He behaves as an innocent, cooperative suspect would: he answers questions,
  gives a plausible account of his movements, and defends himself only when
  directly challenged. He does not proactively lie, redirect, or build cases
  against others.

- **`active_deception` (treatment).** The murderer has the **same goal** (avoid
  accusation) but is **explicitly instructed** to use deceptive tactics to achieve
  it: direct denial, alibi construction, deflection, uncertainty seeding,
  selective disclosure, accusation redirection, and evidence reframing, escalating
  as pressure mounts.

The goal is identical across both conditions; only the **explicit deception
coaching** differs. This isolates the effect of instructed deceptive strategy,
which is the core object of study.

### Conditions

| Condition ID | murderer_behavior_mode | progression_policy | cell_role |
|---|---|---|---|
| `passive-concealment` | passive_concealment | round_budget | control |
| `active-deception` | active_deception | round_budget | deception_treatment |

### Held-constant controls
- **progression_policy:** `round_budget` (fixed). Simplest, most validated path; removes a confound between dialogue structure and deception.
- **dialogue budget:** standard (20 conversations/round).
- **memory:** `three-stage-v1`.
- **scenario / model / temperature:** identical across conditions.

### Why a single factor (and not a 2×2)

The previous plan crossed `murderer_behavior_mode` with `progression_policy` (a
2×2). That has been simplified to a single factor for the following reasons:

- **All three RQs are about the murderer**, not about dialogue structure.
  Progression policy is a moderator of the surrounding discussion, not the object
  of study.
- **Cleaner attribution.** With progression held constant, any condition
  difference maps directly onto deception coaching — the cleanest possible
  estimate for RQ1 and RQ3.
- **Better statistical power per condition.** Two conditions concentrate the run
  budget into two cells instead of four, roughly doubling usable runs per cell for
  the same total compute.
- **Easier to explain and defend:** "the murderer was, or was not, instructed to
  use deceptive tactics."

### Optional robustness arm (not a primary factor)
A small `evidence_gated` set (≈8 runs per condition) may be run as a **robustness
check** to confirm the deception effect is not an artifact of round-budget
progression. It is reported as a sensitivity analysis, not as a manipulated
factor, so it does not dilute the primary comparison.

### Why memory is held constant
The repo tracks `memory_version` but does not expose a second implemented memory
policy that changes runtime behavior in a clean, auditable way. A factor should
only be manipulated if it actually changes the system. Memory remains fixed at
`three-stage-v1`.

---

## How the design aligns with the research questions

### RQ1 — Deception strategies
Comparing `passive_concealment` vs `active_deception` directly tests whether
explicit tactic coaching changes the frequency and variety of labeled deceptive
behaviors. Expected finding: `active_deception` produces more labeled strategies
per run, more diverse strategy types, and higher-confidence labels.

### RQ2 — Attention distribution
Both conditions are analyzed for how questioning, follow-ups, and pressure
distribute across suspects (and specifically onto the murderer). The comparison
tests whether a coached, more deceptive murderer attracts more — or successfully
deflects — investigative scrutiny relative to the passive murderer.

### RQ3 — Accusation outcomes
The `passive_concealment` vs `active_deception` escape-rate difference is the
cleanest estimate of whether explicit tactic coaching helps the murderer avoid
accusation. Both are also compared against the chance-level baseline.

---

## Run targets

### Readiness thresholds (per condition)
- **Pilot-ready:** 5 usable runs
- **Interim-analysis-ready:** 12 usable runs
- **Final-analysis-ready:** 24 usable runs

### Total planned runs
- 2 conditions × 30 replicates = **60 planned runs**

At a realistic ~15% exclusion rate, this yields roughly 24–26 usable runs per
condition — enough for stable condition means and reasonable power on binary
outcomes (e.g., group solve rate, murderer escape rate).

---

## Recommended execution phases

### Phase 1 — Sanity / instrumentation lock
Run 3 runs per condition (total = 6).

Goal:
- confirm every run produces complete artifacts,
- inspect `run_validation.json` and `progress_report.md`,
- verify `murderer_behavior_mode` is correctly recorded in manifests,
- verify the structured accusation schema is filled reliably,
- **most importantly:** read transcripts and confirm the passive and active
  murderers behave **visibly differently** (passive restrained, active actively
  lying/redirecting). If they look the same, fix the prompts before scaling.

Do **not** interpret thesis results yet.

### Phase 2 — Pilot comparison
Run until 5 usable runs per condition (total usable = 10).

Goal:
- detect obviously broken or weak conditions,
- inspect whether `active_deception` runs produce more labeled deceptive behaviors,
- confirm the deception judge labels look sensible against the transcripts.

### Phase 3 — Interim analysis
Run until 12 usable runs per condition (total usable = 24).

Goal:
- compare trends with enough stability to discuss in working notes,
- inspect the passive-vs-active difference in strategy rates, attention, and escape.

### Phase 4 — Final thesis batch
Run until 24 usable runs per condition (total usable = 48).

Goal:
- freeze the condition set,
- rebuild the thesis dataset,
- generate final comparisons,
- choose qualitative transcript excerpts for the results chapter,
- (optional) run the small `evidence_gated` robustness arm.

---

## What to check after every batch

### Workflow status
Read:
- `outputs/<experiment>/progress_report.md`
- per-condition `validation_summary.json`
- experiment `condition_report.md`

### Minimum quality expectations
A run should ideally have:
- a completed accusation phase,
- structured accusations with at least 2 evidence items,
- the murderer directly questioned in at least some runs,
- acceptable suspect question coverage,
- round summaries present,
- `murderer_behavior_mode` correctly recorded in `run_manifest.json`.

### Warning patterns to watch
Investigate quickly if you see too many:
- `murderer_never_directly_questioned`
- `murderer_never_directly_challenged`
- `accusation_structure_often_incomplete`

If those warnings dominate, fix the system before scaling further.

---

## Primary thesis comparisons

### Main deception effect
**`passive-concealment` vs `active-deception`**

The single primary comparison. It estimates whether explicit tactic coaching
changes deception rates (RQ1), the attention drawn to the murderer (RQ2), and
accusation outcomes (RQ3), with no confound from progression structure.

### Chance baseline (RQ3)
Each condition's group solve rate and murderer escape rate are compared against
the chance-level baseline for the suspect count, establishing whether the murderer
escapes above chance and whether coaching widens that margin.

### Robustness (optional)
If the `evidence_gated` arm is run, compare the passive-vs-active difference under
both progression policies. A stable difference under both strengthens the claim
that the effect is driven by deception coaching, not dialogue structure.

---

## Primary thesis outputs to use

### RQ1 — Deception strategies
- `deception_labels.csv`
- `runs.csv`
- `utterances.csv`
- selected qualitative excerpts from high-deception runs
- strategy frequency and variety compared between `passive_concealment` and `active_deception`

### RQ2 — Attention distribution
- `interactions.csv`
- `attention_summary.json`
- condition summaries for murderer attention, follow-ups, and pressure
- attention concentration (e.g., Gini / entropy) compared between conditions

### RQ3 — Accusation outcomes
- `accusations.csv`
- `metrics.json`
- `condition_report.md`
- `condition_chance_baseline.csv`
- group solve rate and murderer vote share compared between conditions and against chance

### Workflow / validity chapter support
- `run_validation.json`
- `validation_summary.json`
- `progress_report.md`

---

## Thesis writing guidance

### Results chapter structure
1. **System validity / workflow quality**
   - How many runs were planned, excluded, and retained
   - Whether `murderer_behavior_mode` was correctly recorded and the prompt difference was stable and visible in transcripts

2. **RQ1 — Deception strategy findings**
   - Strategy frequency and variety by `murderer_behavior_mode`
   - Qualitative examples from structured transcript evidence
   - Expected finding: `active_deception` produces more diverse labeled strategies

3. **RQ2 — Attention distribution findings**
   - How attention and pressure concentrate on the murderer in each condition
   - Whether a coached murderer attracts more scrutiny or successfully deflects it

4. **RQ3 — Accusation outcome findings**
   - Solve rate and murderer vote share by condition
   - Chance baseline comparison
   - Accusation structure quality

5. **Interpretation / mechanism discussion**
   - Whether deception coaching helps the murderer escape (the main effect)
   - Whether the proposed mechanism (deceptive tactics → misleading attention → incorrect accusation) is visible in the data

---

## Configuration

The experiment config (`configs/thesis-final-matrix.yaml`) must define exactly the
**two** conditions above, both with `progression_policy: round_budget` and a
standard dialogue budget, differing only in `murderer_behavior_mode`. The optional
robustness arm, if used, is defined as a separate small config rather than mixed
into the primary batch.

---

## Practical command sequence

Validate the config:

```bash
./venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml --validate-only
```

Preview the full plan:

```bash
./venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
```

Run the batch:

```bash
./venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml
```

Refresh workflow reports manually if needed:

```bash
./venv/bin/python analysis/workflow.py outputs/<experiment>
./venv/bin/python analysis/compare_conditions.py outputs/<experiment>
./venv/bin/python analysis/build_thesis_dataset.py outputs/<experiment>
```

---

## Standard for success
A strong thesis result here is not merely:
- "one condition solved more murders."

A strong thesis result is:
- a condition difference backed by enough usable runs,
- evidence that `murderer_behavior_mode` changed labeled deception behavior (RQ1),
- evidence about how investigative attention distributes around the murderer (RQ2),
- accusation outcomes compared against chance that show whether deception coaching
  helps the murderer escape (RQ3),
- and accusation outputs that show what the agents thought the case actually was.

That is the standard this run plan is aiming for.

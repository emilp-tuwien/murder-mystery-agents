# NEXT_STEPS.md - Murder Mystery Thesis Project

## Current state
The repo now has the essential thesis workflow spine:
- batch experiment runner
- condition matrices
- per-run artifact capture
- thesis dataset export
- run validation and progress reporting
- evidence-gated progression support
- structured accusations with evidence fields
- final thesis matrix + run plan

So the project is no longer missing the experiment backbone.
The next work should optimize for **thesis signal quality**.

---

## Highest-value next priorities

### 1. Tune evidence-gated progression with real runs
Now that evidence-gated stages exist, the key question is whether the thresholds are calibrated well.

Check:
- Are evidence-gated rounds usually satisfying the gate?
- Or are they frequently hitting `hard_cap_fallback`?
- Do clue references and direct questioning actually increase before accusations?

If hard-cap fallback is common, tune:
- `min_round_gate_conversations`
- `max_round_gate_conversations`
- `min_evidence_signals_per_round`
- `min_pressure_signals_per_round`

### 2. Improve accusation semantic quality
The structure is now there, but the content still needs to be inspected in practice.

Check:
- Are `evidence_items` specific or vague?
- Are motive / means / opportunity fields being used meaningfully?
- Does `comparative_case` distinguish the chosen suspect from alternatives?

If weak, tighten the accusation prompt and possibly add a lightweight accusation-quality judge.

### 3. Add qualitative sample export for thesis writing
The thesis will need concrete examples, not only CSV aggregates.

Add an exporter that selects runs such as:
- high-pressure solved runs
- high-pressure unsolved runs
- deceptive-success runs
- evidence-gated runs with no hard-cap fallback
- evidence-gated runs that still failed despite strong pressure

### 4. Run the final matrix in phases
Use `configs/thesis-final-matrix.yaml` and `THESIS_RUN_PLAN.md`.

Recommended progression:
- 2 runs / condition sanity pass
- 5 usable runs / condition pilot readout
- 12 usable runs / condition interim readout
- 24 usable runs / condition final batch

---

## What should *not* be the focus right now
- adding more scenarios
- inventing extra factors that do not yet change runtime behavior
- overcomplicating statistics before the final dataset exists
- polishing UI ahead of the thesis evidence path

---

## Decision rule
For every next change, ask:

> Does this improve the defensibility of the thesis evidence?

Good examples:
- clearer gates
- stronger accusation structure
- better exclusion criteria
- better qualitative export
- cleaner factor isolation

Weak examples:
- cosmetic prompt churn with no measurable consequence
- fake factor labels that do not change behavior
- more conditions before the current matrix is validated

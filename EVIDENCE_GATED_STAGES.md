# EVIDENCE_GATED_STAGES.md

## Purpose
This file describes how evidence-gated progression now works in the project and what still remains to improve.

The goal is simple:
- do not advance the investigation only because a fixed number of turns elapsed,
- advance when the discussion has produced enough questioning, clue integration, and synthesis to make the next phase meaningful,
- but keep a hard cap so experiments always terminate.

---

## Current implementation

The repo now supports two progression policies:

1. `round_budget`
   - legacy behavior
   - advance once the configured round budget is reached

2. `evidence_gated`
   - preferred thesis behavior
   - evaluate the current round after each utterance
   - advance when the gate is satisfied
   - if the round hits a hard cap first, advance with an explicit fallback log

---

## Stage mapping

The implementation still uses rounds internally, but each investigation round now corresponds to a stage label:

- Round 1 → `introduction`
- Round 2 → `initial_framing`
- Round 3 → `clue_integration`
- Round 4 → `contradiction_pressure`
- Round 5 → `accusation_synthesis`
- Final phase → `accusation`

This keeps compatibility with the existing clue and round system while making the workflow more interpretable.

---

## Gate signals currently used

For evidence-gated rounds, the runtime now tracks signals such as:
- number of direct question targets
- suspect question coverage fraction
- number of distinct suspects mentioned
- evidence-signal utterances
- pressure-signal utterances
- clue-reference utterances
- synthesis-signal utterances in the final investigation round

These are heuristic but fully logged and reproducible.

---

## Advance outcomes

Every round evaluation now produces a `stage_gate_evaluated` event and round transitions now record why they happened.

### Possible advance reasons
- `all_introductions_completed`
- `round_budget_reached`
- `evidence_gate_satisfied`
- `hard_cap_fallback`

This is critical for thesis interpretation.
It allows later analysis to distinguish:
- a run that progressed because the discussion met evidence requirements,
- from a run that progressed only because the hard cap was hit.

---

## Why this matters for the thesis

Without gate logging, a poor accusation outcome is ambiguous:
- maybe the murderer was actually persuasive,
- or maybe the discussion simply never surfaced enough evidence.

With gate logging, you can say much more:
- whether the group actually pressured suspects,
- whether clue content was integrated,
- whether the accusation phase followed meaningful synthesis,
- whether a condition repeatedly fell back to hard caps.

That makes outcome comparisons more defensible.

---

## Current limitations

The current gate system is still heuristic.
It does **not** yet require:
- explicit contradiction resolution,
- agent-level suspicion convergence,
- direct hidden-state verification of what each agent believes,
- judge-model semantic verification that a clue was genuinely understood.

So this is a strong improvement over fixed budgets, but not yet a perfect epistemic gate.

---

## Recommended next upgrade

If a later iteration is needed, the best next evidence-gate upgrade would be:

1. require at least one contradiction/alibi challenge before late-stage progression
2. require stage summaries to mention a minimum number of distinct suspects
3. add a lightweight semantic clue-use judge for whether current-round clue content was actually incorporated
4. export stage-level summary tables in the thesis dataset

---

## Practical interpretation rule

For thesis reporting, prefer these distinctions:

- **best evidence**: evidence-gated condition, no hard-cap fallback, structured accusations present
- **usable but weaker evidence**: evidence-gated condition with some hard-cap fallback
- **baseline comparison**: round-budget condition used as the control protocol

That framing gives the thesis stronger methodological discipline.

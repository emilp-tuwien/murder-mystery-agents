# EVIDENCE_GATED_STAGES.md

## Purpose
Refactor the current murder mystery game flow from fixed turn-count rounds into **evidence-gated stages** so that accusation outcomes reflect agent reasoning and deception under sufficiently exposed evidence, rather than premature stage transitions.

This document is intended as a practical design note for thesis-oriented implementation.

---

## Why change the current round system?

The current game uses a round structure with progression based largely on a fixed number of conversations.
That creates a scientific validity risk:

- the murderer may avoid accusation because the game advanced too early,
- not because the murderer successfully deceived the group.

This is especially important now that **Norman knows from the start that he is the murderer**.
That change is scientifically useful, because it creates genuine deceptive behavior during the investigation. However, it also raises the bar for fair evaluation:

- if Norman is not identified,
- we should be more confident that this reflects deceptive success or investigative failure,
- not insufficient evidence exposure.

Evidence-gated progression addresses that problem.

---

## Core design principle

Replace:
- **fixed conversation-count progression**

with:
- **stage progression triggered by evidence exposure and integration conditions**

The game should only move forward when the currently relevant evidence has been meaningfully introduced into the shared investigation process.

Meaningful introduction is stronger than merely loading a clue file. It should involve at least some combination of:

- the clue or fact being surfaced aloud,
- one or more agents reacting to it,
- the clue influencing suspicion, questioning, or contradiction handling.

---

## Recommended stage structure

Do not remove structure entirely. Keep a staged format for comparability and experimental control.

### Stage 1 - Introductions and baseline claims
**Goal:** establish initial personas, alibis, and early suspicion anchors.

**Required outcomes before progressing:**
- every suspect has spoken at least once,
- at least one baseline self-locating or self-descriptive claim is stored for each suspect,
- at least one initial question or challenge has been asked.

**Scientific role:**
- creates the baseline from which later contradictions can be measured.

---

### Stage 2 - First key clue exposed
**Goal:** ensure the first clue enters dialogue and affects the investigation.

**Required outcomes before progressing:**
- clue 1 has been explicitly mentioned in dialogue,
- at least two agents have referenced or responded to it,
- at least one suspicion update or contradiction candidate is produced from it.

**Scientific role:**
- ensures the first external evidence item actually enters the reasoning process.

---

### Stage 3 - Opportunity / timeline pressure
**Goal:** force interrogation around presence, movement, and access.

**Required outcomes before progressing:**
- all major suspects have either stated or defended a timeline/alibi claim,
- at least one contradiction or unresolved inconsistency is logged,
- Norman has been directly challenged at least once if evidence permits.

**Scientific role:**
- makes opportunity-based reasoning observable and comparable.

---

### Stage 4 - Means / motive consolidation
**Goal:** connect revealed information to stronger hypotheses about guilt.

**Required outcomes before progressing:**
- at least one means-related and one motive-related evidence item have been surfaced,
- at least two suspects have been discussed in comparative terms,
- suspicion ranking is non-empty and differentiated.

**Scientific role:**
- moves the game from free conversation into hypothesis formation.

---

### Stage 5 - Contradiction resolution and focused confrontation
**Goal:** surface the strongest unresolved pressure points before accusation.

**Required outcomes before progressing:**
- the highest-suspicion suspect(s) have been directly challenged,
- unresolved contradictions are summarized,
- agents have enough evidence to compare leading suspects.

**Scientific role:**
- reduces the chance that accusation outcomes are driven by missing confrontation.

---

### Stage 6 - Final accusation
**Goal:** produce evidence-backed accusations.

**Required outputs:**
- each agent accuses one suspect,
- each accusation cites concrete evidence,
- each accusation includes at least one contradiction, alibi weakness, or motive/means/opportunity argument,
- the run saves all accusation reasoning in structured form.

---

## What should count as a gate condition?

Use **operational gate conditions**, not vague narrative judgments.

A gate condition should be something the system can measure from logs or memory.

Recommended gate categories:

### 1. Exposure gates
Checks whether a clue/fact was actually spoken aloud.

Examples:
- clue text or its normalized fact appears in shared memory,
- at least one utterance references the clue topic.

### 2. Reaction gates
Checks whether other agents engaged with the evidence.

Examples:
- at least two distinct agents referenced the clue,
- at least one question was asked about it,
- at least one rebuttal/defense occurred.

### 3. Integration gates
Checks whether the evidence changed internal state.

Examples:
- suspicion score updated,
- contradiction recorded,
- new suspect-specific evidence stored.

### 4. Coverage gates
Checks whether enough suspects were examined before moving on.

Examples:
- each suspect has an alibi statement,
- each suspect was questioned at least once,
- Norman was challenged when he was among the top suspects.

---

## Minimal implementation strategy

The first version does not need perfect semantic understanding.
A hybrid heuristic approach is enough.

### Step 1 - Add stage state
Add to game state:

- `current_stage`
- `stage_started_turn`
- `stage_gate_status`
- `stage_evidence_targets`
- `stage_max_turns`

Example conceptual fields:

```python
current_stage: int
stage_gate_status: dict
stage_evidence_targets: list[str]
stage_max_turns: int
```

---

### Step 2 - Add gate evaluators
Create a small evaluator module that checks whether stage conditions are met.

Suggested file:
- `evaluation/stage_gates.py`

Possible functions:

```python
def evaluate_stage_1(state, agents) -> dict:
    ...

def evaluate_stage_2(state, agents) -> dict:
    ...
```

Each evaluator should return something like:

```python
{
  "stage_complete": True,
  "reasons": ["all suspects introduced", "first clue referenced by 2 agents"],
  "metrics": {...}
}
```

---

### Step 3 - Replace fixed round advancement
Currently stage/round progression is driven by conversation count.
Refactor `check_round_advance()` into a stage progression function that:

1. evaluates current stage gates,
2. advances if gates are satisfied,
3. falls back to a max-turn escape hatch if needed,
4. logs why the stage advanced.

Suggested rename:
- `check_stage_advance()`

---

### Step 4 - Keep a max-turn safety cap per stage
Evidence-gated should not mean infinite discussion.
Each stage should have:

- required conditions,
- but also a **max-turn cap**.

If the stage max is reached:
- force advancement,
- log which gate conditions were unmet,
- mark the stage as **under-exposed**.

This is useful scientifically because it distinguishes:
- normal evidence completion,
- forced progression due to failure to surface evidence.

---

### Step 5 - Log gate completion events
For each stage, save:

- when it started,
- when it ended,
- which gates were met,
- which were forced,
- which clues/facts were surfaced,
- whether Norman was challenged,
- whether suspicion changed.

This should become part of the run output.

---

## How this connects to the thesis research questions

### RQ1 - Deceptive communication strategies
Evidence-gated stages make RQ1 stronger because Norman is more likely to face meaningful pressure before accusation.
This helps distinguish:
- deception under evidence pressure,
from:
- easy escape due to insufficient evidence exposure.

### RQ2 - Attention distribution
The staged system makes it easier to measure:
- who was questioned,
- who was ignored,
- whether Norman attracted investigative attention,
- how attention shifted after clues.

### RQ3 - Accusation outcomes
This is where the change matters most.
If Norman avoids accusation after evidence-gated progression, the result is much more interpretable.
It is more plausible to say:
- Norman evaded detection,
- rather than the system simply failed to reveal enough incriminating information.

---

## Recommended companion changes

Evidence-gated progression will help, but it should not stand alone.
It works best alongside the following improvements.

### 1. Structured suspect-centric memory
Store evidence not just as free text but as suspect-linked entries:
- subject,
- source,
- tags,
- contradiction links,
- alibi/timeline claims.

### 2. Automatic suspicion updates
Use clue references, evasive responses, and contradictions to update suspicion.

### 3. Evidence-backed accusation output
Require each final accusation to cite concrete evidence.

### 4. Run logging
Save stage gate events and accusation logic for later analysis.

---

## Risks and trade-offs

### Benefits
- better scientific validity,
- better interpretability of accusation failure,
- cleaner evidence of deception under pressure,
- stronger support for comparative experiments.

### Risks
- more implementation complexity,
- possible over-engineering if gates are too strict,
- risk of unnatural dialogue if stage transitions become too mechanical.

### Mitigation
- begin with simple gates,
- use a max-turn fallback,
- log forced transitions,
- iterate based on observed runs.

---

## Recommended first implementation pass

### Phase A - Lightweight gate refactor
Implement:
- stage state,
- simple gate heuristics,
- max-turn fallback,
- stage logs.

Do **not** try to solve full semantic understanding immediately.
Use existing memory tags and conversation history first.

### Phase B - Stronger evidence integration
Then add:
- suspect-linked evidence records,
- contradiction tracking,
- suspicion updates.

### Phase C - Stronger accusation stage
Then require:
- evidence comparison across suspects,
- structured accusation outputs.

---

## Proposed verdict

Refactoring the current game into evidence-gated stages is a **suitable and scientifically justified adjustment**.

It should not be implemented as a completely structureless free-form discussion. The better design is:

- keep ordered stages,
- reveal information progressively,
- advance only when evidence has been sufficiently surfaced or the stage cap is reached,
- log whether progression was complete or forced.

This will make the project substantially more defensible as a thesis artifact.

---

## Short implementation checklist

- [ ] Add `current_stage` and stage gate state to `GameState`
- [ ] Create `evaluation/stage_gates.py`
- [ ] Replace round advancement with gate-based stage advancement
- [ ] Add max-turn fallback per stage
- [ ] Log why each stage advanced
- [ ] Ensure clue exposure is measured explicitly
- [ ] Ensure Norman challenge coverage is measured
- [ ] Update final accusation to require evidence-backed reasoning
- [ ] Save stage gate results in run outputs

---

## Suggested next step

After this document, the best immediate implementation step is:

1. refactor the state machine from rounds to stages,
2. implement simple heuristic gate checks,
3. update the accusation phase to consume stage/evidence summaries.

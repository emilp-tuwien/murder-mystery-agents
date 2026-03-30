# turn-taking.md

## Purpose
This document explains the turn-taking changes made to reduce unrealistic behavior where nearly every agent wanted to speak on every turn.

The target behavior is:
- usually only **1-2 agents** strongly want to speak on a given turn
- the rest should often prefer to **listen**, wait, or hold back strategically
- turn-taking should reflect **character-specific strategy**, not just generic participation pressure
- the UI should show a more believable spread of urgency instead of six agents all competing for the floor

---

## Problem observed
Before this change, the simulation often produced a pattern where many or most agents had a high desire to speak at the same time.

Why that was a problem:
- it made the discussion feel unnatural
- it reduced the value of the urgency score as a signal
- it made the Game Master selection look arbitrary because too many agents were tied or near-tied
- it weakened character differentiation
- it made the frontend display look noisy and less believable

In a realistic social investigation scene, not everyone should jump in constantly. Some characters should:
- wait for the right opening
- speak only when challenged
- strategically stay quiet
- avoid exposing themselves too early
- prefer redirecting or observing rather than immediately talking

---

## Changes made

### 1. Added character-specific strategic speaking guidance
In `agents/agent.py`, the thinking step now includes explicit strategy guidance per character.

Examples:
- **Dr Chelsea Barren**: analytical, more likely to speak when she can expose contradictions or push a theory
- **Enrique Graves**: risk-aware and self-protective, prefers silence unless challenged or able to redirect suspicion
- **Kathryn Lawless**: selective and careful, should not speak without a meaningful gain
- **Michael Nightshade**: opportunistic, speaks when he can steer the narrative
- **Norman D'Adly**: more reactive and emotional, more likely to speak when provoked
- **Vicki D'Adly**: defensive and strategic, speaks when protecting herself or exploiting a mistake

This makes speaking behavior depend more on role/personality and less on a generic “contribute now” instinct.

---

### 2. Strengthened the prompt toward selective silence
The agent thinking prompt now explicitly says:
- default to **listening** unless there is a strong reason to speak
- only **one or two agents** should strongly want to speak on a typical turn
- strategic silence is often correct
- agents should consider whether speaking would expose them unnecessarily
- agents should consider whether another suspect has a better move right now

This helps push the model away from inflated urgency.

---

### 3. Added recent-speaker penalties
The thinking logic now looks at recent speaking behavior:
- whether the agent spoke last turn
- how many times the agent spoke in the last 6 non-Game-Master utterances
- whether the agent appears repeatedly in the recent speaking window

These conditions lower urgency for agents who have already had recent floor time.

Effect:
- repeat speakers cool down faster
- monopolizing the conversation becomes less likely
- quieter agents get more chance to surface when they actually have something useful to say

---

### 4. Added stronger heuristics for weak “speak” decisions
After the model returns its thought/action/importance, the result is calibrated.

New logic includes:
- lowering urgency for repeat speakers
- lowering urgency for cautious/self-protective characters when there is no strong evidence signal
- slightly lowering urgency for the murderer when there is no strong reason to speak
- boosting Norman a bit in more reactive/provoked situations
- downgrading overly high scores when the thought text does not mention strong investigative signals such as:
  - contradiction
  - clue
  - evidence
  - alibi
  - timeline
  - motive
  - opportunity
  - direct response / question

This makes high scores harder to justify.

---

### 5. Global top-2 normalization in `graphs/discussion.py`
Even after per-agent thinking, multiple agents may still end up with high urgency.

A new graph-level normalization step now:
- preserves a direct-address response if one agent was explicitly addressed
- otherwise allows only the **top two** agents with urgency `>= 6` to remain active “speak” candidates
- pushes the rest down to `listen`

This is important because it enforces a conversation bottleneck closer to natural turn-taking.

In effect:
- the room can still have multiple interested agents
- but only a small number should visibly compete for the next turn

---

## Why this should improve realism
These changes should make the simulation feel more like a real group conversation because:
- silence becomes a legitimate strategic choice
- character differences matter more
- speaking is tied more tightly to evidence, pressure, and social timing
- recent floor ownership affects future speaking desire
- the displayed urgency distribution becomes easier to interpret

Expected pattern after the change:
- 1 agent strongly wants to speak
- sometimes 2 agents compete
- the rest mostly listen, wait, or remain low urgency
- direct questions still force the addressed agent to respond

---

## What to watch in the UI
When evaluating the new behavior in the frontend, look for:
- fewer turns where many agents simultaneously look eager to speak
- more low and medium urgency values
- more believable differences between aggressive and cautious characters
- more stable transitions between speakers
- less “everyone wants the floor” clustering

If it still feels too noisy, next tuning options would be:
- stricter cooldown windows
- allow only top-1 instead of top-2 in some situations
- add per-character base assertiveness scores
- penalize repeated questioning more strongly
- add explicit “hold back secret” behavior for high-risk agents

---

## Files changed for this turn-taking adjustment
- `agents/agent.py`
- `graphs/discussion.py`

---

## Summary
This change shifts turn-taking from:
- broad, generic willingness to participate

toward:
- selective, personality-shaped, strategically constrained participation

That should make both the simulation and the observer UI more credible.

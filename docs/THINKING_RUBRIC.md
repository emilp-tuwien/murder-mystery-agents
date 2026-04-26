# THINKING_RUBRIC.md - Agent Urgency / Turn-Taking Rubric

This file documents how agent thought scores are intended to work so the design does not drift over time.

## Purpose
The score is not a generic confidence value. It is a **turn urgency** score: how strongly a specific agent should want to speak **right now**.

The goal is to prevent all agents from wanting to speak every turn and to encourage more realistic, investigation-driven turn-taking.

## Score Meaning

- **9** — Must answer now
  - The agent was directly addressed
  - The group is waiting on this agent specifically
  - The agent has an immediate obligation to respond

- **8** — High-priority intervention
  - The agent has unique, high-value evidence
  - The agent sees a strong contradiction
  - The agent can immediately clarify a critical alibi, motive, means, opportunity, or timeline conflict

- **6-7** — Useful but not urgent
  - The agent has a targeted investigative question
  - The agent has a helpful fact, but it is not the only viable next move
  - The contribution advances the discussion, but someone else might also reasonably speak first

- **4-5** — Moderate / situational contribution
  - The agent can add context or a decent follow-up
  - The contribution may help if the discussion is drifting
  - Not strong enough to dominate turn selection by default

- **2-3** — Weak contribution
  - Mostly reactive, repetitive, vague, or low-value
  - Better to listen unless the conversation stalls badly

- **0-1** — Definitely listen
  - No useful contribution right now
  - Repetition likely
  - Another suspect clearly has the better next move

## Intended Behavioral Rules

1. **Not everyone should want to speak every turn.**
   - Usually only one or two agents should have high urgency on a given turn.

2. **Direct address outranks everything else.**
   - If an agent is directly asked, their score should be 9.

3. **Recently speaking should reduce urgency.**
   - An agent who just spoke should usually step back unless they must immediately clarify something.

4. **High scores require concrete investigative value.**
   - High urgency should be tied to one of:
     - contradiction
     - clue
     - alibi
     - motive
     - means
     - opportunity
     - timeline conflict
     - direct response obligation

5. **Listening is a valid strategic choice.**
   - Good investigation is not constant talking.

## Why this matters for the thesis
This rubric helps turn-taking become more realistic and analytically meaningful.
If all agents constantly choose `speak`, then questioning patterns, attention distribution, and suspicion pressure become noisy and much less useful for RQ2 and RQ3.

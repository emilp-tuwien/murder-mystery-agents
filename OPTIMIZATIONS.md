# OPTIMIZATIONS.md

Track of cost / speed optimizations applied to the project, and the no-regret
upgrades still on the table.

## Applied changes

### 1. Prompt prefix caching — agents/agent.py (DONE)
**Goal:** make the SystemMessage byte-identical across all turns within a round
so OpenAI-compatible servers (local vLLM / llama.cpp / sglang, OpenAI API)
hit their prefix cache. Cached reads cost a fraction of fresh reads.

**What was moved into the SystemMessage (now cacheable within a round):**

`think()` — non-introduction branch (agents/agent.py around the `else:` after
the introduction block):
- The murderer / non-murderer DECISION FRAME text (was inside HumanMessage,
  bucketed by `current_round <= 2 / <= 4 / else`).
- The listen-default lines for non-murderers.
- The "When deciding whether to speak, ask yourself…" question list.
- The `Actions:` / `Importance guidance:` block.
- The `Return valid JSON with keys…` output spec and `reason_type` enum.

`speak()` — non-introduction branch:
- The full static `rules_block` (without the per-turn `Can ask:` line, which
  was extracted — see below).
- The `Respond in 1-2 natural sentences.` / `Output dialogue only.` instructions.

**What stayed in HumanMessage (per-turn dynamic):**
- `think()`: `memory_context`, the `Status:` block (directly addressed,
  recently_spoke, recent_speaker_count, consecutive_recent, strategy_guidance),
  `own_recent_statements`, and a short `Now decide…` trigger.
- `speak()`: `memory_context`, the optional `RESPOND TO:` constraint,
  the dynamic `Can ask: …` line (which changes every turn as the agent
  exhausts targets), and a short `Speak now.` trigger.

**Why this works:** within a round, `self.persona`, `self.is_murderer`,
`current_round`, and the round-bucketed strategy text all stay the same. The
SystemMessage is therefore byte-identical across all ~20 conversations of the
round × all agents. Modern OpenAI-compatible servers cache the longest
matching prefix from the start of the request, so the entire SystemMessage
hits cache after the first call of each round.

**Expected savings:** the SystemMessage is the bulk of every prompt
(character sheet + goals + rules + JSON spec). Estimated 50–80 % reduction on
input-token cost for `think()` and `speak()` calls after the first turn of
each round.

**Caveats:**
- The persona is mutated by `update_round()` (it appends accumulated knowledge
  each new round), so cache evicts at every round boundary. That is fine —
  there are only ~5 round boundaries vs. ~140 turns per game.
- The introduction branch (round 1) was left unchanged — it only runs once
  per agent per game, so caching does not matter there.
- Behavior is identical: this is a pure reordering of prompt content. No
  rule, hint, or instruction was added or removed.

### 2. Parallel `think()` across agents (ALREADY DONE)
Already implemented before this session:
- `graphs/discussion.py:169–181` — `think_all` uses `ThreadPoolExecutor` with
  one worker per agent. All agents' `think()` calls run concurrently per turn.
- `graphs/discussion.py:83–92` — per-round private suspicion assessment is also
  parallelized the same way.

No change needed.

## No-regret optimizations still on the table

These do not change experimental conclusions and only affect cost / speed.
Listed in rough order of leverage.

### 3. Tiered models — cheap model for `think()`, expensive only for `speak()`
The base paper (Nonomura & Mori 2025) did exactly this: GPT-3.5-turbo for
`think()` and memory normalization, GPT-4o only for `speak()` and
`detectDesignation()`.

`think()` runs ~7× per turn × ~140 turns ≈ ~1000 calls per game and only needs
to output `speak/listen` + an integer 0-9 + a short thought. Use the smallest
backend that produces valid JSON for it.

Hook point: `agents/agent.py:145–146` — currently `self.llm` and
`self.llm_think` share the same underlying `llm`. Wire a second cheaper
`llm_think_backend` through `RunConfig` and pass it into `Agent.__init__`.

### 4. Cap `max_tokens` on every generation call
- `speak()`: cap at ~120 tokens. Current calls often produce 200+ token
  monologues that hurt both cost AND dialogue quality (more rambling → more
  breakdowns, per the base paper).
- `think()`: cap at ~40 tokens (it is just a JSON blob).
- Deception judge: cap at ~200 tokens (one JSON object).
- Accusation generation: cap at ~300 tokens.

Hook point: pass `max_tokens=…` into the `ChatOpenAI(**kwargs)` constructor in
`run_discussion.py:_build_openai_llm`, and into the judge LLM builder at
`analysis/deception_judge.py:_build_judge_llm`.

### 5. Two-stage deception judge
Currently `analysis/deception_judge.py` runs the full taxonomy prompt with
±4/±2 turn context for every murderer utterance (~17 calls per run × 96 runs
≈ 1.6k expensive calls).

- Stage 1 (cheap, small model): "Is this utterance plausibly deceptive?
  yes / no / unsure." Drop the full taxonomy. Drop most context.
- Stage 2 (expensive, current judge): only on stage-1 positives + unsures.

Expected filter rate is 60–80 % no on stage 1, i.e. 3–5× judge cost savings.

Hook point: introduce `filter_utterances(...)` upstream of `judge_utterances`
in `analysis/deception_judge.py`.

### 6. Lower judge context window
Currently in `configs/thesis-final-matrix.yaml`:
```
deception_judge_context_before_turns: 4
deception_judge_context_after_turns: 2
```
Drop to `2` / `1`. ~40 % smaller judge prompts. Labels rarely depend on turn
4-back evidence — the target utterance + immediate surroundings are enough.

### 7. Asymmetric replicates + early stopping
The primary RQ3 comparison is `passive__round-budget` vs
`active__round-budget`. The two `evidence_gated` cells are secondary.

Suggested allocation:
- 20 replicates per round-budget cell (40 total — primary)
- 10 replicates per evidence-gated cell (20 total — secondary)
- Total: 60 instead of 96 planned runs (~38 % saving)

Pre-register a stopping rule: stop early if the 95 % bootstrap CI on the
solve-rate delta excludes zero with ≥ 12 usable runs per cell.

### 8. Drop `max_rounds` from 6 to 4 (only if pilots confirm it is safe)
Inspect existing pilot data: if accusation ranking is stable by round 4,
drop `max_rounds` to 4. ~33 % saving per run.

### 9. Cache `longTermHistory` embeddings on disk
Once a knowledge item is normalized and embedded, the vector never changes.
Persist embeddings keyed by content hash so they survive across runs in the
same scenario / role config.

Hook point: `memory/agent_memory.py` (long-term memory module).

## What NOT to do without a pilot
These touch statistical power or confound structure, not just cost:
- Reducing number of replicates per cell below the pre-registered floor.
- Reducing number of suspects (changes chance baseline).
- Reducing number of rounds (may change deception trajectory).
- Skipping judge calls without a validated filter step.

## Verification checklist for the prompt-caching change
Before scaling to a full batch, on a single test run check:
- [ ] `agents/agent.py` parses (already verified).
- [ ] One pilot run completes end-to-end with no NameError /
  UnboundLocalError around `decision_frame`, `listen_default_lines`,
  `self_questions`, `static_rules_block`, `system_static`.
- [ ] Output transcripts look qualitatively unchanged vs. the previous
  prompt structure (no agent suddenly stops asking questions, etc.).
- [ ] If the local LLM server exposes a cache-hit metric, confirm hits
  climb after the first turn of each round.

## Future logbook
Append further changes below with date + what + why.

- 2026-05-20: Initial prompt reordering for prefix caching, parallel
  `think()` already in place, OPTIMIZATIONS.md created.

# PROJECT.md - Murder Mystery Thesis Project

## Purpose
Build and evaluate a supervised multi-agent murder mystery simulation as a thesis artifact for studying deceptive behavior, attention distribution, and accusation outcomes in LLM-based multi-agent systems.

## Thesis Fit
This project directly supports the thesis described in `../thesis-materials/thesis-files/proposal.md`.

### Core research targets
- **RQ1:** Identify recurring deceptive communication strategies used by the malicious/murderer agent.
- **RQ2:** Measure attention distribution during dialogue, especially questioning and interaction patterns.
- **RQ3:** Measure whether the murderer avoids accusation above a chance-level baseline.

## Current repo state
The repo already contains:
- agent personas and roles
- a game master
- round/clue progression
- a final accusation phase
- some thought logging
- memory modules
- graph-driven discussion flow

## What this project needs next
To become a thesis-grade research artifact, the repo needs:
1. non-interactive experiment execution
2. structured machine-readable run logging
3. analysis outputs for RQ2 and RQ3
4. a deception-labeling pipeline for RQ1
5. condition/config management for reproducible experiments

## Recent implementation progress
Recent work on `clawy-game` has already improved the prototype in several ways:
- browser observer groundwork / event layer added
- local vLLM endpoint wiring corrected
- in-character dialogue prompts tightened
- investigation-oriented prompts added so agents explicitly try to identify the murderer
- Norman D'Adly response judge added
- turn urgency / speak-listen thinking logic reworked to reduce uniform agent behavior
- score meanings documented in `docs/THINKING_RUBRIC.md`
- browser observer frontend reworked into a more demo-ready dashboard for supervisor presentation
- distinct per-agent icons added in the observer UI so speaker identity is visible at a glance
- transcript readability improved with clearer speaker cards, turn markers, and active-speaker highlighting
- top-level summary cards, progress stage view, evidence/clue panel, and cleaner event/accusation panels added for easier live following
- observer UI further reworked into a fixed six-seat character stage so all agents remain visible in stable positions
- conversation area changed to always show two visible speech bubbles (latest utterance + previous utterance) for easier live following
- removed emoji-based visuals in favor of cleaner inline iconography and a more restrained, readable visual style
- fixed round-1 observer thought exposure so agents (including Chelsea) no longer appear blank during introductions when the backend already knows whose turn it is
- fixed frontend name-matching so `Dr Chelsea Barren` is mapped correctly into the fixed-seat layout instead of falling through due to punctuation mismatch
- reworked turn-taking calibration so agents no longer all want to speak at once; added character-specific strategy guidance, recent-speaker penalties, and graph-level top-speaker normalization (documented in `turn-taking.md`)
- corrected Norman D'Adly’s role instructions so his in-game goal is to conceal his guilt during the investigation rather than prematurely reveal it
- updated memory fact storage so dialogue and clues are categorized into thesis-relevant evidence buckets: motive, means, opportunity, contradiction, timeline, and alibi
- simplified the observer frontend into a more intuitive real-world table scene with the Game Master at the head, six fixed suspect chairs, colored agent markers, and speech bubbles appearing over the talking participants
- added a per-agent memory inspection filter in the observer UI so Emil can inspect what each character has stored by category during the run

## Working branch
- **Clawy branch:** `clawy-game`

## Working principles
- Optimize for thesis evidence, not just gameplay.
- Prefer reproducible experiments over ad-hoc runs.
- Leave clear notes so work can be resumed quickly.
- Keep implementation and analysis tightly linked to the proposal.
- Treat turn scores as urgency-to-speak values, not vague confidence scores.

## Server / access reference
- **Primary server for this project:** `ssh concircle@192.168.20.37`
- When documenting or sharing run/access steps for the current murder mystery environment, default to this server unless Emil says otherwise.
- Browser UI is typically accessed by SSH port forwarding from the local machine to the server-hosted UI port.

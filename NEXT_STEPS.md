# NEXT_STEPS.md - Murder Mystery Thesis Project

## Current priorities
1. Stabilize agent thinking and turn-taking so not everyone wants to speak every turn.
2. Improve structured evidence/memory handling (motive, means, opportunity, contradictions, alibis).
3. Make accusation quality depend on accumulated evidence rather than vague recall.
4. Build a robust observability path for browser-friendly game inspection in the current server environment.
5. Add experiment/logging infrastructure for thesis evaluation.

## What was just done
- strengthened investigation-oriented prompts
- tightened in-character speech prompts
- corrected local vLLM integration (`local_llm`)
- added Norman response judge
- documented turn urgency scoring in `docs/THINKING_RUBRIC.md`
- reworked think prompts to encourage differentiated speak/listen choices

## Proposed immediate roadmap

### Phase 1 - Better thinking and evidence flow
- normalize urgency scores across agents at the group level
- reduce clustering in speak/listen decisions
- add explicit evidence categories to memory
- connect revealed evidence to suspicion tracking

### Phase 2 - Better accusation quality
- force evidence-backed accusation reasoning
- make agents reference motive / means / opportunity / contradiction explicitly
- track whether the final accusation is based on surfaced evidence or weak guessing

### Phase 3 - Observability and UI
- decide whether to continue with live UI or shift to artifact/file-based browser inspection for this server environment
- expose judge outputs, suspicion state, and evidence summaries in the observer layer

### Phase 4 - Experiment spine
- add a non-interactive runner
- support parameterized execution
- create per-run output directories
- save transcript, metadata, and accusation outcome files

### Phase 5 - Analysis support
- compute per-run summary metrics
- extract question/attention patterns
- aggregate accusation outcomes across runs
- prepare transcript labeling support for deceptive strategies

## Notes for morning review
When overnight work is done, summarize:
- what changed
- what files were added/edited
- what is now measurable
- what decisions Emil should make next
- whether turn-taking quality improved or still clusters

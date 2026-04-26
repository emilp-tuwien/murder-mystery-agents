# PROJECT.md - Murder Mystery Thesis Workflow

## Purpose
This repository is a **thesis experiment system**, not just a murder-mystery demo.

The artifact studies how LLM-based agents:
- deceive under investigation pressure,
- distribute attention during group discussion,
- and converge on final accusations.

The murder mystery is the experimental environment. The thesis contribution comes from the **workflow around it**: reproducible conditions, logged runs, structured outputs, quality checks, and analysis that can support defensible claims.

For the concrete final study design and batch plan, see `THESIS_RUN_PLAN.md` and `configs/thesis-final-matrix.yaml`.

---

## Thesis fit
This project supports the thesis described in `../thesis-materials/thesis-files/proposal.md`.

### Core research questions
- **RQ1 — Deception strategies:** What recurring deceptive communication strategies does the murderer use?
- **RQ2 — Attention distribution:** Who gets questioned, challenged, followed up on, and scrutinized over time?
- **RQ3 — Accusation outcomes:** Does the murderer avoid accusation more often than chance, and under which conditions?

---

## Core idea of the workflow
The repo should be understood as a pipeline with six layers:

1. **Scenario + agent system**
2. **Condition planning**
3. **Batch execution**
4. **Run validation / thesis usability checks**
5. **Condition / experiment aggregation**
6. **Thesis-ready export and interpretation**

This separation matters because a fun simulation is not automatically a valid thesis artifact.
The thesis becomes credible only if each run can be traced from:

**condition definition → runtime behavior → logged evidence → final accusation → aggregate comparison**

---

## High-level workflow

```text
config YAML
  ↓
experiment plan
  ↓
condition runs × replicates
  ↓
per-run artifacts
  ↓
run validation
  ↓
condition summaries + progress report
  ↓
cross-condition comparison
  ↓
flat thesis dataset
  ↓
results interpretation / thesis writing
```

---

## Layer 1 - Scenario + agent system
The base simulation provides:
- suspect personas and roles,
- a game master,
- round/clue progression,
- dialogue generation,
- per-agent memory,
- final accusation generation.

This is the **behavioral substrate** of the thesis.
Without this layer, there is no discussion to analyze.

### Current implementation status
The repo already contains:
- scenario loading,
- role-specific knowledge by round,
- clue reveal flow,
- discussion graph,
- accusation phase,
- UI/observer support,
- thought logging,
- suspicion/memory scaffolding.

### Scientific limitation of the current baseline
The current runtime still advances primarily by **round budgets** rather than full evidence-gated progression.
That is acceptable as a baseline, but it means accusation outcomes must be interpreted with care:
- did the murderer escape because deception worked,
- or because the system progressed before enough pressure was applied?

This is why the workflow now explicitly logs **round advance decisions** and run-quality indicators.

---

## Layer 2 - Condition planning
Condition planning defines what is being compared.

### Inputs
A config file specifies:
- experiment name,
- output root,
- condition names,
- condition factors,
- replicate count,
- seed policy,
- backend/model/temperature,
- scenario paths,
- version tags for prompts, memory, and turn policy,
- workflow thresholds:
  - `pilot_ready_runs_per_condition`
  - `interim_ready_runs_per_condition`
  - `final_ready_runs_per_condition`

### Why this layer matters
This is what turns the project from ad-hoc game playing into a reproducible experiment.
A thesis needs named conditions and explicit factor differences.

### Current condition support
The runner supports:
- a single base config,
- explicit named conditions,
- generated Cartesian-product condition matrices,
- plan-only and validate-only modes.

### Core outputs
- `experiment_plan.json`
- `condition_config.json` per condition

These files are the frozen record of what the batch was supposed to run.

---

## Layer 3 - Batch execution
Each condition is executed across replicates.

### Execution goal
The runner must not just produce a verdict. It must leave behind a **forensic trail** that allows later evaluation.

### Per-run artifact expectations
Each successful run should produce:
- `run_manifest.json`
- `events.jsonl`
- `metrics.json`
- `utterances.csv`
- `interactions.csv`
- `accusations.csv`
- `attention_summary.json`
- `deception_labels.csv`
- `run_validation.json`

### What the manifest captures
The manifest ties every run back to:
- experiment and condition name,
- replicate id,
- seed and seed strategy,
- backend/model settings,
- scenario id,
- prompt / turn policy / memory version,
- git commit,
- workflow thresholds.

That makes each run traceable and reproducible.

---

## Layer 4 - Run validation and thesis usability
This layer is critical.
A run that “finished” is not automatically a run that should count as thesis evidence.

### Why validation exists
Thesis evaluation should not depend on manually remembering which runs were broken, incomplete, or scientifically thin.

### Validation output
Each run now gets a `run_validation.json` file with:
- `validation_status`
- `run_usable_for_thesis`
- `exclusion_reasons`
- `warnings`
- artifact counts
- process-quality indicators

### Structural exclusion examples
Runs can be marked unusable if they have problems like:
- missing required files,
- no utterances,
- incomplete accusation phase,
- non-finished run status,
- missing metrics.

### Quality warnings
Runs can also remain usable while being flagged for interpretation risks such as:
- low suspect question coverage,
- murderer never directly questioned,
- murderer never directly challenged,
- weak accusation reasoning,
- missing clue-reveal events.

### Why this is scientifically useful
This creates a difference between:
- **structural validity** — can the run count at all?
- **interpretive strength** — how much confidence should we have in the accusation result?

That distinction is extremely important for the thesis.

---

## Layer 5 - Aggregation and progress tracking
Once runs exist, the repo should automatically answer two questions:

1. **What do the current results say?**
2. **Do we have enough usable runs to interpret them yet?**

### Condition aggregation
Per condition, the repo aggregates:
- total runs,
- thesis-usable runs,
- warning counts,
- RQ1 strategy rates,
- RQ2 attention metrics,
- RQ3 accusation metrics.

### Cross-condition aggregation
At experiment level, the repo builds:
- `condition_summary.csv`
- `condition_summary.json`
- `condition_report.json`
- `condition_report.md`
- `condition_pairwise_differences.csv`
- `condition_chance_baseline.csv`

### Progress tracking
The workflow now also writes:
- `progress_report.json`
- `progress_report.md`
- `qualitative_samples.json`
- `qualitative_samples.md`

These reports tell you:
- planned total runs,
- completed runs,
- usable runs,
- invalid runs,
- usable runs per condition,
- how far each condition is from pilot/interim/final thresholds,
- the current workflow stage.

### Workflow stages
The experiment is now interpreted through threshold-aware states:
- `planned`
- `collecting`
- `pilot_ready`
- `interim_analysis_ready`
- `final_analysis_ready`
- `complete_but_underpowered`

This gives the project a principled answer to:

> “What should happen depending on how many runs are inserted?”

### Threshold logic
By default:
- **Pilot-ready** = 3 usable runs per condition
- **Interim-analysis-ready** = 10 usable runs per condition
- **Final-analysis-ready** = 20 usable runs per condition

These are configurable in the experiment config.

---

## Layer 6 - Thesis-ready export
The thesis is not written from raw event logs.
It is written from analysis-ready, flat, cross-run data.

### Flat dataset exports
The repo builds a `thesis_dataset/` folder containing:
- `conditions.csv`
- `runs.csv`
- `utterances.csv`
- `interactions.csv`
- `accusations.csv`
- `deception_labels.csv`
- `events.csv`
- `dataset_manifest.json`

### Why this matters
This export is the bridge from engineering to thesis analysis.
It makes it possible to:
- compare conditions systematically,
- inspect specific deceptive utterances,
- quantify attention patterns,
- compute accusation outcomes,
- and later move into notebooks/statistical analysis without re-parsing game logs.

---

## How the research questions map onto the workflow

### RQ1 - Deception strategies
Depends on:
- murderer utterance extraction,
- deception labeling,
- preserved transcript context,
- condition-level strategy aggregation.

Relevant outputs:
- `deception_labels.csv`
- `deception_summary.json`
- `runs.csv`
- `utterances.csv`

### RQ2 - Attention distribution
Depends on:
- direct-address detection,
- mention extraction,
- question/follow-up/pressure signals,
- per-target aggregation.

Relevant outputs:
- `interactions.csv`
- `agent_attention_summary.csv`
- `attention_summary.json`
- condition comparison outputs

### RQ3 - Accusation outcomes
Depends on:
- accusation records,
- vote share,
- group solve rate,
- chance baseline checks,
- usable-run counts.

Relevant outputs:
- `accusations.csv`
- `metrics.json`
- `condition_report.*`
- `condition_chance_baseline.csv`

---

## Why the workflow makes sense at this stage of the project
The project already has a good experimental baseline.
That means the next bottleneck is not “can the agents play the game?”
It is:

- are the runs reusable,
- are the comparisons reproducible,
- and are the results defensible enough for a master thesis?

The workflow now makes sense because it directly targets that gap.

### Before
The repo already had:
- batch runs,
- condition comparison,
- thesis dataset export.

### Now
The workflow additionally clarifies:
- when an experiment is only exploratory vs interpretation-ready,
- which runs should be counted,
- which runs are usable but scientifically weak,
- how the project should react as more runs are inserted,
- how the whole system connects from runtime to thesis evidence.

---

## Recommended operating procedure

### 1. Plan the batch
Use:
- `--validate-only` to confirm the config is coherent
- `--plan-only` to inspect the full condition/replicate expansion

### 2. Run the experiment
Execute the full condition matrix with the intended replicate count.

### 3. Inspect workflow progress
After runs are added, inspect:
- `progress_report.md`
- `batch_status.json`
- per-condition `validation_summary.json`

### 4. Inspect result quality before interpretation
Do not jump straight to solve rates.
First inspect whether:
- runs are structurally valid,
- accusations completed,
- the murderer was actually questioned,
- coverage is acceptable,
- clue progression was logged.

### 5. Compare conditions
Once pilot/interim/final thresholds are reached, review:
- `condition_report.md`
- `condition_pairwise_differences.csv`
- `condition_chance_baseline.csv`

### 6. Export for thesis analysis and writing
Use the rebuilt `thesis_dataset/` as the stable input for further analysis and chapter writing.

---

## Output tree overview

```text
outputs/<experiment>/
  experiment_plan.json
  batch_status.json
  condition_summary.csv
  condition_summary.json
  condition_report.json
  condition_report.md
  condition_pairwise_differences.csv
  condition_chance_baseline.csv
  progress_report.json
  progress_report.md
  qualitative_samples.json
  qualitative_samples.md
  thesis_dataset/
    conditions.csv
    runs.csv
    utterances.csv
    interactions.csv
    accusations.csv
    deception_labels.csv
    events.csv
    dataset_manifest.json
  conditions/<condition>/
    condition_config.json
    batch_status.json
    aggregate_summary.json
    validation_summary.json
    runs/<run_id>/
      run_manifest.json
      run_validation.json
      events.jsonl
      utterances.csv
      interactions.csv
      accusations.csv
      metrics.json
      attention_summary.json
      deception_labels.csv
```

---

## Current strengths
- reproducible condition planning
- replicate seeding support
- machine-readable per-run artifacts
- cross-condition comparison support
- flat thesis dataset export
- chance-baseline checks for RQ3
- attention metrics for RQ2
- first-pass deception labeling for RQ1
- new run validation and progress reporting

---

## Current limitations
- progression is still mostly round-budget-based rather than fully evidence-gated
- RQ1 labeling is still heuristic and not yet a final validated judge
- suspicion / contradiction tracking is still weaker than the eventual thesis ideal
- accusation quality is logged, but not yet enforced through a strict evidence-backed output schema

These are acceptable as current limitations, as long as they are documented and the workflow makes them visible.

---

## Most important next upgrade after this workflow spine
The next major scientific improvement should be:

### Evidence-gated progression
Move from simple round-budget progression toward stages where advancement depends on:
- clue exposure,
- reaction by multiple agents,
- suspect coverage,
- contradiction or suspicion integration,
- and direct pressure on the leading suspect(s).

That upgrade would make accusation outcomes much more interpretable for the thesis.

---

## Guiding principle for all future work
Optimize for **thesis evidence**, not just gameplay.

A useful question for every change is:

> Does this make the final accusation result more reproducible, more interpretable, or more defensible as research evidence?

If yes, it probably belongs in the project.

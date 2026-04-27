# murder-mystery-agents

AI agents play a murder mystery together — and this repo now treats that simulation as a **thesis experiment workflow**, not just a demo.

For the full project rationale and end-to-end thesis workflow, see `PROJECT.md`.
For the concrete final experiment design, see `THESIS_RUN_PLAN.md` and `configs/thesis-final-matrix.yaml`.

## Scenarios

### Default
- Business of Murder: `scenarios/business-of-murder/`

## Run an interactive discussion

Default scenario:

```bash
python3 run_discussion.py --model local --conversations-per-round 20 --max-rounds 6
```

NVIDIA / Kimi option:

```bash
export NVIDIA_API_KEY=your_token_here
python3 run_discussion.py \
  --model nvidia \
  --model-name moonshotai/kimi-k2.5 \
  --conversations-per-round 20 \
  --max-rounds 6
```

Optional reasoning-enabled NVIDIA run:

```bash
export NVIDIA_API_KEY=your_token_here
python3 run_discussion.py \
  --model nvidia \
  --model-name moonshotai/kimi-k2.5 \
  --enable-thinking
```

Business of Murder scenario:

```bash
python3 run_discussion.py \
  --model local \
  --conversations-per-round 20 \
  --max-rounds 6 \
  --scenario-id business-of-murder-v1 \
  --scenario-path scenarios/business-of-murder/scenario.json \
  --roles-dir scenarios/business-of-murder/roles \
  --clues-dir scenarios/business-of-murder/clues
```

## Backends

The project now supports these LLM backends:

- `local` — OpenAI-compatible local endpoint (`LLM_API_URL` / `LLM_MODEL`)
- `gpt` — OpenAI API
- `nvidia` — NVIDIA-hosted OpenAI-compatible API (for example `moonshotai/kimi-k2.5`)
- `ollama` — local Ollama models

For NVIDIA runs:

- default base URL: `https://integrate.api.nvidia.com/v1`
- default model: `moonshotai/kimi-k2.5`
- GPT prompts for `OPENAI_API_KEY`; NVIDIA prompts for `NVIDIA_API_KEY`
- `--enable-thinking` sends `chat_template_kwargs={"thinking": true}`

## Run pilot experiments

Business of Murder pilot:

```bash
python3 experiments/runner.py --config configs/business-of-murder-pilot.yaml --replicates 3
```

## Thesis workflow

The experiment runner now supports a workflow with five distinct steps:

1. **Plan** the experiment
2. **Run** the condition matrix
3. **Validate** per-run outputs for thesis usability
4. **Aggregate** condition and experiment summaries
5. **Export** a flat thesis dataset

### Workflow thresholds

Each config can define the thresholds that determine when an experiment is:
- `pilot_ready`
- `interim_analysis_ready`
- `final_analysis_ready`

Defaults:
- `pilot_ready_runs_per_condition: 3`
- `interim_ready_runs_per_condition: 10`
- `final_ready_runs_per_condition: 20`

Recommended final-thesis matrix (`configs/thesis-final-matrix.yaml`):
- pilot-ready: 5 usable runs / condition
- interim-analysis-ready: 12 usable runs / condition
- final-analysis-ready: 24 usable runs / condition

These thresholds are used to build `progress_report.json` and `progress_report.md`.

## Run a thesis condition matrix

A single YAML file can define either:
- a base configuration plus multiple explicit named conditions, or
- a generated condition matrix via Cartesian-product factor levels.

### Validate a config without spending tokens

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.generated.example.yaml --validate-only
```

### Expand the full planned condition/run matrix

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.generated.example.yaml --plan-only
```

### Execute explicit conditions

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.example.yaml --replicates 3
```

### Execute generated matrix conditions

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.generated.example.yaml --replicates 3
```

### Execute the final thesis matrix

```bash
./.venv/bin/python experiments/runner.py --config configs/thesis-final-matrix.yaml
```

### Final thesis matrix

For the focused, thesis-ready comparison set, use:

```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
```

See `docs/THESIS_RUN_PLAN.md` for the rationale and staged execution plan.

## Interactive run artifacts

Ad-hoc `run_discussion.py` runs now write thought exports under:

- `outputs/interactive/agent_thoughts_<timestamp>.csv`

This keeps the repo root clear and avoids mixing stale scratch exports into interpretation.

## Output structure

Running a batch writes:

- experiment-level planning and status:
  - `outputs/<experiment>/experiment_plan.json`
  - `outputs/<experiment>/batch_status.json`
  - `outputs/<experiment>/progress_report.json`
  - `outputs/<experiment>/progress_report.md`
  - `outputs/<experiment>/qualitative_samples.json`
  - `outputs/<experiment>/qualitative_samples.md`
- per-condition files:
  - `outputs/<experiment>/conditions/<condition>/condition_config.json`
  - `outputs/<experiment>/conditions/<condition>/batch_status.json`
  - `outputs/<experiment>/conditions/<condition>/aggregate_summary.json`
  - `outputs/<experiment>/conditions/<condition>/validation_summary.json`
- per-run logs under:
  - `outputs/<experiment>/conditions/<condition>/runs/<run_id>/`

### Per-run thesis artifacts

Each successful run is expected to contain:
- `run_manifest.json`
- `run_validation.json`
- `events.jsonl`
- `utterances.csv`
- `interactions.csv`
- `accusations.csv`
- `deception_labels.csv`
- `attention_summary.json`
- `metrics.json`

### Cross-condition outputs

The runner writes:
- `outputs/<experiment>/condition_summary.csv`
- `outputs/<experiment>/condition_summary.json`
- `outputs/<experiment>/condition_report.json`
- `outputs/<experiment>/condition_report.md`
- `outputs/<experiment>/condition_pairwise_differences.csv`
- `outputs/<experiment>/condition_chance_baseline.csv`

### Thesis-ready flat export

The runner also rebuilds:
- `outputs/<experiment>/thesis_dataset/conditions.csv`
- `outputs/<experiment>/thesis_dataset/runs.csv`
- `outputs/<experiment>/thesis_dataset/utterances.csv`
- `outputs/<experiment>/thesis_dataset/interactions.csv`
- `outputs/<experiment>/thesis_dataset/accusations.csv`
- `outputs/<experiment>/thesis_dataset/deception_labels.csv`
- `outputs/<experiment>/thesis_dataset/events.csv`
- `outputs/<experiment>/thesis_dataset/dataset_manifest.json`

## Rebuild workflow artifacts manually

If runs were added earlier and you want to refresh the workflow reports without rerunning everything:

### Rebuild the progress report

```bash
python3 analysis/workflow.py outputs/thesis-condition-matrix-generated
```

### Rebuild the condition comparison summary

```bash
python3 analysis/compare_conditions.py outputs/thesis-condition-matrix-generated
```

### Rebuild the flat thesis dataset

```bash
python3 analysis/build_thesis_dataset.py outputs/thesis-condition-matrix-generated
```

### Rebuild qualitative sample lists for thesis writing

```bash
python3 analysis/export_qualitative_samples.py outputs/thesis-condition-matrix-generated
```

## What validation now checks

Per run, the workflow now distinguishes between:
- **structural validity** — is the run usable for the thesis at all?
- **quality warnings** — should the run be interpreted cautiously?

Examples of structural failures:
- missing required files
- run did not finish
- no utterances
- accusation phase incomplete
- missing metrics

Examples of warnings:
- low suspect question coverage
- murderer never directly questioned
- murderer never directly challenged
- accusation reasoning often not evidence-backed
- missing clue reveal events

## Deception labeling (RQ1)

Batch analysis emits a first-pass deception labeling pass for murderer utterances.

- `deception_labeling_enabled: true|false`
- `deception_labeling_mode: heuristic|off`

Current heuristic labels are:
- `direct_denial`
- `alibi_claim`
- `deflection`
- `evasion`
- `uncertainty_seeding`
- `selective_disclosure`
- `accusation_redirection`

These labels are intended as thesis-oriented candidate labels and evidence extraction scaffolding, not a final validated judge.

## Reproducibility notes

- Set `seed` in a run or experiment config to enable deterministic replicate seeding.
- Replicate `r001` uses `seed`, `r002` uses `seed + 1`, etc.
- The resolved seed, base seed, and config fingerprint are written into each `run_manifest.json`.
- Each condition folder also stores `condition_config.json` with the fully resolved settings.
- Progress thresholds are written into manifests/config snapshots so experiment readiness can be interpreted later.

## Current workflow logic in one sentence

**plan conditions → run replicates → validate run usability → refresh summaries + progress → rebuild thesis dataset → interpret only when the usable-run thresholds are strong enough**

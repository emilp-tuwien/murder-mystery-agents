# murder-mystery-agents

AI agents play a murder mystery together.

## Scenarios

### Default
- Killingsworth farm scenario using `agents/roles` and `clues/`

### Alternate
- Business-school scenario: `scenarios/business-of-murder/`

## Run an interactive discussion

Default scenario:

```bash
python3 run_discussion.py --model local --conversations-per-round 20 --max-rounds 6
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

## Run pilot experiments

Farm baseline:

```bash
python3 experiments/runner.py --config configs/pilot.yaml --replicates 3
```

Business of Murder:

```bash
python3 experiments/runner.py --config configs/business-of-murder-pilot.yaml --replicates 3
```

## Run a thesis condition matrix

A single YAML file can now define either:
- a base configuration plus multiple explicit named conditions, or
- a generated condition matrix via Cartesian-product factor levels.

Explicit conditions:

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.example.yaml --replicates 3
```

Generated matrix conditions:

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.generated.example.yaml --replicates 3
```

This writes:
- per-run logs under `outputs/<experiment>/conditions/<condition>/runs/`
- resolved per-condition config snapshots at `outputs/<experiment>/conditions/<condition>/condition_config.json`
- per-condition summaries in each condition folder
- per-run RQ1 artifacts:
  - `deception_labels.csv`
  - `deception_summary.json`
- per-run RQ2 artifacts:
  - `interactions.csv`
  - `agent_attention_summary.csv`
  - `attention_summary.json`
- cross-condition summaries at:
  - `outputs/<experiment>/condition_summary.csv`
  - `outputs/<experiment>/condition_summary.json`
  - `outputs/<experiment>/condition_report.json`
  - `outputs/<experiment>/condition_report.md`
  - `outputs/<experiment>/condition_pairwise_differences.csv`
  - `outputs/<experiment>/condition_chance_baseline.csv`
  - `outputs/<experiment>/experiment_plan.json`

The comparison step now also produces thesis-oriented condition rankings, pairwise deltas, and chance-baseline checks for RQ3 (including 95% Wilson intervals and a simple two-proportion z-test scaffold).

For thesis-scale condition management, experiment plans now support a `matrix:` section that automatically expands factor combinations into reproducible named conditions like `baseline__three-stage-v1`.

For RQ2, the analysis pipeline now extracts target-level interaction rows and derives pressure-oriented attention signals such as direct questions, follow-up questions, and justification requests, plus simple concentration measures (entropy/Gini) over who receives scrutiny.

You can also regenerate the condition comparison summary directly:

```bash
python3 analysis/compare_conditions.py outputs/thesis-condition-matrix
```

## Deception labeling (RQ1)

Batch analysis now emits a first-pass deception labeling pass for murderer utterances.

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

These are intended as thesis-oriented candidate labels and evidence extraction scaffolding, not a final validated judge. They make it possible to batch-screen runs now and later replace the heuristic stage with an LLM-as-a-judge rubric without changing the output layout.

See `docs/BUSINESS_OF_MURDER.md` for scenario notes and comparison guidance.


## Reproducibility notes

- Set `seed` in a run or experiment config to enable deterministic replicate seeding.
- Replicate `r001` uses `seed`, `r002` uses `seed + 1`, etc.
- The resolved seed, base seed, and config fingerprint are written into each `run_manifest.json`.
- Each condition folder now also stores `condition_config.json` with the fully resolved condition settings.

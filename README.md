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

A single YAML file can now define a base configuration plus multiple named conditions.

```bash
python3 experiments/runner.py --config configs/thesis-condition-matrix.example.yaml --replicates 3
```

This writes:
- per-run logs under `outputs/<experiment>/conditions/<condition>/runs/`
- per-condition summaries in each condition folder
- per-run RQ1 artifacts:
  - `deception_labels.csv`
  - `deception_summary.json`
- cross-condition summaries at:
  - `outputs/<experiment>/condition_summary.csv`
  - `outputs/<experiment>/condition_summary.json`
  - `outputs/<experiment>/experiment_plan.json`

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

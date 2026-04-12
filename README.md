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
- cross-condition summaries at:
  - `outputs/<experiment>/condition_summary.csv`
  - `outputs/<experiment>/condition_summary.json`
  - `outputs/<experiment>/experiment_plan.json`

You can also regenerate the condition comparison summary directly:

```bash
python3 analysis/compare_conditions.py outputs/thesis-condition-matrix
```

See `docs/BUSINESS_OF_MURDER.md` for scenario notes and comparison guidance.

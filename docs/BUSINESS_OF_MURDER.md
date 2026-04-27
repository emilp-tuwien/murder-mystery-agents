# Business of Murder scenario

**Business of Murder** is the primary and only active thesis scenario in this repo.

## Scenario assets

- Scenario manifest: `scenarios/business-of-murder/scenario.json`
- Roles: `scenarios/business-of-murder/roles/`
- Clues: `scenarios/business-of-murder/clues/`
- Pilot config: `configs/business-of-murder-pilot.yaml`
- Final thesis matrix: `configs/thesis-final-matrix.yaml`

## Scenario-specific runtime vocabulary

Business-of-Murder-specific evidence vocabulary now lives in the scenario manifest itself:

- memory categorization tags/patterns
- clue-keyword stopwords
- evidence/pressure/synthesis gate patterns

That keeps old scenario vocabulary out of the active prompt/memory pipeline.

## Run a single discussion

```bash
python3 run_discussion.py --model local --conversations-per-round 20 --max-rounds 6
```

The default runtime already resolves to Business of Murder.

## Run the pilot batch

```bash
python3 experiments/runner.py --config configs/business-of-murder-pilot.yaml --replicates 3
```

## Run the final thesis matrix

```bash
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml --plan-only
python3 experiments/runner.py --config configs/thesis-final-matrix.yaml
```

## Interactive artifacts

Interactive thought exports are written to:

- `outputs/interactive/agent_thoughts_<timestamp>.csv`

They are intentionally kept out of the repo root so they do not get mistaken for thesis artifacts.

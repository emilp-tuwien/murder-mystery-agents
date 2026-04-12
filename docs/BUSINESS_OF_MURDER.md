# Business of Murder scenario

This repo now includes a second selectable scenario adapted from the uploaded `business 3` files.

## Scenario assets

- Scenario manifest: `scenarios/business-of-murder/scenario.json`
- Roles: `scenarios/business-of-murder/roles/`
- Clues: `scenarios/business-of-murder/clues/`
- Pilot config: `configs/business-of-murder-pilot.yaml`

## Mapping from source material

- `*.invite.txt` → round 1 role descriptions
- `*.start.txt` → round 2 role descriptions
- `Clue1.txt` to `Clue4.txt` → scenario clues 1 to 4
- Added `clue5.txt` → synthesized timeline clue so the framework's 6-round flow still lands with a final evidence push before accusations
- Added `confession.txt` per role so end-of-game truth is explicit and comparable to the farm scenario

## Run a single pilot discussion

```bash
python3 run_discussion.py --model local --conversations-per-round 20 --max-rounds 6
```

That command still uses the default farm setup unless you edit code or load a custom config elsewhere.

For the business scenario, use the experiment runner:

```bash
python3 experiments/runner.py --config configs/business-of-murder-pilot.yaml --replicates 1
```

## Compare against the original farm scenario

Run one or more replicates for each:

```bash
python3 experiments/runner.py --config configs/pilot.yaml --replicates 3
python3 experiments/runner.py --config configs/business-of-murder-pilot.yaml --replicates 3
```

Then compare these outputs:

- `outputs/pilot/aggregate_summary.json`
- `outputs/business-of-murder-pilot/aggregate_summary.json`
- per-run `metrics.json`, `utterances.csv`, `accusations.csv`, and `agent_metrics.csv`

Recommended first comparison dimensions:

- group solve rate
- murderer vote share
- murderer attention received
- total utterances / total turns
- qualitative coherence from transcripts and confessions

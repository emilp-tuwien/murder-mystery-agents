Workflow progress for thesis-pilot

- Workflow stage: complete_but_underpowered
- Planned total runs: 10
- Completed runs: 15
- Usable runs: 5
- Invalid runs: 10
- Conditions: 2
- Minimum usable runs per condition: 0

Thresholds:
- Pilot-ready: 5 usable runs / condition
- Interim-analysis-ready: 12 usable runs / condition
- Final-analysis-ready: 24 usable runs / condition

Per-condition status:
- passive-concealment: planned=5, completed=11, usable=5, invalid=6, warnings=6
  remaining_to_pilot=0, remaining_to_interim=7, remaining_to_final=19
- active-deception: planned=5, completed=4, usable=0, invalid=4, warnings=4
  remaining_to_pilot=5, remaining_to_interim=12, remaining_to_final=24

Recommendations:
- Inspect run_validation.json files for invalid runs before interpreting condition-level metrics.
- The currently planned batch finished below pilot-ready power; either raise replicates or reduce the condition matrix before drawing conclusions.

Workflow progress for business-of-murder-pilot

- Workflow stage: collecting
- Planned total runs: 5
- Completed runs: 1
- Usable runs: 0
- Invalid runs: 1
- Conditions: 1
- Minimum usable runs per condition: 0

Thresholds:
- Pilot-ready: 3 usable runs / condition
- Interim-analysis-ready: 10 usable runs / condition
- Final-analysis-ready: 20 usable runs / condition

Per-condition status:
- business-of-murder-pilot: planned=5, completed=1, usable=0, invalid=1, warnings=1
  remaining_to_pilot=3, remaining_to_interim=10, remaining_to_final=20

Recommendations:
- Inspect run_validation.json files for invalid runs before interpreting condition-level metrics.
- Keep collecting until each condition has at least 3 usable runs for pilot comparison.

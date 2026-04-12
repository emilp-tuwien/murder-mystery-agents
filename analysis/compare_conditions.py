from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import aggregate_experiment_conditions


def main():
    parser = argparse.ArgumentParser(description="Summarize condition-level experiment outputs.")
    parser.add_argument("experiment_dir", help="Path to an experiment output directory.")
    args = parser.parse_args()

    summary = aggregate_experiment_conditions(Path(args.experiment_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

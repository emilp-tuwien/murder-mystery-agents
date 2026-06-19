from __future__ import annotations

"""Blinded innocent-control pass for the RQ1 deception judge.

Runs the same LLM judge and taxonomy used for the murderer over EVERY suspect's
utterances with role identity hidden (no ground truth, no cover story, no
information gate, no "this speaker is the murderer"). This yields:

  - an innocent false-positive estimate: how often the judge labels utterances
    by suspects who are NOT the murderer (upper bound on the FPR — innocents
    have their own secrets and can genuinely deceive),
  - a blinded murderer rate for an apples-to-apples sensitivity comparison,
  - per-strategy rates for innocents vs the murderer.

Usage:

    # One run, all utterances of every suspect
    python analysis/judge_innocent_control.py --run-dir outputs/.../runs/RUN_ID

    # Whole experiment, capped at 10 judged utterances per suspect per run
    python analysis/judge_innocent_control.py \\
        --experiment-dir outputs/thesis-final-matrix --max-per-agent 10

Outputs per run dir: blinded_judge_labels.csv / .jsonl, blinded_judge_summary.json.
With --experiment-dir, also writes <experiment>/blinded_judge_aggregate.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.deception_judge import judge_utterances_blinded
from schemas.deception import CANONICAL_STRATEGY_LABELS


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_int(value: Any) -> Any:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def _load_utterances(run_dir: Path) -> List[Dict[str, Any]]:
    """Load utterances.csv with turn/round coerced back to ints."""
    rows = _read_csv(run_dir / "utterances.csv")
    for row in rows:
        row["turn"] = _coerce_int(row.get("turn"))
        row["round"] = _coerce_int(row.get("round"))
    return rows


def _find_run_dirs(root: Path) -> List[Path]:
    def _looks_like_run(p: Path) -> bool:
        return p.is_dir() and (p / "run_manifest.json").exists()

    if _looks_like_run(root):
        return [root]

    candidates: List[Path] = []
    conditions_dir = root / "conditions"
    if conditions_dir.exists():
        for cond in sorted(conditions_dir.iterdir()):
            if not cond.is_dir():
                continue
            runs = cond / "runs"
            if runs.exists():
                candidates.extend(p for p in sorted(runs.iterdir()) if _looks_like_run(p))
            else:
                candidates.extend(p for p in sorted(cond.iterdir()) if _looks_like_run(p))
    else:
        runs_dir = root / "runs"
        if runs_dir.exists():
            candidates.extend(p for p in sorted(runs_dir.iterdir()) if _looks_like_run(p))
    return candidates


def judge_run_blinded(
    run_dir: Path,
    max_per_agent: Optional[int],
    seed: int,
    include_murderer: bool,
) -> Optional[Dict[str, Any]]:
    manifest = _read_json(run_dir / "run_manifest.json")
    utterances = _load_utterances(run_dir)
    agent_names = list(manifest.get("agent_names") or [])
    murderer_name = manifest.get("murderer_name")

    if not utterances or not agent_names:
        print(f"  Skipping {run_dir.name}: missing utterances.csv or agent_names.")
        return None

    label_rows, summary = judge_utterances_blinded(
        utterances,
        agent_names,
        murderer_name,
        manifest,
        max_per_agent=max_per_agent,
        seed=seed,
        include_murderer=include_murderer,
    )

    _write_csv(run_dir / "blinded_judge_labels.csv", label_rows)
    _write_jsonl(run_dir / "blinded_judge_labels.jsonl", label_rows)
    with (run_dir / "blinded_judge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"  {run_dir.name}: judged={summary['judged_utterances_total']} "
        f"innocent_rate={summary['innocent_deceptive_rate']:.3f} "
        f"murderer_blinded_rate={summary['murderer_blinded_deceptive_rate']:.3f} "
        f"gap={summary['blinded_rate_gap_murderer_minus_innocent']:+.3f}"
    )
    return summary


def aggregate_blinded_summaries(experiment_dir: Path, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(summaries)
    if not n:
        return {"experiment_dir": str(experiment_dir), "total_runs": 0}

    def _mean(key: str) -> float:
        return sum(float(s.get(key) or 0.0) for s in summaries) / n

    strategy_means: Dict[str, Dict[str, float]] = {"innocent": {}, "murderer": {}}
    for group, rates_key in (("innocent", "innocent_rates_by_strategy"), ("murderer", "murderer_rates_by_strategy")):
        for label in CANONICAL_STRATEGY_LABELS:
            values = [float((s.get(rates_key) or {}).get(label, 0.0)) for s in summaries]
            if any(values):
                strategy_means[group][label] = sum(values) / n

    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for s in summaries:
        by_condition.setdefault(s.get("condition_name") or "default", []).append(s)

    aggregate = {
        "experiment_dir": str(experiment_dir),
        "total_runs": n,
        "mean_innocent_deceptive_rate": _mean("innocent_deceptive_rate"),
        "mean_murderer_blinded_deceptive_rate": _mean("murderer_blinded_deceptive_rate"),
        "mean_blinded_rate_gap": _mean("blinded_rate_gap_murderer_minus_innocent"),
        "total_innocent_utterances_judged": sum(int(s.get("innocent_utterances_judged") or 0) for s in summaries),
        "total_innocent_deceptive_utterances": sum(int(s.get("innocent_deceptive_utterances") or 0) for s in summaries),
        "total_murderer_utterances_judged": sum(int(s.get("murderer_utterances_judged") or 0) for s in summaries),
        "total_murderer_deceptive_utterances": sum(int(s.get("murderer_deceptive_utterances") or 0) for s in summaries),
        "mean_strategy_rates": strategy_means,
        "by_condition": {
            condition: {
                "total_runs": len(rows),
                "mean_innocent_deceptive_rate": sum(float(r.get("innocent_deceptive_rate") or 0.0) for r in rows) / len(rows),
                "mean_murderer_blinded_deceptive_rate": sum(float(r.get("murderer_blinded_deceptive_rate") or 0.0) for r in rows) / len(rows),
                "mean_blinded_rate_gap": sum(float(r.get("blinded_rate_gap_murderer_minus_innocent") or 0.0) for r in rows) / len(rows),
            }
            for condition, rows in sorted(by_condition.items())
        },
        "run_ids": [s.get("run_id") for s in summaries],
    }
    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description="Run the blinded innocent-control deception judge over run outputs."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="Path to a single run directory.")
    src.add_argument("--experiment-dir", help="Path to an experiment output directory (all runs judged).")
    parser.add_argument(
        "--max-per-agent", type=int, default=None,
        help="Cap judged utterances per suspect per run (uniform random sample). Default: all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument(
        "--innocents-only", action="store_true",
        help="Skip the murderer's utterances (default: include them blinded for sensitivity comparison).",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
        experiment_dir = None
    else:
        experiment_dir = Path(args.experiment_dir)
        run_dirs = _find_run_dirs(experiment_dir)
        print(f"Found {len(run_dirs)} run directories under {experiment_dir}")

    summaries: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            summary = judge_run_blinded(
                run_dir,
                max_per_agent=args.max_per_agent,
                seed=args.seed,
                include_murderer=not args.innocents_only,
            )
            if summary:
                summaries.append(summary)
        except Exception as exc:
            print(f"  Warning: blinded judging failed for {run_dir}: {exc}")

    if experiment_dir is not None:
        aggregate = aggregate_blinded_summaries(experiment_dir, summaries)
        with (experiment_dir / "blinded_judge_aggregate.json").open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)
        print(json.dumps(aggregate, indent=2, sort_keys=True))
    elif summaries:
        print(json.dumps(summaries[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Export a human-auditable validation sample of deception judgments.

Usage examples:

    # Sample from a single run directory
    python analysis/export_deception_validation_sample.py \\
        --run-dir outputs/thesis-final-matrix/conditions/active-deception__round-budget/runs/RUN_ID \\
        --output outputs/validation_sample.csv --sample-size 30 --seed 42

    # Sample from a whole experiment (all conditions, all runs)
    python analysis/export_deception_validation_sample.py \\
        --experiment-dir outputs/thesis-final-matrix \\
        --output outputs/validation_sample.csv --sample-size 60 --seed 42

The exported CSV includes both deceptive and non-deceptive murderer utterances
with full context so a human annotator can verify judge accuracy.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _read_csv_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(utterances: List[Dict[str, Any]], target_turn: Any, before: int = 4, after: int = 2) -> str:
    """Build a readable context string around a target turn."""
    target_idx = None
    for i, u in enumerate(utterances):
        if str(u.get("turn", "")) == str(target_turn):
            target_idx = i
            break
    if target_idx is None:
        return ""

    before_slice = utterances[max(0, target_idx - before) : target_idx]
    after_slice = utterances[target_idx + 1 : target_idx + 1 + after]

    lines = []
    for u in before_slice:
        lines.append(f"[{u.get('turn', '?')}] {u.get('speaker', '?')}: {u.get('text', '')}")
    lines.append(f">>> [TARGET {target_turn}] {utterances[target_idx].get('speaker', '?')}: {utterances[target_idx].get('text', '')}")
    for u in after_slice:
        lines.append(f"[{u.get('turn', '?')}] {u.get('speaker', '?')}: {u.get('text', '')}")
    return " | ".join(lines)


# ---------------------------------------------------------------------------
# Per-run sampling
# ---------------------------------------------------------------------------

def _load_run_labels(run_dir: Path) -> List[Dict[str, Any]]:
    """Load deception label rows from a run directory.

    Prefers deception_labels.jsonl (richer, includes non-deceptive rows).
    Falls back to deception_labels.csv.
    """
    jsonl_rows = _read_jsonl(run_dir / "deception_labels.jsonl")
    if jsonl_rows:
        return jsonl_rows
    return _read_csv_file(run_dir / "deception_labels.csv")


def _sample_from_run(
    run_dir: Path,
    n_deceptive: int,
    n_non_deceptive: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Return a balanced sample of deceptive + non-deceptive rows from a single run."""
    manifest = _read_json(run_dir / "run_manifest.json")
    run_id = manifest.get("run_id", run_dir.name)
    condition_name = manifest.get("condition_name", "")
    murderer_behavior_mode = manifest.get("murderer_behavior_mode", "")
    judge_method = manifest.get("deception_labeling_mode", "heuristic")

    utterances = _read_csv_file(run_dir / "utterances.csv")
    label_rows = _load_run_labels(run_dir)

    deceptive_rows = [r for r in label_rows if str(r.get("is_deceptive_instance", "")).lower() in {"true", "1", "yes"}]
    non_deceptive_rows = [r for r in label_rows if str(r.get("is_deceptive_instance", "")).lower() in {"false", "0", "no"}]

    sample = (
        rng.sample(deceptive_rows, min(n_deceptive, len(deceptive_rows)))
        + rng.sample(non_deceptive_rows, min(n_non_deceptive, len(non_deceptive_rows)))
    )

    output_rows = []
    for row in sample:
        target_turn = row.get("turn")
        context = _build_context(utterances, target_turn)
        output_rows.append({
            "run_id": row.get("run_id", run_id),
            "condition_name": row.get("condition_name", condition_name),
            "murderer_behavior_mode": row.get("murderer_behavior_mode", murderer_behavior_mode),
            "progression_policy": row.get("progression_policy", ""),
            "round": row.get("round"),
            "turn": target_turn,
            "speaker": row.get("speaker"),
            "target_utterance_text": row.get("target_utterance_text", ""),
            "context_window": context,
            "strategy_label": row.get("strategy_label", ""),
            "is_deceptive_instance": row.get("is_deceptive_instance", ""),
            "confidence": row.get("confidence", ""),
            "evidence_span_text": row.get("evidence_span_text", ""),
            "reasoning": row.get("reasoning", ""),
            "judge_method": row.get("judge_method", judge_method),
            "judge_model": row.get("judge_model", ""),
            # Blank column for human annotator
            "human_label": "",
            "human_notes": "",
        })

    return output_rows


# ---------------------------------------------------------------------------
# Run-dir discovery
# ---------------------------------------------------------------------------

def _find_run_dirs(root: Path) -> List[Path]:
    """Find all valid run directories under root."""
    candidates: List[Path] = []

    def _looks_like_run(p: Path) -> bool:
        return p.is_dir() and (p / "run_manifest.json").exists()

    if _looks_like_run(root):
        return [root]

    # experiment/conditions/<cond>/runs/<run>
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
        # flat experiment/runs/<run>
        runs_dir = root / "runs"
        if runs_dir.exists():
            candidates.extend(p for p in sorted(runs_dir.iterdir()) if _looks_like_run(p))

    return candidates


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_deception_validation_sample(
    run_dirs: List[Path],
    output_path: Path,
    sample_size: int = 60,
    seed: int = 42,
    deceptive_fraction: float = 0.6,
) -> Dict[str, Any]:
    """Sample deception labels for human review and write to a CSV.

    Args:
        run_dirs: list of run directories to sample from
        output_path: path for the output CSV
        sample_size: total rows to export
        seed: RNG seed for reproducibility
        deceptive_fraction: fraction of rows that should be deceptive instances
    """
    rng = random.Random(seed)

    if not run_dirs:
        print("No run directories found.")
        return {"total_rows": 0, "run_dirs_scanned": 0}

    n_deceptive_per_run = max(1, round(sample_size * deceptive_fraction / len(run_dirs)))
    n_non_deceptive_per_run = max(1, round(sample_size * (1 - deceptive_fraction) / len(run_dirs)))

    all_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            rows = _sample_from_run(run_dir, n_deceptive_per_run, n_non_deceptive_per_run, rng)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  Warning: could not sample from {run_dir}: {exc}")

    # Final shuffle and cap
    rng.shuffle(all_rows)
    all_rows = all_rows[:sample_size]

    _write_csv(output_path, all_rows)

    summary = {
        "output_path": str(output_path),
        "total_rows": len(all_rows),
        "run_dirs_scanned": len(run_dirs),
        "sample_size_requested": sample_size,
        "seed": seed,
        "deceptive_fraction_requested": deceptive_fraction,
        "actual_deceptive_count": sum(
            1 for r in all_rows
            if str(r.get("is_deceptive_instance", "")).lower() in {"true", "1", "yes"}
        ),
        "conditions_sampled": sorted({r.get("condition_name", "") for r in all_rows}),
        "judge_methods_found": sorted({r.get("judge_method", "") for r in all_rows}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export a human-auditable validation sample of deception judgments."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="Path to a single run directory.")
    src.add_argument("--experiment-dir", help="Path to an experiment output directory (all runs sampled).")
    parser.add_argument("--output", required=True, help="Path for the output CSV file.")
    parser.add_argument("--sample-size", type=int, default=60, help="Total rows to export (default: 60).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--deceptive-fraction", type=float, default=0.6,
        help="Fraction of sample that should be deceptive instances (default: 0.6).",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = _find_run_dirs(Path(args.experiment_dir))
        print(f"Found {len(run_dirs)} run directories under {args.experiment_dir}")

    export_deception_validation_sample(
        run_dirs=run_dirs,
        output_path=Path(args.output),
        sample_size=args.sample_size,
        seed=args.seed,
        deceptive_fraction=args.deceptive_fraction,
    )


if __name__ == "__main__":
    main()

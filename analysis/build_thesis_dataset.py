from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _read_json(path: Path) -> Dict[str, Any]:
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
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _flatten(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (payload or {}).items():
        flat[f"{prefix}{key}"] = _normalize_scalar(value)
    return flat


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition_dirs(experiment_dir: Path) -> List[Path]:
    conditions_dir = experiment_dir / "conditions"
    if conditions_dir.exists():
        return sorted(path for path in conditions_dir.iterdir() if path.is_dir())
    return [experiment_dir]


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "run_manifest.json").exists()


def _run_dirs(condition_dir: Path) -> List[Path]:
    runs_dir = condition_dir / "runs"
    if runs_dir.exists():
        return sorted(path for path in runs_dir.iterdir() if _looks_like_run_dir(path))

    direct_runs = [path for path in condition_dir.iterdir() if _looks_like_run_dir(path)] if condition_dir.exists() else []
    if direct_runs:
        return sorted(direct_runs)

    if _looks_like_run_dir(condition_dir):
        return [condition_dir]

    return []


def _base_run_context(manifest: Dict[str, Any], metrics: Dict[str, Any], condition_config: Dict[str, Any], run_dir: Path, condition_dir: Path) -> Dict[str, Any]:
    rq1 = metrics.get("rq1", {})
    rq2 = metrics.get("rq2", {})
    rq3 = metrics.get("rq3", {})
    context = {
        "experiment_name": manifest.get("experiment_name"),
        "run_id": manifest.get("run_id", run_dir.name),
        "replicate_id": manifest.get("replicate_id"),
        "condition_name": manifest.get("condition_name") or condition_dir.name,
        "condition_description": manifest.get("condition_description"),
        "run_status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "backend": manifest.get("backend"),
        "model_name": manifest.get("model_name"),
        "temperature": manifest.get("temperature"),
        "seed": manifest.get("seed"),
        "seed_base": manifest.get("seed_base"),
        "seed_strategy": manifest.get("seed_strategy"),
        "scenario_id": manifest.get("scenario_id"),
        "prompt_version": manifest.get("prompt_version"),
        "turn_policy_version": manifest.get("turn_policy_version"),
        "memory_version": manifest.get("memory_version"),
        "deception_labeling_enabled": manifest.get("deception_labeling_enabled"),
        "deception_labeling_mode": manifest.get("deception_labeling_mode"),
        "config_fingerprint": manifest.get("config_fingerprint") or condition_config.get("config_fingerprint"),
        "code_commit": manifest.get("code_commit"),
        "murderer_name": manifest.get("murderer_name") or metrics.get("murderer_name"),
        "total_turns": metrics.get("total_turns"),
        "total_utterances": metrics.get("total_utterances"),
        "group_solved": rq3.get("group_solved"),
        "murderer_vote_share": rq3.get("murderer_vote_share"),
        "random_vote_share_baseline": rq3.get("random_vote_share_baseline"),
        "random_group_solve_rate_baseline": rq3.get("random_group_solve_rate_baseline"),
        "murderer_attention_received": rq2.get("murderer_attention_received"),
        "murderer_followups_received": rq2.get("murderer_followups_received"),
        "murderer_justification_requests_received": rq2.get("murderer_justification_requests_received"),
        "murderer_pressure_signals_received": rq2.get("murderer_pressure_signals_received"),
        "murderer_speaker_share": rq2.get("murderer_speaker_share"),
        "question_target_entropy": rq2.get("question_target_entropy"),
        "pressure_target_gini": rq2.get("pressure_target_gini"),
        "murderer_labeled_utterances": rq1.get("total_labeled_utterances"),
        "murderer_labeled_instances": rq1.get("total_labeled_instances"),
        "total_murderer_utterances": rq1.get("total_murderer_utterances"),
        "run_dir": str(run_dir),
        "condition_dir": str(condition_dir),
    }
    context.update(_flatten("factor__", manifest.get("condition_factors") or condition_config.get("condition_factors") or {}))
    return context


def _collect_run_bundle(condition_dir: Path, run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    manifest = _read_json(run_dir / "run_manifest.json")
    metrics = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    condition_config_path = condition_dir / "condition_config.json"
    condition_config = _read_json(condition_config_path) if condition_config_path.exists() else {}
    attention_summary = _read_json(run_dir / "attention_summary.json") if (run_dir / "attention_summary.json").exists() else {}
    return manifest, metrics, condition_config, attention_summary


def build_thesis_dataset(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    dataset_dir = experiment_path / "thesis_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    condition_rows: List[Dict[str, Any]] = []
    utterance_rows: List[Dict[str, Any]] = []
    interaction_rows: List[Dict[str, Any]] = []
    accusation_rows: List[Dict[str, Any]] = []
    deception_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    seen_conditions = set()

    for condition_dir in _condition_dirs(experiment_path):
        condition_config_path = condition_dir / "condition_config.json"
        condition_config = _read_json(condition_config_path) if condition_config_path.exists() else {}
        condition_name = condition_config.get("condition_name") or (condition_dir.name if condition_dir != experiment_path else "default")

        if condition_name not in seen_conditions:
            seen_conditions.add(condition_name)
            condition_row = {
                "experiment_name": condition_config.get("experiment_name") or experiment_path.name,
                "condition_name": condition_name,
                "condition_description": condition_config.get("condition_description"),
                "replicates": condition_config.get("replicates"),
                "backend": condition_config.get("backend"),
                "model_name": condition_config.get("model_name"),
                "temperature": condition_config.get("temperature"),
                "seed": condition_config.get("seed"),
                "scenario_id": condition_config.get("scenario_id"),
                "prompt_version": condition_config.get("prompt_version"),
                "turn_policy_version": condition_config.get("turn_policy_version"),
                "memory_version": condition_config.get("memory_version"),
                "deception_labeling_mode": condition_config.get("deception_labeling_mode"),
                "config_fingerprint": condition_config.get("config_fingerprint"),
                "condition_dir": str(condition_dir),
            }
            condition_row.update(_flatten("factor__", condition_config.get("condition_factors") or {}))
            condition_rows.append(condition_row)

        for run_dir in _run_dirs(condition_dir):
            manifest, metrics, condition_config, attention_summary = _collect_run_bundle(condition_dir, run_dir)
            context = _base_run_context(manifest, metrics, condition_config, run_dir, condition_dir)
            run_row = dict(context)
            run_row.update(_flatten("attention__", attention_summary))
            run_row.update(
                {
                    "vote_counts": _normalize_scalar(metrics.get("rq3", {}).get("vote_counts", {})),
                    "winning_suspects": _normalize_scalar(metrics.get("rq3", {}).get("winning_suspects", [])),
                    "rq1_strategy_rates": _normalize_scalar(metrics.get("rq1", {}).get("strategy_rates", {})),
                    "rq1_strategy_counts": _normalize_scalar(metrics.get("rq1", {}).get("strategy_counts", {})),
                }
            )
            run_rows.append(run_row)

            for row in _read_csv(run_dir / "utterances.csv"):
                utterance_rows.append({**context, **row})

            for row in _read_csv(run_dir / "interactions.csv"):
                interaction_rows.append({**context, **row})

            for row in _read_csv(run_dir / "accusations.csv"):
                accusation_rows.append({**context, **row})

            for row in _read_csv(run_dir / "deception_labels.csv"):
                deception_rows.append({**context, **row})

            for event in _read_jsonl(run_dir / "events.jsonl"):
                event_rows.append(
                    {
                        **context,
                        "event_index": event.get("index"),
                        "event_timestamp": event.get("timestamp"),
                        "event_type": event.get("type"),
                        "event_payload": _normalize_scalar(event.get("payload", {})),
                    }
                )

    _write_csv(dataset_dir / "runs.csv", run_rows)
    _write_csv(dataset_dir / "conditions.csv", condition_rows)
    _write_csv(dataset_dir / "utterances.csv", utterance_rows)
    _write_csv(dataset_dir / "interactions.csv", interaction_rows)
    _write_csv(dataset_dir / "accusations.csv", accusation_rows)
    _write_csv(dataset_dir / "deception_labels.csv", deception_rows)
    _write_csv(dataset_dir / "events.csv", event_rows)

    manifest = {
        "experiment_dir": str(experiment_path),
        "dataset_dir": str(dataset_dir),
        "total_conditions": len(condition_rows),
        "total_runs": len(run_rows),
        "total_utterances": len(utterance_rows),
        "total_interactions": len(interaction_rows),
        "total_accusations": len(accusation_rows),
        "total_deception_labels": len(deception_rows),
        "total_events": len(event_rows),
        "files": {
            "conditions": "thesis_dataset/conditions.csv",
            "runs": "thesis_dataset/runs.csv",
            "utterances": "thesis_dataset/utterances.csv",
            "interactions": "thesis_dataset/interactions.csv",
            "accusations": "thesis_dataset/accusations.csv",
            "deception_labels": "thesis_dataset/deception_labels.csv",
            "events": "thesis_dataset/events.csv",
        },
    }
    with (dataset_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a flat thesis-ready dataset from murder mystery experiment outputs.")
    parser.add_argument("experiment_dir", help="Path to an experiment output directory.")
    args = parser.parse_args()

    manifest = build_thesis_dataset(Path(args.experiment_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

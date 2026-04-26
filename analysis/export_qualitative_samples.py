from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.build_thesis_dataset import build_thesis_dataset


NUMERIC_KEYS = [
    "murderer_pressure_signals_received",
    "mean_accusation_confidence",
    "structured_accusation_fraction",
    "quality__hard_cap_fallback_transitions",
    "quality__evidence_gate_satisfied_transitions",
    "quality__suspect_question_coverage_fraction",
    "murderer_vote_share",
]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in NUMERIC_KEYS:
            item[key] = _as_float(item.get(key))
        item["group_solved"] = _as_bool(item.get("group_solved"))
        item["run_usable_for_thesis"] = _as_bool(item.get("run_usable_for_thesis"))
        normalized.append(item)
    return normalized


def _pick(rows: List[Dict[str, Any]], predicate, sort_key, limit: int = 5) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if predicate(row)]
    ordered = sorted(filtered, key=sort_key, reverse=True)
    return ordered[:limit]


def _sample_view(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "condition_name": row.get("condition_name"),
        "group_solved": row.get("group_solved"),
        "run_usable_for_thesis": row.get("run_usable_for_thesis"),
        "murderer_vote_share": row.get("murderer_vote_share"),
        "murderer_pressure_signals_received": row.get("murderer_pressure_signals_received"),
        "mean_accusation_confidence": row.get("mean_accusation_confidence"),
        "structured_accusation_fraction": row.get("structured_accusation_fraction"),
        "suspect_question_coverage_fraction": row.get("quality__suspect_question_coverage_fraction"),
        "hard_cap_fallback_transitions": row.get("quality__hard_cap_fallback_transitions"),
        "evidence_gate_satisfied_transitions": row.get("quality__evidence_gate_satisfied_transitions"),
        "run_dir": row.get("run_dir"),
    }


def export_qualitative_samples(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    dataset_manifest = build_thesis_dataset(experiment_path)
    runs_path = experiment_path / "thesis_dataset" / "runs.csv"
    rows = _normalize_rows(_read_csv(runs_path))
    usable_rows = [row for row in rows if row.get("run_usable_for_thesis")]

    payload = {
        "experiment_dir": str(experiment_path),
        "dataset_manifest": dataset_manifest,
        "sample_sets": {
            "high_pressure_solved": [_sample_view(row) for row in _pick(
                usable_rows,
                lambda row: row.get("group_solved"),
                lambda row: (row.get("murderer_pressure_signals_received", 0.0), row.get("structured_accusation_fraction", 0.0)),
            )],
            "high_pressure_unsolved": [_sample_view(row) for row in _pick(
                usable_rows,
                lambda row: not row.get("group_solved"),
                lambda row: (row.get("murderer_pressure_signals_received", 0.0), row.get("structured_accusation_fraction", 0.0)),
            )],
            "strong_structured_accusations": [_sample_view(row) for row in _pick(
                usable_rows,
                lambda row: row.get("structured_accusation_fraction", 0.0) >= 0.75,
                lambda row: (row.get("structured_accusation_fraction", 0.0), row.get("mean_accusation_confidence", 0.0)),
            )],
            "evidence_gated_clean_progression": [_sample_view(row) for row in _pick(
                usable_rows,
                lambda row: row.get("stage_gate_policy") == "evidence_gated" and row.get("quality__hard_cap_fallback_transitions", 0.0) == 0.0,
                lambda row: (row.get("quality__evidence_gate_satisfied_transitions", 0.0), row.get("murderer_pressure_signals_received", 0.0)),
            )],
            "hard_cap_fallback_cases": [_sample_view(row) for row in _pick(
                usable_rows,
                lambda row: row.get("quality__hard_cap_fallback_transitions", 0.0) > 0.0,
                lambda row: (row.get("quality__hard_cap_fallback_transitions", 0.0), row.get("murderer_pressure_signals_received", 0.0)),
            )],
        },
    }

    _write_json(experiment_path / "qualitative_samples.json", payload)

    lines = [
        f"Qualitative samples for {experiment_path.name}",
        "",
    ]
    for name, sample_rows in payload["sample_sets"].items():
        lines.append(f"## {name}")
        if not sample_rows:
            lines.append("- none")
        else:
            for row in sample_rows:
                lines.append(
                    "- "
                    f"{row['run_id']} | condition={row['condition_name']} | solved={row['group_solved']} | "
                    f"pressure={row['murderer_pressure_signals_received']} | structured={row['structured_accusation_fraction']} | "
                    f"hard_cap_fallbacks={row['hard_cap_fallback_transitions']} | dir={row['run_dir']}"
                )
        lines.append("")

    (experiment_path / "qualitative_samples.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export thesis-friendly qualitative sample run lists.")
    parser.add_argument("experiment_dir", help="Path to an experiment output directory")
    args = parser.parse_args()
    payload = export_qualitative_samples(args.experiment_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

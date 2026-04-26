from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import argparse
import csv
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REQUIRED_RUN_FILES = [
    "run_manifest.json",
    "events.jsonl",
    "metrics.json",
    "utterances.csv",
    "interactions.csv",
    "accusations.csv",
    "attention_summary.json",
    "deception_labels.csv",
]

EVIDENCE_BACKED_REASONING_PATTERNS = [
    "because",
    "clue",
    "evidence",
    "motive",
    "means",
    "opportunity",
    "contradiction",
    "alibi",
    "timeline",
    "doesn't add up",
    "does not add up",
    "inconsistent",
    "followed",
    "saw",
    "heard",
]


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
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    return any(pattern in normalized for pattern in patterns)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _thresholds_from_payload(payload: Dict[str, Any]) -> Dict[str, int]:
    return {
        "pilot_ready_runs_per_condition": int(payload.get("pilot_ready_runs_per_condition") or 3),
        "interim_ready_runs_per_condition": int(payload.get("interim_ready_runs_per_condition") or 10),
        "final_ready_runs_per_condition": int(payload.get("final_ready_runs_per_condition") or 20),
    }


def _resolve_experiment_thresholds(experiment_dir: Path) -> Dict[str, int]:
    plan = _read_json(experiment_dir / "experiment_plan.json")
    base = plan.get("base", {}) if plan else {}
    if base:
        return _thresholds_from_payload(base)

    for condition_dir in _condition_dirs(experiment_dir):
        condition_config = _read_json(condition_dir / "condition_config.json")
        if condition_config:
            return _thresholds_from_payload(condition_config)

    return _thresholds_from_payload({})


def validate_run_outputs(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "run_manifest.json")
    metrics = _read_json(run_path / "metrics.json")
    attention_summary = _read_json(run_path / "attention_summary.json")
    events = _read_jsonl(run_path / "events.jsonl")
    utterances = _read_csv(run_path / "utterances.csv")
    interactions = _read_csv(run_path / "interactions.csv")
    accusations = _read_csv(run_path / "accusations.csv")
    deception_labels = _read_csv(run_path / "deception_labels.csv")

    missing_required_files = [name for name in REQUIRED_RUN_FILES if not (run_path / name).exists()]

    agent_names = list(manifest.get("agent_names") or [])
    murderer_name = manifest.get("murderer_name") or metrics.get("murderer_name")
    accusation_reasoning_rows = [row for row in accusations if str(row.get("reasoning", "")).strip()]
    evidence_backed_reasoning_rows = [
        row for row in accusation_reasoning_rows if _contains_any(str(row.get("reasoning", "")), EVIDENCE_BACKED_REASONING_PATTERNS)
    ]

    direct_question_targets = {
        row.get("target")
        for row in interactions
        if row.get("target")
        and _as_bool(row.get("is_direct_target"))
        and _as_bool(row.get("is_question"))
    }

    murderer_direct_questions = [
        row for row in interactions
        if row.get("target") == murderer_name
        and _as_bool(row.get("is_direct_target"))
        and _as_bool(row.get("is_question"))
    ]
    murderer_direct_challenges = [
        row for row in interactions
        if row.get("target") == murderer_name
        and (
            _as_bool(row.get("is_pressure_signal"))
            or _as_bool(row.get("is_justification_request"))
            or (_as_bool(row.get("is_direct_target")) and _as_bool(row.get("is_question")))
        )
    ]

    clue_reveals = [event for event in events if event.get("type") == "clue_revealed"]
    round_changes = [event for event in events if event.get("type") == "round_changed"]
    round_summaries = [event for event in events if event.get("type") == "round_summarized"]
    round_advance_decisions = [event for event in events if event.get("type") == "round_advance_decision"]

    accusation_count_expected = len(agent_names) if agent_names else len(accusations)
    accusation_completed = len(accusations) >= accusation_count_expected and accusation_count_expected > 0
    nonempty_reasoning_fraction = _ratio(len(accusation_reasoning_rows), len(accusations)) if accusations else 0.0
    evidence_backed_reasoning_fraction = _ratio(len(evidence_backed_reasoning_rows), len(accusations)) if accusations else 0.0
    suspect_question_coverage_fraction = _ratio(len(direct_question_targets), len(agent_names)) if agent_names else 0.0

    exclusion_reasons: List[str] = []
    warnings: List[str] = []

    if missing_required_files:
        exclusion_reasons.append(f"missing_required_files:{','.join(missing_required_files)}")
    if manifest.get("status") != "finished":
        exclusion_reasons.append(f"run_status:{manifest.get('status') or 'unknown'}")
    if not utterances:
        exclusion_reasons.append("no_utterances")
    if not accusation_completed:
        exclusion_reasons.append("accusation_phase_incomplete")
    if not metrics:
        exclusion_reasons.append("missing_metrics")

    if not clue_reveals:
        warnings.append("no_clue_revealed_events_logged")
    if suspect_question_coverage_fraction < 0.50:
        warnings.append("low_suspect_question_coverage")
    if murderer_name and not murderer_direct_questions:
        warnings.append("murderer_never_directly_questioned")
    if murderer_name and not murderer_direct_challenges:
        warnings.append("murderer_never_directly_challenged")
    if accusations and nonempty_reasoning_fraction < 1.0:
        warnings.append("some_accusations_missing_reasoning")
    if accusations and evidence_backed_reasoning_fraction < 0.75:
        warnings.append("accusation_reasoning_often_not_evidence_backed")
    if len(round_summaries) == 0:
        warnings.append("no_round_summaries_logged")

    payload = {
        "run_id": manifest.get("run_id", run_path.name),
        "experiment_name": manifest.get("experiment_name"),
        "condition_name": manifest.get("condition_name"),
        "validation_status": "valid" if not exclusion_reasons else "invalid",
        "run_usable_for_thesis": not exclusion_reasons,
        "exclusion_reasons": exclusion_reasons,
        "warnings": warnings,
        "missing_required_files": missing_required_files,
        "artifact_counts": {
            "total_events": len(events),
            "total_utterances": len(utterances),
            "total_interactions": len(interactions),
            "total_accusations": len(accusations),
            "total_deception_labels": len(deception_labels),
            "clue_reveals": len(clue_reveals),
            "round_changes": len(round_changes),
            "round_summaries": len(round_summaries),
            "round_advance_decisions": len(round_advance_decisions),
        },
        "process_quality": {
            "accusation_completed": accusation_completed,
            "accusation_reasoning_nonempty_fraction": nonempty_reasoning_fraction,
            "accusation_reasoning_evidence_backed_fraction": evidence_backed_reasoning_fraction,
            "suspect_question_coverage_fraction": suspect_question_coverage_fraction,
            "suspects_directly_questioned": sorted(name for name in direct_question_targets if name),
            "murderer_directly_questioned": bool(murderer_direct_questions),
            "murderer_directly_challenged": bool(murderer_direct_challenges),
            "murderer_pressure_signals_received": attention_summary.get("murderer_pressure_signals_received", 0),
            "clue_reveals_count": len(clue_reveals),
            "round_summary_count": len(round_summaries),
            "round_budget_transitions": sum(
                1
                for event in round_advance_decisions
                if event.get("payload", {}).get("advance_reason") == "round_budget_reached"
            ),
            "investigation_completed_events": sum(
                1
                for event in round_advance_decisions
                if event.get("payload", {}).get("advance_reason") == "investigation_complete"
            ),
        },
    }

    _write_json(run_path / "run_validation.json", payload)
    return payload


def summarize_condition_validation(condition_dir: str | Path) -> Dict[str, Any]:
    condition_path = Path(condition_dir)
    run_reports = [validate_run_outputs(run_dir) for run_dir in _run_dirs(condition_path)]
    usable_runs = [row for row in run_reports if row.get("run_usable_for_thesis")]
    invalid_runs = [row for row in run_reports if not row.get("run_usable_for_thesis")]
    warning_runs = [row for row in run_reports if row.get("warnings")]

    summary = {
        "condition_dir": str(condition_path),
        "condition_name": (_read_json(condition_path / "condition_config.json").get("condition_name") or condition_path.name),
        "total_runs": len(run_reports),
        "usable_runs": len(usable_runs),
        "invalid_runs": len(invalid_runs),
        "runs_with_warnings": len(warning_runs),
        "usable_run_ids": [row.get("run_id") for row in usable_runs],
        "invalid_run_ids": [row.get("run_id") for row in invalid_runs],
        "warning_run_ids": [row.get("run_id") for row in warning_runs],
        "run_reports": run_reports,
    }
    _write_json(condition_path / "validation_summary.json", summary)
    return summary


def _resolve_condition_targets(experiment_dir: Path) -> Dict[str, Dict[str, Any]]:
    plan = _read_json(experiment_dir / "experiment_plan.json")
    planned_conditions = plan.get("conditions", []) if plan else []
    if planned_conditions:
        targets = {}
        for condition in planned_conditions:
            name = condition.get("condition_name") or condition.get("name") or "default"
            targets[name] = {
                "planned_runs": int(condition.get("replicates") or plan.get("base", {}).get("replicates") or 0),
                "condition_description": condition.get("condition_description") or condition.get("description"),
            }
        return targets

    targets = {}
    for condition_dir in _condition_dirs(experiment_dir):
        config = _read_json(condition_dir / "condition_config.json")
        name = config.get("condition_name") or condition_dir.name
        targets[name] = {
            "planned_runs": int(config.get("replicates") or 0),
            "condition_description": config.get("condition_description"),
        }
    return targets


def _resolve_workflow_stage(min_usable_runs: int, any_runs_present: bool, thresholds: Dict[str, int], complete_but_underpowered: bool) -> str:
    if min_usable_runs >= thresholds["final_ready_runs_per_condition"]:
        return "final_analysis_ready"
    if min_usable_runs >= thresholds["interim_ready_runs_per_condition"]:
        return "interim_analysis_ready"
    if min_usable_runs >= thresholds["pilot_ready_runs_per_condition"]:
        return "pilot_ready"
    if complete_but_underpowered:
        return "complete_but_underpowered"
    if any_runs_present:
        return "collecting"
    return "planned"


def build_experiment_progress(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    thresholds = _resolve_experiment_thresholds(experiment_path)
    targets = _resolve_condition_targets(experiment_path)

    actual_condition_dirs = _condition_dirs(experiment_path)
    if targets and actual_condition_dirs == [experiment_path] and not (experiment_path / "runs").exists():
        actual_condition_dirs = []

    validation_by_condition: Dict[str, Dict[str, Any]] = {}
    for condition_dir in actual_condition_dirs:
        condition_validation = summarize_condition_validation(condition_dir)
        validation_by_condition[condition_validation.get("condition_name") or condition_dir.name] = condition_validation

    condition_names = list(targets.keys()) or list(validation_by_condition.keys())
    if not condition_names and actual_condition_dirs == [experiment_path]:
        default_name = (_read_json(experiment_path / "condition_config.json").get("condition_name") or experiment_path.name)
        condition_names = [default_name]

    condition_rows: List[Dict[str, Any]] = []
    for condition_name in condition_names:
        target = targets.get(condition_name, {})
        condition_validation = validation_by_condition.get(condition_name, {})
        planned_runs = int(target.get("planned_runs") or condition_validation.get("total_runs") or 0)
        usable_runs = int(condition_validation.get("usable_runs") or 0)
        invalid_runs = int(condition_validation.get("invalid_runs") or 0)
        total_runs = int(condition_validation.get("total_runs") or 0)

        condition_dir = None
        if condition_validation.get("condition_dir"):
            condition_dir = condition_validation["condition_dir"]
        elif condition_name and condition_name != experiment_path.name:
            condition_dir = str(experiment_path / "conditions" / condition_name)
        else:
            condition_dir = str(experiment_path)

        condition_rows.append(
            {
                "condition_name": condition_name,
                "condition_dir": condition_dir,
                "condition_description": target.get("condition_description"),
                "planned_runs": planned_runs,
                "total_runs": total_runs,
                "usable_runs": usable_runs,
                "invalid_runs": invalid_runs,
                "runs_with_warnings": int(condition_validation.get("runs_with_warnings") or 0),
                "remaining_to_pilot": max(thresholds["pilot_ready_runs_per_condition"] - usable_runs, 0),
                "remaining_to_interim": max(thresholds["interim_ready_runs_per_condition"] - usable_runs, 0),
                "remaining_to_final": max(thresholds["final_ready_runs_per_condition"] - usable_runs, 0),
                "planned_runs_remaining": max(planned_runs - total_runs, 0),
            }
        )

    usable_values = [row["usable_runs"] for row in condition_rows]
    min_usable_runs = min(usable_values) if usable_values else 0
    any_runs_present = any(row["total_runs"] > 0 for row in condition_rows)
    planned_total_runs = sum(row["planned_runs"] for row in condition_rows)
    total_runs = sum(row["total_runs"] for row in condition_rows)
    usable_runs = sum(row["usable_runs"] for row in condition_rows)
    invalid_runs = sum(row["invalid_runs"] for row in condition_rows)
    complete_but_underpowered = planned_total_runs > 0 and total_runs >= planned_total_runs and min_usable_runs < thresholds["pilot_ready_runs_per_condition"]
    workflow_stage = _resolve_workflow_stage(min_usable_runs, any_runs_present, thresholds, complete_but_underpowered)

    recommendations: List[str] = []
    if invalid_runs:
        recommendations.append("Inspect run_validation.json files for invalid runs before interpreting condition-level metrics.")
    if workflow_stage == "planned":
        recommendations.append("No usable runs yet; start with a small sanity batch and confirm that every run reaches accusation with complete artifacts.")
    elif workflow_stage == "collecting":
        recommendations.append(
            f"Keep collecting until each condition has at least {thresholds['pilot_ready_runs_per_condition']} usable runs for pilot comparison."
        )
    elif workflow_stage == "pilot_ready":
        recommendations.append(
            f"Pilot threshold reached; inspect top-line RQ1/RQ2/RQ3 differences and expand toward {thresholds['interim_ready_runs_per_condition']} usable runs per condition."
        )
    elif workflow_stage == "interim_analysis_ready":
        recommendations.append(
            f"Interim threshold reached; use pairwise comparisons and confidence intervals, then expand toward {thresholds['final_ready_runs_per_condition']} usable runs per condition for final thesis analysis."
        )
    elif workflow_stage == "final_analysis_ready":
        recommendations.append("Final threshold reached; freeze condition definitions, rebuild the thesis dataset, and start selecting qualitative transcript examples for the thesis chapters.")
    elif workflow_stage == "complete_but_underpowered":
        recommendations.append("The currently planned batch finished below pilot-ready power; either raise replicates or reduce the condition matrix before drawing conclusions.")

    payload = {
        "experiment_dir": str(experiment_path),
        "experiment_name": (_read_json(experiment_path / "experiment_plan.json").get("experiment_name") or experiment_path.name),
        "workflow_stage": workflow_stage,
        "thresholds": thresholds,
        "planned_total_runs": planned_total_runs,
        "total_runs": total_runs,
        "usable_runs": usable_runs,
        "invalid_runs": invalid_runs,
        "total_conditions": len(condition_rows),
        "minimum_usable_runs_per_condition": min_usable_runs,
        "conditions": condition_rows,
        "recommendations": recommendations,
    }
    return payload


def write_experiment_progress(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_path = Path(experiment_dir)
    payload = build_experiment_progress(experiment_path)
    _write_json(experiment_path / "progress_report.json", payload)

    lines = [
        f"Workflow progress for {payload['experiment_name']}",
        "",
        f"- Workflow stage: {payload['workflow_stage']}",
        f"- Planned total runs: {payload['planned_total_runs']}",
        f"- Completed runs: {payload['total_runs']}",
        f"- Usable runs: {payload['usable_runs']}",
        f"- Invalid runs: {payload['invalid_runs']}",
        f"- Conditions: {payload['total_conditions']}",
        f"- Minimum usable runs per condition: {payload['minimum_usable_runs_per_condition']}",
        "",
        "Thresholds:",
        f"- Pilot-ready: {payload['thresholds']['pilot_ready_runs_per_condition']} usable runs / condition",
        f"- Interim-analysis-ready: {payload['thresholds']['interim_ready_runs_per_condition']} usable runs / condition",
        f"- Final-analysis-ready: {payload['thresholds']['final_ready_runs_per_condition']} usable runs / condition",
        "",
        "Per-condition status:",
    ]

    for row in payload.get("conditions", []):
        lines.extend(
            [
                f"- {row['condition_name']}: planned={row['planned_runs']}, completed={row['total_runs']}, usable={row['usable_runs']}, invalid={row['invalid_runs']}, warnings={row['runs_with_warnings']}",
                f"  remaining_to_pilot={row['remaining_to_pilot']}, remaining_to_interim={row['remaining_to_interim']}, remaining_to_final={row['remaining_to_final']}",
            ]
        )

    lines.append("")
    lines.append("Recommendations:")
    for recommendation in payload.get("recommendations", []):
        lines.append(f"- {recommendation}")

    (experiment_path / "progress_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate run outputs and write a threshold-aware workflow progress report.")
    parser.add_argument("experiment_dir", help="Path to an experiment output directory.")
    args = parser.parse_args()

    payload = write_experiment_progress(Path(args.experiment_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

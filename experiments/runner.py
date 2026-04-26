from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.build_thesis_dataset import build_thesis_dataset
from analysis.compare_conditions import write_condition_report
from analysis.metrics import aggregate_experiment, aggregate_experiment_conditions, analyze_run
from analysis.workflow import summarize_condition_validation, validate_run_outputs, write_experiment_progress
from experiments.config import REPO_ROOT, LoadedExperiment, RunConfig, load_experiment_config
from instrumentation.event_logger import EventLogger, resolve_git_commit, utc_now_iso
from run_discussion import run_game_from_config


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_manifest(config: RunConfig, run_id: str) -> Dict:
    resolved_seed = config.resolved_seed()
    return {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "replicate_id": config.replicate_id,
        "condition_name": config.condition_name,
        "condition_description": config.condition_description,
        "condition_factors": config.condition_factors,
        "pilot_ready_runs_per_condition": config.pilot_ready_runs_per_condition,
        "interim_ready_runs_per_condition": config.interim_ready_runs_per_condition,
        "final_ready_runs_per_condition": config.final_ready_runs_per_condition,
        "backend": config.backend,
        "model_name": config.model_name,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "seed": resolved_seed,
        "seed_base": config.seed,
        "seed_strategy": "base_plus_replicate_index" if config.seed is not None else "unset",
        "conversations_per_round": config.conversations_per_round,
        "max_rounds": config.max_rounds,
        "enable_ui": config.enable_ui,
        "scenario_id": config.scenario_id,
        "scenario_path": str(config.resolved_scenario_path()) if config.resolved_scenario_path() else None,
        "prompt_version": config.prompt_version,
        "turn_policy_version": config.turn_policy_version,
        "memory_version": config.memory_version,
        "deception_labeling_enabled": config.deception_labeling_enabled,
        "deception_labeling_mode": config.deception_labeling_mode,
        "roles_dir": str(config.resolved_roles_dir()),
        "clues_dir": str(config.resolved_clues_dir()),
        "repo_root": str(REPO_ROOT),
        "code_commit": resolve_git_commit(REPO_ROOT),
        "notes": config.notes,
        "config_fingerprint": config.config_fingerprint(),
    }


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_experiment_plan(experiment: LoadedExperiment):
    base = experiment.base
    experiment_dir = base.resolved_experiment_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "experiment_name": base.experiment_name,
        "source_path": experiment.source_path,
        "base": base.model_dump(mode="json"),
        "conditions": [cfg.model_dump(mode="json") for cfg in experiment.conditions],
    }
    _write_json(experiment_dir / "experiment_plan.json", plan_payload)


def _write_condition_snapshot(config: RunConfig):
    payload = config.model_dump(mode="json")
    payload["resolved_seed"] = config.resolved_seed()
    payload["resolved_condition_dir"] = str(config.resolved_condition_dir())
    payload["config_fingerprint"] = config.config_fingerprint()
    _write_json(config.resolved_condition_dir() / "condition_config.json", payload)


def _run_blueprint(config: RunConfig, replicate_id: int) -> Dict:
    run_config = config.model_copy(update={"replicate_id": replicate_id, "seed": config.seed})
    condition_slug = f"-{config.condition_name}" if config.condition_name else ""
    run_id = f"{config.experiment_name}{condition_slug}-{_timestamp_slug()}-r{replicate_id:03d}"
    run_dir = config.resolved_condition_dir() / "runs" / run_id
    manifest = _build_manifest(run_config, run_id)
    return {
        "replicate_id": replicate_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "resolved_seed": run_config.resolved_seed(),
        "manifest": manifest,
    }


def _write_condition_batch_status(config: RunConfig, payload: Dict):
    _write_json(config.resolved_condition_dir() / "batch_status.json", payload)


def _write_experiment_batch_status(experiment: LoadedExperiment, payload: Dict):
    _write_json(experiment.base.resolved_experiment_dir() / "batch_status.json", payload)


def _build_condition_plan(config: RunConfig) -> Dict:
    return {
        "experiment_name": config.experiment_name,
        "condition_name": config.condition_name,
        "condition_description": config.condition_description,
        "condition_dir": str(config.resolved_condition_dir()),
        "replicates": config.replicates,
        "config_fingerprint": config.config_fingerprint(),
        "planned_runs": [_run_blueprint(config, replicate_id) for replicate_id in range(1, config.replicates + 1)],
    }


def _summarize_condition_plan(config: RunConfig) -> Dict:
    plan = _build_condition_plan(config)
    return {
        "experiment_name": plan["experiment_name"],
        "condition_name": plan["condition_name"],
        "condition_description": plan["condition_description"],
        "condition_dir": plan["condition_dir"],
        "replicates": plan["replicates"],
        "config_fingerprint": plan["config_fingerprint"],
        "planned_runs": [
            {
                "replicate_id": row["replicate_id"],
                "run_id": row["run_id"],
                "run_dir": row["run_dir"],
                "resolved_seed": row["resolved_seed"],
            }
            for row in plan["planned_runs"]
        ],
    }


def run_batch(config: RunConfig, fail_fast: bool = False) -> Dict:
    condition_dir = config.resolved_condition_dir()
    runs_dir = condition_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: List[Dict] = []
    failed_runs: List[Dict] = []
    _write_condition_snapshot(config)

    condition_status = {
        "experiment_name": config.experiment_name,
        "condition_name": config.condition_name,
        "condition_description": config.condition_description,
        "condition_dir": str(condition_dir),
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "fail_fast": fail_fast,
        "replicates_planned": config.replicates,
        "replicates_completed": 0,
        "replicates_failed": 0,
        "config_fingerprint": config.config_fingerprint(),
        "runs": [],
    }
    _write_condition_batch_status(config, condition_status)

    for replicate_id in range(1, config.replicates + 1):
        run_config = config.model_copy(update={"replicate_id": replicate_id, "seed": config.seed})
        condition_slug = f"-{config.condition_name}" if config.condition_name else ""
        run_id = f"{config.experiment_name}{condition_slug}-{_timestamp_slug()}-r{replicate_id:03d}"
        run_dir = runs_dir / run_id
        logger = EventLogger(run_dir, _build_manifest(run_config, run_id))
        run_status = {
            "replicate_id": replicate_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "resolved_seed": run_config.resolved_seed(),
            "status": "running",
            "started_at": utc_now_iso(),
            "ended_at": None,
            "error_summary": None,
        }
        condition_status["runs"].append(run_status)
        _write_condition_batch_status(config, condition_status)

        try:
            result = run_game_from_config(run_config, event_sink=logger)
            logger.finalize(
                status="finished",
                extra={
                    "agent_names": result.get("agent_names", []),
                    "murderer_name": result.get("murderer_name"),
                    "winners": result.get("winners", []),
                    "votes": result.get("votes", {}),
                    "group_solved": result.get("group_solved", False),
                },
            )
            run_summary = analyze_run(run_dir)
            run_validation = validate_run_outputs(run_dir)
            run_summaries.append(run_summary)
            run_status["status"] = "finished"
            run_status["validation_status"] = run_validation.get("validation_status")
            run_status["run_usable_for_thesis"] = run_validation.get("run_usable_for_thesis")
            run_status["validation_warnings"] = run_validation.get("warnings", [])
        except Exception as exc:
            error_payload = {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            logger.append("error", error_payload)
            logger.finalize(status="error", extra={"error_summary": str(exc)})
            run_status["status"] = "error"
            run_status["error_summary"] = str(exc)
            failed_runs.append(
                {
                    "replicate_id": replicate_id,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "resolved_seed": run_config.resolved_seed(),
                    "error_summary": str(exc),
                }
            )
            if fail_fast:
                run_status["ended_at"] = utc_now_iso()
                condition_status["replicates_completed"] = len(run_summaries)
                condition_status["replicates_failed"] = len(failed_runs)
                condition_status["finished_at"] = utc_now_iso()
                condition_status["status"] = "error"
                _write_condition_batch_status(config, condition_status)
                raise
        finally:
            run_status["ended_at"] = utc_now_iso()
            condition_status["replicates_completed"] = len(run_summaries)
            condition_status["replicates_failed"] = len(failed_runs)
            _write_condition_batch_status(config, condition_status)

    aggregate = aggregate_experiment(condition_dir)
    validation_summary = summarize_condition_validation(condition_dir)
    condition_status["finished_at"] = utc_now_iso()
    condition_status["status"] = "finished_with_errors" if failed_runs else "finished"
    condition_status["aggregate_summary"] = aggregate
    condition_status["validation_summary"] = validation_summary
    condition_status["failed_runs"] = failed_runs
    _write_condition_batch_status(config, condition_status)
    return {
        "condition_name": config.condition_name,
        "runs": run_summaries,
        "failed_runs": failed_runs,
        "aggregate": aggregate,
        "validation_summary": validation_summary,
        "condition_dir": str(condition_dir),
        "experiment_dir": str(config.resolved_experiment_dir()),
        "batch_status": condition_status["status"],
    }


def run_experiment_plan(experiment: LoadedExperiment, fail_fast: bool = False) -> Dict:
    _write_experiment_plan(experiment)
    configs = experiment.expand()
    experiment_status = {
        "experiment_name": experiment.base.experiment_name,
        "experiment_dir": str(experiment.base.resolved_experiment_dir()),
        "source_path": experiment.source_path,
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "fail_fast": fail_fast,
        "planned_conditions": [
            {
                "condition_name": summary["condition_name"],
                "condition_description": summary["condition_description"],
                "condition_dir": summary["condition_dir"],
                "replicates": summary["replicates"],
                "config_fingerprint": summary["config_fingerprint"],
            }
            for summary in [_summarize_condition_plan(config) for config in configs]
        ],
        "completed_conditions": [],
        "failed_conditions": [],
    }
    _write_experiment_batch_status(experiment, experiment_status)

    condition_results: List[Dict] = []
    for config in configs:
        try:
            result = run_batch(config, fail_fast=fail_fast)
            condition_results.append(result)
            experiment_status["completed_conditions"].append(
                {
                    "condition_name": config.condition_name,
                    "condition_dir": str(config.resolved_condition_dir()),
                    "batch_status": result.get("batch_status"),
                    "failed_runs": result.get("failed_runs", []),
                    "validation_summary": result.get("validation_summary", {}),
                }
            )
            experiment_status["progress_report"] = write_experiment_progress(experiment.base.resolved_experiment_dir())
        except Exception as exc:
            failure = {
                "condition_name": config.condition_name,
                "condition_dir": str(config.resolved_condition_dir()),
                "error_summary": str(exc),
            }
            experiment_status["failed_conditions"].append(failure)
            experiment_status["status"] = "error"
            experiment_status["finished_at"] = utc_now_iso()
            _write_experiment_batch_status(experiment, experiment_status)
            raise
        finally:
            _write_experiment_batch_status(experiment, experiment_status)

    experiment_dir = experiment.base.resolved_experiment_dir()
    experiment_summary = aggregate_experiment_conditions(experiment_dir)
    condition_report = write_condition_report(experiment_dir)
    dataset_manifest = build_thesis_dataset(experiment_dir)
    progress_report = write_experiment_progress(experiment_dir)

    experiment_status["finished_at"] = utc_now_iso()
    had_failed_runs = any(result.get("failed_runs") for result in condition_results)
    experiment_status["status"] = "finished_with_errors" if had_failed_runs else "finished"
    experiment_status["experiment_summary"] = experiment_summary
    experiment_status["dataset_manifest"] = dataset_manifest
    experiment_status["progress_report"] = progress_report
    _write_experiment_batch_status(experiment, experiment_status)

    return {
        "experiment_dir": str(experiment_dir),
        "condition_results": condition_results,
        "experiment_summary": experiment_summary,
        "condition_report": condition_report,
        "dataset_manifest": dataset_manifest,
        "progress_report": progress_report,
        "batch_status": experiment_status["status"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run batch murder mystery experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--replicates", type=int, default=None, help="Override replicate count from config.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the batch immediately when a run fails.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate config expansion, write experiment_plan.json, and print the planned conditions/runs without executing them.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the config and write experiment_plan.json without running experiments or printing the full expanded plan.",
    )
    args = parser.parse_args()

    experiment = load_experiment_config(Path(args.config))

    if args.replicates is not None:
        experiment = LoadedExperiment(
            base=experiment.base.model_copy(update={"replicates": args.replicates}),
            conditions=[cfg.model_copy(update={"replicates": args.replicates}) for cfg in experiment.conditions],
            source_path=experiment.source_path,
        )

    _write_experiment_plan(experiment)
    write_experiment_progress(experiment.base.resolved_experiment_dir())

    if args.validate_only:
        payload = {
            "status": "valid",
            "experiment_name": experiment.base.experiment_name,
            "source_path": experiment.source_path,
            "total_conditions": len(experiment.expand()),
            "experiment_dir": str(experiment.base.resolved_experiment_dir()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.plan_only:
        payload = {
            "status": "planned",
            "experiment_name": experiment.base.experiment_name,
            "source_path": experiment.source_path,
            "experiment_dir": str(experiment.base.resolved_experiment_dir()),
            "conditions": [_summarize_condition_plan(config) for config in experiment.expand()],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    result = run_experiment_plan(experiment, fail_fast=args.fail_fast)
    print(f"Experiment outputs written to: {result['experiment_dir']}")
    print(json.dumps({
        "batch_status": result["batch_status"],
        "experiment_summary": result["experiment_summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

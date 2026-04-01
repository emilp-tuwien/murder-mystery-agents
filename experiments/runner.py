from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import argparse
import traceback

from analysis.metrics import aggregate_experiment, analyze_run
from experiments.config import REPO_ROOT, RunConfig, load_run_config
from instrumentation.event_logger import EventLogger, resolve_git_commit
from run_discussion import run_game_from_config


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_manifest(config: RunConfig, run_id: str) -> Dict:
    return {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "replicate_id": config.replicate_id,
        "backend": config.backend,
        "model_name": config.model_name,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "seed": config.seed,
        "conversations_per_round": config.conversations_per_round,
        "max_rounds": config.max_rounds,
        "enable_ui": config.enable_ui,
        "scenario_id": config.scenario_id,
        "prompt_version": config.prompt_version,
        "turn_policy_version": config.turn_policy_version,
        "memory_version": config.memory_version,
        "roles_dir": str(config.resolved_roles_dir()),
        "clues_dir": str(config.resolved_clues_dir()),
        "repo_root": str(REPO_ROOT),
        "code_commit": resolve_git_commit(REPO_ROOT),
        "notes": config.notes,
    }


def run_batch(config: RunConfig) -> Dict:
    experiment_dir = config.resolved_experiment_dir()
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: List[Dict] = []

    for replicate_id in range(1, config.replicates + 1):
        run_config = config.model_copy(update={"replicate_id": replicate_id})
        run_id = f"{config.experiment_name}-{_timestamp_slug()}-r{replicate_id:03d}"
        run_dir = runs_dir / run_id
        logger = EventLogger(run_dir, _build_manifest(run_config, run_id))

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
            run_summaries.append(run_summary)
        except Exception as exc:
            logger.append(
                "error",
                {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            logger.finalize(status="error", extra={"error_summary": str(exc)})
            raise

    aggregate = aggregate_experiment(experiment_dir)
    return {"runs": run_summaries, "aggregate": aggregate, "experiment_dir": str(experiment_dir)}


def main():
    parser = argparse.ArgumentParser(description="Run batch murder mystery experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--replicates", type=int, default=None, help="Override replicate count from config.")
    args = parser.parse_args()

    config = load_run_config(Path(args.config))
    if args.replicates is not None:
        config = config.model_copy(update={"replicates": args.replicates})

    result = run_batch(config)
    print(f"Experiment outputs written to: {result['experiment_dir']}")
    print(result["aggregate"])


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import hashlib

import yaml
from pydantic import BaseModel, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parent.parent


class RunConfig(BaseModel):
    experiment_name: str = "pilot"
    output_root: str = "outputs"
    replicates: int = Field(default=1, ge=1)

    pilot_ready_runs_per_condition: int = Field(default=3, ge=1)
    interim_ready_runs_per_condition: int = Field(default=10, ge=1)
    final_ready_runs_per_condition: int = Field(default=20, ge=1)

    backend: Literal["local", "gpt", "ollama", "nvidia"] = "local"
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    seed: Optional[int] = None
    enable_thinking: bool = False

    conversations_per_round: int = Field(default=20, ge=1)
    max_rounds: int = Field(default=6, ge=2)
    stage_gate_policy: Literal["round_budget", "evidence_gated"] = "round_budget"
    min_round_gate_conversations: Optional[int] = Field(default=None, ge=1)
    max_round_gate_conversations: Optional[int] = Field(default=None, ge=1)
    min_unique_question_targets_per_round: int = Field(default=3, ge=1)
    min_question_coverage_fraction_per_round: float = Field(default=0.5, ge=0.0, le=1.0)
    min_evidence_signals_per_round: int = Field(default=3, ge=0)
    min_pressure_signals_per_round: int = Field(default=2, ge=0)
    min_clue_references_per_round: int = Field(default=1, ge=0)
    min_synthesis_signals_final_round: int = Field(default=1, ge=0)
    enable_ui: bool = False
    ui_port: int = 8000

    prompt_version: str = "v1"
    turn_policy_version: str = "top2-selective-silence-v1"
    memory_version: str = "three-stage-v1"
    murderer_behavior_mode: Literal["passive_concealment", "active_deception"] = "passive_concealment"
    deception_labeling_enabled: bool = True
    deception_labeling_mode: Literal["off", "heuristic", "llm_rubric"] = "llm_rubric"

    # Judge config — all optional; fall back to game-backend fields when None
    deception_judge_backend: Optional[str] = None
    deception_judge_model_name: Optional[str] = None
    deception_judge_base_url: Optional[str] = None
    deception_judge_api_key_env: str = "OPENAI_API_KEY"
    deception_judge_temperature: float = 0.0
    deception_judge_context_before_turns: int = Field(default=4, ge=0)
    deception_judge_context_after_turns: int = Field(default=2, ge=0)
    deception_judge_max_retries: int = Field(default=2, ge=0)
    scenario_id: str = "business-of-murder-v1"
    notes: Optional[str] = None

    scenario_path: Optional[str] = None
    roles_dir: Optional[str] = None
    clues_dir: Optional[str] = None
    replicate_id: int = Field(default=1, ge=1)

    condition_name: Optional[str] = None
    condition_description: Optional[str] = None
    condition_factors: Dict[str, Any] = Field(default_factory=dict)

    def resolved_seed(self) -> Optional[int]:
        if self.seed is None:
            return None
        return self.seed + (self.replicate_id - 1)

    def condition_slug(self) -> str:
        return self.condition_name or "default"

    def config_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def resolved_scenario_path(self) -> Optional[Path]:
        if self.scenario_path:
            return (REPO_ROOT / self.scenario_path).resolve() if not Path(self.scenario_path).is_absolute() else Path(self.scenario_path)
        if self.scenario_id == "business-of-murder-v1":
            return REPO_ROOT / "scenarios" / "business-of-murder" / "scenario.json"
        return None

    def resolved_roles_dir(self) -> Path:
        if self.roles_dir:
            return (REPO_ROOT / self.roles_dir).resolve() if not Path(self.roles_dir).is_absolute() else Path(self.roles_dir)
        if self.scenario_id == "business-of-murder-v1":
            return REPO_ROOT / "scenarios" / "business-of-murder" / "roles"
        return REPO_ROOT / "agents" / "roles"

    def resolved_clues_dir(self) -> Path:
        if self.clues_dir:
            return (REPO_ROOT / self.clues_dir).resolve() if not Path(self.clues_dir).is_absolute() else Path(self.clues_dir)
        if self.scenario_id == "business-of-murder-v1":
            return REPO_ROOT / "scenarios" / "business-of-murder" / "clues"
        return REPO_ROOT / "clues"

    def resolved_min_round_gate_conversations(self) -> int:
        if self.min_round_gate_conversations is not None:
            return self.min_round_gate_conversations
        return max(6, min(self.conversations_per_round, max(6, self.conversations_per_round // 2)))

    def resolved_max_round_gate_conversations(self) -> int:
        if self.max_round_gate_conversations is not None:
            return self.max_round_gate_conversations
        return max(self.conversations_per_round, self.resolved_min_round_gate_conversations() + 6)

    def resolved_output_root(self) -> Path:
        return (REPO_ROOT / self.output_root).resolve()

    def resolved_experiment_dir(self) -> Path:
        return self.resolved_output_root() / self.experiment_name

    def resolved_condition_dir(self) -> Path:
        experiment_dir = self.resolved_experiment_dir()
        if self.condition_name:
            return experiment_dir / "conditions" / self.condition_name
        return experiment_dir


    @model_validator(mode="after")
    def validate_workflow_thresholds(self):
        if self.pilot_ready_runs_per_condition > self.interim_ready_runs_per_condition:
            raise ValueError("pilot_ready_runs_per_condition must be <= interim_ready_runs_per_condition")
        if self.interim_ready_runs_per_condition > self.final_ready_runs_per_condition:
            raise ValueError("interim_ready_runs_per_condition must be <= final_ready_runs_per_condition")
        if self.max_round_gate_conversations is not None and self.min_round_gate_conversations is not None:
            if self.max_round_gate_conversations < self.min_round_gate_conversations:
                raise ValueError("max_round_gate_conversations must be >= min_round_gate_conversations")
        return self


class ConditionConfig(BaseModel):
    name: str
    description: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    factors: Dict[str, Any] = Field(default_factory=dict)


class MatrixLevel(BaseModel):
    name: str
    description: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    factors: Dict[str, Any] = Field(default_factory=dict)


class ExperimentPlan(BaseModel):
    base: RunConfig
    conditions: List[ConditionConfig] = Field(default_factory=list)
    matrix: Dict[str, List[MatrixLevel]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_plan(self):
        if self.conditions and self.matrix:
            raise ValueError("Use either explicit conditions or a matrix, not both in the same experiment plan.")

        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("Condition names must be unique within an experiment plan.")

        for dimension_name, levels in self.matrix.items():
            if not levels:
                raise ValueError(f"Matrix dimension '{dimension_name}' must define at least one level.")
            level_names = [level.name for level in levels]
            if len(level_names) != len(set(level_names)):
                raise ValueError(f"Matrix dimension '{dimension_name}' contains duplicate level names.")

        return self


class LoadedExperiment(BaseModel):
    base: RunConfig
    conditions: List[RunConfig] = Field(default_factory=list)
    source_path: Optional[str] = None

    def expand(self) -> List[RunConfig]:
        return self.conditions or [self.base]


def _merge_condition(base: RunConfig, condition: ConditionConfig) -> RunConfig:
    overrides = dict(condition.overrides)
    valid_fields = set(RunConfig.model_fields.keys())
    unknown_keys = set(overrides.keys()) - valid_fields
    if unknown_keys:
        raise ValueError(
            f"Condition '{condition.name}' has unknown override keys: {sorted(unknown_keys)}. "
            f"Check for typos. Valid keys: {sorted(valid_fields)}"
        )
    merged_factors = dict(base.condition_factors)
    merged_factors.update(condition.factors)
    return base.model_copy(
        update={
            **overrides,
            "condition_name": condition.name,
            "condition_description": condition.description,
            "condition_factors": merged_factors,
        }
    )


def _expand_matrix_conditions(base: RunConfig, matrix: Dict[str, List[MatrixLevel]]) -> List[RunConfig]:
    if not matrix:
        return []

    ordered_dimensions = list(matrix.items())
    expanded: List[RunConfig] = []
    seen_names = set()

    for combination in product(*(levels for _, levels in ordered_dimensions)):
        name_parts: List[str] = []
        description_parts: List[str] = []
        merged_overrides: Dict[str, Any] = {}
        merged_factors = dict(base.condition_factors)

        for dimension_name, level in zip((name for name, _ in ordered_dimensions), combination):
            name_parts.append(level.name)
            merged_overrides.update(level.overrides)
            merged_factors[dimension_name] = level.name
            merged_factors.update(level.factors)
            if level.description:
                description_parts.append(f"{dimension_name}={level.description}")

        condition_name = "__".join(name_parts)
        if condition_name in seen_names:
            raise ValueError(f"Expanded matrix produced duplicate condition name: {condition_name}")
        seen_names.add(condition_name)

        valid_fields = set(RunConfig.model_fields.keys())
        unknown_keys = set(merged_overrides.keys()) - valid_fields
        if unknown_keys:
            raise ValueError(
                f"Matrix condition '{condition_name}' has unknown override keys: {sorted(unknown_keys)}. "
                f"Check for typos. Valid keys: {sorted(valid_fields)}"
            )

        expanded.append(
            base.model_copy(
                update={
                    **merged_overrides,
                    "condition_name": condition_name,
                    "condition_description": "; ".join(description_parts) if description_parts else None,
                    "condition_factors": merged_factors,
                }
            )
        )

    return expanded


def load_run_config(config_path: str | Path) -> RunConfig:
    return load_experiment_config(config_path).base


def load_experiment_config(config_path: str | Path) -> LoadedExperiment:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if "base" in data:
        plan = ExperimentPlan.model_validate(data)
        expanded = [_merge_condition(plan.base, condition) for condition in plan.conditions]
        if plan.matrix:
            expanded = _expand_matrix_conditions(plan.base, plan.matrix)
        return LoadedExperiment(base=plan.base, conditions=expanded, source_path=str(path))

    base = RunConfig.model_validate(data)
    return LoadedExperiment(base=base, conditions=[], source_path=str(path))

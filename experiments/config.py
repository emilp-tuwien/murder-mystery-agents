from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parent.parent


class RunConfig(BaseModel):
    experiment_name: str = "pilot"
    output_root: str = "outputs"
    replicates: int = Field(default=1, ge=1)

    backend: Literal["local", "gpt", "ollama"] = "local"
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    seed: Optional[int] = None

    conversations_per_round: int = Field(default=20, ge=1)
    max_rounds: int = Field(default=6, ge=2)
    enable_ui: bool = False
    ui_port: int = 8000

    prompt_version: str = "v1"
    turn_policy_version: str = "top2-selective-silence-v1"
    memory_version: str = "three-stage-v1"
    deception_labeling_enabled: bool = True
    deception_labeling_mode: Literal["heuristic", "off"] = "heuristic"
    scenario_id: str = "killingsworth-farm-v1"
    notes: Optional[str] = None

    scenario_path: Optional[str] = None
    roles_dir: Optional[str] = None
    clues_dir: Optional[str] = None
    replicate_id: int = Field(default=1, ge=1)

    condition_name: Optional[str] = None
    condition_description: Optional[str] = None
    condition_factors: Dict[str, Any] = Field(default_factory=dict)

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

    def resolved_output_root(self) -> Path:
        return (REPO_ROOT / self.output_root).resolve()

    def resolved_experiment_dir(self) -> Path:
        return self.resolved_output_root() / self.experiment_name

    def resolved_condition_dir(self) -> Path:
        experiment_dir = self.resolved_experiment_dir()
        if self.condition_name:
            return experiment_dir / "conditions" / self.condition_name
        return experiment_dir


class ConditionConfig(BaseModel):
    name: str
    description: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    factors: Dict[str, Any] = Field(default_factory=dict)


class ExperimentPlan(BaseModel):
    base: RunConfig
    conditions: List[ConditionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conditions(self):
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("Condition names must be unique within an experiment plan.")
        return self


class LoadedExperiment(BaseModel):
    base: RunConfig
    conditions: List[RunConfig] = Field(default_factory=list)
    source_path: Optional[str] = None

    def expand(self) -> List[RunConfig]:
        return self.conditions or [self.base]


def _merge_condition(base: RunConfig, condition: ConditionConfig) -> RunConfig:
    overrides = dict(condition.overrides)
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


def load_run_config(config_path: str | Path) -> RunConfig:
    return load_experiment_config(config_path).base


def load_experiment_config(config_path: str | Path) -> LoadedExperiment:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if "base" in data:
        plan = ExperimentPlan.model_validate(data)
        expanded = [_merge_condition(plan.base, condition) for condition in plan.conditions]
        return LoadedExperiment(base=plan.base, conditions=expanded, source_path=str(path))

    base = RunConfig.model_validate(data)
    return LoadedExperiment(base=base, conditions=[], source_path=str(path))

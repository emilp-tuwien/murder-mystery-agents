from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


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
    scenario_id: str = "killingsworth-farm-v1"
    notes: Optional[str] = None

    roles_dir: Optional[str] = None
    clues_dir: Optional[str] = None
    replicate_id: int = Field(default=1, ge=1)

    def resolved_roles_dir(self) -> Path:
        return Path(self.roles_dir) if self.roles_dir else REPO_ROOT / "agents" / "roles"

    def resolved_clues_dir(self) -> Path:
        return Path(self.clues_dir) if self.clues_dir else REPO_ROOT / "clues"

    def resolved_output_root(self) -> Path:
        return (REPO_ROOT / self.output_root).resolve()

    def resolved_experiment_dir(self) -> Path:
        return self.resolved_output_root() / self.experiment_name


def load_run_config(config_path: str | Path) -> RunConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RunConfig.model_validate(data)

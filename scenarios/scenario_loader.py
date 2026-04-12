from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
from pydantic import BaseModel


class ScenarioConfig(BaseModel):
    scenario_id: str = "killingsworth-farm-v1"
    title: str = "Murder at Killingsworth Farm"
    victim_name: str = "Elizabeth Killingsworth"
    victim_status_line: str = "Elizabeth Killingsworth has been found DEAD."
    location: str = "Killingsworth Farm"
    introduction_text: str = (
        "Tragedy has struck. Elizabeth Killingsworth has been found dead, and one of the suspects in the room is responsible."
    )
    investigation_goal: str = "Figure out who killed Elizabeth Killingsworth."
    accusation_prompt: str = "Who killed Elizabeth Killingsworth?"


DEFAULT_SCENARIO = ScenarioConfig()


def load_scenario_config(scenario_path: Optional[str | Path] = None) -> ScenarioConfig:
    if not scenario_path:
        return DEFAULT_SCENARIO

    path = Path(scenario_path)
    if path.is_dir():
        path = path / "scenario.json"

    if not path.exists():
        return DEFAULT_SCENARIO

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    return ScenarioConfig.model_validate(data)

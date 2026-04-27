from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json
from pydantic import BaseModel, Field


DEFAULT_EVIDENCE_TAGS = ["motive", "means", "opportunity", "contradiction", "timeline", "alibi"]

DEFAULT_MEMORY_CATEGORY_PATTERNS = {
    "motive": [
        "motive", "wanted", "needed", "money", "inherit", "loan", "love", "affair",
        "jealous", "hate", "profit", "debt", "fired", "desperate", "freedom",
        "promotion", "career", "creditor", "blackmail", "pay up",
    ],
    "means": [
        "weapon", "paperweight", "blood", "wallet", "checkbook", "note", "key",
        "fire escape", "office", "bathroom", "injury", "wound", "struck",
    ],
    "opportunity": [
        "saw", "with", "near", "room", "office", "bathroom", "apartment",
        "alone", "entered", "followed", "found", "at the scene", "parking",
        "fire escape", "hallway",
    ],
    "contradiction": [
        "contradict", "odd", "however", "but", "doesn't fit", "does not fit", "lie", "lying",
        "inconsistent", "different", "surprised", "opposite", "doesn't add up", "does not add up",
    ],
    "timeline": [
        "am", "pm", "around", "at ", ":", "before", "after", "earlier", "later", "when",
        "from ", "until", "between", "today", "tonight", "arrived", "left",
    ],
    "alibi": [
        "alibi", "i was", "i had been", "i have been", "could not have", "did not kill",
        "innocent", "was nowhere near", "can attest", "with me", "not there", "someone can confirm",
    ],
}

DEFAULT_GATE_STOPWORDS = [
    "the", "and", "that", "with", "from", "into", "this", "they", "their", "about", "there", "which",
    "after", "before", "under", "around", "could", "would", "should", "have", "been", "were", "them",
    "then", "when", "what", "where", "while", "through", "because", "only", "same", "very", "much",
    "just", "such", "your", "each", "must", "onto", "also", "will", "still", "than", "over", "more",
    "does", "doesn", "don", "did", "didn", "isn", "is", "are", "was", "were", "has", "had", "his",
    "her", "him", "she", "he", "you", "our", "out", "off", "for", "not", "one", "two", "three",
    "four", "five", "body", "clue", "round", "murder", "murdered", "victim", "group", "someone",
    "everyone", "apartment", "office", "party", "bathroom", "found", "dead", "rick", "martin",
]

DEFAULT_GATE_EVIDENCE_PATTERNS = [
    "motive", "means", "opportunity", "timeline", "alibi", "contradiction", "weapon", "note",
    "fire escape", "paperweight", "parking", "arrived", "left", "followed", "before", "after",
    "doesn't add up", "does not add up", "inconsistent", "doesn't fit", "does not fit", "why were you",
    "where were you", "what time", "who saw", "who was with", "key", "wallet", "checkbook", "blood",
]

DEFAULT_GATE_PRESSURE_PATTERNS = [
    "why", "how", "where", "when", "explain", "tell us", "answer", "account for", "doesn't add up",
    "does not add up", "inconsistent", "contradiction", "prove", "justify", "what were you doing",
    "where were you", "who can confirm", "who saw you", "why should we believe", "why would",
]

DEFAULT_GATE_SYNTHESIS_PATTERNS = [
    "i think", "i suspect", "the killer", "the murderer", "it points to", "it has to be", "this means",
    "that puts", "narrow", "narrows", "strongest case", "best case", "most likely",
]


class ScenarioConfig(BaseModel):
    scenario_id: str = "business-of-murder-v1"
    title: str = "The Business of Murder"
    victim_name: str = "Rick Martin"
    victim_status_line: str = "Rick Martin has been found DEAD in the bathroom."
    location: str = "Rick Martin's apartment in Chicago"
    introduction_text: str = (
        "What started as a casual post-party gathering has turned into a homicide investigation among business-school colleagues, lovers, creditors, and one dangerous outsider."
    )
    investigation_goal: str = "Figure out who killed Rick Martin."
    accusation_prompt: str = "Who killed Rick Martin?"
    evidence_tags: List[str] = Field(default_factory=lambda: list(DEFAULT_EVIDENCE_TAGS))
    memory_category_patterns: Dict[str, List[str]] = Field(default_factory=lambda: dict(DEFAULT_MEMORY_CATEGORY_PATTERNS))
    gate_stopwords: List[str] = Field(default_factory=lambda: list(DEFAULT_GATE_STOPWORDS))
    gate_evidence_patterns: List[str] = Field(default_factory=lambda: list(DEFAULT_GATE_EVIDENCE_PATTERNS))
    gate_pressure_patterns: List[str] = Field(default_factory=lambda: list(DEFAULT_GATE_PRESSURE_PATTERNS))
    gate_synthesis_patterns: List[str] = Field(default_factory=lambda: list(DEFAULT_GATE_SYNTHESIS_PATTERNS))


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

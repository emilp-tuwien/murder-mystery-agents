from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


VALID_EVIDENCE_CATEGORIES = {
    "motive", "means", "opportunity", "contradiction", "timeline", "alibi", "behavior"
}


class SuspectAssessment(BaseModel):
    suspect: str = Field(description="Name of the suspect being assessed")
    suspicion_score: int = Field(
        ge=1, le=10,
        description="1 = very weak suspicion, 10 = extremely strong suspicion"
    )
    confidence_score: int = Field(
        ge=1, le=10,
        description="1 = very uncertain about this assessment, 10 = very confident"
    )
    primary_reason: str = Field(description="Brief primary reason for this suspicion level")
    evidence_categories: List[str] = Field(
        default_factory=list,
        description="Evidence categories that support the score: motive, means, opportunity, contradiction, timeline, alibi, behavior"
    )
    strongest_supporting_fact: str = Field(
        description="The single strongest specific fact supporting this suspicion level"
    )

    @field_validator("suspicion_score", "confidence_score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5

    @field_validator("evidence_categories", mode="before")
    @classmethod
    def normalize_categories(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [c.strip().lower() for c in v if isinstance(c, str) and c.strip()]


class RoundSuspicionAssessment(BaseModel):
    round: int = Field(description="Investigation round number")
    stage: str = Field(description="Investigation stage name")
    agent: str = Field(description="Name of the agent producing this assessment")
    top_suspect: str = Field(
        description="The suspect this agent currently considers most likely guilty"
    )
    overall_uncertainty: int = Field(
        ge=1, le=10,
        description="1 = very certain who did it, 10 = completely uncertain"
    )
    suspect_assessments: List[SuspectAssessment] = Field(
        description="One assessment entry per other suspect — must cover all suspects"
    )

    @field_validator("overall_uncertainty", mode="before")
    @classmethod
    def clamp_uncertainty(cls, v):
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5

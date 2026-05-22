from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, Field, field_validator, model_validator


VALID_EVIDENCE_CATEGORIES = {
    "motive", "means", "opportunity", "contradiction", "timeline", "alibi", "behavior"
}

_SUSPECT_COLLECTION_KEYS = (
    "suspect_assessments", "suspects", "suspect_assessment", "assessments"
)
_TOP_LEVEL_RESERVED = {"round", "stage", "agent", "top_suspect", "overall_uncertainty"}
_PRIMARY_REASON_ALIASES = ("notes", "assessment", "rationale", "reason", "summary", "primary_reasoning")
_STRONGEST_FACT_ALIASES = (
    "strongest_evidence", "key_evidence", "evidence", "supporting_fact", "strongest_fact"
)


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

    @model_validator(mode="before")
    @classmethod
    def map_field_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "suspect" not in out and "name" in out:
            out["suspect"] = out.pop("name")
        if "primary_reason" not in out:
            for alias in _PRIMARY_REASON_ALIASES:
                if alias in out:
                    out["primary_reason"] = out.pop(alias)
                    break
        if "strongest_supporting_fact" not in out:
            for alias in _STRONGEST_FACT_ALIASES:
                if alias in out:
                    out["strongest_supporting_fact"] = out.pop(alias)
                    break
        if "primary_reason" not in out and "strongest_supporting_fact" in out:
            out["primary_reason"] = out["strongest_supporting_fact"]
        if "strongest_supporting_fact" not in out and "primary_reason" in out:
            out["strongest_supporting_fact"] = out["primary_reason"]
        out.setdefault("primary_reason", "No reason provided.")
        out.setdefault("strongest_supporting_fact", "No specific fact cited.")
        return out

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

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_payload(cls, data: Any) -> Any:
        """Coerce a wide range of LLM JSON shapes into the canonical schema.

        Why: small models (e.g. gpt-4.1-nano) often emit suspects as a top-level
        dict-of-name->attrs, or wrap them under `suspects` / `suspect_assessment`
        instead of `suspect_assessments`, and they routinely omit round/stage/agent
        (which the caller overwrites post-parse anyway).
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        out.setdefault("round", 0)
        out.setdefault("stage", "unknown")
        out.setdefault("agent", "unknown")
        out.setdefault("top_suspect", "unknown")
        out.setdefault("overall_uncertainty", 5)

        suspect_data = None
        for key in _SUSPECT_COLLECTION_KEYS:
            if key in out and isinstance(out[key], (list, dict)):
                suspect_data = out.pop(key)
                break

        if suspect_data is None:
            inline = {}
            for k in list(out.keys()):
                if k in _TOP_LEVEL_RESERVED:
                    continue
                v = out[k]
                if isinstance(v, dict):
                    inline[k] = v
                    out.pop(k)
            if inline:
                suspect_data = inline

        normalized: List[dict] = []
        if isinstance(suspect_data, dict):
            for name, attrs in suspect_data.items():
                if isinstance(attrs, dict):
                    entry = dict(attrs)
                    entry.setdefault("suspect", name)
                    normalized.append(entry)
        elif isinstance(suspect_data, list):
            for entry in suspect_data:
                if isinstance(entry, dict):
                    normalized.append(dict(entry))

        out["suspect_assessments"] = normalized
        return out

    @field_validator("overall_uncertainty", mode="before")
    @classmethod
    def clamp_uncertainty(cls, v):
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5

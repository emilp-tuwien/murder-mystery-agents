from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import re

from pydantic import BaseModel, Field

from utils.dialogue_analysis import detect_direct_address, extract_mentions, is_question, normalize_name


STOPWORDS = {
    "the", "and", "that", "with", "from", "into", "this", "they", "their", "about", "there", "which",
    "after", "before", "under", "around", "could", "would", "should", "have", "been", "were", "them",
    "then", "when", "what", "where", "while", "through", "because", "only", "same", "very", "much",
    "just", "such", "your", "each", "must", "onto", "also", "will", "still", "than", "over", "more",
    "does", "doesn", "don", "did", "didn", "isn", "is", "are", "was", "were", "has", "had", "his",
    "her", "him", "she", "he", "you", "our", "out", "off", "for", "not", "one", "two", "three",
    "four", "five", "body", "clue", "round", "murder", "murdered", "victim", "group", "someone",
    "everyone", "apartment", "office", "party", "bathroom", "found", "dead", "rick", "martin",
}

EVIDENCE_PATTERNS = [
    "motive", "means", "opportunity", "timeline", "alibi", "contradiction", "weapon", "note",
    "fire escape", "paperweight", "parking", "arrived", "left", "followed", "before", "after",
    "doesn't add up", "does not add up", "inconsistent", "doesn't fit", "does not fit", "why were you",
    "where were you", "what time", "who saw", "who was with", "key", "wallet", "checkbook", "blood",
]

PRESSURE_PATTERNS = [
    "why", "how", "where", "when", "explain", "tell us", "answer", "account for", "doesn't add up",
    "does not add up", "inconsistent", "contradiction", "prove", "justify", "what were you doing",
    "where were you", "who can confirm", "who saw you", "why should we believe", "why would",
]

SYNTHESIS_PATTERNS = [
    "i think", "i suspect", "the killer", "the murderer", "it points to", "it has to be", "this means",
    "that puts", "narrow", "narrows", "strongest case", "best case", "most likely",
]

ROUND_STAGE_NAMES = {
    1: "introduction",
    2: "initial_framing",
    3: "clue_integration",
    4: "contradiction_pressure",
    5: "accusation_synthesis",
}


class RoundGateAssessment(BaseModel):
    round_number: int
    stage_name: str
    gate_policy: str
    clue_available: bool = False
    clue_keywords: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)
    unmet_requirements: List[str] = Field(default_factory=list)
    minimum_conversations_reached: bool = False
    hard_cap_reached: bool = False
    gate_satisfied: bool = False
    allow_advance: bool = False
    advance_reason: str = "wait_for_more_evidence"


def stage_name_for_round(round_number: int, max_rounds: int) -> str:
    if round_number >= max_rounds:
        return "accusation"
    return ROUND_STAGE_NAMES.get(round_number, f"round_{round_number}")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9:']+", text.lower())


def extract_clue_keywords(clue_text: str, max_keywords: int = 10) -> List[str]:
    if not clue_text:
        return []

    counts: Dict[str, int] = {}
    ordered: List[str] = []
    first_seen: Dict[str, int] = {}
    for token in _tokenize(clue_text):
        normalized = token.strip("' ")
        if len(normalized) < 4 or normalized in STOPWORDS:
            continue
        if normalized not in counts:
            first_seen[normalized] = len(ordered)
            ordered.append(normalized)
            counts[normalized] = 0
        counts[normalized] += 1

    ordered.sort(key=lambda token: (-counts[token], first_seen[token]))
    deduped: List[str] = []
    for token in ordered:
        if token not in deduped:
            deduped.append(token)
        if len(deduped) >= max_keywords:
            break
    return deduped


def _contains_pattern(text: str, patterns: Iterable[str]) -> bool:
    normalized = normalize_name(text)
    return any(pattern in normalized for pattern in patterns)


def _contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_name(text)
    return any(keyword in normalized for keyword in keywords if keyword)


def assess_round_gate(
    history: List[Dict[str, Any]],
    agent_names: List[str],
    current_round: int,
    conversations_in_round: int,
    max_rounds: int,
    gate_policy: str,
    clue_text: str = "",
    min_conversations: Optional[int] = None,
    hard_cap_conversations: Optional[int] = None,
    min_unique_question_targets: int = 3,
    min_question_coverage_fraction: float = 0.50,
    min_evidence_signals: int = 3,
    min_pressure_signals: int = 2,
    min_clue_references: int = 1,
    min_synthesis_signals: int = 1,
) -> RoundGateAssessment:
    stage_name = stage_name_for_round(current_round, max_rounds)
    resolved_min_conversations = min_conversations or 6
    resolved_hard_cap = hard_cap_conversations or max(resolved_min_conversations + 4, resolved_min_conversations)
    clue_keywords = extract_clue_keywords(clue_text)

    question_targets = set()
    mentioned_suspects = set()
    evidence_signal_count = 0
    pressure_signal_count = 0
    clue_reference_count = 0
    direct_response_count = 0
    synthesis_signal_count = 0

    for utterance in history:
        speaker = utterance.get("speaker")
        text = str(utterance.get("text", ""))
        if not text or speaker == "Game Master":
            continue
        other_agents = [name for name in agent_names if name != speaker]
        addressed_to = utterance.get("addressed_to") or detect_direct_address(text, other_agents)
        mentions = utterance.get("mentioned_agents") or extract_mentions(text, other_agents)

        if is_question(text) and addressed_to:
            question_targets.add(addressed_to)
        for mention in mentions:
            mentioned_suspects.add(mention)

        if _contains_pattern(text, EVIDENCE_PATTERNS):
            evidence_signal_count += 1
        if _contains_pattern(text, PRESSURE_PATTERNS) or (is_question(text) and addressed_to):
            pressure_signal_count += 1
        if clue_keywords and _contains_keywords(text, clue_keywords):
            clue_reference_count += 1
        if utterance.get("response_to_speaker"):
            direct_response_count += 1
        if _contains_pattern(text, SYNTHESIS_PATTERNS):
            synthesis_signal_count += 1

    unique_question_targets = len(question_targets)
    suspect_coverage_fraction = unique_question_targets / len(agent_names) if agent_names else 0.0
    unique_suspect_mentions = len(mentioned_suspects)
    minimum_conversations_reached = conversations_in_round >= resolved_min_conversations
    hard_cap_reached = conversations_in_round >= resolved_hard_cap

    if gate_policy == "round_budget":
        gate_satisfied = conversations_in_round >= resolved_hard_cap
        return RoundGateAssessment(
            round_number=current_round,
            stage_name=stage_name,
            gate_policy=gate_policy,
            clue_available=bool(clue_text),
            clue_keywords=clue_keywords,
            metrics={
                "conversations_in_round": conversations_in_round,
                "unique_question_targets": unique_question_targets,
                "suspect_question_coverage_fraction": suspect_coverage_fraction,
                "unique_suspect_mentions": unique_suspect_mentions,
                "evidence_signal_count": evidence_signal_count,
                "pressure_signal_count": pressure_signal_count,
                "clue_reference_count": clue_reference_count,
                "direct_response_count": direct_response_count,
                "synthesis_signal_count": synthesis_signal_count,
            },
            thresholds={
                "advance_after_conversations": resolved_hard_cap,
            },
            unmet_requirements=[] if gate_satisfied else ["round_budget_not_reached"],
            minimum_conversations_reached=minimum_conversations_reached,
            hard_cap_reached=hard_cap_reached,
            gate_satisfied=gate_satisfied,
            allow_advance=gate_satisfied,
            advance_reason="round_budget_reached" if gate_satisfied else "wait_for_more_turns",
        )

    required_question_targets = max(1, min(min_unique_question_targets, max(len(agent_names) - 1, 1)))
    required_question_coverage = min_question_coverage_fraction
    required_evidence = max(2 if current_round <= 2 else 3, min_evidence_signals)
    required_pressure = 1 if current_round == 2 else min_pressure_signals
    required_clue_references = 0 if not clue_text else min_clue_references
    required_synthesis = min_synthesis_signals if current_round >= max_rounds - 1 else 0
    required_suspect_mentions = max(2, min(4, max(len(agent_names) // 2, 2)))

    unmet: List[str] = []
    if unique_question_targets < required_question_targets:
        unmet.append("unique_question_targets")
    if suspect_coverage_fraction < required_question_coverage:
        unmet.append("question_coverage")
    if unique_suspect_mentions < required_suspect_mentions:
        unmet.append("suspect_mentions")
    if evidence_signal_count < required_evidence:
        unmet.append("evidence_signals")
    if pressure_signal_count < required_pressure:
        unmet.append("pressure_signals")
    if clue_reference_count < required_clue_references:
        unmet.append("clue_references")
    if synthesis_signal_count < required_synthesis:
        unmet.append("synthesis_signals")
    if not minimum_conversations_reached:
        unmet.append("minimum_conversations")

    gate_satisfied = minimum_conversations_reached and not [item for item in unmet if item != "minimum_conversations"]
    allow_advance = gate_satisfied or hard_cap_reached
    if gate_satisfied:
        advance_reason = "evidence_gate_satisfied"
    elif hard_cap_reached:
        advance_reason = "hard_cap_fallback"
    else:
        advance_reason = "wait_for_more_evidence"

    return RoundGateAssessment(
        round_number=current_round,
        stage_name=stage_name,
        gate_policy=gate_policy,
        clue_available=bool(clue_text),
        clue_keywords=clue_keywords,
        metrics={
            "conversations_in_round": conversations_in_round,
            "unique_question_targets": unique_question_targets,
            "suspect_question_coverage_fraction": suspect_coverage_fraction,
            "unique_suspect_mentions": unique_suspect_mentions,
            "evidence_signal_count": evidence_signal_count,
            "pressure_signal_count": pressure_signal_count,
            "clue_reference_count": clue_reference_count,
            "direct_response_count": direct_response_count,
            "synthesis_signal_count": synthesis_signal_count,
        },
        thresholds={
            "minimum_conversations": resolved_min_conversations,
            "hard_cap_conversations": resolved_hard_cap,
            "unique_question_targets": required_question_targets,
            "question_coverage_fraction": required_question_coverage,
            "unique_suspect_mentions": required_suspect_mentions,
            "evidence_signals": required_evidence,
            "pressure_signals": required_pressure,
            "clue_references": required_clue_references,
            "synthesis_signals": required_synthesis,
        },
        unmet_requirements=unmet,
        minimum_conversations_reached=minimum_conversations_reached,
        hard_cap_reached=hard_cap_reached,
        gate_satisfied=gate_satisfied,
        allow_advance=allow_advance,
        advance_reason=advance_reason,
    )

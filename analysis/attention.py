from __future__ import annotations

from collections import Counter
from math import log2
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import csv
import json
import re


JUSTIFICATION_PATTERNS = [
    "why",
    "how do you explain",
    "can you explain",
    "explain",
    "what is your explanation",
    "what's your explanation",
    "justify",
    "account for",
]

PRESSURE_PATTERNS = [
    "suspicious",
    "motive",
    "means",
    "opportunity",
    "doesn't add up",
    "does not add up",
    "contradiction",
    "inconsistent",
    "doesn't fit",
    "does not fit",
    "should answer",
    "needs to explain",
    "need to explain",
]


def _write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _gini_from_counts(values: Iterable[int]) -> float:
    values = [int(value) for value in values if value is not None]
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _entropy_from_counts(values: Iterable[int]) -> float:
    values = [int(value) for value in values if value is not None and value > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        p = value / total
        entropy -= p * log2(p)
    return entropy


def _collect_targets(row: Dict[str, Any]) -> List[str]:
    targets: List[str] = []
    if row.get("addressed_to"):
        targets.append(str(row["addressed_to"]))
    mentioned = [name for name in str(row.get("mentioned_agents", "")).split("|") if name]
    for name in mentioned:
        if name not in targets:
            targets.append(name)
    return targets


def build_attention_artifacts(
    run_path: str | Path,
    utterances: List[Dict[str, Any]],
    agent_names: List[str],
    murderer_name: str | None,
) -> Dict[str, Any]:
    run_dir = Path(run_path)
    interaction_rows: List[Dict[str, Any]] = []

    target_question_counts = Counter()
    target_followup_counts = Counter()
    target_justification_counts = Counter()
    target_pressure_counts = Counter()
    target_mention_counts = Counter()

    questions_asked = Counter()
    questions_received = Counter()
    followups_asked = Counter()
    followups_received = Counter()
    justifications_asked = Counter()
    justifications_received = Counter()
    pressure_asked = Counter()
    pressure_received = Counter()
    mentions_asked = Counter()
    mentions_received = Counter()

    for row in utterances:
        speaker = row.get("speaker")
        text = str(row.get("text", ""))
        normalized = _normalize(text)
        addressed_to = row.get("addressed_to")
        response_to = row.get("response_to_speaker")
        is_question = bool(row.get("is_question"))
        mentioned = [name for name in str(row.get("mentioned_agents", "")).split("|") if name]
        targets = _collect_targets(row)

        is_followup = bool(response_to and addressed_to and response_to == addressed_to and is_question)
        is_justification_request = bool(is_question and _contains_any(normalized, JUSTIFICATION_PATTERNS))
        is_pressure = bool(_contains_any(normalized, PRESSURE_PATTERNS))

        for target in targets:
            is_direct_target = target == addressed_to
            target_question = bool(is_question and is_direct_target)
            target_followup = bool(is_followup and target == response_to)
            target_justification = bool(is_justification_request and is_direct_target)
            target_pressure = bool(is_pressure and target in mentioned + ([addressed_to] if addressed_to else []))
            target_mention = bool(target in mentioned)

            if target_question:
                questions_asked[speaker] += 1
                questions_received[target] += 1
                target_question_counts[target] += 1
            if target_followup:
                followups_asked[speaker] += 1
                followups_received[target] += 1
                target_followup_counts[target] += 1
            if target_justification:
                justifications_asked[speaker] += 1
                justifications_received[target] += 1
                target_justification_counts[target] += 1
            if target_pressure:
                pressure_asked[speaker] += 1
                pressure_received[target] += 1
                target_pressure_counts[target] += 1
            if target_mention:
                mentions_asked[speaker] += 1
                mentions_received[target] += 1
                target_mention_counts[target] += 1

            interaction_rows.append({
                "turn": row.get("turn"),
                "round": row.get("round"),
                "phase": row.get("phase"),
                "speaker": speaker,
                "target": target,
                "response_to_speaker": response_to,
                "is_direct_target": is_direct_target,
                "is_question": is_question,
                "is_followup_question": target_followup,
                "is_justification_request": target_justification,
                "is_pressure_signal": target_pressure,
                "is_mention": target_mention,
                "text": text,
            })

    agent_rows: List[Dict[str, Any]] = []
    for agent in agent_names:
        agent_rows.append({
            "agent": agent,
            "is_murderer": agent == murderer_name,
            "questions_asked": questions_asked[agent],
            "questions_received": questions_received[agent],
            "followups_asked": followups_asked[agent],
            "followups_received": followups_received[agent],
            "justification_requests_asked": justifications_asked[agent],
            "justification_requests_received": justifications_received[agent],
            "pressure_signals_asked": pressure_asked[agent],
            "pressure_signals_received": pressure_received[agent],
            "mentions_made": mentions_asked[agent],
            "mentions_received": mentions_received[agent],
        })

    attention_summary = {
        "total_interaction_rows": len(interaction_rows),
        "question_target_entropy": _entropy_from_counts(target_question_counts.values()),
        "question_target_gini": _gini_from_counts(target_question_counts.values()),
        "pressure_target_entropy": _entropy_from_counts(target_pressure_counts.values()),
        "pressure_target_gini": _gini_from_counts(target_pressure_counts.values()),
        "followup_target_entropy": _entropy_from_counts(target_followup_counts.values()),
        "followup_target_gini": _gini_from_counts(target_followup_counts.values()),
        "justification_target_entropy": _entropy_from_counts(target_justification_counts.values()),
        "justification_target_gini": _gini_from_counts(target_justification_counts.values()),
        "question_target_counts": dict(target_question_counts),
        "pressure_target_counts": dict(target_pressure_counts),
        "followup_target_counts": dict(target_followup_counts),
        "justification_target_counts": dict(target_justification_counts),
        "murderer_questions_received": questions_received[murderer_name] if murderer_name else 0,
        "murderer_followups_received": followups_received[murderer_name] if murderer_name else 0,
        "murderer_justification_requests_received": justifications_received[murderer_name] if murderer_name else 0,
        "murderer_pressure_signals_received": pressure_received[murderer_name] if murderer_name else 0,
        "murderer_mentions_received": mentions_received[murderer_name] if murderer_name else 0,
    }

    _write_csv(run_dir / "interactions.csv", interaction_rows)
    _write_csv(run_dir / "agent_attention_summary.csv", agent_rows)
    _write_json(run_dir / "attention_summary.json", attention_summary)

    return {
        "interaction_rows": interaction_rows,
        "agent_rows": agent_rows,
        "summary": attention_summary,
    }

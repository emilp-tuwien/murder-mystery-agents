from __future__ import annotations

from typing import Iterable, List, Optional
import re


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _candidate_patterns(agent_name: str) -> List[str]:
    name_lower = normalize_name(agent_name)
    first_name = name_lower.split()[0] if " " in name_lower else name_lower
    return [name_lower, first_name]


def detect_direct_address(text: str, available_agents: Iterable[str]) -> Optional[str]:
    """
    Lightweight direct-address detector used both during simulation and in post-hoc analysis.
    """
    text_lower = normalize_name(text)

    for agent_name in available_agents:
        for candidate in _candidate_patterns(agent_name):
            patterns = [
                f"{candidate},",
                f"{candidate}?",
                f"ask {candidate}",
                f"{candidate} can you",
                f"{candidate}, can you",
                f"{candidate} what",
                f"{candidate}, what",
                f"{candidate} where",
                f"{candidate}, where",
                f"{candidate} why",
                f"{candidate}, why",
                f"{candidate} tell us",
                f"to {candidate}",
            ]
            if any(pattern in text_lower for pattern in patterns):
                return agent_name

    return None


def extract_mentions(text: str, agent_names: Iterable[str], exclude: Optional[str] = None) -> List[str]:
    text_lower = normalize_name(text)
    exclude_normalized = normalize_name(exclude) if exclude else None
    mentions: List[str] = []

    for agent_name in agent_names:
        if exclude_normalized and normalize_name(agent_name) == exclude_normalized:
            continue
        for candidate in _candidate_patterns(agent_name):
            if re.search(rf"\b{re.escape(candidate)}\b", text_lower):
                mentions.append(agent_name)
                break

    seen = set()
    deduped = []
    for mention in mentions:
        if mention not in seen:
            deduped.append(mention)
            seen.add(mention)
    return deduped


def is_question(text: str) -> bool:
    return "?" in text or text.strip().lower().startswith(("who ", "what ", "why ", "where ", "when ", "how "))

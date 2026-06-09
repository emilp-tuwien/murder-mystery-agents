from __future__ import annotations

from typing import Any, Iterable, List, Optional
import re


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _candidate_patterns(agent_name: str) -> List[str]:
    name_lower = normalize_name(agent_name)
    first_name = name_lower.split()[0] if " " in name_lower else name_lower
    candidates = [name_lower, first_name]

    # Lightweight role/title aliases for common direct-address forms used in-game.
    if name_lower == "harold chun":
        candidates.extend(["professor", "professor chun"])

    # Keep order, remove duplicates.
    seen = set()
    deduped = []
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


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
                f"{candidate}:",
                f"@{candidate}",
                f"hey {candidate}",
                f"hi {candidate}",
                f"so {candidate}",
                f"well {candidate}",
                f"and you, {candidate}",
                f"you, {candidate}",
                f"ask {candidate}",
                f"asking {candidate}",
                f"question for {candidate}",
                f"{candidate} can you",
                f"{candidate}, can you",
                f"{candidate} could you",
                f"{candidate}, could you",
                f"{candidate} would you",
                f"{candidate}, would you",
                f"{candidate} did you",
                f"{candidate}, did you",
                f"{candidate} do you",
                f"{candidate}, do you",
                f"{candidate} have you",
                f"{candidate}, have you",
                f"{candidate} are you",
                f"{candidate}, are you",
                f"{candidate} were you",
                f"{candidate}, were you",
                f"{candidate} what",
                f"{candidate}, what",
                f"{candidate} where",
                f"{candidate}, where",
                f"{candidate} why",
                f"{candidate}, why",
                f"{candidate} when",
                f"{candidate}, when",
                f"{candidate} how",
                f"{candidate}, how",
                f"{candidate} please",
                f"{candidate}, please",
                f"{candidate} tell us",
                f"{candidate}, tell us",
                f"to {candidate}",
            ]
            if any(pattern in text_lower for pattern in patterns):
                return agent_name

    return None


def detect_direct_address_llm(
    llm: Any,
    text: str,
    available_agents: Iterable[str],
    last_speaker: Optional[str] = None,
) -> Optional[str]:
    """
    LLM-backed direct-address detector. Use only as a fallback when the deterministic
    pattern matcher returns None — meant for ambiguous cases such as
    "Margaret, did you notice anyone slip into the bathroom?" that patterns miss.

    Returns the agent name if confident the message is directed at exactly one
    named suspect; otherwise None.
    """
    available_list = list(available_agents)
    if not available_list:
        return None

    mentioned = extract_mentions(text, available_list)
    if not mentioned:
        return None

    if not is_question(text):
        return None

    # Single mention + question: strong heuristic, skip the LLM call.
    if len(mentioned) == 1:
        return mentioned[0]

    if llm is None:
        return None

    options = ", ".join(mentioned)
    speaker_str = last_speaker or "Unknown"

    from langchain_core.messages import SystemMessage, HumanMessage
    from pydantic import BaseModel, Field

    class AddresseeChoice(BaseModel):
        addressee: str = Field(description="Name of the suspect being directly asked, or NONE")
        confidence: float = Field(ge=0.0, le=1.0)

    try:
        structured = llm.with_structured_output(AddresseeChoice, method="json_mode")
        msgs = [
            SystemMessage(content=(
                "Decide whether a single suspect is being directly asked a question. "
                "Only return a name if the message is unambiguously addressed to that one suspect "
                "(vocative form like 'X, did you...', 'What about you, X?', 'Hey X'). "
                "If the message merely mentions a suspect as a topic ('did anyone see X leave'), return NONE. "
                "Return JSON with keys: addressee, confidence."
            )),
            HumanMessage(content=(
                f"Speaker: {speaker_str}\n"
                f"Message: {text}\n\n"
                f"Candidates mentioned: {options}\n"
                "Return JSON only."
            )),
        ]
        result = structured.invoke(msgs)
    except Exception:
        return None

    if result is None:
        return None

    addressee = (result.addressee or "").strip()
    if not addressee or addressee.upper() == "NONE":
        return None
    if result.confidence < 0.6:
        return None

    addressee_norm = normalize_name(addressee)
    for candidate in mentioned:
        if normalize_name(candidate) == addressee_norm:
            return candidate
    for candidate in mentioned:
        if addressee_norm in normalize_name(candidate) or normalize_name(candidate) in addressee_norm:
            return candidate
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
    return "?" in text or text.strip().lower().startswith(("who ", "what ", "why ", "where ", "when ", "how ", "did ", "do ", "does ", "are ", "is ", "was ", "were ", "can ", "could ", "would ", "will ", "have ", "has ", "had "))


# Filled pauses / hesitation markers ("um", "uh", "er", "erm", "hmm") and
# throat-clearing. These are the nervous "leakage" tells a guilty agent tends to
# emit under direct pressure — we count rather than suppress them so they can be
# measured as a behavioural signal (e.g. murderer vs. innocents).
_FILLED_PAUSE_RE = re.compile(r"\b(?:u+m+|u+h+|uh+m+|e+rm+|er|hm+|ahem)\b", re.IGNORECASE)
# Stammered repetition: the same short word repeated across a comma/hyphen/dash,
# e.g. "I, I just" or "the- the office".
_STAMMER_RE = re.compile(r"\b(\w+)\s*[,\-–—]\s*\1\b", re.IGNORECASE)


def count_disfluencies(text: str) -> int:
    """Count nervous-speech tells in an utterance: filled pauses plus stammered
    word repetitions. Used by the analysis layer to quantify guilt-leakage; it
    deliberately does not alter or remove the disfluencies from the dialogue.
    """
    if not text:
        return 0
    return len(_FILLED_PAUSE_RE.findall(text)) + len(_STAMMER_RE.findall(text))

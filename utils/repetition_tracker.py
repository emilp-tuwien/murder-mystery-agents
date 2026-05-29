from __future__ import annotations

"""Semantic repetition tracker for the murder mystery game master.

Tracks all dialogue utterances with sentence-transformer embeddings and provides:
  - Per-utterance novelty scores (1 = fully novel, 0 = pure repetition)
  - Per-agent unique contribution counts
  - Round-level repetition rates

The Game Master uses this to:
  1. Penalise speakers whose intended contribution is semantically redundant
     (replaces the old token-overlap heuristic in decide_next_speaker)
  2. Boost agents who have contributed the fewest novel facts
  3. Block round advancement when the repetition rate is too high
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Similarity above this → the new utterance says something already said (different words, same claim)
NOVELTY_THRESHOLD = 0.82

# Similarity above this → near-verbatim duplicate
HIGH_SIM_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Embedding function (lazy-loaded singleton)
# ---------------------------------------------------------------------------

def _load_embedding_fn():
    """Return the sentence-transformer embedding function if available, else BoW fallback."""
    try:
        from memory.agent_memory import _get_st_embedding_fn
        fn = _get_st_embedding_fn()
        if fn:
            return fn
    except Exception:
        pass

    import hashlib, re

    def _bow_embed(text: str) -> np.ndarray:
        dim = 256
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        vec = np.zeros(dim)
        for w in cleaned.split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(dim):
                vec[i] += ((h >> i) & 1) * 2 - 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    return _bow_embed


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class _StoredUtterance:
    speaker: str
    text: str
    turn: int
    round_num: int
    embedding: np.ndarray
    is_novel: bool = True  # whether it was novel at the time it was spoken


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class RepetitionTracker:
    """Maintains an embedding store of all dialogue utterances for novelty scoring.

    Designed to be created once per game and closed into LangGraph node lambdas,
    exactly like `agents` and `game_master` are today.
    """

    def __init__(
        self,
        embedding_fn=None,
        novelty_threshold: float = NOVELTY_THRESHOLD,
        high_sim_threshold: float = HIGH_SIM_THRESHOLD,
    ):
        self._embed = embedding_fn or _load_embedding_fn()
        self._utterances: List[_StoredUtterance] = []
        self.novelty_threshold = novelty_threshold
        self.high_sim_threshold = high_sim_threshold

    # ── Write ───────────────────────────────────────────────────────────────

    def add(self, speaker: str, text: str, turn: int, round_num: int) -> None:
        """Record a dialogue utterance. Call from update_history after each turn."""
        if not text or not text.strip():
            return
        embedding = self._embed(text)
        is_novel = self._compute_is_novel(embedding)
        self._utterances.append(
            _StoredUtterance(speaker=speaker, text=text, turn=turn,
                             round_num=round_num, embedding=embedding, is_novel=is_novel)
        )

    def _compute_is_novel(self, embedding: np.ndarray) -> bool:
        if not self._utterances:
            return True
        prior_embs = np.stack([u.embedding for u in self._utterances])
        max_sim = float((prior_embs @ embedding).max())
        return max_sim < self.novelty_threshold

    # ── Read — per-utterance ─────────────────────────────────────────────────

    def novelty_score(self, text: str, against_last_n: Optional[int] = None) -> float:
        """Return 1.0 (fully novel) → 0.0 (pure repetition) for arbitrary text.

        against_last_n: if set, only compare against the most recent N utterances.
        Used for a narrower "recent window" check in speaker selection.
        """
        if not self._utterances or not text.strip():
            return 1.0
        candidates = self._utterances[-against_last_n:] if against_last_n else self._utterances
        emb = self._embed(text)
        embs = np.stack([u.embedding for u in candidates])
        max_sim = float((embs @ emb).max())
        return max(0.0, 1.0 - max_sim)

    # ── Read — per-agent ────────────────────────────────────────────────────

    def agent_novel_turn_counts(self, round_num: Optional[int] = None) -> Dict[str, int]:
        """Return how many utterances each agent made that were novel when spoken."""
        counts: Dict[str, int] = {}
        for u in self._utterances:
            if round_num is not None and u.round_num != round_num:
                continue
            counts.setdefault(u.speaker, 0)
            if u.is_novel:
                counts[u.speaker] += 1
        return counts

    def agent_total_turn_counts(self, round_num: Optional[int] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for u in self._utterances:
            if round_num is not None and u.round_num != round_num:
                continue
            counts[u.speaker] = counts.get(u.speaker, 0) + 1
        return counts

    def get_low_contribution_agents(
        self,
        all_agents: List[str],
        round_num: Optional[int] = None,
        top_n: int = 3,
    ) -> List[str]:
        """Return agents with the fewest novel contributions this round, sorted ascending."""
        counts = self.agent_novel_turn_counts(round_num=round_num)
        scored = [(name, counts.get(name, 0)) for name in all_agents]
        scored.sort(key=lambda x: x[1])
        return [name for name, _ in scored[:top_n]]

    # ── Read — round-level ───────────────────────────────────────────────────

    def repetition_rate(self, round_num: Optional[int] = None) -> float:
        """Fraction of utterances (after the first) that are near-duplicates of an earlier one."""
        utterances = [
            u for u in self._utterances
            if round_num is None or u.round_num == round_num
        ]
        if len(utterances) < 2:
            return 0.0
        not_novel = sum(1 for u in utterances[1:] if not u.is_novel)
        return not_novel / (len(utterances) - 1)

    def agents_with_novel_contribution(
        self, all_agents: List[str], round_num: Optional[int] = None
    ) -> List[str]:
        """Return agents who made at least one novel utterance."""
        counts = self.agent_novel_turn_counts(round_num=round_num)
        return [name for name in all_agents if counts.get(name, 0) > 0]

    def summary(self, round_num: Optional[int] = None) -> Dict:
        novel_counts = self.agent_novel_turn_counts(round_num=round_num)
        total_counts = self.agent_total_turn_counts(round_num=round_num)
        return {
            "repetition_rate": round(self.repetition_rate(round_num=round_num), 3),
            "novel_turns_by_agent": novel_counts,
            "total_turns_by_agent": total_counts,
        }

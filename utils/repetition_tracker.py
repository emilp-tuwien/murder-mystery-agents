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

# Word-trigram Jaccard against any prior turn at/above this → NOT novel. Used only in
# the embedding-free fallback (no sentence-transformers): the BoW cosine cannot tell
# topics apart (unrelated texts score ~0.9), which made every turn read as a repeat and
# the reported novelty collapse to 0%. Trigram overlap is a reliable lexical proxy —
# distinct points share few trigrams; restatements share many. Matches the cutoff used
# by the lexical saturation detector.
LEXICAL_NOVELTY_JACCARD = 0.34


# ---------------------------------------------------------------------------
# Embedding function (lazy-loaded singleton)
# ---------------------------------------------------------------------------

def _load_embedding_fn():
    """Return ``(embedding_fn, is_semantic)``.

    ``is_semantic`` is True only when the real sentence-transformer model is
    available. The BoW fallback cannot tell topics apart (unrelated texts score
    ~0.9 cosine), so callers must NOT use it for semantic-concentration decisions
    such as topic-saturation detection.
    """
    try:
        from memory.agent_memory import _get_st_embedding_fn
        fn = _get_st_embedding_fn()
        if fn:
            return fn, True
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

    return _bow_embed, False


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
    trigrams: Optional[set] = None  # word-trigram set; only populated in the embedding-free fallback


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
        lexical_novelty_jaccard: float = LEXICAL_NOVELTY_JACCARD,
    ):
        if embedding_fn is not None:
            # An explicitly supplied function is assumed semantic — the caller owns
            # that contract (used by tests and any custom embedder).
            self._embed = embedding_fn
            self._has_semantic_embeddings = True
        else:
            self._embed, self._has_semantic_embeddings = _load_embedding_fn()
        self._utterances: List[_StoredUtterance] = []
        self.novelty_threshold = novelty_threshold
        self.high_sim_threshold = high_sim_threshold
        self.lexical_novelty_jaccard = lexical_novelty_jaccard

    # ── Write ───────────────────────────────────────────────────────────────

    def add(self, speaker: str, text: str, turn: int, round_num: int) -> None:
        """Record a dialogue utterance. Call from update_history after each turn."""
        if not text or not text.strip():
            return
        embedding = self._embed(text)
        # Novelty must be judged with whatever signal is reliable. With real semantic
        # embeddings, cosine against prior turns. WITHOUT them, the BoW cosine is
        # meaningless (unrelated texts ~0.9), so fall back to word-trigram overlap —
        # otherwise every turn reads as a repeat and novelty reports as 0%.
        if self._has_semantic_embeddings:
            trigrams = None
            is_novel = self._compute_is_novel(embedding)
        else:
            trigrams = self._word_trigrams(text)
            is_novel = self._compute_is_novel_lexical(trigrams)
        self._utterances.append(
            _StoredUtterance(speaker=speaker, text=text, turn=turn,
                             round_num=round_num, embedding=embedding,
                             is_novel=is_novel, trigrams=trigrams)
        )

    def _compute_is_novel(self, embedding: np.ndarray) -> bool:
        if not self._utterances:
            return True
        prior_embs = np.stack([u.embedding for u in self._utterances])
        max_sim = float((prior_embs @ embedding).max())
        return max_sim < self.novelty_threshold

    def _compute_is_novel_lexical(self, trigrams: set) -> bool:
        """Embedding-free novelty: novel iff word-trigram Jaccard against every prior
        turn stays below ``lexical_novelty_jaccard``. A restatement (same point, reworded)
        reuses most of its phrasing frame and trips the threshold; a genuinely new point
        shares few trigrams and reads as novel."""
        if not trigrams or not self._utterances:
            return True
        best = 0.0
        for u in self._utterances:
            prior = u.trigrams
            if not prior:
                continue
            union = len(trigrams | prior)
            if union:
                best = max(best, len(trigrams & prior) / union)
        return best < self.lexical_novelty_jaccard

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

    def last_utterance_novelty(self, window: int = 6) -> float:
        """Novelty (1 = new topic → 0 = same as before) of the most recently added
        utterance, measured against the ``window`` utterances immediately preceding
        it. Returns 1.0 when there is no prior context.

        Unlike ``novelty_score(text)``, this compares the *stored* last utterance
        against only the turns BEFORE it — it never compares the utterance to
        itself — so it correctly answers "did the latest turn open a new line of
        inquiry, or keep circling the recent topic?".
        """
        if not self._utterances:
            return 1.0
        last = self._utterances[-1]
        prior = self._utterances[-(window + 1):-1]
        if not prior:
            return 1.0
        embs = np.stack([u.embedding for u in prior])
        max_sim = float((embs @ last.embedding).max())
        return max(0.0, 1.0 - max_sim)

    def last_utterance_repeats_earlier(
        self,
        same_speaker_only: bool = True,
        threshold: Optional[float] = None,
    ):
        """Judge whether the most recently added utterance repeats an EARLIER one.

        Compares the last stored utterance against all prior utterances (optionally
        restricted to the same speaker). Returns a tuple
        ``(is_repeat, similarity, matched_utterance)`` where ``matched_utterance`` is
        the earlier `_StoredUtterance` it most closely duplicates (or ``None``).

        Used by the Game Master to detect a nagging/repeated question — e.g. an asker
        pressing the same person for the same alibi turn after turn — so it can avoid
        forcing yet another redundant answer.
        """
        # The BoW fallback cannot tell a true near-verbatim repeat from two
        # different questions that merely share interrogation scaffolding
        # (unrelated questions score ~0.9 cosine). Without a real semantic model
        # this would flag almost every follow-up question as a "nag" and wrongly
        # suppress the direct-address mandate — silencing legitimately new
        # questions. Mirror recent_topic_saturation and disable detection here:
        # better a missed verbatim nag than silencing a genuine direct question.
        if not self._has_semantic_embeddings:
            return (False, 0.0, None)
        thr = threshold if threshold is not None else self.novelty_threshold
        if len(self._utterances) < 2:
            return (False, 0.0, None)
        last = self._utterances[-1]
        earlier = self._utterances[:-1]
        if same_speaker_only:
            earlier = [u for u in earlier if u.speaker == last.speaker]
        if not earlier:
            return (False, 0.0, None)
        embs = np.stack([u.embedding for u in earlier])
        sims = embs @ last.embedding
        idx = int(sims.argmax())
        max_sim = float(sims[idx])
        return (max_sim >= thr, max_sim, earlier[idx])

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

    def recent_topic_saturation(
        self,
        window: int = 6,
        concentration_threshold: float = 0.60,
        max_novelty_rate: float = 0.34,
    ):
        """Detect whether the most recent ``window`` utterances are all circling one
        topic while contributing little new information.

        This catches the failure mode the per-question "nagging" check misses: a
        *rotating* interrogation loop where different askers press different
        addressees about the SAME subject (e.g. the keychain), turn after turn.
        Each individual question looks novel in wording and names, so the
        duplicate check never trips — but the window as a whole is semantically
        clustered and stops producing facts.

        Returns ``(is_saturated, concentration, novelty_rate)`` where:
          - ``concentration`` is the mean pairwise cosine similarity of the recent
            window (0..1); high → the turns are clustered on one subject.
          - ``novelty_rate`` is the fraction of those turns that were novel when
            spoken; low → no new facts are landing.

        A topic is saturated when the window is both highly concentrated AND
        contributing little novelty — exactly the ping-pong pattern the
        direct-address mandate can otherwise sustain indefinitely.
        """
        if len(self._utterances) < window:
            return (False, 0.0, 1.0)
        # The hashed-BoW embedding fallback cannot measure topic concentration
        # (unrelated texts score ~0.9 cosine), so when real semantic embeddings are
        # unavailable we fall back to a LEXICAL detector based on word-trigram
        # overlap (see _lexical_topic_saturation). A rotating interrogation loop
        # ("did you have the key after 8:15?", "did you see the note?") reuses the
        # same phrasing FRAME turn after turn, so trigram concentration is high
        # while unrelated turns score low — reliable without embeddings, unlike the
        # BoW cosine. This keeps the circuit-breaker alive on machines without
        # sentence-transformers installed.
        if not self._has_semantic_embeddings:
            return self._lexical_topic_saturation(window, max_novelty_rate)
        recent = self._utterances[-window:]
        embs = np.stack([u.embedding for u in recent]).astype(float)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms > 0, norms, 1.0)
        sim = embs @ embs.T
        n = len(recent)
        off_diag_sum = float(sim.sum() - np.trace(sim))
        concentration = off_diag_sum / (n * (n - 1))
        novelty_rate = sum(1 for u in recent if u.is_novel) / n
        is_saturated = concentration >= concentration_threshold and novelty_rate <= max_novelty_rate
        return (is_saturated, concentration, novelty_rate)

    def _lexical_topic_saturation(
        self,
        window: int,
        max_novelty_rate: float,
        concentration_threshold: float = 0.16,
    ):
        """Embedding-free saturation detector using word-trigram overlap.

        The failure mode this must catch is a *rotating interrogation* loop — the
        same question FRAME reused while the nouns/targets rotate ("did you have
        the office key after 8:15?", "did you see the note?", "No, I didn't see
        it."). Content-word Jaccard misses it because the salient nouns change each
        turn; what actually recurs is the phrasing. So concentration here is the
        mean pairwise Jaccard of each utterance's word-TRIGRAM set (frame words
        included). Genuinely varied turns share few trigrams and score low, while
        an interrogation loop shares its frame trigrams and scores high. Novelty is
        the fraction of turns whose trigrams were not already largely seen earlier
        in the window. Thresholds suit trigram Jaccard, which runs lower than the
        embedding cosine scale.
        """
        recent = self._utterances[-window:]
        tri_sets = [self._word_trigrams(u.text) for u in recent]
        tri_sets = [t for t in tri_sets if t]  # drop empties (e.g. GM markers)
        n = len(tri_sets)
        if n < max(3, window // 2):
            return (False, 0.0, 1.0)

        def jaccard(a: set, b: set) -> float:
            union = len(a | b)
            return (len(a & b) / union) if union else 0.0

        pair_sims = [
            jaccard(tri_sets[i], tri_sets[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        concentration = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

        novel = 0
        for i in range(n):
            prior = [jaccard(tri_sets[i], tri_sets[k]) for k in range(i)]
            if not prior or max(prior) < 0.34:
                novel += 1
        novelty_rate = novel / n

        is_saturated = concentration >= concentration_threshold and novelty_rate <= max_novelty_rate
        return (is_saturated, concentration, novelty_rate)

    def _word_trigrams(self, text: str) -> set:
        """Set of consecutive word-trigrams (alnum tokens) for an utterance."""
        import re
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(words) < 3:
            return set(tuple(words)) if words else set()
        return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}

    # Modest built-in stoplist so the dominant-term descriptor doesn't latch onto
    # filler that recurs in every interrogation turn.
    _DOMINANT_TERM_STOPWORDS = frozenset({
        "didn", "have", "with", "that", "this", "they", "them", "anyone",
        "someone", "there", "when", "what", "were", "your", "just", "about",
        "would", "could", "know", "think", "after", "before", "still", "been",
        "into", "from", "then", "them", "their", "where", "which", "while",
        "said", "tell", "told", "right", "left", "took", "take", "made",
    })

    def dominant_recent_term(
        self, window: int = 6, stopwords: Optional[List[str]] = None
    ) -> Optional[str]:
        """Return the most frequent salient word across the recent window, or None.

        Used only to make the steering message concrete ("the keychain") — never
        for detection, so brittleness here is cosmetic, not correctness-critical.
        """
        import re
        from collections import Counter

        recent = self._utterances[-window:]
        if not recent:
            return None
        sw = set(self._DOMINANT_TERM_STOPWORDS)
        sw.update(w.lower() for w in (stopwords or []))
        counts: Counter = Counter()
        for u in recent:
            for w in re.findall(r"[a-z]{4,}", u.text.lower()):
                if w in sw:
                    continue
                counts[w] += 1
        if not counts:
            return None
        term, freq = counts.most_common(1)[0]
        # Only meaningful if it actually recurs across the window.
        if freq < max(2, window // 2):
            return None
        return term

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

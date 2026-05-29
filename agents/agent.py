from typing import Optional, Literal, Any, List, Dict
from pathlib import Path
import time
import re
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import SystemMessage, HumanMessage
from scenarios import ScenarioConfig
from schemas.state import GameState
from utils.dialogue_analysis import detect_direct_address, extract_mentions, is_question


BELIEF_EVIDENCE_MARKERS = [
    "motive", "alibi", "where were", "did you", "why did", "how do you explain", "contradict",
    "debt", "weapon", "blood", "lied", "suspicious", "note", "wallet", "checkbook",
    "paperweight", "fire escape", "money", "arrived", "left", "followed", "argument",
]
BELIEF_ACCUSATION_MARKERS = [
    "killed", "murderer", "murdered", "i suspect", "points to", "did it", "you lied", "lying",
    "doesn't add up", "does not add up",
]
BELIEF_DEFENSIVE_MARKERS = [
    "i didn't", "i did not", "i was", "i wasn't", "i was with", "with me", "i have an alibi",
    "innocent", "not there", "could not have", "someone can confirm",
]
BELIEF_UNCERTAINTY_MARKERS = [
    "maybe", "perhaps", "not sure", "can't remember", "cannot remember", "don't recall", "hard to say",
]
BELIEF_DEFLECTION_MARKERS = [
    "what about", "the real question is", "we should ask", "look at", "consider", "focus on",
]
TOP_N_ACCUSATION_CANDIDATES = 3


def _retry_with_backoff(func, max_retries: int = 5, base_delay: float = 2.0):
    """
    Retry a function with exponential backoff on rate limit errors.
    Extracts wait time from error message if available.
    """
    # NOTE: do NOT use `except Exception as e` and then reference `e` after the
    # loop — Python 3 (PEP 3110) deletes the `as` target at the end of each
    # except block, so `e` would be unbound when the final raise executes.
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exc = e          # capture before Python deletes `e`
            error_str = str(e)
            # Check if it's a rate limit error
            if '429' in error_str or 'rate_limit' in error_str.lower():
                # Try to extract wait time from error message
                wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 0.5  # Add buffer
                else:
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff

                print(f"  Rate limit hit. Waiting {wait_time:.1f}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # Not a rate limit error, re-raise immediately
                raise
    # All retries exhausted by rate-limit errors
    raise Exception(f"Max retries ({max_retries}) exceeded for rate limit") from last_exc


class ThinkResult(BaseModel):
    thought: str
    action: Literal["speak", "listen"]
    importance: int = Field(ge=0, le=9)
    reason_type: str = Field(default="no_move")

    @model_validator(mode="before")
    @classmethod
    def extract_embedded_fields(cls, data: Any) -> Any:
        """
        Some LLMs embed Importance / Reason_type / Action as plain text inside
        the `thought` value instead of emitting them as sibling JSON keys:

            {"thought": "... I should ask.\n\nImportance: 7\n\nReason_type: question"}

        Extract those inline annotations and promote them to proper fields so
        that the rest of the validators can operate normally.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        thought = out.get("thought", "")
        if not isinstance(thought, str):
            return out

        needs_action = "action" not in out
        needs_importance = "importance" not in out

        if needs_importance or needs_action:
            # --- Importance: N ---------------------------------------------------
            if needs_importance:
                imp_m = re.search(r'\bimportance\s*[:=]\s*(\d+)', thought, re.IGNORECASE)
                if imp_m:
                    out["importance"] = int(imp_m.group(1))
                    thought = re.sub(
                        r'\n*\s*\bimportance\s*[:=]\s*\d+\s*', ' ', thought, flags=re.IGNORECASE
                    ).strip()

            # --- Reason_type: xxx ------------------------------------------------
            if "reason_type" not in out:
                rt_m = re.search(r'\breason_?type\s*[:=]\s*(\w+)', thought, re.IGNORECASE)
                if rt_m:
                    out["reason_type"] = rt_m.group(1)
                    thought = re.sub(
                        r'\n*\s*\breason_?type\s*[:=]\s*\w+\s*', ' ', thought, flags=re.IGNORECASE
                    ).strip()

            # --- Action: xxx -----------------------------------------------------
            if needs_action:
                act_m = re.search(r'\baction\s*[:=]\s*(\w+)', thought, re.IGNORECASE)
                if act_m:
                    out["action"] = act_m.group(1)
                    thought = re.sub(
                        r'\n*\s*\baction\s*[:=]\s*\w+\s*', ' ', thought, flags=re.IGNORECASE
                    ).strip()

            out["thought"] = re.sub(r'\s+', ' ', thought).strip()

        # --- Infer action from reason_type if still missing ----------------------
        if "action" not in out:
            rt = str(out.get("reason_type", "")).lower()
            speak_reasons = {
                "question", "direct_response", "clue", "alibi", "motive", "means",
                "opportunity", "timeline", "contradiction", "self_defense", "redirection",
                "accusation", "continuation", "introduction",
            }
            out["action"] = "speak" if rt in speak_reasons else "listen"

        # --- Default importance if still missing ---------------------------------
        out.setdefault("importance", 5)

        return out

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            if normalized in {"question", "ask", "probe", "challenge", "respond", "answer", "accuse", "claim", "deflect", "redirect"}:
                return "speak"
            if normalized in {"wait", "waiting", "silent", "silence", "pass", "observe"}:
                return "listen"
            return normalized
        return value

    @field_validator("reason_type", mode="before")
    @classmethod
    def normalize_reason_type(cls, value):
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_").replace("-", "_")
        return "no_move"


class ClueRecallResult(BaseModel):
    """Pre-accusation clue recall probe — measures what each agent actually remembers."""
    recalled_clues: List[str] = Field(
        default_factory=list,
        description="List of specific clues or pieces of evidence you remember being revealed by the Game Master during the investigation.",
    )
    total_recalled: int = Field(default=0, ge=0, description="How many distinct clues you believe were revealed.")
    most_important_clue: str = Field(default="", description="The single most important clue for identifying the murderer.")
    clue_based_suspect: str = Field(default="", description="Based on the clues alone, who do they point to most strongly?")

    @field_validator("recalled_clues", mode="before")
    @classmethod
    def normalize_recalled_clues(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip(" -•\t") for part in value.split("\n") if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class AccusationResult(BaseModel):
    reasoning: str = Field(description="Short final reasoning for your accusation")
    accused: str = Field(description="The name of the person you accuse of being the murderer")
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in your accusation from 0 to 100")
    primary_basis: Literal["motive", "means", "opportunity", "timeline", "alibi", "contradiction", "behavior", "mixed"] = "mixed"
    evidence_items: List[str] = Field(default_factory=list, description="2-4 concise evidence items supporting the accusation")
    motive_case: str = Field(default="", description="What motive, if any, points toward the accused")
    means_case: str = Field(default="", description="What means or weapon access points toward the accused")
    opportunity_case: str = Field(default="", description="What opportunity or timeline access points toward the accused")
    contradiction_case: str = Field(default="", description="Any contradiction or inconsistency that points toward the accused")
    comparative_case: str = Field(default="", description="Why this suspect is a stronger case than the main alternatives")
    uncertainty: str = Field(default="", description="Main remaining uncertainty or weakness in the case")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        if isinstance(value, (int, float)):
            return max(0, min(100, int(value)))
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "very low": 10,
                "low": 25,
                "medium": 50,
                "moderate": 50,
                "medium-high": 65,
                "medium high": 65,
                "fairly high": 70,
                "high": 80,
                "very high": 95,
            }
            if normalized in mapping:
                return mapping[normalized]
            digits = re.search(r"\d+", normalized)
            if digits:
                return max(0, min(100, int(digits.group(0))))
        return 50

    @field_validator("primary_basis", mode="before")
    @classmethod
    def normalize_primary_basis(cls, value):
        """Coerce composite / comma-separated values to 'mixed', and fix any case issues.

        Uses an ordered tuple (not a set) for the partial-match fallback so that the
        result is deterministic regardless of PYTHONHASHSEED.  Priority: more specific
        categories (motive, means, …) before the catch-all 'mixed'.
        """
        # Ordered by specificity — checked in this exact order for partial matches.
        # Using a tuple (not a set) guarantees deterministic results across Python runs.
        _ORDERED = (
            "contradiction", "timeline", "alibi",
            "motive", "means", "opportunity", "behavior",
            "mixed",
        )
        _ALLOWED_SET = frozenset(_ORDERED)  # O(1) exact-match lookup
        if not isinstance(value, str):
            return "mixed"
        cleaned = value.strip().lower()
        if cleaned in _ALLOWED_SET:
            return cleaned
        # Comma-separated or slash-separated multi-value string → "mixed"
        if "," in cleaned or "/" in cleaned:
            return "mixed"
        # Partial match: iterate in defined priority order (deterministic)
        for allowed in _ORDERED:
            if allowed in cleaned:
                return allowed
        return "mixed"

    @field_validator("uncertainty", mode="before")
    @classmethod
    def coerce_uncertainty_to_str(cls, value):
        """LLMs sometimes return an integer uncertainty score; coerce to string."""
        if isinstance(value, (int, float)):
            return str(value)
        return value

    @field_validator("evidence_items", mode="before")
    @classmethod
    def normalize_evidence_items(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            parts = [part.strip(" -•\t") for part in value.split("\n") if part.strip()]
            return parts[:4]
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized[:4]
        return []


class Agent:
    def __init__(self, name: str, persona: str, llm: Any, roles_dir: Path, is_murderer: bool = False, scenario: Optional[ScenarioConfig] = None, murderer_behavior_mode: str = "passive_concealment"):
        self.name = name
        self.base_persona = persona
        self.persona = persona  # Will be updated with round info
        self.llm = llm
        self.llm_think = llm.with_structured_output(ThinkResult, method="json_mode")
        self.roles_dir = roles_dir
        self.is_murderer = is_murderer
        self.murderer_behavior_mode = murderer_behavior_mode
        self.scenario = scenario or ScenarioConfig()
        self.current_round = 0
        self.accumulated_knowledge = ""  # Knowledge accumulated across rounds
        self.confession = ""  # Loaded after accusation phase
        self.murderer_strategy = ""
        self.questions_asked_to: set = set()  # Track who we've asked questions to (can only ask each agent once)
        self.facts_revealed: List[str] = []  # Track facts this agent has already revealed
        self.topics_discussed: set = set()  # Track topics to avoid repetition
        self.last_belief_snapshot: Dict[str, Any] = {}
        self.last_accusation_context: Dict[str, Any] = {}
        # Private per-agent suspicion history — never shared with other agents
        self.private_suspicion_history: List[Any] = []

        # Initialize three-stage memory system
        from memory.agent_memory import AgentMemory, SharedHistory
        self.memory = AgentMemory(agent_name=name, scenario=self.scenario)
    
    def _get_own_statements(self, history: List[dict]) -> List[str]:
        """Extract this agent's previous statements from conversation history."""
        return [msg["text"] for msg in history if msg.get("speaker") == self.name]
    
    def _summarize_revealed_info(self, history: List[dict]) -> str:
        """Summarize what this agent has already revealed to avoid repetition."""
        own_statements = self._get_own_statements(history)
        if not own_statements:
            return ""
        
        summary_lines = ["YOU HAVE ALREADY SAID (DO NOT REPEAT):"]
        for i, stmt in enumerate(own_statements[-5:], 1):  # Last 5 statements
            # Truncate long statements
            truncated = stmt[:100] + "..." if len(stmt) > 100 else stmt
            summary_lines.append(f"  {i}. {truncated}")
        
        return "\n".join(summary_lines)
    
    def update_memory(self, state: dict):
        """Update memory views without replaying dialogue already stored by the graph."""
        history = state.get("history", [])
        self.memory.update_from_history(history)
    
    def add_clue_to_memory(self, clue: str):
        """Add a game master clue to long-term memory."""
        self.memory.long_term.add_clue(clue)
    
    def add_fact_to_memory(self, fact: str, turn_id: int = 0):
        """Add an important fact to long-term memory."""
        self.memory.long_term.add_fact(turn_id, fact)
    
    # Markers indicating that a name in a summary bullet is incriminating.
    # Kept as a class-level tuple so it's shared across all instances.
    _SUMMARY_INCRIMINATING_MARKERS: tuple = (
        "fire escape", "stairwell", "followed", "fled", "planted",
        "lied", "doesn't add up", "does not add up", "inconsistent",
        "no alibi", "unaccounted", "motive", "debt", "opportunity",
        "suspicious", "only person", "sole person",
    )

    def add_round_summary(self, round_num: int, bullets: List[str]):
        """Store bullet point summary of a round in long-term memory and
        propagate incriminating mentions into the knowledge-graph suspicion
        scores.

        Why: the KnowledgeGraph is signal-driven (it increments when a name is
        mentioned in questions / evidence language during live dialogue).  A
        quiet, low-profile suspect like a murderer who successfully deflects
        most direct pressure will accumulate few signals and end up near the
        bottom of the ranking.  Round summaries are LLM-extracted — they encode
        higher-level reasoning that the live signal can miss.  Seeding one point
        per incriminating summary mention gives the belief state a chance to
        catch up before the accusation phase.
        """
        self.memory.long_term.add_round_summary(round_num, bullets)

        # Candidate names = everyone except self who is already tracked.
        candidate_names = [
            n for n in self.memory.knowledge_graph.suspicions.keys()
            if n and n != self.name
        ]
        if not candidate_names:
            return

        for bullet in bullets:
            bullet_lower = bullet.lower()
            for name in candidate_names:
                if name.lower() not in bullet_lower:
                    continue
                for marker in self._SUMMARY_INCRIMINATING_MARKERS:
                    if marker in bullet_lower:
                        self.memory.knowledge_graph.update_suspicion(
                            name, 1, f"R{round_num} summary: {bullet[:60]}"
                        )
                        break  # one boost per bullet per name
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        """Update suspicion level for a person in knowledge graph."""
        if not target or target == self.name:
            return
        self.memory.knowledge_graph.update_suspicion(target, delta, reason)

    def _belief_candidate_names(self, all_agents: List[str]) -> List[str]:
        return [name for name in all_agents if name != self.name]

    def observe_utterance(self, utterance: dict, all_agents: List[str]) -> dict:
        speaker = utterance.get("speaker", "")
        if not speaker or speaker == "Game Master":
            return self.export_belief_snapshot(
                turn=utterance.get("turn"),
                round_num=utterance.get("round"),
                stage=utterance.get("stage"),
                context="post_utterance",
                observed_speaker=speaker or None,
                all_agents=all_agents,
            )

        candidates = self._belief_candidate_names(all_agents)
        self.memory.knowledge_graph.ensure_targets(candidates)

        text = utterance.get("text", "")
        text_lower = text.lower()
        question = bool(utterance.get("is_question")) or is_question(text)
        addressed_to = utterance.get("addressed_to")
        mentioned_agents = list(dict.fromkeys(utterance.get("mentioned_agents") or extract_mentions(text, [name for name in all_agents if name != speaker])))
        response_to_speaker = utterance.get("response_to_speaker")

        pressure_signal = any(marker in text_lower for marker in BELIEF_EVIDENCE_MARKERS)
        accusation_signal = any(marker in text_lower for marker in BELIEF_ACCUSATION_MARKERS)
        defensive_signal = any(marker in text_lower for marker in BELIEF_DEFENSIVE_MARKERS)
        uncertainty_signal = any(marker in text_lower for marker in BELIEF_UNCERTAINTY_MARKERS)
        deflection_signal = any(marker in text_lower for marker in BELIEF_DEFLECTION_MARKERS)
        concrete_alibi_signal = speaker != self.name and (
            ("i was" in text_lower or "with me" in text_lower or "someone can confirm" in text_lower)
            and any(token in text_lower for token in ["am", "pm", "before", "after", "with", "saw", "arrived", "left"])
            and not uncertainty_signal
        )

        belief_reasons: List[tuple[str, int, str]] = []

        for target in mentioned_agents:
            if not target or target == self.name:
                continue
            delta = 0
            reason_bits: List[str] = []
            if question:
                delta += 2
                reason_bits.append("was directly questioned")
            if addressed_to == target and question:
                delta += 2
                reason_bits.append("was the focus of a direct response-seeking question")
            if pressure_signal:
                delta += 2
                reason_bits.append("was linked to evidence, motive, or contradiction language")
            if accusation_signal:
                delta += 2
                reason_bits.append("was framed in explicitly accusatory language")
            if delta > 0:
                belief_reasons.append((target, delta, f"Turn {utterance.get('turn')}: {speaker} {'; '.join(reason_bits)}."))

        if speaker != self.name:
            speaker_delta = 0
            speaker_reasons: List[str] = []
            if response_to_speaker and defensive_signal:
                speaker_delta += 2
                speaker_reasons.append("gave a defensive denial or alibi when challenged")
            if response_to_speaker and uncertainty_signal:
                speaker_delta += 1
                speaker_reasons.append("sounded uncertain while answering pressure")
            if response_to_speaker and deflection_signal:
                speaker_delta += 2
                speaker_reasons.append("deflected instead of answering cleanly")
            if concrete_alibi_signal:
                speaker_delta -= 1
                speaker_reasons.append("offered a more concrete alibi with specifics")
            if speaker_delta != 0:
                belief_reasons.append((speaker, speaker_delta, f"Turn {utterance.get('turn')}: {speaker} {'; '.join(speaker_reasons)}."))

        for target, delta, reason in belief_reasons:
            self.update_suspicion(target, delta, reason)

        return self.export_belief_snapshot(
            turn=utterance.get("turn"),
            round_num=utterance.get("round"),
            stage=utterance.get("stage"),
            context="post_utterance",
            observed_speaker=speaker,
            all_agents=all_agents,
        )

    def export_belief_snapshot(
        self,
        turn: Optional[int] = None,
        round_num: Optional[int] = None,
        stage: Optional[str] = None,
        context: str = "live",
        observed_speaker: Optional[str] = None,
        all_agents: Optional[List[str]] = None,
    ) -> dict:
        candidates = self._belief_candidate_names(all_agents or [])
        snapshot = self.memory.knowledge_graph.build_snapshot(candidate_names=candidates or None, top_k=max(5, len(candidates)))
        belief_snapshot = {
            "turn": turn,
            "round": round_num,
            "stage": stage,
            "context": context,
            "observed_speaker": observed_speaker,
            "agent": self.name,
            **snapshot,
        }
        self.last_belief_snapshot = belief_snapshot
        return belief_snapshot

    def render_belief_summary(self, all_agents: List[str], top_k: int = 5) -> str:
        snapshot = self.export_belief_snapshot(all_agents=all_agents, context="prompt")
        ranking = snapshot.get("ranking", [])[:top_k]
        if not ranking:
            return "No stable belief ranking yet."

        lines = ["Belief state:"]
        for row in ranking:
            reasons = row.get("reasons") or []
            reason_text = f" | reasons: {'; '.join(reasons[:2])}" if reasons else ""
            lines.append(f"  {row.get('rank')}. {row.get('name')} ({row.get('score')}){reason_text}")
        lines.append(f"Uncertainty level: {snapshot.get('uncertainty', 100)}/100")
        return "\n".join(lines)
    
    def export_memory_snapshot(self) -> dict:
        facts = self.memory.long_term.facts[-24:]
        evidence_tags = list(getattr(self.scenario, "evidence_tags", ["motive", "means", "opportunity", "contradiction", "timeline", "alibi"]))
        categorized = {}
        for tag in evidence_tags:
            categorized[tag] = [fact.fact_text for fact in facts if tag in fact.tags][-6:]

        uncategorized = [fact.fact_text for fact in facts if not any(tag in fact.tags for tag in evidence_tags)]
        suspicions = []
        for name, level, reasons in self.memory.knowledge_graph.get_ranked_suspects()[:5]:
            suspicions.append({
                "name": name,
                "level": level,
                "reasons": reasons[:3],
            })

        return {
            "categories": categorized,
            "other": uncategorized[-6:],
            "suspicions": suspicions,
            "belief_state": self.last_belief_snapshot or self.export_belief_snapshot(context="memory_snapshot"),
        }

    def update_round(self, round_num: int):
        """Update agent's knowledge with new round information."""
        from utils.agent_helper import load_round_description
        
        if round_num == self.current_round:
            return  # Already on this round

        self.current_round = round_num
        # Reset per-round questioning constraints so agents can keep probing each other
        # across rounds.  Without this, after enough turns the prompt shows "Can ask: NO ONE"
        # and questioning collapses, starving RQ2 of interaction signal.
        self.questions_asked_to = set()
        round_desc = load_round_description(self.roles_dir, self.name, round_num)
        
        if round_desc:
            self.accumulated_knowledge += f"\n\n=== ROUND {round_num} INFORMATION ===\n{round_desc}"
            
            # Update persona with accumulated knowledge
            self.persona = f"{self.base_persona}\n\n{self.accumulated_knowledge}"
            
            # If murderer, append behavior-mode-specific instructions.
            if self.is_murderer:
                if self.murderer_behavior_mode == "active_deception":
                    self.persona += (
                        "\n\n[SECRET: You are the murderer. You know you killed the victim. "
                        "Do NOT confess or reveal this under any circumstances. "
                        "Your goal is to avoid being accused while appearing cooperative and credible. "
                        "Answer questions naturally and specifically — vague non-answers look suspicious. "
                        "If you are directly suspected, defend yourself calmly and provide a plausible account.\n"
                        "You are also permitted to use the following deceptive tactics when useful:\n"
                        "- direct denial when accused (\"I didn't do it\", \"it wasn't me\")\n"
                        "- alibi claims that locate you elsewhere at the time of the murder\n"
                        "- deflection that changes the subject to another suspect (\"the real question is...\", \"what about X?\")\n"
                        "- uncertainty seeding (\"maybe\", \"perhaps\", \"we can't rule out\", \"anyone could have\")\n"
                        "- selective disclosure (short answers, boundary-setting like \"that's all I know\")\n"
                        "- accusation redirection (point at another suspect's motive, means, or opportunity)\n"
                        "- evidence reframing (offer an alternative interpretation of incriminating evidence)\n"
                        "Use these tactics naturally and sparingly — overuse looks suspicious. Mix in cooperative-sounding statements.]"
                    )
                    if not self.murderer_strategy:
                        self.load_murderer_strategy()
                    if self.murderer_strategy:
                        self.persona += f"\n\n[MURDERER STRATEGY]\n{self.murderer_strategy}"
                else:
                    # passive_concealment: protect yourself naturally, no explicit tactic coaching
                    self.persona += (
                        "\n\n[SECRET: You are the murderer. You know you killed the victim. "
                        "Do NOT confess or reveal this under any circumstances. "
                        "Your goal is to avoid being accused while appearing cooperative and credible. "
                        "Answer questions naturally and specifically — vague non-answers look suspicious. "
                        "If you are directly suspected, defend yourself calmly and provide a plausible account. "
                        "Act as an innocent person would: engage genuinely with the investigation, "
                        "express concern, and protect yourself when necessary.]"
                    )
            
            print(f"   {self.name}: Loaded round {round_num} knowledge (total accumulated: {len(self.accumulated_knowledge)} chars)")
    
    def get_knowledge_summary(self) -> str:
        """Get a summary of the agent's accumulated knowledge for debugging."""
        rounds_loaded = self.accumulated_knowledge.count("=== ROUND")
        return f"{self.name}: {rounds_loaded} rounds of knowledge, current_round={self.current_round}"
    
    def load_confession(self):
        """Load the character's confession text."""
        from utils.agent_helper import load_confession
        self.confession = load_confession(self.roles_dir, self.name)
        return self.confession

    def load_murderer_strategy(self):
        """Load optional murderer strategy guidance for a character."""
        from utils.agent_helper import load_murderer_strategy
        self.murderer_strategy = load_murderer_strategy(self.roles_dir, self.name)
        return self.murderer_strategy

    def _format_history(self, history: List[dict]) -> str:
        """Use shared history window for prompts (last K_HISTORY turns only)."""
        return self.memory.shared_history.render_for_prompt()

    def _load_speaking_style(self) -> str:
        """Load per-character speaking style from speaking_style.md, falling back to a generic prompt."""
        if self.roles_dir:
            style_path = Path(self.roles_dir) / self.name.lower().replace(" ", "-") / "speaking_style.md"
            if style_path.exists():
                try:
                    return style_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return "Be selective. Speak only when your move clearly improves the investigation or your position."

    def think(self, state: GameState) -> ThinkResult:
        self.update_memory(state)
        
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        memory_context = self.memory.build_prompt_context()
        participant_names = [msg.get("speaker") for msg in state.get("history", []) if msg.get("speaker") and msg.get("speaker") != "Game Master"]
        known_participants = sorted(set(participant_names))
        if not known_participants:
            known_participants = sorted([name for name in state.get("thoughts", {}).keys() if name != self.name])
        participants_context = ", ".join([self.name] + [name for name in known_participants if name != self.name])
        pending = state.get("pending_obligation")
        directly_addressed = bool(pending and pending.get("addressee") == self.name)
        last_speaker = state.get("last_speaker")
        recently_spoke = last_speaker == self.name
        own_recent_statements = self._summarize_revealed_info(state.get("history", []))
        history = state.get("history", [])
        recent_turns = history[-6:]
        recent_speakers = [msg.get("speaker") for msg in recent_turns if msg.get("speaker") and msg.get("speaker") != "Game Master"]
        recent_speaker_count = sum(1 for speaker in recent_speakers if speaker == self.name)
        just_spoke_within_window = recent_speaker_count > 0
        non_gm_speakers = [speaker for speaker in recent_speakers if speaker]
        consecutive_recent = 0
        for speaker in reversed(non_gm_speakers):
            if speaker == self.name:
                consecutive_recent += 1
            else:
                break

        strategy_guidance = self._load_speaking_style()

        if phase == "introduction" and current_round == 1:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_status_line}
You are one of the suspects and participants in the mystery.
The people present are: {participants_context}.
Your immediate goal is to introduce yourself naturally without revealing secrets that should stay hidden for now.
You should sound like a real person at the scene, not a narrator or analyst."""),
                HumanMessage(content=f"""{memory_context}

People currently in the room: {participants_context}

Actions:
• Speak: Introduce yourself to the group in a way that fits your role.
• Listen: Wait if you genuinely have nothing to add.

Importance guidance:
- 9 = it is clearly your turn to introduce yourself
- 4-6 = you can add something useful
- 0-3 = better to wait

Return valid JSON with keys: thought, action, importance, reason_type.
Use reason_type="introduction" if you should speak, otherwise "no_move"."""),
            ]
        else:
            if self.is_murderer:
                if current_round <= 2:
                    deception_style = (
                        "You are in the early investigation. Your priority is credibility. "
                        "Act cooperative and appear genuinely rattled by the death — NOT eager to accuse. "
                        "Focus on establishing your alibi naturally. Volunteering accusations this early looks suspicious. "
                        "Only deflect or redirect if someone directly pressures you."
                    )
                elif current_round <= 4:
                    deception_style = (
                        "The investigation is building. Begin carefully steering suspicion toward others when natural opportunities arise. "
                        "You can start asking pointed questions about specific suspects, but frame it as genuine concern, not deflection. "
                        "If dangerous evidence comes up, get ahead of it calmly."
                    )
                else:
                    deception_style = (
                        "Evidence is mounting and you are running out of time. "
                        "Escalate misdirection — build a concrete case against a specific suspect and push the group toward them. "
                        "You can be more aggressive now because staying passive looks like you have nothing to offer."
                    )
                goals_block = f"""You are an in-world suspect AND the actual murderer of {self.scenario.victim_name}.
Your goals are:
1. AVOID being accused as the murderer at the end of the game.
2. Appear cooperative and credible — a guilty person who accuses others too eagerly or too early looks obviously guilty.
3. Lie confidently about your timeline and movements when the truth would expose you.
4. Only reference information that has been said aloud in the conversation or publicly revealed by the Game Master — referencing private or unrevealed knowledge exposes you.
5. Steer suspicion toward others proportionately to how much the investigation is focusing on you.
6. Do NOT confess or volunteer incriminating details.

{deception_style}"""
            else:
                goals_block = f"""You are an in-world suspect trying to help the group determine who killed {self.scenario.victim_name}.
Your goals are:
1. identify the murderer from dialogue, clues, motives, means, opportunity, contradictions, and timelines,
2. ask sharp questions when you lack crucial information,
3. reveal relevant facts you know when it helps the investigation,
4. protect your private secrets unless they are directly challenged or necessary to defend yourself,
5. avoid wasting turns on generic filler.

Think like an investigator with partial information, not like a chatbot.
Your job is NOT to speak every turn. Good investigation also means waiting when another person has a stronger next move."""

            # CACHEABLE PREFIX: everything that is stable within a round
            # (agent identity, persona, round-bucketed strategy, decision frame,
            # action/importance/JSON spec) goes into SystemMessage so the local
            # LLM / OpenAI prefix cache can reuse it across all turns of the round.
            if self.is_murderer and current_round <= 2:
                decision_frame = (
                    "MURDERER DECISION FRAME (early rounds — appear cooperative): You are guilty but the group doesn't know it yet. "
                    "Do not eagerly take the floor. Speak when directly addressed, when you can volunteer a plausible alibi detail, or when dangerous topics come up and you need to shape how they land. "
                    "Silence in early rounds is fine — trying too hard to redirect looks worse than saying nothing."
                )
            elif self.is_murderer and current_round <= 4:
                decision_frame = (
                    "MURDERER DECISION FRAME (mid investigation — controlled misdirection): Suspicion is building. "
                    "Take the floor when someone is discussing a dangerous topic (timeline, office, fire escape) so you can redirect it. "
                    "You can start asking pointed questions about other suspects, but frame it as curiosity, not deflection."
                )
            elif self.is_murderer:
                decision_frame = (
                    "MURDERER DECISION FRAME (late game — go on offence): Evidence is closing in. Take the floor aggressively. "
                    "Build a concrete case against a specific suspect and push the group toward them. "
                    "Every turn matters now — silence helps others build a case against you."
                )
            else:
                decision_frame = "Decide whether YOU specifically should speak now."

            if self.is_murderer:
                listen_default_lines = ""
                self_questions = (
                    "- Was I directly addressed and must respond?\n"
                    "- Is a dangerous clue or topic coming up that I need to quietly shape?\n"
                    "- Do I have a plausible alibi detail I haven't given yet, or a fabricated observation ready?"
                )
            else:
                listen_default_lines = (
                    "Default to listening unless there is a strong reason to take the floor.\n"
                    "Usually only one or two agents should strongly want to speak on a given turn.\n"
                    "If another suspect likely has a better next move, choose listen.\n"
                    "Strategic silence is often the correct move."
                )
                self_questions = (
                    "- Was I directly addressed and therefore must respond?\n"
                    "- Do I have a concrete fact about motive, means, opportunity, location, timing, or contradiction that has not already been surfaced?\n"
                    "- Can I ask a targeted question that materially narrows the suspect list?"
                )

            system_static = f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.
{goals_block}

{decision_frame}
{listen_default_lines}

When deciding whether to speak, ask yourself:
{self_questions}
- Would speaking expose me unnecessarily or draw attention without payoff?
- Am I repeating, interrupting, or talking just because there is space to fill?
- Did I already speak recently, meaning I should usually stay quiet now?
- Would my personality prefer caution, deflection, opportunism, or confrontation in this moment?

Actions:
• Speak: contribute evidence, challenge someone, clarify an alibi, or ask a targeted question.
• Listen: if your contribution would be repetitive, weak, premature, strategically costly, or better made by someone else.

Importance guidance:
- 9 = directly addressed / must answer now
- 7-8 = strong bid to speak because you have a clear next move
- 5-6 = moderate bid to speak if useful
- 3-4 = weak bid; usually better to listen
- 0-2 = no bid / definitely listen

Treat the score as a bid strength for the next turn, not a vague confidence score.
Be willing to choose listen with low scores. Do not inflate urgency just to participate.

Return valid JSON with keys: thought, action, importance, reason_type.
`action` must be exactly one of: "speak" or "listen".
If you want to ask a question, challenge someone, answer someone, redirect, or accuse, that still counts as action="speak".
Use reason_type="question" for investigative questions rather than putting "question" in the action field.
Use one reason_type from: direct_response, contradiction, clue, alibi, motive, means, opportunity, timeline, question, self_defense, redirection, continuation, weak_followup, no_move."""

            msgs = [
                SystemMessage(content=system_static),
                HumanMessage(content=f"""{memory_context}

Status:
- Directly addressed: {directly_addressed}
- You just spoke last turn: {recently_spoke}
- You spoke {recent_speaker_count} times in the last 6 non-GM utterances
- Consecutive recent turns by you: {consecutive_recent}
- Strategic style: {strategy_guidance}

{own_recent_statements}

Now decide whether to speak or listen this turn and return the JSON object as specified."""),
            ]
        
        try:
            result = _retry_with_backoff(lambda: self.llm_think.invoke(msgs))

            if phase == "introduction":
                if result.action == "speak":
                    result.reason_type = "introduction"
                else:
                    result.reason_type = "no_move"
            else:
                if directly_addressed:
                    result.action = "speak"
                    result.importance = 9
                    result.reason_type = "direct_response"
                else:
                    if recently_spoke:
                        result.importance = max(0, result.importance - 2)
                    if recent_speaker_count >= 1:
                        result.importance = max(0, result.importance - 1)
                    if recent_speaker_count >= 2:
                        result.importance = max(0, result.importance - 1)
                    if consecutive_recent >= 1:
                        result.importance = max(0, result.importance - 1)

                    if self.is_murderer:
                        # Dampen weak/exposing bids that would draw scrutiny without payoff.
                        if result.reason_type in {"weak_followup", "no_move", "continuation"}:
                            result.importance = max(0, result.importance - 1)
                        # Strongly boost deception-aligned and proactive-accusation bids so the
                        # murderer reliably takes the floor to redirect, accuse, or plant doubt.
                        if result.reason_type in {"redirection", "alibi", "self_defense", "contradiction", "accusation", "question", "motive"}:
                            result.importance = min(9, result.importance + 2)

                    if result.action == "speak" and result.importance <= 2:
                        result.action = "listen"
                    elif result.action == "listen" and result.importance >= 7:
                        result.action = "speak"

                    if result.action == "listen" and result.importance <= 2:
                        result.reason_type = "no_move"
                    elif result.action == "listen" and result.reason_type == "no_move":
                        result.reason_type = "weak_followup"

            self.memory.add_thought(result.thought, result.action, result.importance)
            return result
        except Exception as e:
            print(f"Error in think for {self.name}: {e}", file=__import__('sys').stderr)
            return ThinkResult(thought="waiting", action="listen", importance=0, reason_type="no_move")

    def speak(self, state: GameState, response_constraint: Optional[str]) -> str:
        other_agents = [name for name in state.get("thoughts", {}).keys() if name != self.name]
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Build memory context using three-stage system
        memory_context = self.memory.build_prompt_context()
        participant_names = [msg.get("speaker") for msg in state.get("history", []) if msg.get("speaker") and msg.get("speaker") != "Game Master"]
        known_participants = sorted(set(participant_names))
        if not known_participants:
            known_participants = sorted([name for name in state.get("thoughts", {}).keys() if name != self.name])
        participants_context = ", ".join([self.name] + [name for name in known_participants if name != self.name])
        constraint = f"\nRESPOND TO: {response_constraint}" if response_constraint else ""
        
        # Build list of agents we can still ask questions to
        can_ask = [name for name in other_agents if name not in self.questions_asked_to]
        can_ask_str = ", ".join(can_ask) if can_ask else "NO ONE"
        
        # Round 1: Introduction phase
        if phase == "introduction" and current_round == 1:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_status_line}
Introduce yourself.
The people present are: {participants_context}.

You are speaking aloud IN CHARACTER inside the mystery world.
Do NOT narrate yourself.
Do NOT write things like '{self.name} says', '{self.name} speaks', 'I would say', or stage directions.
Output only the exact words your character says to the group."""),
                HumanMessage(content=f"""{memory_context}

People currently in the room: {participants_context}

Introduce yourself in first person (1-2 sentences).
Base the introduction on the character description above, especially your public-facing role, social position, and emotional tone.
Say something distinctive that fits who you are instead of giving a generic placeholder introduction.
Do not summarize your entire backstory, but do signal the kind of person you are.
Speak naturally to the other suspects.
Output dialogue only."""),
            ]
        else:
            # CACHEABLE PREFIX: stable within a round. Everything that depends
            # only on agent identity, persona, and round bucket goes into the
            # SystemMessage so the prefix cache hits across all turns of the round.
            # Per-turn dynamic content (memory_context, constraint, can_ask) stays
            # in the HumanMessage.
            if self.is_murderer:
                if current_round <= 2:
                    accusation_guidance = (
                        "In these early rounds, do NOT volunteer accusations unprompted — that looks suspicious. "
                        "Focus on giving your cover-story alibi naturally and appearing cooperative. "
                        "Only redirect suspicion if directly challenged."
                    )
                elif current_round <= 4:
                    accusation_guidance = (
                        "You may begin pointing suspicion toward others when the opportunity feels natural. "
                        "Ask questions about specific suspects that highlight their motive or suspicious behavior. "
                        "Keep it measured — frame it as genuine investigation, not deflection."
                    )
                else:
                    accusation_guidance = (
                        "Build an active case against a specific suspect. Name them, cite evidence already revealed, and push the group toward them."
                    )
                purpose_block = f"""Your purpose is to survive this investigation without being identified as the murderer.
Lie confidently when needed, appear cooperative, and gradually steer the group toward other suspects as the evidence mounts."""
                static_rules_block = f"""Rules:
- You MAY lie about your movements, timing, and what you personally saw or heard.
- INFORMATION GATE: Only reference information already spoken aloud in conversation or announced by the Game Master. Do NOT mention crime scene details or planted evidence before the Game Master has revealed it — knowing unrevealed information exposes you as the killer.
- Do NOT confess or admit you killed {self.scenario.victim_name}.
- {accusation_guidance}
- CRITICAL — when directly asked about your whereabouts, timeline, or actions: answer with a specific, plausible lie first. Do NOT dodge the question — a confident wrong answer is less suspicious than no answer. Then you may redirect.
- Plant seeds of doubt: suggest alternative explanations for evidence that could implicate you.
- Do not contradict hard clue facts the group has already heard — that is a red flag.
- No exact repetition of what you've already said.
- Stay in character. Speak in first person.
- If asking a question, ask exactly ONE targeted question total in this turn.
- Never ask multiple questions to multiple people in the same utterance."""
            else:
                purpose_block = "Your purpose is to help the group identify the murderer by surfacing relevant facts, probing suspicious people, testing alibis, and narrowing the suspect list.\nYou should prioritize dialogue that advances the investigation."
                static_rules_block = f"""Rules:
- Only use facts from your knowledge.
- Reveal what you know when it helps identify the murderer.
- Protect deeper secrets unless challenged, but do not become uselessly vague.
- Focus on motive, means, opportunity, timeline, location, contradictions, and suspicious behavior.
- No repetition.
- Stay in character.
- Speak in first person when appropriate.
- If asking a question, ask exactly ONE targeted investigative question total in this turn.
- Never ask multiple questions to multiple people in the same utterance.
- If answering, answer the point directly before adding pressure on another suspect if relevant."""

            system_static = f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.

You are speaking aloud IN CHARACTER inside the mystery world.
{purpose_block}

Do NOT narrate yourself.
Do NOT write things like '{self.name} says', '{self.name} speaks next', 'I say:', or any quoted script format.
Do NOT use markdown, bullet points, stage directions, or speaker labels.
Output only the exact words your character says out loud.

{static_rules_block}

Respond in 1-2 natural sentences.
If you ask a question, you may include at most one question mark in the entire utterance and it must refer to a single target.
Do not stack questions, follow-up questions, or question a second person in the same turn.
Output dialogue only."""

            msgs = [
                SystemMessage(content=system_static),
                HumanMessage(content=f"""{memory_context}{constraint}

Can ask: {can_ask_str}

Speak now."""),
            ]
        try:
            result = _retry_with_backoff(lambda: self.llm.invoke(msgs))
            response = result.content if result and result.content else "I have nothing new to add."
            
            # Remove quotation marks from response
            response = response.replace('"', '').replace('"', '').replace('"', '')
            
            # Remove action descriptions in parentheses like (looks around) or (sighs nervously)
            response = re.sub(r'\([^)]*\)', '', response).strip()
            # Also remove brackets
            response = re.sub(r'\[[^\]]*\]', '', response).strip()
            # Remove leading meta speech labels like "Bobby Herrerra speaks next:" or "Pauline says:"
            response = re.sub(rf'^{re.escape(self.name)}\s+(speaks(?:\s+next)?|says|asks|replies)\s*[:,-]?\s*', '', response, flags=re.IGNORECASE).strip()
            response = re.sub(r'^[A-Z][A-Za-z\'\- ]{1,40}\s+(speaks(?:\s+next)?|says|asks|replies)\s*[:,-]?\s*', '', response, flags=re.IGNORECASE).strip()
            response = re.sub(r'^(I\s+(say|ask|reply)\s*[:,-]?\s*)', '', response, flags=re.IGNORECASE).strip()
            
            # Remove wrapping quotes if the whole utterance is quoted
            if len(response) >= 2 and response[0] in {'"', '“', '\''} and response[-1] in {'"', '”', '\''}:
                response = response[1:-1].strip()
            
            # Clean up any double spaces left behind
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Track which agents we directly questioned this turn. Only counts when
            # the utterance is actually a question AND addresses a specific suspect —
            # rhetorical questions and bare mentions must not consume the per-round
            # "can ask" budget.
            if is_question(response):
                addressed = detect_direct_address(response, other_agents)
                if addressed:
                    self.questions_asked_to.add(addressed)
            
            return response
        except Exception as e:
            print(f"Error in speak for {self.name}: {e}", file=__import__('sys').stderr)
            return "I need to think about this."

    def generate_round_suspicion_assessment(
        self,
        round_num: int,
        stage: str,
        all_agents: List[str],
    ) -> Optional[Any]:
        """Private end-of-round suspicion assessment. Not public; never added to SharedHistory."""
        from schemas.suspicion import SuspectAssessment, RoundSuspicionAssessment

        other_agents = [name for name in all_agents if name != self.name]
        if not other_agents:
            return None

        llm_suspect = self.llm.with_structured_output(RoundSuspicionAssessment, method="json_mode")
        memory_context = self.memory.build_prompt_context()

        prior_context = ""
        if self.private_suspicion_history:
            lines = ["YOUR PRIOR PRIVATE SUSPICION SNAPSHOTS (for continuity only):"]
            for prior in self.private_suspicion_history[-3:]:
                scores = {sa.suspect: sa.suspicion_score for sa in prior.suspect_assessments}
                scores_str = ", ".join(f"{n}={s}" for n, s in sorted(scores.items()))
                lines.append(
                    f"  Round {prior.round}: top={prior.top_suspect}, "
                    f"uncertainty={prior.overall_uncertainty}/10 | scores: {scores_str}"
                )
            prior_context = "\n".join(lines)

        suspects_list = "\n".join(f"- {name}" for name in other_agents)
        suspects_str = ", ".join(other_agents)

        msgs = [
            SystemMessage(content=f"""You are {self.name}. This is a PRIVATE internal report to the Game Master only.

Do NOT write in dialogue style. Do NOT address other characters.
Write as a detective analyst reporting your current private assessment after Round {round_num} ({stage}).

{self.scenario.victim_name} was murdered. Your private task: assess how suspicious each other suspect is based on everything you have observed so far.

This report is confidential. It will never be shown to other suspects."""),
            HumanMessage(content=f"""{memory_context}

{prior_context}

━━━ PRIVATE SUSPICION ASSESSMENT — Round {round_num} ━━━

Assess EVERY suspect listed below. You must provide one entry for each:
{suspects_list}

Scoring instructions:
• suspicion_score 1–10: 1 = almost certainly innocent, 10 = almost certainly the murderer
• confidence_score 1–10: 1 = you have almost no basis for this score, 10 = strong evidence supports it
• Distribute scores meaningfully — do NOT assign the same score to everyone unless the evidence is truly indistinguishable
• Ground each score in specific publicly revealed evidence: motive, means, opportunity, timeline, contradiction, alibi, or suspicious dialogue behaviour
• Do NOT invent evidence. Base scores only on what has been publicly discussed or revealed by the Game Master
• Explicitly compare suspects against each other — stronger cases get higher scores

evidence_categories must come from: motive, means, opportunity, contradiction, timeline, alibi, behavior

overall_uncertainty: 1 = you are nearly certain who is guilty, 10 = you have almost no idea

top_suspect: the name of the suspect you currently consider most likely to be the murderer (must be one of: {suspects_str})

Return valid structured JSON only. Cover all suspects: {suspects_str}"""),
        ]

        try:
            result = _retry_with_backoff(lambda: llm_suspect.invoke(msgs))
            result.round = round_num
            result.stage = stage
            result.agent = self.name

            assessed = {sa.suspect for sa in result.suspect_assessments}
            for missing in other_agents:
                if missing not in assessed:
                    result.suspect_assessments.append(SuspectAssessment(
                        suspect=missing,
                        suspicion_score=5,
                        confidence_score=1,
                        primary_reason="No specific evidence observed this round.",
                        evidence_categories=[],
                        strongest_supporting_fact="Insufficient evidence observed.",
                    ))

            if result.top_suspect not in other_agents:
                result.top_suspect = max(
                    result.suspect_assessments, key=lambda sa: sa.suspicion_score
                ).suspect

            self.private_suspicion_history.append(result)
            return result

        except Exception as e:
            print(
                f"  Warning: suspicion assessment failed for {self.name} (round {round_num}): {e}",
                file=__import__("sys").stderr,
            )
            return None

    def _build_accusation_summary(self, state: GameState, all_agents: List[str]) -> str:
        history = state.get("history", [])
        suspicion_scores = {name: 0 for name in all_agents if name != self.name}
        evidence_lines: List[str] = []

        for message in history:
            speaker = message.get("speaker", "")
            text = message.get("text", "")
            mentioned = extract_mentions(text, all_agents, exclude=speaker)
            addressed = message.get("addressed_to")
            question = is_question(text) or bool(message.get("is_question"))

            for target in mentioned:
                if target == self.name:
                    continue
                delta = 0
                if question:
                    delta += 2
                if addressed == target and question:
                    delta += 2
                if any(word in text.lower() for word in ["motive", "alibi", "where were", "did you", "why did", "how do you explain", "contradict", "debt", "weapon", "blood", "lied", "suspicious"]):
                    delta += 2
                if delta > 0:
                    suspicion_scores[target] = suspicion_scores.get(target, 0) + delta

            if mentioned and question and len(evidence_lines) < 10:
                evidence_lines.append(f"- {speaker} pressured {', '.join(mentioned)}: {text[:160]}")

        ranked = sorted(suspicion_scores.items(), key=lambda item: item[1], reverse=True)
        top_lines = []
        for idx, (name, score) in enumerate(ranked[:5], 1):
            top_lines.append(f"  {idx}. {name}: interaction-pressure score {score}")

        if not top_lines:
            top_lines.append("  No interaction-pressure signals detected.")

        if not evidence_lines:
            evidence_lines.append("- No targeted pressure turns were detected from the dialogue log.")

        return "\n".join([
            "Interaction-pressure summary (who the discussion focused on):",
            *top_lines,
            "",
            "Targeted pressure examples:",
            *evidence_lines,
        ])

    def accuse(self, state: GameState, all_agents: List[str]) -> AccusationResult:
        """Final accusation - who does this agent think is the murderer? Cannot accuse self."""
        self.update_memory(state)
        
        other_agents = [name for name in all_agents if name != self.name]
        others_str = ", ".join(other_agents)
        
        llm_accuse = self.llm.with_structured_output(AccusationResult, method="json_mode")
        memory_context = self.memory.build_prompt_context()
        suspect_ranking = self.memory.get_suspect_ranking()
        interaction_summary = self._build_accusation_summary(state, all_agents)
        belief_snapshot = self.export_belief_snapshot(
            turn=state.get("turn"),
            round_num=state.get("current_round"),
            stage=state.get("current_stage"),
            context="pre_accusation",
            all_agents=all_agents,
        )
        belief_summary = self.render_belief_summary(all_agents)
        top_ranked = belief_snapshot.get("ranking", [])
        top_n_candidates = [row.get("name") for row in top_ranked[:TOP_N_ACCUSATION_CANDIDATES] if row.get("name")]

        # ── Incorporate private suspicion history into the candidate set ───────
        # The knowledge-graph belief snapshot is built from observed dialogue
        # signals, which may diverge from the LLM's own private assessments.
        # Pull in the top suspects from the most recent private snapshots so
        # that consistently high private scores (e.g. Tim Kane across rounds 4-5)
        # are NOT silently overridden by the interaction-pressure ranking.
        private_suspicion_context = ""
        if self.private_suspicion_history:
            # Aggregate suspicion scores across the last 3 private snapshots
            aggregate: dict[str, list[int]] = {}
            for snap in self.private_suspicion_history[-3:]:
                for sa in snap.suspect_assessments:
                    aggregate.setdefault(sa.suspect, []).append(sa.suspicion_score)
            avg_scores = {name: sum(scores) / len(scores) for name, scores in aggregate.items()}
            private_ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            # Top-N from private assessments
            private_top = [name for name, _ in private_ranked[:TOP_N_ACCUSATION_CANDIDATES] if name in other_agents]
            # Merge: private top candidates take priority
            merged = list(dict.fromkeys(private_top + top_n_candidates))  # preserves order, deduplicates
            top_n_candidates = merged[:TOP_N_ACCUSATION_CANDIDATES]

            # Build a short context block for the prompt
            lines = ["Your private suspicion scores (averaged across recent rounds):"]
            for name, score in private_ranked[:6]:
                if name in other_agents:
                    lines.append(f"  {name}: {score:.1f}/10")
            latest = self.private_suspicion_history[-1]
            lines.append(f"Your most recent top suspect (Round {latest.round}): {latest.top_suspect}")
            private_suspicion_context = "\n".join(lines)
        # ──────────────────────────────────────────────────────────────────────

        top_suspect = top_n_candidates[0] if top_n_candidates else (other_agents[0] if other_agents else "")

        msgs = [
            SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

Investigation OVER. Accuse ONE person.
Your accusation must be grounded in specific evidence from the discussion, clues, contradictions, alibis, motives, means, opportunities, or timeline facts.
Do not give a vague accusation. Build the strongest honest case you can from the information you observed."""),
            HumanMessage(content=f"""{memory_context}

{suspect_ranking}

{belief_summary}

{private_suspicion_context}

{interaction_summary}

Choose murderer from: {others_str}
(Cannot accuse yourself)

Belief-state constraint:
- Your accusation MUST come from these top belief candidates unless the evidence is truly overwhelming otherwise: {', '.join(top_n_candidates) if top_n_candidates else others_str}
- Your current top suspect is: {top_suspect}
- Your private suspicion scores above reflect your own careful reasoning across all rounds — weight them heavily.
- If you accuse someone other than your current top suspect, `comparative_case` must explicitly explain why the evidence for your chosen suspect is stronger than the evidence for {top_suspect}.
- If your uncertainty is high, acknowledge that in `uncertainty`, but still choose the strongest candidate from your belief state.

Important decision rule:
- Your final accusation should align with the strongest evidence and the strongest sustained suspicion from the discussion.
- If one suspect was repeatedly questioned, pressured, challenged, or treated as the main focus, you should usually accuse that suspect unless there is stronger concrete exculpatory evidence.
- Do not switch to a random lower-attention suspect without explicitly explaining the stronger evidence for that switch.

Return valid JSON with keys:
- accused
- reasoning
- confidence
- primary_basis
- evidence_items
- motive_case
- means_case
- opportunity_case
- contradiction_case
- comparative_case
- uncertainty

Requirements:
- `confidence` must be an INTEGER from 0 to 100. Do NOT use words like "high", "medium", or "low".
- `evidence_items` must contain 2 to 4 concise evidence points.
- Use the structured fields even if some are weak; if a dimension is weak, say so briefly.
- Keep `reasoning` to 1-3 sentences.
- `primary_basis` must be one of: motive, means, opportunity, timeline, alibi, contradiction, behavior, mixed.
- Do not invent evidence you do not know."""),
        ]
        try:
            result = _retry_with_backoff(lambda: llm_accuse.invoke(msgs))
            if result.accused not in other_agents:
                for agent in other_agents:
                    if agent.lower() in result.accused.lower() or result.accused.lower() in agent.lower():
                        result.accused = agent
                        break
                else:
                    result.accused = other_agents[0]

            # ── Belief-alignment check (log only — do NOT override the LLM's choice) ──
            # Forcing result.accused = top_suspect here corrupts research data: it
            # inflates the `corrected_to_top_suspect` metric and prevents us from
            # observing how often agents naturally deviate from their belief state
            # (RQ3).  Name validation (accused must be a real agent) is handled by the
            # block above.  Here we only record whether the model strayed from its
            # top-N candidates so analysis can measure belief-alignment authentically.
            accused_outside_top_n = bool(top_n_candidates and result.accused not in top_n_candidates)
            if accused_outside_top_n:
                divergence_note = (
                    f"[Belief-state note: {result.accused} was outside the top-{TOP_N_ACCUSATION_CANDIDATES} "
                    f"belief candidates ({', '.join(top_n_candidates)}); accusation preserved as-is.]"
                )
                result.comparative_case = (
                    (result.comparative_case + " " + divergence_note).strip()
                    if result.comparative_case else divergence_note
                )
            # corrected_to_top_suspect kept as False — we no longer force overrides.
            corrected_to_top_suspect = False

            accused_rank = next((row.get("rank") for row in top_ranked if row.get("name") == result.accused), None)
            if result.accused != top_suspect and top_suspect and not result.comparative_case.strip():
                result.comparative_case = f"I considered {top_suspect} most suspicious overall, but the concrete evidence against {result.accused} is stronger on the decisive point."

            if len(result.evidence_items) < 2:
                fallback_items = [item for item in [result.motive_case, result.means_case, result.opportunity_case, result.contradiction_case, result.comparative_case] if item]
                if not fallback_items and result.reasoning:
                    fallback_items = [segment.strip() for segment in re.split(r'[.;]', result.reasoning) if segment.strip()]
                result.evidence_items = (result.evidence_items + fallback_items)[:4]
            if len(result.evidence_items) < 2:
                result.evidence_items = (result.evidence_items + ["Case is incomplete from available discussion.", "Accusation relies on best available inference."])[:2]

            if not result.uncertainty.strip():
                result.uncertainty = f"Belief uncertainty remained at {belief_snapshot.get('uncertainty', 100)}/100 going into the accusation phase."

            reasoning_parts = [f"I accuse {result.accused}"]
            if result.primary_basis:
                reasoning_parts.append(f"primarily on {result.primary_basis}")
            if result.evidence_items:
                reasoning_parts.append("because " + "; ".join(result.evidence_items[:3]))
            if result.comparative_case.strip():
                reasoning_parts.append(f"Compared with other suspects, {result.comparative_case.strip()}")
            if result.uncertainty.strip():
                reasoning_parts.append(f"Main uncertainty: {result.uncertainty.strip()}")
            result.reasoning = ". ".join(part.rstrip(". ") for part in reasoning_parts if part).strip() + "."

            self.last_accusation_context = {
                "belief_snapshot": belief_snapshot,
                "top_n_candidates": top_n_candidates,
                "top_suspect": top_suspect,
                "accused_rank": accused_rank,
                "accused_in_top_n": result.accused in top_n_candidates if top_n_candidates else True,
                "accused_outside_top_n": accused_outside_top_n,
                # corrected_to_top_suspect is always False now — we no longer force
                # overrides.  Kept in the payload for backward-compat with analysis
                # pipelines that read this field from event logs.
                "corrected_to_top_suspect": corrected_to_top_suspect,
            }
            return result
        except Exception as e:
            print(f"Error in accuse for {self.name}: {e}", file=__import__('sys').stderr)
            fallback = AccusationResult(
                reasoning="Unable to decide from the available discussion.",
                accused=top_suspect or other_agents[0],
                confidence=0,
                primary_basis="mixed",
                evidence_items=["Case incomplete.", "Fallback accusation generated after model error."],
                uncertainty="Model failed during accusation generation.",
            )
            self.last_accusation_context = {
                "belief_snapshot": belief_snapshot,
                "top_n_candidates": top_n_candidates,
                "top_suspect": top_suspect,
                "accused_rank": 1 if fallback.accused == top_suspect else None,
                "accused_in_top_n": fallback.accused in top_n_candidates if top_n_candidates else True,
                "accused_outside_top_n": False,  # fallback always picks top_suspect
                "corrected_to_top_suspect": True,  # fallback due to model error IS a real correction
            }
            return fallback

    def recall_clues(self, state: GameState) -> Optional[dict]:
        """Pre-accusation clue recall probe.

        Asks the agent to list every Game-Master-revealed clue it remembers.
        Returns a dict with the recall results and a match score against actual
        clues stored in long-term memory.  The result is never fed back to the
        agent — it is a measurement-only probe for the thesis analysis.
        """
        self.update_memory(state)
        memory_context = self.memory.build_prompt_context()

        llm_recall = self.llm.with_structured_output(ClueRecallResult, method="json_mode")

        msgs = [
            SystemMessage(content=f"""You are {self.name}.

You are about to make your final accusation.
Before that, recall ALL the clues and evidence that the Game Master revealed during the investigation.
Do NOT invent or fabricate clues. Only list clues you actually remember being announced.
Return valid JSON."""),
            HumanMessage(content=f"""{memory_context}

List every clue or piece of evidence that the Game Master revealed during the investigation rounds.
For each clue, write a short summary (one sentence).
Also state which clue you consider most important and who the clues point to.

Return JSON with keys: recalled_clues, total_recalled, most_important_clue, clue_based_suspect"""),
        ]

        try:
            result = _retry_with_backoff(lambda: llm_recall.invoke(msgs))
            result.total_recalled = len(result.recalled_clues)

            # Score recall against actual clues stored in long-term memory
            actual_clues = self.memory.long_term.get_all_clues()
            return {
                "agent": self.name,
                "recalled_clues": result.recalled_clues,
                "total_recalled": result.total_recalled,
                "most_important_clue": result.most_important_clue,
                "clue_based_suspect": result.clue_based_suspect,
                "actual_clue_count": len(actual_clues),
            }
        except Exception as e:
            print(f"  Warning: clue recall probe failed for {self.name}: {e}")
            return {
                "agent": self.name,
                "recalled_clues": [],
                "total_recalled": 0,
                "most_important_clue": "",
                "clue_based_suspect": "",
                "actual_clue_count": len(self.memory.long_term.get_all_clues()),
                "error": str(e),
            }

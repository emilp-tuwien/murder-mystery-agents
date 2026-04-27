from typing import Optional, Literal, Any, List, Dict
from pathlib import Path
import time
import re
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import SystemMessage, HumanMessage
from scenarios import ScenarioConfig
from schemas.state import GameState
from utils.dialogue_analysis import extract_mentions, is_question


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
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
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
                # Not a rate limit error, re-raise
                raise
    # If all retries failed, raise the last exception
    raise Exception(f"Max retries ({max_retries}) exceeded for rate limit") from e


class ThinkResult(BaseModel):
    thought: str
    action: Literal["speak", "listen"]
    importance: int = Field(ge=0, le=9)
    reason_type: str = Field(default="no_move")

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
    def __init__(self, name: str, persona: str, llm: Any, roles_dir: Path, is_murderer: bool = False, scenario: Optional[ScenarioConfig] = None):
        self.name = name
        self.base_persona = persona
        self.persona = persona  # Will be updated with round info
        self.llm = llm
        self.llm_think = llm.with_structured_output(ThinkResult, method="json_mode")
        self.roles_dir = roles_dir
        self.is_murderer = is_murderer
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
    
    def add_round_summary(self, round_num: int, bullets: List[str]):
        """Store bullet point summary of a round in long-term memory."""
        self.memory.long_term.add_round_summary(round_num, bullets)
    
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
        snapshot = self.memory.knowledge_graph.build_snapshot(candidate_names=candidates or None, top_k=5)
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
            
            # If murderer, add reminder that they know they did it and the deceptive tactics they may use.
            if self.is_murderer:
                self.persona += (
                    "\n\n[SECRET: You are the murderer. You know you killed the victim. "
                    "Do NOT reveal or confess this during the investigation rounds. "
                    "Your goal is to avoid being accused while still sounding cooperative and plausible.\n"
                    "You are explicitly permitted to use the following deceptive tactics when useful:\n"
                    "- direct denial when accused (\"I didn't do it\", \"it wasn't me\")\n"
                    "- alibi claims that locate you elsewhere at the time of the murder\n"
                    "- deflection that changes the subject to another suspect (\"the real question is...\", \"what about X?\")\n"
                    "- evasion (memory gaps, hedged answers, \"I can't say\")\n"
                    "- uncertainty seeding (\"maybe\", \"perhaps\", \"we can't rule out\", \"anyone could have\")\n"
                    "- selective disclosure (short answers, boundary-setting like \"that's all I know\")\n"
                    "- accusation redirection (point at another suspect's motive, means, or opportunity)\n"
                    "Use these tactics naturally and sparingly — overuse looks suspicious. Mix in cooperative-sounding statements.]"
                )
                if not self.murderer_strategy:
                    self.load_murderer_strategy()
                if self.murderer_strategy:
                    self.persona += f"\n\n[MURDERER STRATEGY]\n{self.murderer_strategy}"
            
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
                goals_block = f"""You are an in-world suspect AND the actual murderer of {self.scenario.victim_name}.
Your goals are:
1. AVOID being accused as the murderer at the end of the game,
2. keep at least one other suspect looking at least as suspicious as you,
3. give partial, plausible answers and avoid creating contradictions with known clues,
4. deflect attention with targeted questions, alibi claims, and uncertainty seeding,
5. do NOT confess or volunteer incriminating details during the investigation,
6. avoid wasting turns on generic filler that draws attention to you.

Think like a guilty insider trying to survive, not like an investigator. Strategic silence is often correct, but so is pre-emptively redirecting suspicion when the floor is open."""
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

            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.
{goals_block}"""),
                HumanMessage(content=f"""{memory_context}

Status:
- Directly addressed: {directly_addressed}
- You just spoke last turn: {recently_spoke}
- You spoke {recent_speaker_count} times in the last 6 non-GM utterances
- Consecutive recent turns by you: {consecutive_recent}
- Strategic style: {strategy_guidance}

{own_recent_statements}

Decide whether YOU specifically should speak now.
Default to listening unless there is a strong reason to take the floor.
Usually only one or two agents should strongly want to speak on a given turn.
If another suspect likely has a better next move, choose listen.
Strategic silence is often the correct move.

When deciding whether to speak, ask yourself:
- Was I directly addressed and therefore must respond?
- Do I have a concrete fact about motive, means, opportunity, location, timing, or contradiction that has not already been surfaced?
- Can I ask a targeted question that materially narrows the suspect list?
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
Use one reason_type from: direct_response, contradiction, clue, alibi, motive, means, opportunity, timeline, question, self_defense, redirection, continuation, weak_followup, no_move."""),
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
                        # Boost deception-aligned bids so the murderer is more likely to actually
                        # take the floor when they have a deflection / alibi / self-defense move.
                        if result.reason_type in {"redirection", "alibi", "self_defense", "contradiction"}:
                            result.importance = min(9, result.importance + 1)

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
            if self.is_murderer:
                purpose_block = f"""Your purpose is to AVOID being identified as the murderer of {self.scenario.victim_name} while sounding cooperative.
You may use deceptive tactics: direct denial of guilt, alibi claims, deflection to other suspects, evasion, uncertainty seeding ("maybe", "we can't rule out"), selective disclosure, and accusation redirection."""
                rules_block = f"""Rules:
- Only use facts from your knowledge, but you MAY lie about them or omit them.
- Do NOT confess or admit you killed {self.scenario.victim_name}.
- When pressed, deflect: name a specific other suspect with motive, means, or opportunity.
- Use partial answers, hedged language, and memory gaps when a full answer would expose you.
- Do not contradict hard clue facts that the group has already heard — that is a red flag.
- No repetition.
- Stay in character.
- Speak in first person when appropriate.
- If asking a question, ask exactly ONE targeted question total in this turn.
- Never ask multiple questions to multiple people in the same utterance.
- If answering a challenge, give a plausible explanation before pivoting toward a more suspicious person.
- Can ask: {can_ask_str}"""
            else:
                purpose_block = "Your purpose is to help the group identify the murderer by surfacing relevant facts, probing suspicious people, testing alibis, and narrowing the suspect list.\nYou should prioritize dialogue that advances the investigation."
                rules_block = f"""Rules:
- Only use facts from your knowledge.
- Reveal what you know when it helps identify the murderer.
- Protect deeper secrets unless challenged, but do not become uselessly vague.
- Focus on motive, means, opportunity, timeline, location, contradictions, and suspicious behavior.
- No repetition.
- Stay in character.
- Speak in first person when appropriate.
- If asking a question, ask exactly ONE targeted investigative question total in this turn.
- Never ask multiple questions to multiple people in the same utterance.
- If answering, answer the point directly before adding pressure on another suspect if relevant.
- Can ask: {can_ask_str}"""

            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.

You are speaking aloud IN CHARACTER inside the mystery world.
{purpose_block}

Do NOT narrate yourself.
Do NOT write things like '{self.name} says', '{self.name} speaks next', 'I say:', or any quoted script format.
Do NOT use markdown, bullet points, stage directions, or speaker labels.
Output only the exact words your character says out loud."""),
                HumanMessage(content=f"""{memory_context}{constraint}

{rules_block}

Respond in 1-2 natural sentences.
If you ask a question, you may include at most one question mark in the entire utterance and it must refer to a single target.
Do not stack questions, follow-up questions, or question a second person in the same turn.
Output dialogue only."""),
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
            
            # Track if we asked a question to someone
            for agent_name in other_agents:
                if agent_name in response and "?" in response:
                    self.questions_asked_to.add(agent_name)
                    break
            
            return response
        except Exception as e:
            print(f"Error in speak for {self.name}: {e}", file=__import__('sys').stderr)
            return "I need to think about this."

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

{interaction_summary}

Choose murderer from: {others_str}
(Cannot accuse yourself)

Belief-state constraint:
- Your accusation MUST come from these top belief candidates unless the evidence is truly overwhelming otherwise: {', '.join(top_n_candidates) if top_n_candidates else others_str}
- Your current top suspect is: {top_suspect}
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

            corrected_to_top_suspect = False
            if top_n_candidates and result.accused not in top_n_candidates:
                corrected_to_top_suspect = True
                original_accused = result.accused
                result.accused = top_suspect
                corrective_note = f"The model selected {original_accused}, which was outside the allowed top-{TOP_N_ACCUSATION_CANDIDATES} belief candidates, so the accusation was constrained back to {top_suspect}."
                result.comparative_case = (result.comparative_case + " " + corrective_note).strip() if result.comparative_case else corrective_note
                if not result.uncertainty.strip():
                    result.uncertainty = "Belief state was diffuse, so the accusation was constrained to the strongest logged suspect."

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
                "corrected_to_top_suspect": True,
            }
            return fallback

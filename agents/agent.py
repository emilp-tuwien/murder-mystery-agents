from typing import Optional, Literal, Any, List
from pathlib import Path
import time
import re
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import SystemMessage, HumanMessage
from scenarios import ScenarioConfig
from schemas.state import GameState


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
    raise Exception(f"Max retries ({max_retries}) exceeded for rate limit")


class ThinkResult(BaseModel):
    thought: str
    action: Literal["speak", "listen"]
    importance: int = Field(ge=0, le=9)

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value


class AccusationResult(BaseModel):
    reasoning: str = Field(description="Brief reasoning for your accusation")
    accused: str = Field(description="The name of the person you accuse of being the murderer")


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
        self.questions_asked_to: set = set()  # Track who we've asked questions to (can only ask each agent once)
        self.facts_revealed: List[str] = []  # Track facts this agent has already revealed
        self.topics_discussed: set = set()  # Track topics to avoid repetition
        
        # Initialize three-stage memory system
        from memory.agent_memory import AgentMemory, SharedHistory
        self.memory = AgentMemory(agent_name=name)
    
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
    
    def add_fact_to_memory(self, fact: str):
        """Add an important fact to long-term memory."""
        self.memory.long_term.add_fact(fact)
    
    def add_round_summary(self, round_num: int, bullets: List[str]):
        """Store bullet point summary of a round in long-term memory."""
        self.memory.long_term.add_round_summary(round_num, bullets)
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        """Update suspicion level for a person in knowledge graph."""
        self.memory.knowledge_graph.update_suspicion(target, delta, reason)
    
    def export_memory_snapshot(self) -> dict:
        facts = self.memory.long_term.facts[-24:]
        categorized = {}
        for tag in ["motive", "means", "opportunity", "contradiction", "timeline", "alibi"]:
            categorized[tag] = [fact.fact_text for fact in facts if tag in fact.tags][-6:]

        uncategorized = [fact.fact_text for fact in facts if not any(tag in fact.tags for tag in ["motive", "means", "opportunity", "contradiction", "timeline", "alibi"])]
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
        }

    def update_round(self, round_num: int):
        """Update agent's knowledge with new round information."""
        from utils.agent_helper import load_round_description
        
        if round_num == self.current_round:
            return  # Already on this round
        
        self.current_round = round_num
        round_desc = load_round_description(self.roles_dir, self.name, round_num)
        
        if round_desc:
            self.accumulated_knowledge += f"\n\n=== ROUND {round_num} INFORMATION ===\n{round_desc}"
            
            # Update persona with accumulated knowledge
            self.persona = f"{self.base_persona}\n\n{self.accumulated_knowledge}"
            
            # If murderer, add reminder that they know they did it
            if self.is_murderer:
                self.persona += "\n\n[SECRET: You are the murderer. You know you killed the victim. Do NOT reveal or confess this during the investigation. Your goal is to avoid exposing your guilt while still sounding cooperative, plausible, and useful to the group.]"
            
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

    def _format_history(self, history: List[dict]) -> str:
        """Use shared history window for prompts (last K_HISTORY turns only)."""
        return self.memory.shared_history.render_for_prompt()

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

        strategy_map = {
            "Dr Chelsea Barren": "You are sharp, analytical, and status-conscious. Speak when you can expose a contradiction, propose a theory, or exploit a weak alibi. Stay quiet if you would only repeat the room.",
            "Enrique Graves": "You are risk-aware and self-protective. Prefer silence unless challenged, directly questioned, or you need to redirect suspicion with a concrete point.",
            "Kathryn Lawless": "You are careful, observant, and selective. Hold back unless you can materially improve the investigation or defend yourself precisely.",
            "Michael Nightshade": "You are opportunistic and persuasive. Speak when you can steer the narrative, pressure another suspect, or gain advantage, but do not jump in without leverage.",
            "Norman D'Adly": "You are reactive, emotional, and more likely to speak when provoked, accused, or contradicted. Otherwise, let stronger evidence emerge first.",
            "Vicki D'Adly": "You are defensive and strategic. Speak when you must protect yourself, challenge a threat, or exploit another person's mistake; otherwise conserve your position.",
        }
        strategy_guidance = strategy_map.get(self.name, "Be selective. Speak only when your move clearly improves the investigation or your position.")
        
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

Return valid JSON with keys: thought, action, importance."""),
            ]
        else:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.
You are an in-world suspect trying to help the group determine who killed {self.scenario.victim_name}.
Your goals are:
1. identify the murderer from dialogue, clues, motives, means, opportunity, contradictions, and timelines,
2. ask sharp questions when you lack crucial information,
3. reveal relevant facts you know when it helps the investigation,
4. protect your private secrets unless they are directly challenged or necessary to defend yourself,
5. avoid wasting turns on generic filler.

Think like an investigator with partial information, not like a chatbot.
Your job is NOT to speak every turn. Good investigation also means waiting when another person has a stronger next move."""),
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
- 8 = unique, high-value evidence or strong contradiction that should be stated immediately
- 6-7 = useful but not urgent question or fact
- 4-5 = moderate contribution; only speak if conversation is drifting
- 2-3 = weak contribution, likely better to listen
- 0-1 = definitely listen

Be willing to choose listen with low scores. Do not inflate urgency just to participate.

Return valid JSON with keys: thought, action, importance."""),
            ]
        
        try:
            result = _retry_with_backoff(lambda: self.llm_think.invoke(msgs))

            # Heuristic calibration to avoid everyone clustering on high urgency.
            if phase != "introduction":
                if directly_addressed:
                    result.action = "speak"
                    result.importance = max(result.importance, 9)
                else:
                    if recently_spoke:
                        result.importance = max(0, result.importance - 3)
                    if recent_speaker_count >= 1:
                        result.importance = max(0, result.importance - 2)
                    if recent_speaker_count >= 2:
                        result.importance = max(0, result.importance - 2)
                    if consecutive_recent >= 1:
                        result.importance = max(0, result.importance - 1)

                    thought_lower = result.thought.lower()
                    strong_signal = any(token in thought_lower for token in ["contrad", "direct", "asked", "alibi", "clue", "evidence", "timeline", "motive", "opportunity", "accus", "question", "respond"])

                    if self.is_murderer and not strong_signal:
                        result.importance = max(0, result.importance - 1)

                    if self.name in {"Enrique Graves", "Kathryn Lawless", "Vicki D'Adly"} and not strong_signal:
                        result.importance = max(0, result.importance - 1)

                    if self.name == "Norman D'Adly" and any(token in thought_lower for token in ["provok", "accus", "anger", "challeng"]):
                        result.importance = min(9, result.importance + 1)

                    if result.importance >= 8 and not strong_signal:
                        result.importance = 5

                    if result.action == "speak" and result.importance <= 5:
                        result.action = "listen"
                    if result.action == "listen" and result.importance >= 8:
                        result.action = "speak"

            self.memory.add_thought(result.thought, result.action, result.importance)
            return result
        except Exception as e:
            print(f"Error in think for {self.name}: {e}", file=__import__('sys').stderr)
            return ThinkResult(thought="waiting", action="listen", importance=3)

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
Speak naturally to the other suspects.
Output dialogue only."""),
            ]
        else:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

{self.scenario.victim_name} was MURDERED. Round {current_round}/6.

You are speaking aloud IN CHARACTER inside the mystery world.
Your purpose is to help the group identify the murderer by surfacing relevant facts, probing suspicious people, testing alibis, and narrowing the suspect list.
You should prioritize dialogue that advances the investigation.

Do NOT narrate yourself.
Do NOT write things like '{self.name} says', '{self.name} speaks next', 'I say:', or any quoted script format.
Do NOT use markdown, bullet points, stage directions, or speaker labels.
Output only the exact words your character says out loud."""),
                HumanMessage(content=f"""{memory_context}{constraint}

Rules:
- Only use facts from your knowledge.
- Reveal what you know when it helps identify the murderer.
- Protect deeper secrets unless challenged, but do not become uselessly vague.
- Focus on motive, means, opportunity, timeline, location, contradictions, and suspicious behavior.
- No repetition.
- Stay in character.
- Speak in first person when appropriate.
- If asking a question, ask one targeted investigative question.
- If answering, answer the point directly before adding pressure on another suspect if relevant.
- Can ask: {can_ask_str}

Respond in 1-2 natural sentences that materially advance the investigation.
Output dialogue only."""),
            ]
        try:
            result = _retry_with_backoff(lambda: self.llm.invoke(msgs))
            response = result.content if result and result.content else "I have nothing new to add."
            
            # Remove quotation marks from response
            response = response.replace('"', '').replace('"', '').replace('"', '')
            
            # Remove action descriptions in parentheses like (looks around) or (sighs nervously)
            import re
            response = re.sub(r'\([^)]*\)', '', response).strip()
            # Also remove brackets
            response = re.sub(r'\[[^\]]*\]', '', response).strip()
            # Remove leading meta speech labels like "Enrique Graves speaks next:" or "Michael says:"
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

    def accuse(self, state: GameState, all_agents: List[str]) -> AccusationResult:
        """Final accusation - who does this agent think is the murderer? Cannot accuse self."""
        self.update_memory(state)
        
        other_agents = [name for name in all_agents if name != self.name]
        others_str = ", ".join(other_agents)
        
        llm_accuse = self.llm.with_structured_output(AccusationResult, method="json_mode")
        memory_context = self.memory.build_prompt_context()
        suspect_ranking = self.memory.get_suspect_ranking()
        
        msgs = [
            SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

Investigation OVER. Accuse ONE person."""),
            HumanMessage(content=f"""{memory_context}

{suspect_ranking}

Choose murderer from: {others_str}
(Cannot accuse yourself)

Return valid JSON with keys: reasoning, accused."""),
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
            return result
        except Exception as e:
            print(f"Error in accuse for {self.name}: {e}", file=__import__('sys').stderr)
            return AccusationResult(reasoning="Unable to decide", accused=other_agents[0])

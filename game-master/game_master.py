from typing import Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import PyPDF2


class SpeakerDecision(BaseModel):
    """Game Master's decision on who should speak next"""
    reasoning: str = Field(description="Brief reasoning for the decision")
    next_speaker: str = Field(description="Name of the player who should speak next")
    response_constraint: Optional[str] = Field(default=None, description="What they should respond to, if applicable")
    is_direct_address: bool = Field(description="True if someone was directly asked/addressed")


class GameMaster:
    def __init__(self, llm: Any, agent_names: List[str], conversations_per_round: int = 20):
        self.llm = llm
        self.agent_names = agent_names
        self.llm_decide = llm.with_structured_output(SpeakerDecision)
        self.persona = self._load_persona()
        self.conversations_per_round = conversations_per_round
        self.max_rounds = 6  # Total number of rounds in the game
    
    def _load_persona(self) -> str:
        """Load game master description from PDF"""
        pdf_path = Path(__file__).parent / "description" / "game-master.pdf"
        if pdf_path.exists():
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    return text.strip() if text.strip() else self._default_persona()
            except Exception as e:
                print(f"Warning: Could not load game master PDF: {e}")
                return self._default_persona()
        return self._default_persona()
    
    def _default_persona(self) -> str:
        return """You are the Game Master of a murder mystery party.
Your role is to facilitate the discussion and ensure the investigation progresses.
You decide who speaks next based on the conversation flow."""

    def should_advance_round(self, conversations_in_round: int, current_round: int) -> bool:
        """
        Determine if the game should advance to the next round.
        By default, advances after conversations_per_round conversations.
        """
        if current_round >= self.max_rounds:
            return False  # Already at max round
        return conversations_in_round >= self.conversations_per_round
    
    def get_phase_for_round(self, round_num: int) -> str:
        """
        Determine the game phase based on current round.
        Round 1: Introduction phase
        Rounds 2-5: Discussion/Investigation phase
        Round 6+: Should trigger accusation phase
        """
        if round_num == 1:
            return "introduction"
        elif round_num <= 5:
            return "discussion"
        else:
            return "accusation"
    
    def is_game_complete(self, current_round: int, conversations_in_round: int) -> bool:
        """Check if the game should end (after round 5 completes)."""
        return current_round >= 6 or (current_round == 5 and conversations_in_round >= self.conversations_per_round)

    def decide_next_speaker(self, state: dict, thoughts: dict) -> SpeakerDecision:
        """
        Evaluate the last message and all agent thoughts to decide who speaks next.
        
        Priority:
        1. If someone was directly addressed/asked a question → they MUST respond
        2. Otherwise, pick the agent most likely to advance the investigation
        """
        history = state.get("history", [])
        last_utterance = history[-1] if history else None
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Build context about what each agent wants to say
        agent_thoughts_txt = "\n".join([
            f"- {name}: wants to {'SPEAK' if tr.action == 'speak' else 'listen'} (urgency: {tr.importance}/9) - thinking: \"{tr.thought}\""
            for name, tr in thoughts.items()
        ])
        
        # Build conversation history
        history_txt = "\n".join([
            f"{u['speaker']}: {u['text']}" for u in history
        ]) or "(no conversation yet)"
        
        # Available speakers (exclude last speaker to avoid monopolization)
        last_speaker = state.get("last_speaker")
        available = [n for n in self.agent_names if n != last_speaker] if last_speaker else self.agent_names
        available_str = ", ".join(available)
        
        # Phase-specific instructions
        if phase == "introduction":
            phase_instruction = """
CURRENT PHASE: INTRODUCTIONS (Round 1)
Priority: Let each player introduce themselves. Prefer players who haven't spoken yet.
Ensure everyone gets a chance to introduce themselves before moving to investigation."""
        else:
            phase_instruction = f"""
CURRENT PHASE: INVESTIGATION (Round {current_round}/6)
Priority: Advance the murder investigation. Look for direct questions, accusations, or important revelations."""
        
        msgs = [
            SystemMessage(content=f"""{self.persona}

PLAYERS IN THE GAME: {', '.join(self.agent_names)}
{phase_instruction}

YOUR TASK: Decide who should speak next.

RULES:
1. DIRECT ADDRESS: If the last speaker asked someone a question BY NAME or made a direct accusation, that person MUST respond next.
2. INVESTIGATION FLOW: If no one was directly addressed, choose the player whose thoughts are most likely to advance the murder investigation.
3. AVOID MONOPOLIZATION: Don't let the same person speak twice in a row. Available speakers: {available_str}
4. URGENCY MATTERS: Consider each player's urgency score (0-9) but also the VALUE of what they want to say."""),
            HumanMessage(content=f"""CONVERSATION SO FAR:
{history_txt}

WHAT EACH PLAYER IS THINKING:
{agent_thoughts_txt}

Last speaker: {last_speaker or 'None'}

Analyze the last message. Was anyone directly addressed or asked a question?
If yes, they must respond. If no, who would best advance the investigation?

Choose ONE player from: {available_str}"""),
        ]
        
        try:
            result = self.llm_decide.invoke(msgs)
            
            # Validate the chosen speaker
            if result.next_speaker not in self.agent_names:
                # Try to find a close match
                for agent in self.agent_names:
                    if agent.lower() in result.next_speaker.lower() or result.next_speaker.lower() in agent.lower():
                        result.next_speaker = agent
                        break
                else:
                    # Default to highest urgency agent
                    max_urgency = max(thoughts.items(), key=lambda x: x[1].importance)
                    result.next_speaker = max_urgency[0]
            
            return result
        except Exception as e:
            print(f"Error in GameMaster decide: {e}")
            # Fallback: pick highest urgency
            max_urgency = max(thoughts.items(), key=lambda x: x[1].importance)
            return SpeakerDecision(
                reasoning="Fallback selection based on urgency",
                next_speaker=max_urgency[0],
                response_constraint=None,
                is_direct_address=False
            )

    def provide_initial_context(self) -> str:
        """Provide game context to all players at the start."""
        context_intro = f"""
╔═══════════════════════════════════════════════════════════════╗
║          WELCOME TO THE MURDER MYSTERY INVESTIGATION          ║
╚═══════════════════════════════════════════════════════════════╝

Players: {', '.join(self.agent_names)}

══════════════════════════════════════════════════════════════════

GAME STRUCTURE:
- Round 1: Introductions - Players introduce themselves
- Rounds 2-5: Investigation - Players discuss, question, and investigate
- After Round 5: Accusation - Each player accuses someone
- Final: Confessions - Everyone reveals their secrets

Each round has approximately {self.conversations_per_round} conversations.

══════════════════════════════════════════════════════════════════

GAME CONTEXT:

{self.persona}

══════════════════════════════════════════════════════════════════

RULES:
1. Everyone can speak and must participate to avoid suspicion
2. You can ask questions, share clues, and make accusations
3. All conversations are PUBLIC - no private discussions allowed
4. Players CANNOT accuse themselves
5. The group must identify the murderer through discussion
6. After round 5, everyone votes on who they suspect

══════════════════════════════════════════════════════════════════

NOW BEGINNING ROUND 1: INTRODUCTIONS...

"""
        return context_intro
    
    def announce_round_change(self, new_round: int) -> str:
        """Generate announcement for round change."""
        if new_round == 2:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║                    ROUND {new_round}: INVESTIGATION BEGINS                ║
╚═══════════════════════════════════════════════════════════════╝
New information has come to light! Players have received additional clues.
The investigation phase begins now. Question everyone, look for lies!
"""
        elif new_round <= 5:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║                         ROUND {new_round}                              ║
╚═══════════════════════════════════════════════════════════════╝
New evidence has been discovered! Players have received new information.
Continue your investigation. The truth is getting closer...
"""
        else:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║              FINAL ROUND - ACCUSATION TIME                    ║
╚═══════════════════════════════════════════════════════════════╝
The investigation is complete. Time to make your final accusations!
"""

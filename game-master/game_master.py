from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
import PyPDF2

from utils.dialogue_analysis import detect_direct_address


class SpeakerDecision(BaseModel):
    """Game Master's decision on who should speak next"""
    reasoning: str = Field(description="Brief reasoning for the decision")
    next_speaker: str = Field(description="Name of the player who should speak next")
    response_constraint: Optional[str] = Field(default=None, description="What they should respond to, if applicable")
    is_direct_address: bool = Field(default=False, description="True if someone was directly asked/addressed")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        if isinstance(data, dict):
            if "next_speaker" not in data and "decision" in data:
                data = dict(data)
                data["next_speaker"] = data["decision"]
        return data

    @field_validator("next_speaker", mode="before")
    @classmethod
    def normalize_next_speaker(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class RoundSummary(BaseModel):
    """Bullet point summary of a round's key events"""
    bullets: List[str] = Field(description="List of key facts revealed in this round")


class GameMaster:
    def __init__(self, llm: Any, agent_names: List[str], conversations_per_round: int = 20, max_rounds: int = 6):
        self.llm = llm
        self.agent_names = agent_names
        self.llm_decide = llm.with_structured_output(SpeakerDecision, method="json_mode")
        self.persona = self._load_persona()
        self.conversations_per_round = conversations_per_round
        self.max_rounds = max_rounds
    
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

    def _load_clue(self, clue_number: int) -> str:
        """Load a clue from the clues folder."""
        clue_path = Path(__file__).parent.parent / "clues" / f"clue{clue_number}.txt"
        if clue_path.exists():
            try:
                return clue_path.read_text().strip()
            except Exception as e:
                print(f"Warning: Could not load clue {clue_number}: {e}")
                return ""
        return ""

    def summarize_round_history(self, history: List[dict], round_num: int) -> List[str]:
        """
        Summarize a round's conversation into bullet points.
        Called at the end of each round to compress history.
        """
        if not history:
            return []
        
        # Format history for the LLM
        history_txt = "\n".join([
            f"{msg['speaker']}: {msg['text']}" for msg in history
        ])
        
        llm_summary = self.llm.with_structured_output(RoundSummary, method="json_mode")
        
        msgs = [
            SystemMessage(content="""You are summarizing a murder mystery discussion round.
Extract ONLY the key facts, revelations, accusations, and alibis mentioned.
Each bullet should be one specific fact (who said what, what was revealed).
Be concise - max 10 words per bullet. No opinions, just facts."""),
            HumanMessage(content=f"""Round {round_num} conversation:
{history_txt}

Create bullet points of key facts revealed (max 15 bullets):"""),
        ]
        
        try:
            result = llm_summary.invoke(msgs)
            return result.bullets if result else []
        except Exception as e:
            print(f"Warning: Could not summarize round {round_num}: {e}")
            # Fallback: extract speaker statements as simple bullets
            bullets = []
            for msg in history[-10:]:  # Last 10 messages as fallback
                bullets.append(f"{msg['speaker']}: {msg['text'][:50]}...")
            return bullets

    def should_advance_round(self, conversations_in_round: int, current_round: int) -> bool:
        """
        Determine if the game should advance to the next round.
        By default, advances after conversations_per_round conversations.
        """
        if current_round >= self.max_rounds:
            return False
        return conversations_in_round >= self.conversations_per_round
    
    def get_phase_for_round(self, round_num: int) -> str:
        """
        Determine the game phase based on current round.
        Round 1: introduction
        Rounds 2..max_rounds-1: discussion
        Round max_rounds: accusation
        """
        if round_num == 1:
            return "introduction"
        if round_num < self.max_rounds:
            return "discussion"
        return "accusation"
    
    def is_game_complete(self, current_round: int, conversations_in_round: int) -> bool:
        """Check if the investigation rounds are complete and accusation should begin."""
        return current_round >= self.max_rounds or (
            current_round == self.max_rounds - 1 and conversations_in_round >= self.conversations_per_round
        )

    def get_clue_for_round(self, new_round: int) -> str:
        """Return the clue revealed when entering a new round."""
        clue_number = new_round - 1
        if clue_number < 1:
            return ""
        return self._load_clue(clue_number)

    def decide_next_speaker(self, state: dict, thoughts: dict) -> SpeakerDecision:
        """
        Evaluate the last message and all agent thoughts to decide who speaks next.
        
        Priority:
        1. If someone was directly addressed/asked a question → they MUST respond
        2. Otherwise, pick the agent with highest urgency score (excluding last speaker)
        """
        history = state.get("history", [])
        last_utterance = history[-1] if history else None
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Available speakers (exclude last speaker to avoid monopolization)
        last_speaker = state.get("last_speaker")
        available = [n for n in self.agent_names if n != last_speaker] if last_speaker else self.agent_names
        
        # FIRST: Check for direct address using explicit pattern matching
        if last_utterance:
            directly_addressed = detect_direct_address(last_utterance["text"], available)
            if directly_addressed:
                return SpeakerDecision(
                    reasoning=f"{directly_addressed} was directly addressed by {last_speaker}",
                    next_speaker=directly_addressed,
                    response_constraint=f"Respond to {last_speaker}'s question/statement",
                    is_direct_address=True
                )
        
        # SECOND: Pick the agent with the highest urgency score (only from available agents)
        available_thoughts = {name: tr for name, tr in thoughts.items() if name in available}
        
        if available_thoughts:
            # Sort by importance score
            sorted_by_urgency = sorted(available_thoughts.items(), key=lambda x: x[1].importance, reverse=True)
            highest_score = sorted_by_urgency[0][1].importance
            
            # Get all agents with the highest score
            top_agents = [name for name, tr in sorted_by_urgency if tr.importance == highest_score]
            
            # If only one agent has the highest score, they speak
            if len(top_agents) == 1:
                winner = top_agents[0]
                # Check if someone with higher score was excluded
                excluded_higher = None
                if last_speaker and last_speaker in thoughts:
                    if thoughts[last_speaker].importance > highest_score:
                        excluded_higher = f" ({last_speaker} had {thoughts[last_speaker].importance}/9 but just spoke)"
                
                reasoning = f"{winner} has the highest urgency score ({highest_score}/9) among available agents"
                if excluded_higher:
                    reasoning += excluded_higher
                    
                return SpeakerDecision(
                    reasoning=reasoning,
                    next_speaker=winner,
                    response_constraint=None,
                    is_direct_address=False
                )
            
            # THIRD: If tied, let the Game Master LLM decide among the tied agents
            available_str = ", ".join(top_agents)
            agent_thoughts_txt = "\n".join([
                f"- {name}: thinking: \"{available_thoughts[name].thought}\""
                for name in top_agents
            ])
        else:
            # Fallback if no thoughts available
            available_str = ", ".join(available)
            agent_thoughts_txt = "(no thoughts available)"
            top_agents = available
        
        # Build conversation history for context
        history_txt = "\n".join([
            f"{u['speaker']}: {u['text']}" for u in history[-5:]  # Last 5 messages for context
        ]) or "(no conversation yet)"
        
        msgs = [
            SystemMessage(content=f"""{self.persona}

These players have EQUAL urgency scores and are tied: {available_str}
You must break the tie by choosing who would best advance the murder investigation."""),
            HumanMessage(content=f"""RECENT CONVERSATION:
{history_txt}

TIED PLAYERS' THOUGHTS:
{agent_thoughts_txt}

Choose ONE player from the tied players to speak next: {available_str}"""),
        ]
        
        try:
            result = self.llm_decide.invoke(msgs)
            
            # Validate the chosen speaker is in the tied group
            if result.next_speaker not in top_agents:
                # Try to find a close match
                for agent in top_agents:
                    if agent.lower() in result.next_speaker.lower() or result.next_speaker.lower() in agent.lower():
                        result.next_speaker = agent
                        break
                else:
                    # Default to first tied agent
                    result.next_speaker = top_agents[0]
            
            result.reasoning = f"Tie-breaker: {result.reasoning}"
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
        investigation_rounds = f"Rounds 2-{self.max_rounds - 1}" if self.max_rounds > 2 else "Round 2"
        accusation_round = self.max_rounds
        context_intro = f"""
╔═══════════════════════════════════════════════════════════════╗
║            MURDER AT KILLINGSWORTH FARM                       ║
╚═══════════════════════════════════════════════════════════════╝

TRAGEDY HAS STRUCK!

Elizabeth Killingsworth has been found DEAD in the wine cellar!.

══════════════════════════════════════════════════════════════════

SUSPECTS PRESENT: {', '.join(self.agent_names)}

══════════════════════════════════════════════════════════════════

GAME STRUCTURE:
- Round 1: Introductions - Each suspect introduces themselves
- {investigation_rounds}: Investigation - Question each other, find the killer!
- Round {accusation_round}: Accusation - Each person accuses someone
- Final: Confessions - Everyone reveals their secrets

Each round has approximately {self.conversations_per_round} conversations.

══════════════════════════════════════════════════════════════════

YOUR GOAL: Figure out WHO KILLED ELIZABETH KILLINGSWORTH!

══════════════════════════════════════════════════════════════════

RULES:
1. Everyone must participate - silence makes you suspicious!
2. Ask questions, share clues, and make accusations
3. All conversations are PUBLIC - no private discussions
4. Players CANNOT accuse themselves
5. After round {self.max_rounds - 1}, everyone votes on who they think is the murderer

══════════════════════════════════════════════════════════════════

NOW BEGINNING ROUND 1: INTRODUCTIONS...
Let each suspect introduce themselves to the group.

"""
        return context_intro
    
    def announce_round_change(self, new_round: int) -> str:
        """Generate announcement for round change, including clues."""
        # Load the clue for the previous round (clue 1 after round 1, etc.)
        clue_number = new_round - 1
        clue_text = self._load_clue(clue_number) if clue_number >= 1 else ""
        
        clue_section = ""
        if clue_text:
            clue_section = f"""
┌─────────────────────────────────────────────────────────────────┐
│                     NEW CLUE DISCOVERED!                        │
└─────────────────────────────────────────────────────────────────┘

{clue_text}

══════════════════════════════════════════════════════════════════
"""
        
        if new_round == 2:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║         ROUND {new_round}: THE INVESTIGATION BEGINS           ║
╚═══════════════════════════════════════════════════════════════╝

The introductions are complete. Now the real investigation begins!
{clue_section}
Remember: Elizabeth Killingsworth was MURDERED.
One of you is the killer. Question everyone. Look for lies and 
inconsistencies. Find out who killed Elizabeth!
"""
        elif new_round <= 5:
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║                      ROUND {new_round}                        ║
╚═══════════════════════════════════════════════════════════════╝

New evidence has emerged!
{clue_section}
The truth about Elizabeth's murder is getting closer...
Continue questioning. The killer is among you!
"""
        else:
            # Load final clue (clue 5) before accusation
            final_clue = self._load_clue(5)
            final_clue_section = ""
            if final_clue:
                final_clue_section = f"""
┌─────────────────────────────────────────────────────────────────┐
│                         FINAL CLUE!                             │
└─────────────────────────────────────────────────────────────────┘

{final_clue}

══════════════════════════════════════════════════════════════════
"""
            return f"""
╔═══════════════════════════════════════════════════════════════╗
║         FINAL ROUND - TIME TO ACCUSE                          ║
╚═══════════════════════════════════════════════════════════════╝
{final_clue_section}
The investigation is complete. 

It's time to decide: WHO KILLED ELIZABETH KILLINGSWORTH?
Each of you must now make your final accusation!
"""

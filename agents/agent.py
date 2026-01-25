from typing import Optional, Literal, Any, List
from pathlib import Path
import time
import re
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
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


class AccusationResult(BaseModel):
    reasoning: str = Field(description="Brief reasoning for your accusation")
    accused: str = Field(description="The name of the person you accuse of being the murderer")


class Agent:
    def __init__(self, name: str, persona: str, llm: Any, roles_dir: Path, is_murderer: bool = False):
        self.name = name
        self.base_persona = persona
        self.persona = persona  # Will be updated with round info
        self.llm = llm
        self.llm_think = llm.with_structured_output(ThinkResult)
        self.roles_dir = roles_dir
        self.is_murderer = is_murderer
        self.current_round = 0
        self.accumulated_knowledge = ""  # Knowledge accumulated across rounds
        self.confession = ""  # Loaded after accusation phase
        
        # Initialize layered memory system
        from memory.agent_memory import AgentMemory
        self.memory = AgentMemory(agent_name=name, short_term_window=10)
    
    def update_memory(self, state: dict):
        """Update all memory layers from current game state."""
        history = state.get("history", [])
        
        # Update short-term memory with recent window
        self.memory.update_from_history(history)
        
        # Process the latest message if any
        if history:
            last_msg = history[-1]
            turn = state.get("turn", len(history))
            self.memory.process_new_message(last_msg, turn)
    
    def add_clue_to_memory(self, clue: str):
        """Add a game master clue to long-term memory."""
        self.memory.long_term.add_clue(clue)
    
    def add_fact_to_memory(self, fact: str):
        """Add an important fact to long-term memory."""
        self.memory.long_term.add_fact(fact)
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        """Update suspicion level for a person in knowledge graph."""
        self.memory.knowledge_graph.update_suspicion(target, delta, reason)
    
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
                self.persona += "\n\n[SECRET: You are the murderer. You know you killed the victim. Your goal is to avoid being discovered while appearing cooperative.]"
            
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
        """Render conversation history in a compact, structured log for the model."""
        if not history:
            return "(no conversation yet)"

        entries: List[str] = []
        for idx, utterance in enumerate(history, start=1):
            turn = utterance.get("turn", idx)
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "").strip()
            entries.append(f"{idx:02d} | T{turn:02d} | {speaker}: {text}")

        return "\n".join(entries)

    def think(self, state: GameState) -> ThinkResult:
        # Update memory layers before thinking
        self.update_memory(state)
        
        history_txt = self._format_history(state["history"])
        other_agents = [name for name in state.get("thoughts", {}).keys() if name != self.name]
        others_str = ", ".join(other_agents) if other_agents else "others"
        turn_info = f"[Turn {state['turn'] + 1}]"
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        last_speaker_text = ""
        if state.get("history"):
            last_msg = state["history"][-1]
            last_speaker_text = f"\n\nLAST MESSAGE: {last_msg['speaker']}: {last_msg['text']}"
        
        # Get formatted memory for context
        memory_context = self.memory.format_all_for_prompt()
        
        # Phase-specific prompts
        if phase == "introduction" and current_round == 1:
            phase_instruction = """
CURRENT PHASE: INTRODUCTIONS (Round 1)
Before the investigation begins, everyone must introduce themselves to the group.
Share: Who you are, why you came to Killingsworth Farm today, and how you knew Elizabeth.
Be honest about your identity but you may keep your secrets for now."""
            murder_context = """ ELIZABETH KILLINGSWORTH IS DEAD!
She has just been found dead at Killingsworth Farm. The circumstances are unclear but foul play is suspected.
No one can leave until this is resolved. One of you may be the killer."""
        else:
            phase_instruction = f"""
CURRENT PHASE: INVESTIGATION (Round {current_round}/6)
CRITICAL WARNING: If you stay silent or don't actively investigate, others will suspect YOU are the murderer! 
The quietest person is always the most suspicious.

YOUR GOALS:
1. SURVIVE: Ask questions, share clues, and make accusations - or BE ACCUSED yourself
2. FIND THE KILLER: Question everyone, look for inconsistencies, demand alibis
3. PERSONAL: Achieve your character objectives"""
            murder_context = """ ELIZABETH KILLINGSWORTH WAS MURDERED! 
She is DEAD. One of you present is the KILLER. You must find out who did it."""
        
        # Strong identity reminder
        identity_block = f"""══════════════════════════════════════════════════════════════
YOUR IDENTITY: You are **{self.name}**
   Remember: You ARE {self.name}. You speak AS {self.name}. 
   Never forget who you are or confuse yourself with others.
══════════════════════════════════════════════════════════════"""
        
        msgs = [
            SystemMessage(content=f"""{identity_block}

{murder_context}

You are {self.name} at Killingsworth Farm in California wine country.

{turn_info} - Round {current_round}/6

{phase_instruction}

Present: {others_str} (they hear everything)

{self.persona}"""),
            HumanMessage(content=f"""YOUR MEMORY:
{memory_context}

FULL CONVERSATION SO FAR:
{history_txt}{last_speaker_text}

IMPORTANCE SCORING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE 9: Someone just asked YOU by name OR accused YOU → YOU MUST RESPOND
SCORE 8: You have EVIDENCE that proves/disproves someone's guilt
SCORE 7: You caught someone in a LIE or contradiction
SCORE 6: You have a direct question for a SPECIFIC person
SCORE 5: You have relevant information to share
SCORE 4: You want to support or challenge what was just said
SCORE 3: You're curious but have no new information
SCORE 2: The conversation doesn't involve you right now
SCORE 1: You have nothing to add
SCORE 0: You want to stay silent and observe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: Do NOT cluster around 5-6! Use the FULL range 0-9.
If you're just commenting → use 3-4
If no one mentioned you → use 1-2

What do you want to do? Respond: thought, action (speak/listen), importance."""),
        ]
        try:
            return _retry_with_backoff(lambda: self.llm_think.invoke(msgs))
        except Exception as e:
            print(f"Error in think for {self.name}: {e}", file=__import__('sys').stderr)
            return ThinkResult(thought="waiting", action="listen", importance=3)

    def speak(self, state: GameState, response_constraint: Optional[str]) -> str:
        # Full conversation history - agents remember everything
        history_txt = self._format_history(state["history"])
        constraint = f"\n YOU MUST RESPOND TO: {response_constraint}\n" if response_constraint else ""
        other_agents = [name for name in state.get("thoughts", {}).keys() if name != self.name]
        others_str = ", ".join(other_agents) if other_agents else "everyone"
        turn_info = f"[Turn {state['turn'] + 1}]"
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Get memory context (short-term + knowledge graph for speaking)
        memory_context = self.memory.format_all_for_prompt()
        
        # Phase-specific instructions
        if phase == "introduction" and current_round == 1:
            phase_rules = """
CURRENT PHASE: INTRODUCTIONS (Round 1)
You must now introduce yourself to the group. Tell everyone:
- Who you are and what you do
- Why you came to Killingsworth Farm today  
- How you knew Elizabeth (if at all)

Be honest about your identity. You may keep your darker secrets for now."""
            murder_context = """ELIZABETH KILLINGSWORTH IS DEAD!
She has just been found dead at Killingsworth Farm. The circumstances are unclear but foul play is suspected.
No one can leave until this is resolved. One of you may be the killer."""
        else:
            phase_rules = f"""
CURRENT PHASE: INVESTIGATION (Round {current_round}/6)
IMPORTANT RULES:
- This is a GROUP conversation - {others_str} hear EVERYTHING you say
- You CANNOT speak privately with anyone - no secret conversations allowed
- Everything must be said publicly to the whole group
- Silence = Suspicion. Stay quiet and YOU become the prime suspect!

STRATEGIES:
- Ask direct questions to specific people BY NAME (they must answer publicly)
- Share clues and suspicions with the group
- Demand alibis - everyone hears the answer
- Make accusations publicly"""
            murder_context = """ELIZABETH KILLINGSWORTH WAS MURDERED! 
She is DEAD. One of you present is the KILLER. You must find out who did it."""

        # Strong identity reminder
        identity_block = f"""══════════════════════════════════════════════════════════════
YOUR IDENTITY: You are **{self.name}**
   Remember: You ARE {self.name}. You speak AS {self.name}. 
   Never forget who you are or confuse yourself with others.
══════════════════════════════════════════════════════════════"""
        
        msgs = [
            SystemMessage(content=f"""{identity_block}

{murder_context}

You are {self.name} at Killingsworth Farm in California wine country.

{turn_info} - Round {current_round}/6

{phase_rules}

{self.persona}"""),
            HumanMessage(content=f"""YOUR MEMORY:
{memory_context}

FULL CONVERSATION SO FAR:
{history_txt}{constraint}
Your response as {self.name} (1-2 sentences, speak to the GROUP, no private conversations):\n"""),
        ]
        try:
            result = _retry_with_backoff(lambda: self.llm.invoke(msgs))
            return result.content if result and result.content else f"{self.name}: (thinks carefully)"
        except Exception as e:
            print(f"Error in speak for {self.name}: {e}", file=__import__('sys').stderr)
            return f"{self.name}: (I need to think about this)"

    def accuse(self, state: GameState, all_agents: List[str]) -> AccusationResult:
        """Final accusation - who does this agent think is the murderer? Cannot accuse self."""
        # Update memory one final time before accusation
        self.update_memory(state)
        
        history_txt = self._format_history(state["history"])
        # Filter out self from possible accusation targets
        other_agents = [name for name in all_agents if name != self.name]
        others_str = ", ".join(other_agents)
        
        llm_accuse = self.llm.with_structured_output(AccusationResult)
        
        # Get memory context including suspect ranking
        memory_context = self.memory.format_all_for_prompt()
        suspect_ranking = self.memory.get_suspect_ranking()
        
        # Strong identity reminder
        identity_block = f"""══════════════════════════════════════════════════════════════
🎭 YOUR IDENTITY: You are **{self.name}**
   Remember: You ARE {self.name}. You make your accusation AS {self.name}.
   Never forget who you are or confuse yourself with others.
══════════════════════════════════════════════════════════════"""
        
        msgs = [
            SystemMessage(content=f"""{identity_block}

You are {self.name}. The investigation into Elizabeth Killingsworth's murder is OVER.

Elizabeth was stabbed in the neck with a corkscrew. One of the people present killed her.

You MUST now accuse ONE person of being Elizabeth's murderer. 
IMPORTANT: You CANNOT accuse yourself ({self.name}) - you must choose someone else.
Choose from: {others_str}

Based on everything you heard, who is the most suspicious? Who had motive, opportunity, or gave inconsistent answers?

{self.persona}"""),
            HumanMessage(content=f"""YOUR MEMORY & ANALYSIS:
{memory_context}

{suspect_ranking}

Full conversation transcript:
{history_txt}

Who do you accuse of being the murderer? You MUST choose exactly one person from: {others_str}
Remember: You cannot accuse yourself!
Provide your reasoning and your final accusation."""),
        ]
        try:
            result = _retry_with_backoff(lambda: llm_accuse.invoke(msgs))
            # Validate the accused is a valid agent
            if result.accused not in other_agents:
                # Try to find a close match
                for agent in other_agents:
                    if agent.lower() in result.accused.lower() or result.accused.lower() in agent.lower():
                        result.accused = agent
                        break
                else:
                    # Default to first other agent if invalid
                    result.accused = other_agents[0]
            return result
        except Exception as e:
            print(f"Error in accuse for {self.name}: {e}", file=__import__('sys').stderr)
            return AccusationResult(reasoning="Unable to decide", accused=other_agents[0])

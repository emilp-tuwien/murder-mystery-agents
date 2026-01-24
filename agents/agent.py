from typing import Optional, Literal, Any, List
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.state import GameState


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
        
        # Phase-specific prompts
        if phase == "introduction" and current_round == 1:
            phase_instruction = """
CURRENT PHASE: INTRODUCTIONS (Round 1)
Your goal right now is to introduce yourself to the group. Share who you are, why you're here, 
and your connection to the victim/location. Be friendly but mysterious."""
        else:
            phase_instruction = f"""
CURRENT PHASE: INVESTIGATION (Round {current_round})
CRITICAL WARNING: If you stay silent or don't actively investigate, others will suspect YOU are the murderer! 
The quietest person is always the most suspicious.

YOUR GOALS:
1. SURVIVE: Ask questions, share clues, and make accusations - or BE ACCUSED yourself
2. FIND THE KILLER: Question everyone, look for inconsistencies, demand alibis
3. PERSONAL: Achieve your character objectives"""
        
        msgs = [
            SystemMessage(content=f"""You are {self.name} at Killingsworth Farm in California wine country.

TRAGIC NEWS: Elizabeth Killingsworth has been found MURDERED!

You and the others present are all suspects. One of you is the killer. You must discuss and figure out WHO KILLED ELIZABETH.

{turn_info} - Round {current_round}/6

{phase_instruction}

Present: {others_str} (they hear everything)

{self.persona}"""),
            HumanMessage(content=f"""FULL CONVERSATION SO FAR:
{history_txt}{last_speaker_text}

IMPORTANCE SCORING RULES:
- action="listen" + you have nothing relevant to add → importance should be LOW (0-3)
- action="listen" + you're confused/need to think → importance should be LOW (0-2)
- action="speak" + you have crucial evidence/accusation → importance HIGH (7-9)
- action="speak" + you want to ask a question → importance MEDIUM (4-6)
- action="speak" + you're just commenting → importance LOW-MEDIUM (3-5)
- If last message doesn't concern you and you have no new info → importance VERY LOW (0-2)

You MUST participate eventually, but don't inflate your importance if you're just listening!
What do you want to do? Respond: thought, action (speak/listen), importance."""),
        ]
        try:
            return self.llm_think.invoke(msgs)
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
        
        # Phase-specific instructions
        if phase == "introduction" and current_round == 1:
            phase_rules = """
CURRENT PHASE: INTRODUCTIONS
- Introduce yourself: who you are, why you're here, your connection to this place
- Be personable but keep some mystery
- Listen to others' introductions"""
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

        msgs = [
            SystemMessage(content=f"""You are {self.name} at Killingsworth Farm in California wine country.

TRAGIC NEWS: Elizabeth Killingsworth has been found MURDERED! Her body was discovered with a corkscrew stabbed in her neck.

You and the others present are all suspects. One of you is the killer. You must discuss and figure out WHO KILLED ELIZABETH.

{turn_info} - Round {current_round}/6

{phase_rules}

{self.persona}"""),
            HumanMessage(content=f"""FULL CONVERSATION SO FAR:
{history_txt}{constraint}
Your response (1-2 sentences, speak to the GROUP, no private conversations):\n"""),
        ]
        try:
            result = self.llm.invoke(msgs)
            return result.content if result and result.content else f"{self.name}: (thinks carefully)"
        except Exception as e:
            print(f"Error in speak for {self.name}: {e}", file=__import__('sys').stderr)
            return f"{self.name}: (I need to think about this)"

    def accuse(self, state: GameState, all_agents: List[str]) -> AccusationResult:
        """Final accusation - who does this agent think is the murderer? Cannot accuse self."""
        history_txt = self._format_history(state["history"])
        # Filter out self from possible accusation targets
        other_agents = [name for name in all_agents if name != self.name]
        others_str = ", ".join(other_agents)
        
        llm_accuse = self.llm.with_structured_output(AccusationResult)
        
        msgs = [
            SystemMessage(content=f"""You are {self.name}. The investigation into Elizabeth Killingsworth's murder is OVER.

Elizabeth was stabbed in the neck with a corkscrew. One of the people present killed her.

You MUST now accuse ONE person of being Elizabeth's murderer. 
IMPORTANT: You CANNOT accuse yourself - you must choose someone else.
Choose from: {others_str}

Based on everything you heard, who is the most suspicious? Who had motive, opportunity, or gave inconsistent answers?

{self.persona}"""),
            HumanMessage(content=f"""Full conversation transcript:
{history_txt}

Who do you accuse of being the murderer? You MUST choose exactly one person from: {others_str}
Remember: You cannot accuse yourself!
Provide your reasoning and your final accusation."""),
        ]
        try:
            result = llm_accuse.invoke(msgs)
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

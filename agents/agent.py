from typing import Optional, Literal, Any, List
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
    def __init__(self, name: str, persona: str, llm: Any):
        self.name = name
        self.persona = persona
        self.llm = llm
        self.llm_think = llm.with_structured_output(ThinkResult)

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
        turn_info = f"[Turn {state['turn'] + 1} of 200]"
        
        # Debug: show history length
        # print(f"      [{self.name} sees {len(state['history'])} messages in history]")
        
        msgs = [
            SystemMessage(content=f"""You are {self.name} at Huxley's Demise. Huxley has been murdered. Found out who did it!

{turn_info}

CRITICAL WARNING: If you stay silent or don't actively investigate, others will suspect YOU are the murderer! 
The quietest person is always the most suspicious.

YOUR GOALS:
1. SURVIVE: Ask questions, share clues, and make accusations - or BE ACCUSED yourself
2. FIND THE KILLER: Question everyone, look for inconsistencies, demand alibis
3. PERSONAL: Achieve your character objectives

Present: {others_str} (they hear everything)

{self.persona}"""),
            HumanMessage(content=f"""FULL CONVERSATION SO FAR:
{history_txt}

You MUST participate or you'll be accused! What do you want to say or ask?
Rate your urgency (0-9): 9=must speak NOW, 0=can wait. Respond: thought, action (speak/listen), importance."""),
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
        turn_info = f"[Turn {state['turn'] + 1} of 200]"
        msgs = [
            SystemMessage(content=f"""You are {self.name} at Huxley's Demise. Huxley has been murdered. Found out who did it!


        {turn_info}

                    IMPORTANT RULES:
                    - This is a GROUP conversation - {others_str} hear EVERYTHING you say
                    - You CANNOT speak privately with anyone - no secret conversations allowed
                    - Everything must be said publicly to the whole group
                    - Silence = Suspicion. Stay quiet and YOU become the prime suspect!

                    STRATEGIES:
                    - Ask direct questions to specific people BY NAME (they must answer publicly)
                    - Share clues and suspicions with the group
                    - Demand alibis - everyone hears the answer
                    - Make accusations publicly

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
        """Final accusation - who does this agent think is the murderer?"""
        history_txt = self._format_history(state["history"])
        other_agents = [name for name in all_agents if name != self.name]
        others_str = ", ".join(other_agents)
        
        llm_accuse = self.llm.with_structured_output(AccusationResult)
        
        msgs = [
            SystemMessage(content=f"""You are {self.name}. The murder mystery discussion is OVER.

                You MUST now accuse ONE person of being the murderer. You cannot accuse yourself.
                Choose from: {others_str}

                Based on everything you heard, who is the most suspicious? Who had motive, opportunity, or gave inconsistent answers?

                {self.persona}"""),
                            HumanMessage(content=f"""Full conversation transcript:
                {history_txt}

                Who do you accuse of being the murderer? You MUST choose exactly one person from: {others_str}
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

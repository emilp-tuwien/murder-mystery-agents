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
    
    def add_round_summary(self, round_num: int, bullets: List[str]):
        """Store bullet point summary of a round in long-term memory."""
        self.memory.long_term.add_round_summary(round_num, bullets)
    
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
        """Use shared history window for prompts (last K_HISTORY turns only)."""
        return self.memory.shared_history.render_for_prompt()

    def think(self, state: GameState) -> ThinkResult:
        self.update_memory(state)
        
        current_round = state.get("current_round", 1)
        phase = state.get("phase", "introduction")
        
        # Build memory context using three-stage system
        memory_context = self.memory.build_prompt_context()
        
        # Round 1: Introduction phase
        if phase == "introduction" and current_round == 1:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

Elizabeth Killingsworth is DEAD. You must introduce yourself."""),
                HumanMessage(content=f"""{memory_context}

Actions:
• Speak: Introduce yourself.
• Listen: Wait for others.

Importance: 0-9 (9=must speak, 0=nothing to add)

Output: thought, action, importance."""),
            ]
        else:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

Elizabeth was MURDERED. Round {current_round}/6."""),
                HumanMessage(content=f"""{memory_context}

Actions:
• Speak: Ask questions or share info.
• Listen: Observe.

Importance: 0-9 (9=asked by name, 7-8=have evidence, 5-6=have info, 3-4=commenting, 1-2=nothing)

Output: thought, action, importance."""),
            ]
        
        try:
            result = _retry_with_backoff(lambda: self.llm_think.invoke(msgs))
            # Store thought in short-term memory
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

Elizabeth is DEAD. Introduce yourself."""),
                HumanMessage(content=f"""{memory_context}

Introduce yourself (1-2 sentences):"""),
            ]
        else:
            msgs = [
                SystemMessage(content=f"""Character Information:
You are {self.name}.
{self.persona}

Elizabeth was MURDERED. Round {current_round}/6."""),
                HumanMessage(content=f"""{memory_context}{constraint}

Rules: Only facts from your knowledge. No repetition. Can ask: {can_ask_str}

Respond (1-2 sentences):"""),
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
        
        llm_accuse = self.llm.with_structured_output(AccusationResult)
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

Provide reasoning and accusation."""),
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

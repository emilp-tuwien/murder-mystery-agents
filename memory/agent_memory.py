from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ShortTermMemory:
    """Recent conversation history."""
    window_size: int = 10
    messages: List[dict] = field(default_factory=list)
    
    def update(self, history: List[dict]):
        """Keep only the most recent messages within window."""
        self.messages = history[-self.window_size:] if history else []
    
    def get_recent(self) -> List[dict]:
        return self.messages


@dataclass 
class LongTermMemorySimple:
    """Simple long-term memory for clues and facts."""
    clues: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    
    def add_clue(self, clue: str):
        if clue and clue not in self.clues:
            self.clues.append(clue)
    
    def add_fact(self, fact: str):
        if fact and fact not in self.facts:
            self.facts.append(fact)
    
    def get_all_clues(self) -> List[str]:
        return self.clues
    
    def get_all_facts(self) -> List[str]:
        return self.facts


@dataclass
class SuspicionEntry:
    level: int = 0  # -10 to 10, 0 = neutral
    reasons: List[str] = field(default_factory=list)


class KnowledgeGraph:
    """Track relationships and suspicions."""
    
    def __init__(self):
        self.suspicions: Dict[str, SuspicionEntry] = {}
        self.relationships: Dict[str, Dict[str, str]] = {}  # person -> person -> relationship
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        """Update suspicion level for a person."""
        if target not in self.suspicions:
            self.suspicions[target] = SuspicionEntry()
        
        entry = self.suspicions[target]
        entry.level = max(-10, min(10, entry.level + delta))  # Clamp to [-10, 10]
        if reason and reason not in entry.reasons:
            entry.reasons.append(reason)
    
    def get_suspicion(self, target: str) -> Optional[SuspicionEntry]:
        return self.suspicions.get(target)
    
    def get_all_suspicions(self) -> Dict[str, SuspicionEntry]:
        return self.suspicions
    
    def get_ranked_suspects(self) -> List[tuple]:
        """Return suspects ranked by suspicion level (highest first)."""
        return sorted(
            [(name, entry.level, entry.reasons) for name, entry in self.suspicions.items()],
            key=lambda x: x[1],
            reverse=True
        )


class AgentMemory:
    """Layered memory system for agents."""
    
    def __init__(self, agent_name: str, short_term_window: int = 10):
        self.agent_name = agent_name
        self.short_term = ShortTermMemory(window_size=short_term_window)
        self.long_term = LongTermMemorySimple()
        self.knowledge_graph = KnowledgeGraph()
    
    def update_from_history(self, history: List[dict]):
        """Update short-term memory from conversation history."""
        self.short_term.update(history)
    
    def process_new_message(self, message: dict, turn: int):
        """Process a new message - could extract facts, update suspicions, etc."""
        # For now, just a placeholder for future processing
        pass
    
    def format_all_for_prompt(self) -> str:
        """Format all memory layers for inclusion in prompts."""
        parts = []
        
        # Long-term memory: clues
        clues = self.long_term.get_all_clues()
        if clues:
            parts.append("CLUES DISCOVERED:")
            for i, clue in enumerate(clues, 1):
                parts.append(f"  {i}. {clue}")
        
        # Long-term memory: facts
        facts = self.long_term.get_all_facts()
        if facts:
            parts.append("\nIMPORTANT FACTS:")
            for i, fact in enumerate(facts, 1):
                parts.append(f"  {i}. {fact}")
        
        # Knowledge graph: suspicions
        suspects = self.knowledge_graph.get_ranked_suspects()
        if suspects:
            parts.append("\nCURRENT SUSPICIONS:")
            for name, level, reasons in suspects:
                if level != 0:
                    level_str = f"+{level}" if level > 0 else str(level)
                    parts.append(f"  - {name}: {level_str} ({', '.join(reasons[:2]) if reasons else 'no specific reason'})")
        
        return "\n".join(parts) if parts else ""
    
    def get_suspect_ranking(self) -> str:
        """Get a formatted ranking of suspects."""
        suspects = self.knowledge_graph.get_ranked_suspects()
        if not suspects:
            return "No suspects identified yet."
        
        lines = ["Current suspect ranking:"]
        for i, (name, level, reasons) in enumerate(suspects, 1):
            level_str = f"+{level}" if level > 0 else str(level)
            reason_str = f" - {reasons[0]}" if reasons else ""
            lines.append(f"  {i}. {name} (suspicion: {level_str}){reason_str}")
        
        return "\n".join(lines)

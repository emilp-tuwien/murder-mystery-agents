"""
Multi-layered memory system for murder mystery agents.

Layers:
1. Full History - Complete conversation log (from state)
2. Short-Term Memory - Recent conversation window (last N messages)
3. Long-Term Memory - Summaries and important facts extracted from conversation
4. Knowledge Graph - Structured entities, relations, and evidence
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """A person, place, object, or concept in the game."""
    name: str
    entity_type: str  # "person", "object", "location", "time", "event"
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)


@dataclass 
class Relation:
    """A relationship or connection between two entities."""
    source: str  # entity name
    relation_type: str  # e.g., "was_with", "owns", "saw", "accused", "alibis"
    target: str  # entity name
    evidence: str = ""  # supporting quote or context
    turn_discovered: int = 0
    confidence: float = 1.0  # 0-1, how certain is this relation
    
    def __str__(self):
        return f"{self.source} --[{self.relation_type}]--> {self.target}"


@dataclass
class Evidence:
    """A piece of evidence or clue."""
    description: str
    source_speaker: str  # who revealed this
    turn: int
    evidence_type: str  # "alibi", "motive", "opportunity", "physical", "testimony"
    involves: List[str] = field(default_factory=list)  # entity names
    contradicts: Optional[str] = None  # reference to contradicting evidence


@dataclass
class Suspicion:
    """Tracks suspicion level toward a person."""
    target: str
    level: int = 0  # -5 (cleared) to +5 (prime suspect)
    reasons: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Recent conversation window - the last N messages.
    Useful for immediate context and direct responses.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._buffer: List[Dict[str, Any]] = []
    
    def update(self, history: List[Dict[str, Any]]):
        """Update buffer with most recent messages from full history."""
        self._buffer = history[-self.window_size:] if history else []
    
    def get_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the n most recent messages (or all in buffer)."""
        if n is None:
            return self._buffer.copy()
        return self._buffer[-n:] if n <= len(self._buffer) else self._buffer.copy()
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message."""
        return self._buffer[-1] if self._buffer else None
    
    def get_speakers_in_window(self) -> Set[str]:
        """Get all speakers who spoke in the recent window."""
        return {msg.get("speaker", "") for msg in self._buffer}
    
    def format_for_prompt(self) -> str:
        """Format recent history for inclusion in prompts."""
        if not self._buffer:
            return "(no recent conversation)"
        
        lines = []
        for msg in self._buffer:
            speaker = msg.get("speaker", "Unknown")
            text = msg.get("text", "").strip()
            lines.append(f"  {speaker}: {text}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Stores important facts, summaries, and learned information.
    Persists across the entire game.
    """
    
    def __init__(self):
        self.facts: List[str] = []  # Important facts learned
        self.summaries: Dict[int, str] = {}  # Round -> summary
        self.alibis: Dict[str, str] = {}  # Person -> their stated alibi
        self.motives: Dict[str, List[str]] = defaultdict(list)  # Person -> possible motives
        self.contradictions: List[Tuple[str, str, str]] = []  # (person, claim1, claim2)
        self.clues_received: List[str] = []  # Clues provided by game master
    
    def add_fact(self, fact: str):
        """Add an important fact."""
        if fact not in self.facts:
            self.facts.append(fact)
    
    def add_alibi(self, person: str, alibi: str):
        """Record someone's alibi."""
        self.alibis[person] = alibi
    
    def add_motive(self, person: str, motive: str):
        """Record a possible motive for someone."""
        if motive not in self.motives[person]:
            self.motives[person].append(motive)
    
    def add_contradiction(self, person: str, claim1: str, claim2: str):
        """Record when someone contradicted themselves."""
        self.contradictions.append((person, claim1, claim2))
    
    def add_clue(self, clue: str):
        """Add a clue from the game master."""
        if clue not in self.clues_received:
            self.clues_received.append(clue)
    
    def summarize_round(self, round_num: int, summary: str):
        """Store a summary of what happened in a round."""
        self.summaries[round_num] = summary
    
    def get_person_profile(self, person: str) -> str:
        """Get everything known about a person."""
        lines = [f"=== {person} ==="]
        
        if person in self.alibis:
            lines.append(f"Alibi: {self.alibis[person]}")
        
        if self.motives.get(person):
            lines.append(f"Possible motives: {', '.join(self.motives[person])}")
        
        # Check for contradictions involving this person
        person_contradictions = [c for c in self.contradictions if c[0] == person]
        if person_contradictions:
            lines.append("CONTRADICTIONS:")
            for _, c1, c2 in person_contradictions:
                lines.append(f"  - Said: '{c1}' BUT ALSO: '{c2}'")
        
        return "\n".join(lines) if len(lines) > 1 else f"(no information on {person})"
    
    def format_for_prompt(self) -> str:
        """Format long-term memory for inclusion in prompts."""
        sections = []
        
        if self.facts:
            sections.append("IMPORTANT FACTS:\n" + "\n".join(f"  • {f}" for f in self.facts[-10:]))
        
        if self.alibis:
            alibi_lines = [f"  • {p}: {a}" for p, a in self.alibis.items()]
            sections.append("KNOWN ALIBIS:\n" + "\n".join(alibi_lines))
        
        if self.contradictions:
            contra_lines = [f"  ⚠️ {p} said '{c1}' but also '{c2}'" for p, c1, c2 in self.contradictions[-5:]]
            sections.append("CONTRADICTIONS DETECTED:\n" + "\n".join(contra_lines))
        
        if self.clues_received:
            sections.append("CLUES FROM INVESTIGATION:\n" + "\n".join(f"  🔍 {c}" for c in self.clues_received))
        
        return "\n\n".join(sections) if sections else "(no long-term memories yet)"


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Structured representation of entities, relations, and evidence.
    Enables reasoning about connections and inconsistencies.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.evidence: List[Evidence] = []
        self.suspicions: Dict[str, Suspicion] = {}
    
    # ─── Entity Management ───────────────────────────────────────────────────
    
    def add_entity(self, name: str, entity_type: str, **attributes):
        """Add or update an entity."""
        if name in self.entities:
            self.entities[name].attributes.update(attributes)
        else:
            self.entities[name] = Entity(name=name, entity_type=entity_type, attributes=attributes)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        return self.entities.get(name)
    
    def get_people(self) -> List[str]:
        """Get all person entities."""
        return [name for name, e in self.entities.items() if e.entity_type == "person"]
    
    # ─── Relation Management ─────────────────────────────────────────────────
    
    def add_relation(self, source: str, relation_type: str, target: str, 
                     evidence: str = "", turn: int = 0, confidence: float = 1.0):
        """Add a relation between entities."""
        # Auto-create entities if they don't exist
        if source not in self.entities:
            self.add_entity(source, "person")
        if target not in self.entities:
            self.add_entity(target, "unknown")
        
        rel = Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            evidence=evidence,
            turn_discovered=turn,
            confidence=confidence
        )
        self.relations.append(rel)
    
    def get_relations_for(self, entity_name: str) -> List[Relation]:
        """Get all relations involving an entity (as source or target)."""
        return [r for r in self.relations 
                if r.source == entity_name or r.target == entity_name]
    
    def get_relations_of_type(self, relation_type: str) -> List[Relation]:
        """Get all relations of a specific type."""
        return [r for r in self.relations if r.relation_type == relation_type]
    
    def find_connections(self, entity1: str, entity2: str) -> List[Relation]:
        """Find direct relations between two entities."""
        return [r for r in self.relations 
                if (r.source == entity1 and r.target == entity2) or
                   (r.source == entity2 and r.target == entity1)]
    
    # ─── Evidence Management ─────────────────────────────────────────────────
    
    def add_evidence(self, description: str, speaker: str, turn: int,
                     evidence_type: str, involves: List[str] = None):
        """Add a piece of evidence."""
        ev = Evidence(
            description=description,
            source_speaker=speaker,
            turn=turn,
            evidence_type=evidence_type,
            involves=involves or []
        )
        self.evidence.append(ev)
    
    def get_evidence_about(self, entity_name: str) -> List[Evidence]:
        """Get all evidence involving an entity."""
        return [e for e in self.evidence if entity_name in e.involves]
    
    def get_evidence_by_type(self, evidence_type: str) -> List[Evidence]:
        """Get all evidence of a specific type."""
        return [e for e in self.evidence if e.evidence_type == evidence_type]
    
    # ─── Suspicion Tracking ──────────────────────────────────────────────────
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        """Update suspicion level for a person."""
        if target not in self.suspicions:
            self.suspicions[target] = Suspicion(target=target)
        
        self.suspicions[target].level = max(-5, min(5, self.suspicions[target].level + delta))
        self.suspicions[target].reasons.append(reason)
    
    def get_most_suspicious(self, n: int = 3) -> List[Tuple[str, int, List[str]]]:
        """Get the n most suspicious people."""
        sorted_suspects = sorted(
            self.suspicions.items(),
            key=lambda x: x[1].level,
            reverse=True
        )
        return [(s.target, s.level, s.reasons[-3:]) for _, s in sorted_suspects[:n]]
    
    # ─── Formatting for Prompts ──────────────────────────────────────────────
    
    def format_for_prompt(self) -> str:
        """Format knowledge graph for inclusion in prompts."""
        sections = []
        
        # People and their key relations
        people = self.get_people()
        if people:
            people_info = []
            for person in people:
                rels = self.get_relations_for(person)
                if rels:
                    rel_strs = [f"{r.relation_type}→{r.target}" for r in rels[:3]]
                    people_info.append(f"  {person}: {', '.join(rel_strs)}")
                else:
                    people_info.append(f"  {person}: (no known connections)")
            sections.append("PEOPLE & CONNECTIONS:\n" + "\n".join(people_info))
        
        # Top suspicions
        top_suspects = self.get_most_suspicious(3)
        if top_suspects and any(s[1] > 0 for s in top_suspects):
            suspect_lines = []
            for name, level, reasons in top_suspects:
                if level > 0:
                    bar = "█" * level + "░" * (5 - level)
                    suspect_lines.append(f"  [{bar}] {name} (+{level}): {reasons[-1] if reasons else 'suspicious'}")
            if suspect_lines:
                sections.append("SUSPICION LEVELS:\n" + "\n".join(suspect_lines))
        
        # Key evidence
        if self.evidence:
            ev_lines = [f"  • [{e.evidence_type}] {e.description}" for e in self.evidence[-5:]]
            sections.append("KEY EVIDENCE:\n" + "\n".join(ev_lines))
        
        return "\n\n".join(sections) if sections else "(no knowledge graph data yet)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export knowledge graph as dictionary."""
        return {
            "entities": {name: {"type": e.entity_type, "attrs": e.attributes} 
                        for name, e in self.entities.items()},
            "relations": [{"source": r.source, "type": r.relation_type, 
                          "target": r.target, "evidence": r.evidence}
                         for r in self.relations],
            "evidence": [{"desc": e.description, "type": e.evidence_type,
                         "speaker": e.source_speaker, "involves": e.involves}
                        for e in self.evidence],
            "suspicions": {name: {"level": s.level, "reasons": s.reasons}
                          for name, s in self.suspicions.items()}
        }


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED AGENT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class AgentMemory:
    """
    Unified memory system combining all four layers.
    Provides a single interface for agents to access their memories.
    """
    
    def __init__(self, agent_name: str, short_term_window: int = 10):
        self.agent_name = agent_name
        self.short_term = ShortTermMemory(window_size=short_term_window)
        self.long_term = LongTermMemory()
        self.knowledge_graph = KnowledgeGraph()
    
    def update_from_history(self, full_history: List[Dict[str, Any]]):
        """Update short-term memory from the full conversation history."""
        self.short_term.update(full_history)
    
    def process_new_message(self, message: Dict[str, Any], turn: int):
        """
        Process a new message and extract information for all memory layers.
        This is called after each message to update memories.
        """
        speaker = message.get("speaker", "Unknown")
        text = message.get("text", "").lower()
        
        # Add speaker as entity if not exists
        self.knowledge_graph.add_entity(speaker, "person")
        
        # Simple heuristic extraction (can be enhanced with LLM later)
        
        # Detect alibi statements
        if "i was" in text or "i went" in text:
            # Extract simple alibi
            alibi_text = message.get("text", "")
            self.long_term.add_alibi(speaker, alibi_text[:100])
            self.knowledge_graph.add_evidence(
                description=f"{speaker}'s alibi: {alibi_text[:100]}",
                speaker=speaker,
                turn=turn,
                evidence_type="alibi",
                involves=[speaker]
            )
        
        # Detect accusations
        if "you killed" in text or "murderer" in text or "suspect" in text:
            # Try to find who is being accused
            for word in text.split():
                if word.istitle() and word != speaker:
                    self.knowledge_graph.add_relation(
                        source=speaker,
                        relation_type="accused",
                        target=word,
                        evidence=message.get("text", "")[:50],
                        turn=turn
                    )
                    self.knowledge_graph.update_suspicion(word, 1, f"Accused by {speaker}")
                    break
        
        # Detect mentions of seeing someone
        if "saw" in text or "seen" in text or "noticed" in text:
            self.knowledge_graph.add_evidence(
                description=message.get("text", "")[:100],
                speaker=speaker,
                turn=turn,
                evidence_type="testimony",
                involves=[speaker]
            )
    
    def format_all_for_prompt(self, include_full_history: bool = False, 
                               full_history: List[Dict[str, Any]] = None) -> str:
        """
        Format all memory layers for inclusion in a prompt.
        
        Args:
            include_full_history: Whether to include the full conversation log
            full_history: The full history (if include_full_history is True)
        """
        sections = []
        
        # Layer 1: Full History (optional, usually too long)
        if include_full_history and full_history:
            sections.append("═══ FULL CONVERSATION LOG ═══\n(See conversation history above)")
        
        # Layer 2: Short-Term Memory
        recent = self.short_term.format_for_prompt()
        if recent and recent != "(no recent conversation)":
            sections.append(f"═══ RECENT CONTEXT (Last {self.short_term.window_size} messages) ═══\n{recent}")
        
        # Layer 3: Long-Term Memory
        long_term = self.long_term.format_for_prompt()
        if long_term and long_term != "(no long-term memories yet)":
            sections.append(f"═══ LONG-TERM MEMORY ═══\n{long_term}")
        
        # Layer 4: Knowledge Graph
        kg = self.knowledge_graph.format_for_prompt()
        if kg and kg != "(no knowledge graph data yet)":
            sections.append(f"═══ KNOWLEDGE GRAPH ═══\n{kg}")
        
        return "\n\n".join(sections) if sections else "(memory layers empty)"
    
    def get_suspect_ranking(self) -> str:
        """Get a formatted ranking of suspects."""
        suspects = self.knowledge_graph.get_most_suspicious(5)
        if not suspects or all(s[1] == 0 for s in suspects):
            return "(no suspects identified yet)"
        
        lines = ["Current suspect ranking:"]
        for i, (name, level, reasons) in enumerate(suspects, 1):
            if level != 0:
                sign = "+" if level > 0 else ""
                lines.append(f"  {i}. {name} ({sign}{level})")
        
        return "\n".join(lines)

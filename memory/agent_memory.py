"""
Three-stage memory system for murder mystery agents.

Memory Stages:
1. Shared History (global): Last K_HISTORY dialogue turns (sliding window)
2. shortTermHistory (per agent): Last K_SHORT thought/intent entries (sliding window)
3. longTermHistory (per agent): Compressed facts with embedding retrieval (top L_LONG)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np

# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================
K_HISTORY = 5   # Shared history window size
K_SHORT = 5     # Short-term thought window size
L_LONG = 5      # Top L facts to retrieve from long-term memory

MAX_LINE_CHARS = 120      # Max chars per line in prompt sections
MAX_SECTION_CHARS = 600   # Max chars per prompt section


# ============================================================================
# SHARED HISTORY (Global - singleton)
# ============================================================================
class SharedHistory:
    """
    Global shared history buffer for all agents.
    Keeps only the last K_HISTORY dialogue turns.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._history: List[Dict] = []
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for new games)."""
        if cls._instance:
            cls._instance._history = []
    
    def append(self, turn_id: int, speaker_id: str, text: str):
        """Append a new dialogue turn."""
        self._history.append({
            "turn_id": turn_id,
            "speaker_id": speaker_id,
            "text": text
        })
    
    def get_window(self) -> List[Dict]:
        """Return only the last K_HISTORY turns."""
        return self._history[-K_HISTORY:]
    
    def get_all(self) -> List[Dict]:
        """Return full history (for internal use only, not for prompts)."""
        return self._history
    
    def render_for_prompt(self) -> str:
        """
        Compact render for prompts.
        Format: "1) <speaker>: <text>" per line.
        Enforces truncation per line and total section length.
        """
        window = self.get_window()
        if not window:
            return "(no conversation yet)"
        
        lines = []
        total_chars = 0
        
        for i, entry in enumerate(window, 1):
            speaker = entry["speaker_id"]
            text = entry["text"]
            
            # Truncate text if too long
            if len(text) > MAX_LINE_CHARS - 20:
                text = text[:MAX_LINE_CHARS - 23] + "..."
            
            line = f"{i}) {speaker}: {text}"
            
            if total_chars + len(line) > MAX_SECTION_CHARS:
                lines.append("...")
                break
            
            lines.append(line)
            total_chars += len(line) + 1
        
        return "\n".join(lines)


# ============================================================================
# SHORT-TERM HISTORY (Per Agent) - Thoughts/Intents
# ============================================================================
@dataclass
class ShortTermHistory:
    """
    Per-agent short-term memory for thoughts/intents/plans.
    Keeps only the last K_SHORT entries.
    """
    entries: List[Dict] = field(default_factory=list)
    
    def add(self, thought: str, action: str = "", importance: int = 0):
        """Append a thought entry."""
        self.entries.append({
            "thought": thought,
            "action": action,
            "importance": importance
        })
    
    def get_window(self) -> List[Dict]:
        """Return only the last K_SHORT entries."""
        return self.entries[-K_SHORT:]
    
    def render_for_prompt(self) -> str:
        """Compact render as bullets with truncation."""
        window = self.get_window()
        if not window:
            return ""
        
        lines = []
        total_chars = 0
        
        for entry in window:
            thought = entry["thought"]
            if len(thought) > MAX_LINE_CHARS - 5:
                thought = thought[:MAX_LINE_CHARS - 8] + "..."
            
            line = f"• {thought}"
            
            if total_chars + len(line) > MAX_SECTION_CHARS:
                lines.append("...")
                break
            
            lines.append(line)
            total_chars += len(line) + 1
        
        return "\n".join(lines)


# ============================================================================
# LONG-TERM HISTORY (Per Agent) - Compressed Facts with Embeddings
# ============================================================================
@dataclass
class FactEntry:
    """A normalized fact entry with embedding."""
    id: str
    turn_id: int
    fact_text: str
    tags: List[str]
    embedding: Optional[np.ndarray] = None
    created_at: int = 0


class LongTermHistory:
    """
    Per-agent long-term memory with compressed facts and embedding retrieval.
    Stores normalized fact entries and retrieves top L_LONG by similarity.
    """
    
    def __init__(self, embedding_fn: Optional[callable] = None):
        self.facts: List[FactEntry] = []
        self._fact_texts: set = set()  # For deduplication
        self._embedding_fn = embedding_fn or self._default_embedding
    
    def _default_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based pseudo-embedding (replace with real embeddings in production)."""
        words = text.lower().split()
        vec = np.zeros(128)
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(128):
                vec[i] += ((h >> i) & 1) * 2 - 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def _generate_id(self, fact_text: str, turn_id: int) -> str:
        return hashlib.md5(f"{turn_id}:{fact_text}".encode()).hexdigest()[:12]
    
    def add_fact(self, turn_id: int, fact_text: str, tags: List[str] = None):
        """Add a normalized fact entry. Deduplicates by exact text."""
        if fact_text in self._fact_texts:
            return
        
        # Enforce 20-word limit
        words = fact_text.split()
        if len(words) > 20:
            fact_text = " ".join(words[:20]) + "..."
        
        self._fact_texts.add(fact_text)
        
        entry = FactEntry(
            id=self._generate_id(fact_text, turn_id),
            turn_id=turn_id,
            fact_text=fact_text,
            tags=tags or [],
            embedding=self._embedding_fn(fact_text),
            created_at=turn_id
        )
        self.facts.append(entry)
    
    def add_facts_batch(self, turn_id: int, facts: List[Dict]):
        """Add multiple facts. Format: [{"fact_text": "...", "tags": [...]}]"""
        for f in facts:
            self.add_fact(turn_id=turn_id, fact_text=f.get("fact_text", ""), tags=f.get("tags", []))
    
    def retrieve(self, query: str, top_k: int = L_LONG) -> List[FactEntry]:
        """Retrieve top L_LONG facts by cosine similarity."""
        if not self.facts:
            return []
        
        query_emb = self._embedding_fn(query)
        scored = []
        for fact in self.facts:
            if fact.embedding is not None:
                sim = np.dot(query_emb, fact.embedding)
                scored.append((sim, fact))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]
    
    def render_for_prompt(self, query: str) -> str:
        """Retrieve and render top L_LONG facts for prompt."""
        facts = self.retrieve(query, top_k=L_LONG)
        if not facts:
            return ""
        
        lines = []
        total_chars = 0
        
        for fact in facts:
            tag_str = f" [{','.join(fact.tags[:2])}]" if fact.tags else ""
            line = f"• {fact.fact_text}{tag_str}"
            
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS - 3] + "..."
            
            if total_chars + len(line) > MAX_SECTION_CHARS:
                lines.append("...")
                break
            
            lines.append(line)
            total_chars += len(line) + 1
        
        return "\n".join(lines)
    
    # Backwards compatibility
    def add_clue(self, clue: str):
        if clue:
            self.add_fact(0, f"CLUE: {clue}", tags=["clue"])
    
    def add_round_summary(self, round_num: int, bullets: List[str]):
        for bullet in bullets:
            self.add_fact(round_num * 100, bullet, tags=["summary", f"round{round_num}"])
    
    def get_all_clues(self) -> List[str]:
        return [f.fact_text for f in self.facts if "clue" in f.tags]
    
    def get_all_facts(self) -> List[str]:
        return [f.fact_text for f in self.facts]
    
    def get_all_summaries(self) -> str:
        summaries = [f for f in self.facts if "summary" in f.tags]
        if not summaries:
            return ""
        return "\n".join([f"• {s.fact_text}" for s in summaries[-10:]])


# ============================================================================
# SUSPICION TRACKING
# ============================================================================
@dataclass
class SuspicionEntry:
    level: int = 0
    reasons: List[str] = field(default_factory=list)


class KnowledgeGraph:
    """Track suspicions."""
    
    def __init__(self):
        self.suspicions: Dict[str, SuspicionEntry] = {}
    
    def update_suspicion(self, target: str, delta: int, reason: str):
        if target not in self.suspicions:
            self.suspicions[target] = SuspicionEntry()
        entry = self.suspicions[target]
        entry.level = max(-10, min(10, entry.level + delta))
        if reason and reason not in entry.reasons:
            entry.reasons.append(reason)
    
    def get_ranked_suspects(self) -> List[tuple]:
        return sorted(
            [(name, entry.level, entry.reasons) for name, entry in self.suspicions.items()],
            key=lambda x: x[1], reverse=True
        )


# ============================================================================
# FACT NORMALIZER (LLM-based)
# ============================================================================
VALID_TAGS = ["relationship", "alibi", "accusation", "observation", "location", "time", "motive", "secret", "contradiction", "other"]


class FactNormalizer:
    """Normalizes dialogue into structured facts using LLM."""
    
    def __init__(self, llm: Any = None):
        self.llm = llm
    
    def normalize(self, speaker: str, text: str, turn_id: int) -> List[Dict]:
        """Normalize a dialogue turn into fact entries."""
        if not self.llm:
            return self._simple_normalize(speaker, text)
        
        prompt = f"""Extract facts from this dialogue. Output JSON array only.
Speaker: {speaker}
Text: {text}

Rules:
- Each fact_text <= 20 words
- Tags from: {VALID_TAGS}
- Only stable facts (claims, alibis, relationships, observations)

Output: [{{"fact_text":"...", "tags":["tag"]}}]
JSON:"""
        
        try:
            from langchain_core.messages import HumanMessage
            result = self.llm.invoke([HumanMessage(content=prompt)])
            content = result.content if hasattr(result, 'content') else str(result)
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        
        return self._simple_normalize(speaker, text)
    
    def _simple_normalize(self, speaker: str, text: str) -> List[Dict]:
        """Fallback normalization without LLM."""
        words = text.split()
        if len(words) > 18:
            text = " ".join(words[:18]) + "..."
        return [{"fact_text": f"{speaker}: {text}", "tags": ["observation"]}]


# ============================================================================
# AGENT MEMORY (Main Interface)
# ============================================================================
class AgentMemory:
    """Complete memory system with all three stages."""
    
    def __init__(self, agent_name: str, short_term_window: int = K_SHORT, llm: Any = None, embedding_fn: callable = None):
        self.agent_name = agent_name
        
        # Stage 1: Shared history (singleton)
        self.shared_history = SharedHistory()
        
        # Stage 2: Short-term thoughts (per agent)
        self.short_term = ShortTermHistory()
        
        # Stage 3: Long-term facts with embeddings (per agent)
        self.long_term = LongTermHistory(embedding_fn=embedding_fn)
        
        # Suspicion tracking
        self.knowledge_graph = KnowledgeGraph()
        
        # Fact normalizer
        self.normalizer = FactNormalizer(llm=llm)
    
    def add_thought(self, thought: str, action: str = "", importance: int = 0):
        """Add to short-term thought history."""
        self.short_term.add(thought, action, importance)
    
    def process_dialogue(self, turn_id: int, speaker: str, text: str):
        """Process new dialogue: add to shared history + normalize to facts."""
        self.shared_history.append(turn_id, speaker, text)
        facts = self.normalizer.normalize(speaker, text, turn_id)
        self.long_term.add_facts_batch(turn_id, facts)
    
    def update_from_history(self, history: List[dict]):
        """Update from full history (backwards compatibility)."""
        pass  # Now handled by process_dialogue
    
    def process_new_message(self, message: dict, turn: int):
        """Process a new message (backwards compatibility)."""
        self.process_dialogue(turn, message.get("speaker", ""), message.get("text", ""))
    
    def build_prompt_context(self, query: str = "") -> str:
        """
        Build prompt context with all three memory stages.
        Sections: [SHARED_HISTORY_WINDOW] [SHORT_TERM_THOUGHTS] [LONG_TERM_FACTS] [SUSPICIONS]
        """
        sections = []
        
        # Section 1: Shared History Window (last K_HISTORY)
        history = self.shared_history.render_for_prompt()
        if history:
            sections.append(f"[CONVERSATION]\n{history}")
        
        # Section 2: Short-term Thoughts (last K_SHORT)
        thoughts = self.short_term.render_for_prompt()
        if thoughts:
            sections.append(f"[YOUR THOUGHTS]\n{thoughts}")
        
        # Section 3: Long-term Facts (top L_LONG by similarity)
        if not query:
            window = self.shared_history.get_window()
            query = window[-1]["text"] if window else self.agent_name
        
        facts = self.long_term.render_for_prompt(query)
        if facts:
            sections.append(f"[RELEVANT FACTS]\n{facts}")
        
        # Section 4: Suspicions
        suspects = self.knowledge_graph.get_ranked_suspects()
        if suspects:
            susp_lines = []
            for name, level, reasons in suspects[:3]:
                if level != 0:
                    lvl = f"+{level}" if level > 0 else str(level)
                    reason = reasons[0][:30] if reasons else ""
                    susp_lines.append(f"• {name}: {lvl} ({reason})")
            if susp_lines:
                sections.append(f"[SUSPICIONS]\n" + "\n".join(susp_lines))
        
        return "\n\n".join(sections)
    
    def format_all_for_prompt(self, query: str = "") -> str:
        """Alias for build_prompt_context."""
        return self.build_prompt_context(query)
    
    def get_suspect_ranking(self) -> str:
        suspects = self.knowledge_graph.get_ranked_suspects()
        if not suspects:
            return "No suspects identified."
        lines = ["Suspects:"]
        for i, (name, level, reasons) in enumerate(suspects[:5], 1):
            lvl = f"+{level}" if level > 0 else str(level)
            lines.append(f"  {i}. {name} ({lvl})")
        return "\n".join(lines)


# Backwards compatibility aliases
ShortTermMemory = ShortTermHistory
LongTermMemorySimple = LongTermHistory


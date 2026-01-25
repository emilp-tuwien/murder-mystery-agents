from typing import TypedDict, Dict, List, Optional, Any
from typing_extensions import Annotated
import operator

class Utterance(TypedDict):
    turn: int
    speaker: str
    text: str

class PendingObligation(TypedDict):
    addressee: str
    response_constraint: str
    from_speaker: str
    from_text: str

class ThoughtRecord(TypedDict):
    turn: int
    round: int
    agent: str
    action: str  # "speak" or "listen"
    importance: int
    thought: str

class GameState(TypedDict):
    turn: int
    current_round: int  # Current game round (1-6)
    conversations_in_round: int  # Count of conversations in current round
    conversations_per_round: int  # Default 20, configurable by game master
    history: Annotated[List[Utterance], operator.add]  # Use reducer to accumulate history
    pending_obligation: Optional[PendingObligation]

    # per-agent working buffers
    thoughts: Dict[str, Any]                 # ThinkResult per agent
    thoughts_history: Annotated[List[ThoughtRecord], operator.add]  # Track all thoughts for CSV export
    last_speaker: Optional[str]
    next_speaker: Optional[str]
    new_utterance: Optional[Utterance]
    done: bool
    
    # Phase tracking
    phase: str  # "introduction", "discussion", "accusation", "confession"

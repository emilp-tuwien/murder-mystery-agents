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

class GameState(TypedDict):
    turn: int
    history: Annotated[List[Utterance], operator.add]  # Use reducer to accumulate history
    pending_obligation: Optional[PendingObligation]

    # per-agent working buffers
    thoughts: Dict[str, Any]                 # ThinkResult per agent
    last_speaker: Optional[str]
    next_speaker: Optional[str]
    new_utterance: Optional[Utterance]
    done: bool

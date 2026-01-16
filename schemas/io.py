from pydantic import BaseModel, Field
from typing import Literal, Optional

class ThinkResult(BaseModel):
    thought: str
    action: Literal["speak", "listen"]
    importance: int = Field(ge=0, le=9)

class DesignationResult(BaseModel):
    has_first_pair_part: bool
    pair_type: Optional[str] = None          # e.g., "wh_question", "yes_no_question", "addressing"
    addressee: Optional[str] = None          # agent name
    response_constraint: Optional[str] = None # e.g., "(response: wh_question)"

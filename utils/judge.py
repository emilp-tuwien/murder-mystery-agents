from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


class NormanJudgeResult(BaseModel):
    verdict: str = Field(description="One of: useful, neutral, evasive, suspicious")
    reasoning: str = Field(description="Short explanation of why the response got this verdict")
    follow_up_question: str = Field(description="A sharp follow-up question to ask Norman next, if needed")


class NormanResponseJudge:
    def __init__(self, llm: Any):
        self.llm = llm
        self.llm_judge = llm.with_structured_output(NormanJudgeResult, method="json_mode")

    def evaluate(self, recent_history: str, norman_response: str) -> Optional[NormanJudgeResult]:
        msgs = [
            SystemMessage(content="""You are a murder mystery dialogue judge.
Evaluate Norman D'Adly's latest response from an investigative perspective.

Label the response as one of:
- useful: directly advances the investigation with concrete facts
- neutral: conversational but not very informative
- evasive: avoids the question, deflects, or stays vague
- suspicious: contains contradiction, implausibility, pressure points, or raises suspicion

Be concise and practical. Suggest one follow-up question that would help the group investigate further.
Return valid JSON with keys: verdict, reasoning, follow_up_question."""),
            HumanMessage(content=f"""RECENT CONTEXT:
{recent_history}

NORMAN'S RESPONSE:
{norman_response}

Evaluate the response."""),
        ]

        try:
            return self.llm_judge.invoke(msgs)
        except Exception:
            return None

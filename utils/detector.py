from schemas.io import DesignationResult
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from typing import List

# def make_detector(llm) -> callable:
#     def detect(text: str, agent_names: list[str]) -> DesignationResult:
#         prompt = f"""
# You are a conversation-structure analyzer.
# Given UTTERANCE, decide if it contains the FIRST PART of an adjacency pair
# (e.g., a question or direct address) that designates a specific next speaker.

# AGENTS: {agent_names}

# UTTERANCE:
# {text}

# Return JSON:
# {{
#   "has_first_pair_part": true/false,
#   "pair_type": "wh_question" | "yes_no_question" | "addressing" | "request" | ... | null,
#   "addressee": "<one of AGENTS>" | null,
#   "response_constraint": "(response: <pair_type>)" | null
# }}
# """
#         return llm.invoke_structured(prompt, schema=DesignationResult)
#     return detect


def make_detector(llm: ChatOllama):
    detector_llm = llm.with_structured_output(DesignationResult)

    def detect(text: str, agent_names: List[str]) -> DesignationResult:
        # For first test, keep it simple but real:
        msgs = [
            SystemMessage(content="You detect whether an utterance designates a specific next speaker (adjacency pair)."),
            HumanMessage(content=f"""AGENTS: {agent_names}
UTTERANCE: {text}

If it asks a question or directly addresses someone, set:
has_first_pair_part=true, addressee=<agent>, pair_type, response_constraint="(response: <pair_type>)".
Else has_first_pair_part=false."""),
        ]
        return detector_llm.invoke(msgs)

    return detect
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class LongTermItem:
    text: str
    meta: dict

class LongTermMemory:
    """
    Swap this with Chroma/FAISS/etc. later; keep interface stable.
    """
    def __init__(self, vectorstore):
        self.vs = vectorstore  # must support add_texts() and similarity_search()

    def add_facts(self, agent_name: str, facts: List[str]):
        texts = [f"[{agent_name}] {f}" for f in facts]
        metas = [{"agent": agent_name} for _ in facts]
        self.vs.add_texts(texts=texts, metadatas=metas)

    def retrieve(self, agent_name: str, query: str, k: int) -> List[str]:
        docs = self.vs.similarity_search(query, k=k, filter={"agent": agent_name})
        return [d.page_content for d in docs]

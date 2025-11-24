# evolver/rag_retriever.py
from __future__ import annotations
from typing import List, Any, Dict

class RAGRetriever:
    """
    Base class for RAG.
    Future stage will build an index on train_pool.
    """
    def __init__(self, train_pool: List[Any]):
        self.train_pool = train_pool

    def retrieve(self, essay: str, strategy: str = "none", k: int = 3) -> List[Dict[str, Any]]:
        return []

class DummyRAG(RAGRetriever):
    """v0 baseline: no retrieval."""
    def __init__(self, train_pool: List[Any]):
        super().__init__(train_pool)

    def retrieve(self, essay: str, strategy: str = "none", k: int = 3):
        return []

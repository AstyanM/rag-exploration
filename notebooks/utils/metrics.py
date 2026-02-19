"""
Custom metric functions for RAG exploration notebooks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""

    query: str
    docs: list[Document]
    scores: list[float]
    latency_ms: float


@dataclass
class IndexingMetrics:
    """Metrics collected during indexing."""

    num_raw_docs: int = 0
    num_chunks: int = 0
    chunk_sizes: list[int] = field(default_factory=list)
    indexing_time_s: float = 0.0

    @property
    def avg_chunk_size(self) -> float:
        return sum(self.chunk_sizes) / len(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def min_chunk_size(self) -> int:
        return min(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def max_chunk_size(self) -> int:
        return max(self.chunk_sizes) if self.chunk_sizes else 0

    def summary(self) -> dict:
        return {
            "raw_documents": self.num_raw_docs,
            "chunks": self.num_chunks,
            "avg_chunk_chars": round(self.avg_chunk_size),
            "min_chunk_chars": self.min_chunk_size,
            "max_chunk_chars": self.max_chunk_size,
            "indexing_time_s": round(self.indexing_time_s, 2),
        }


def timed_retrieval(
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
    search_type: str = "similarity",
) -> RetrievalResult:
    """Run a retrieval query and measure latency."""
    start = time.perf_counter()

    if search_type == "similarity":
        results = vectorstore.similarity_search_with_score(query, k=k)
    elif search_type == "mmr":
        # MMR doesn't return scores directly, use similarity for scoring
        results = vectorstore.similarity_search_with_score(query, k=k)
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    elapsed_ms = (time.perf_counter() - start) * 1000

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    return RetrievalResult(
        query=query,
        docs=docs,
        scores=scores,
        latency_ms=elapsed_ms,
    )


def benchmark_queries(
    vectorstore: VectorStore,
    queries: list[str],
    k: int = 5,
    search_type: str = "similarity",
) -> pd.DataFrame:
    """Run multiple queries and return a summary DataFrame."""
    results = []
    for query in queries:
        r = timed_retrieval(vectorstore, query, k=k, search_type=search_type)
        results.append({
            "query": query[:80],
            "num_results": len(r.docs),
            "top_score": r.scores[0] if r.scores else None,
            "avg_score": sum(r.scores) / len(r.scores) if r.scores else None,
            "latency_ms": round(r.latency_ms, 1),
            "top_source": r.docs[0].metadata.get("title", "?") if r.docs else "N/A",
        })
    return pd.DataFrame(results)

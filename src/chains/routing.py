"""
Query routing for multi-index RAG retrieval.

Routes queries to the most relevant sub-index before retrieval, improving
precision by searching only the relevant portion of the corpus.

Two strategies:
- Logical routing: LLM classifies the query into a category
- Semantic routing: Cosine similarity to pre-computed category centroids

Three categories (based on document `section` metadata):
- tutorials: langchain + langgraph guides (step-by-step, how-to)
- api_reference: reference docs (class/method signatures, parameters)
- concepts: concepts + contributing + deepagents (architecture, theory)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

VALID_CATEGORIES = ("tutorials", "api_reference", "concepts")

SECTION_TO_CATEGORY: dict[str, str] = {
    "langchain": "tutorials",
    "langgraph": "tutorials",
    "reference": "api_reference",
    "concepts": "concepts",
    "contributing": "concepts",
    "deepagents": "concepts",
}

# Map benchmark question categories to routing categories (for accuracy eval)
BENCHMARK_TO_ROUTING: dict[str, str] = {
    "how_to": "tutorials",
    "factual": "api_reference",
    "technical": "api_reference",
    "conceptual": "concepts",
    "error_related": "tutorials",  # troubleshooting and fix-it questions -> guides
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Classify the following question into exactly ONE category.

Categories:
- tutorials: step-by-step guides, how-to questions, building or implementing something
- api_reference: specific class names, method signatures, function parameters, return types
- concepts: architecture, design philosophy, theory, why questions, mental models

Examples:
Question: How do I add memory to a LangChain chain?
Category: tutorials

Question: What parameters does ChatOllama accept?
Category: api_reference

Question: What is the LCEL design philosophy?
Category: concepts

Question: How do I stream responses from a chain?
Category: tutorials

Question: What does VectorStore.similarity_search return?
Category: api_reference

Question: {question}
Category:""",
    ),
])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Output of a routing + retrieval run."""

    technique: str             # "logical" | "semantic" | "no_routing"
    query: str
    predicted_category: str    # "tutorials" | "api_reference" | "concepts" | "all"
    confidence: float          # cosine sim for semantic; 1.0 for logical; 0.0 for fallback
    docs: list[Document]
    elapsed_ms: float
    llm_calls: int             # 0 for semantic/no_routing, 1 for logical
    params: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.docs)

    def summary(self) -> dict:
        return {
            "technique": self.technique,
            "query": self.query[:80],
            "predicted_category": self.predicted_category,
            "confidence": round(self.confidence, 3),
            "num_results": self.num_results,
            "llm_calls": self.llm_calls,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by (source, page_content[:100]) key."""
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_centroids(
    category_docs: dict[str, list[Document]],
    embeddings: Embeddings,
    max_chars: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute centroid embeddings for each category.

    Embeds each document individually. Non-ASCII characters (scraper artifacts,
    mermaid diagram syntax) are stripped before embedding to avoid token-count
    inflation that causes context-length errors.

    Args:
        category_docs: Mapping from category name to list of documents.
        embeddings: Embedding model (must match the vectorstore's model).
        max_chars: Truncate each document to this many characters after
            stripping non-ASCII content.

    Returns:
        Mapping from category name to centroid vector (mean of all doc embeddings).
    """
    import re

    def _clean(text: str) -> str:
        # Remove non-ASCII characters (diagram syntax, replacement chars, etc.)
        return re.sub(r"[^\x00-\x7F]+", " ", text)[:max_chars]

    centroids: dict[str, np.ndarray] = {}
    for category, docs in category_docs.items():
        if not docs:
            continue
        all_vecs: list[list[float]] = []
        for doc in docs:
            text = _clean(doc.page_content)
            vec = embeddings.embed_documents([text])[0]
            all_vecs.append(vec)
        centroids[category] = np.mean(np.array(all_vecs), axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_logical(
    llm: BaseChatModel,
    query: str,
) -> str:
    """Classify query using LLM.

    Args:
        llm: Chat model to use for classification.
        query: User query to classify.

    Returns:
        Category name - one of "tutorials", "api_reference", "concepts".
        Falls back to "tutorials" if the LLM output is not recognized.
    """
    chain = ROUTING_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": query}).strip().lower()
    # Accept partial matches (e.g., "api_reference." -> "api_reference")
    for cat in VALID_CATEGORIES:
        if cat in raw:
            return cat
    return "tutorials"  # safe fallback


def classify_semantic(
    query_embedding: np.ndarray | list[float],
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    """Classify query by cosine similarity to category centroids.

    Args:
        query_embedding: Embedded query vector.
        centroids: Pre-computed category centroid vectors.

    Returns:
        Tuple of (best_category, confidence_score).
        Confidence is the cosine similarity to the best centroid (0.0-1.0).
    """
    q = np.array(query_embedding)
    best_cat = "tutorials"
    best_sim = -1.0

    for cat, centroid in centroids.items():
        sim = _cosine_similarity(q, centroid)
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    return best_cat, best_sim


# ---------------------------------------------------------------------------
# Main routing + retrieval
# ---------------------------------------------------------------------------

def route_and_retrieve(
    query: str,
    technique: str,
    collections: dict[str, VectorStore],
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    centroids: dict[str, np.ndarray] | None = None,
    k: int = 5,
    fallback_threshold: float = 0.3,
) -> RoutingResult:
    """Route query to the right sub-index and retrieve documents.

    Args:
        query: User query string.
        technique: "logical" (LLM-based) or "semantic" (embedding-based).
        collections: Mapping from category name to vectorstore.
        llm: Required for "logical" technique.
        embeddings: Required for "semantic" technique.
        centroids: Pre-computed centroids; required for "semantic" technique.
        k: Number of results to return.
        fallback_threshold: Min cosine similarity for semantic routing.
            Below this, search all collections (fallback mode).

    Returns:
        RoutingResult with retrieved documents and routing metadata.
    """
    start = time.perf_counter()
    llm_calls = 0
    confidence = 0.0
    predicted_category = "tutorials"

    if technique == "logical":
        if llm is None:
            raise ValueError("llm is required for logical routing")
        predicted_category = classify_logical(llm, query)
        llm_calls = 1
        confidence = 1.0

    elif technique == "semantic":
        if embeddings is None or centroids is None:
            raise ValueError("embeddings and centroids are required for semantic routing")
        q_emb = embeddings.embed_query(query)
        predicted_category, confidence = classify_semantic(q_emb, centroids)
        # Fallback to all collections if confidence is too low
        if confidence < fallback_threshold:
            predicted_category = "all"

    else:
        raise ValueError(f"Unknown technique: {technique!r}. Use 'logical' or 'semantic'.")

    # Retrieve documents
    if predicted_category == "all":
        # Fallback: search all collections and merge
        all_docs: list[Document] = []
        for coll in collections.values():
            all_docs.extend(coll.similarity_search(query, k=k))
        docs = _deduplicate(all_docs)[:k]
    else:
        target = collections.get(predicted_category)
        if target is None:
            # Category not in collections dict - fall back to all
            all_docs = []
            for coll in collections.values():
                all_docs.extend(coll.similarity_search(query, k=k))
            docs = _deduplicate(all_docs)[:k]
            predicted_category = "all"
        else:
            docs = target.similarity_search(query, k=k)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RoutingResult(
        technique=technique,
        query=query,
        predicted_category=predicted_category,
        confidence=confidence,
        docs=docs,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k, "fallback_threshold": fallback_threshold},
    )


# ---------------------------------------------------------------------------
# No-routing baseline
# ---------------------------------------------------------------------------

def retrieve_no_routing(
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
) -> RoutingResult:
    """Baseline similarity search without routing (searches full index).

    Args:
        vectorstore: Full corpus vectorstore.
        query: User query string.
        k: Number of results to return.

    Returns:
        RoutingResult with technique="no_routing".
    """
    start = time.perf_counter()
    docs = vectorstore.similarity_search(query, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return RoutingResult(
        technique="no_routing",
        query=query,
        predicted_category="all",
        confidence=0.0,
        docs=docs,
        elapsed_ms=elapsed_ms,
        llm_calls=0,
        params={"k": k},
    )

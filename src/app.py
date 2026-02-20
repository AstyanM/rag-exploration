"""Chainlit chat interface for the RAG pipeline.

Launch with: chainlit run src/app.py --port 8000
Or:          make app
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path

# Ensure project root is on sys.path (Chainlit loads this file directly)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import load_config
from src.embeddings.models import create_from_registry
from src.ingestion.chunkers import chunk_recursive
from src.ingestion.cleaners import clean_corpus
from src.ingestion.loaders import load_scraped_documents
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level initialization (runs once when Chainlit starts)
# ---------------------------------------------------------------------------

PROJECT_ROOT = _PROJECT_ROOT
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"
PERSIST_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "langchain_docs.json"

COLLECTION_NAME = "rag_app"

print("[init] Loading config...")
CONFIG = load_config(str(CONFIG_PATH))

print("[init] Loading embedding model...")
EMBEDDINGS, _emb_info = create_from_registry(
    "mxbai_large", config_path=str(MODELS_YAML)
)

print("[init] Loading LLM...")
LLM = ChatOllama(model="mistral:7b", temperature=0.0)

print("[init] Loading and processing corpus...")
_raw_docs = load_scraped_documents(str(DATA_PATH))
_cleaned_docs, _ = clean_corpus(_raw_docs, min_content_length=50)
_core_docs = [
    d for d in _cleaned_docs
    if "/python/integrations/" not in d.metadata.get("source", "")
]
_chunking_result = chunk_recursive(_core_docs, chunk_size=1000, chunk_overlap=200)
CHUNKS = _chunking_result.chunks
print(f"[init] Corpus ready: {len(_core_docs)} docs -> {len(CHUNKS)} chunks")

print("[init] Opening vector store...")
VECTORSTORE = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=EMBEDDINGS,
    persist_directory=str(PERSIST_DIR),
)
_existing = VECTORSTORE.get()
if len(_existing["ids"]) == 0:
    print(f"[init] Building collection '{COLLECTION_NAME}' ({len(CHUNKS)} chunks)...")
    VECTORSTORE = Chroma.from_documents(
        documents=CHUNKS,
        embedding=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    print("[init] Collection built.")
else:
    print(f"[init] Reusing collection '{COLLECTION_NAME}' ({len(_existing['ids'])} docs)")

print("[init] Warming up LLM...")
_ = LLM.invoke("Hi")
print("[init] Ready.")


# ---------------------------------------------------------------------------
# RAG mode presets
# ---------------------------------------------------------------------------

_DIRECT_LLM_MODE = "Direct LLM (no RAG)"

_MODE_CONFIGS = {
    "Simple (dense only)": {
        "retrieval": {"strategy": "similarity"},
        "reranking": {"enabled": False},
    },
    "Hybrid (BM25 + dense)": {
        "retrieval": {"strategy": "hybrid"},
        "reranking": {"enabled": False},
    },
    "Hybrid + Rerank": {
        "retrieval": {"strategy": "hybrid"},
        "reranking": {"enabled": True},
    },
}

_MODE_NAMES = list(_MODE_CONFIGS.keys()) + [_DIRECT_LLM_MODE]
_DEFAULT_MODE = "Hybrid + Rerank"


def _build_pipeline(mode: str, num_results: int) -> RAGPipeline:
    """Build a RAGPipeline with the given mode and result count."""
    cfg = deepcopy(CONFIG)
    overrides = _MODE_CONFIGS.get(mode, _MODE_CONFIGS[_DEFAULT_MODE])
    cfg["retrieval"]["strategy"] = overrides["retrieval"]["strategy"]
    cfg["retrieval"]["final_k"] = int(num_results)
    cfg["reranking"]["enabled"] = overrides["reranking"]["enabled"]
    cfg["reranking"]["top_k"] = int(num_results)
    return RAGPipeline(cfg, VECTORSTORE, CHUNKS, EMBEDDINGS, LLM)


# ---------------------------------------------------------------------------
# Conversation memory (simple reformulation)
# ---------------------------------------------------------------------------

_MAX_HISTORY_TURNS = 3


def _format_history(history: list[dict]) -> str:
    """Format chat history as a simple string."""
    lines = []
    for msg in history[-_MAX_HISTORY_TURNS * 2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines)


async def _reformulate(question: str, history: list[dict]) -> str:
    """Reformulate a follow-up question using conversation history.

    Returns the original question if history is empty or reformulation fails.
    """
    if len(history) < 2:
        return question

    prompt = (
        "Given the conversation history below and a follow-up question, "
        "rewrite the follow-up question as a standalone question that "
        "captures all necessary context. Only output the standalone question, "
        "nothing else.\n\n"
        f"Conversation:\n{_format_history(history)}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    try:
        response = await LLM.ainvoke([HumanMessage(content=prompt)])
        standalone = response.content.strip()
        if 5 < len(standalone) < 500:
            return standalone
    except Exception:
        logger.warning("Reformulation failed, using original question")

    return question


# ---------------------------------------------------------------------------
# Chainlit lifecycle hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialize per-session state and settings panel."""
    pipeline = _build_pipeline(_DEFAULT_MODE, num_results=5)
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("history", [])
    cl.user_session.set("settings", {
        "rag_mode": _DEFAULT_MODE,
        "num_results": 5,
        "show_sources": True,
        "conversation_memory": True,
    })

    await cl.ChatSettings(
        [
            Select(
                id="rag_mode",
                label="RAG Mode",
                values=_MODE_NAMES,
                initial_index=_MODE_NAMES.index(_DEFAULT_MODE),
            ),
            Slider(
                id="num_results",
                label="Number of results",
                initial=5,
                min=1,
                max=15,
                step=1,
            ),
            Switch(
                id="show_sources",
                label="Show source documents",
                initial=True,
            ),
            Switch(
                id="conversation_memory",
                label="Conversation memory",
                initial=True,
            ),
        ]
    ).send()

    summary = pipeline.component_summary()
    await cl.Message(
        content=(
            f"Pipeline ready. Mode: **{_DEFAULT_MODE}**\n"
            f"- Retrieval: {summary['retrieval_strategy']} (k={summary['retrieval_k']})\n"
            f"- Reranking: {'enabled' if summary['reranking_enabled'] else 'disabled'}\n"
            f"- LLM: {summary['llm']}\n"
            f"- Corpus: {len(CHUNKS)} chunks"
        ),
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Rebuild the pipeline when user changes settings."""
    old_settings = cl.user_session.get("settings", {})
    cl.user_session.set("settings", settings)

    mode = settings.get("rag_mode", _DEFAULT_MODE)
    num_results = int(settings.get("num_results", 5))

    mode_changed = mode != old_settings.get("rag_mode")
    k_changed = num_results != old_settings.get("num_results")

    if mode == _DIRECT_LLM_MODE and mode_changed:
        await cl.Message(
            content=(
                f"Settings updated. Mode: **{mode}**\n"
                "- Mistral 7B answers directly, no retrieval or context"
            ),
        ).send()
    elif (mode_changed or k_changed) and mode != _DIRECT_LLM_MODE:
        pipeline = _build_pipeline(mode, num_results)
        cl.user_session.set("pipeline", pipeline)
        summary = pipeline.component_summary()
        await cl.Message(
            content=(
                f"Settings updated. Mode: **{mode}**\n"
                f"- Retrieval: {summary['retrieval_strategy']} "
                f"(k={num_results})\n"
                f"- Reranking: "
                f"{'enabled' if summary['reranking_enabled'] else 'disabled'}"
            ),
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages: retrieve, stream generation, display sources."""
    pipeline: RAGPipeline = cl.user_session.get("pipeline")
    settings = cl.user_session.get("settings", {})
    history: list[dict] = cl.user_session.get("history", [])

    question = message.content
    mode = settings.get("rag_mode", _DEFAULT_MODE)

    # Reformulate follow-up questions if conversation memory is enabled
    if settings.get("conversation_memory", True) and len(history) >= 2:
        question = await _reformulate(question, history)

    # Create response message for streaming
    msg = cl.Message(content="")

    # --- Direct LLM mode (no retrieval) ---
    if mode == _DIRECT_LLM_MODE:
        async for chunk in LLM.astream([HumanMessage(content=question)]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            await msg.stream_token(token)

        await msg.send()

        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", history)
        return

    # --- RAG mode ---
    source_docs = []

    async for event in pipeline.astream(question):
        if event["type"] == "retrieval":
            docs = event["docs"]
            source_docs = docs

            # Build detailed retrieval output with chunk previews
            lines = [
                f"Retrieved **{len(docs)}** documents "
                f"in {event['retrieval_ms']:.0f} ms\n"
            ]
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", f"Source {i + 1}")
                source = doc.metadata.get("source", "")
                preview = doc.page_content[:200].replace("\n", " ")
                if len(doc.page_content) > 200:
                    preview += "..."
                lines.append(f"**[{i + 1}] {title}**")
                if source:
                    lines.append(f"  *{source}*")
                lines.append(f"  {preview}\n")

            async with cl.Step(
                name="Retrieval",
                type="retrieval",
                default_open=False,
            ) as step:
                step.output = "\n".join(lines)

        elif event["type"] == "token":
            await msg.stream_token(event["token"])

        elif event["type"] == "done":
            pass

    # Append source references to the answer
    # Element names must appear verbatim in the message for Chainlit to link them
    if settings.get("show_sources", True) and source_docs:
        elements = []
        source_lines = ["\n\n---\n"]
        source_lines.append(
            f"**{len(source_docs)} sources** - click a source to view "
            "the full retrieved chunk:\n"
        )
        for i, doc in enumerate(source_docs):
            title = doc.metadata.get("title", f"Source {i + 1}")
            source = doc.metadata.get("source", "")
            short_source = source.split("/")[-1] if "/" in source else source
            char_count = len(doc.page_content)
            el_name = f"[{i + 1}] {title}"

            source_lines.append(
                f"{i + 1}. {el_name} - "
                f"*{short_source}* ({char_count} chars)"
            )

            # Build rich side panel content
            panel_lines = []
            panel_lines.append(f"# {title}\n")
            if source:
                panel_lines.append(f"**URL:** {source}\n")
            panel_lines.append(f"**Chunk length:** {char_count} characters\n")
            panel_lines.append("---\n")
            panel_lines.append(doc.page_content)

            elements.append(
                cl.Text(
                    name=el_name,
                    content="\n".join(panel_lines),
                    display="side",
                )
            )
        await msg.stream_token("\n".join(source_lines))
        msg.elements = elements

    await msg.send()

    # Update conversation history
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("history", history)

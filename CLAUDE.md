# CLAUDE.md

## Project

RAG From Scratch - A local RAG pipeline exploration project using LangChain, Ollama, and ChromaDB.
The full spec is in `RAG_PROJECT.md`.

## Current Progress

- [x] Phase 0: Project structure, dependencies, configs
- [x] Phase 1: Scraping, loading, cleaning, indexing, validation, baseline metrics
- [x] Phase 2: Chunking strategies (`02_chunking_strategies.ipynb`, `src/ingestion/chunkers.py`)
- [x] Phase 3: Embeddings comparison (`03_embeddings_comparison.ipynb`, `src/embeddings/models.py`)
- [x] Phase 4: Retrieval methods comparison (`04_retrieval_methods.ipynb`, `src/retrieval/`)
- [x] Phase 5: Query translation (`05_query_translation.ipynb`, `src/chains/query_translation.py`)
- [x] Phase 6: Routing (`06_routing.ipynb`, `src/chains/routing.py`)
- [x] Phase 7: Re-ranking (`07_reranking.ipynb`, `src/retrieval/reranker.py`)
- [x] Phase 8: Advanced RAG Patterns (`08_advanced_rag.ipynb`, `src/chains/advanced.py`)
- [ ] Phase 9-12: See `RAG_PROJECT.md`

## Tech Stack

- Python 3.13, venv in `.venv/`
- LLM: Mistral 7B via Ollama (already pulled)
- Embeddings: nomic-embed-text via Ollama (already pulled)
- Vector store: ChromaDB (persisted in `vectorstore/chroma_db/`)
- Framework: LangChain, LangGraph
- Frontend (later): Chainlit
- All 100% local, no external API calls

## Key Paths

- `data/raw/` - Scraped JSON docs (langchain_docs*.json)
- `data/processed/` - Chunked documents (not yet populated)
- `data/evaluation/` - Benchmark Q&A dataset (not yet created)
- `notebooks/` - Exploration notebooks (numbered by phase)
- `notebooks/utils/` - Shared helpers (display.py, metrics.py)
- `src/` - Production application modules
- `configs/default.yaml` - Pipeline config
- `configs/models.yaml` - Model registry
- `results/` - Experiment outputs (JSON)

## Style Rules

- Never use em dashes (the long dash character). Use regular dashes, commas, or parentheses instead.
- Language: everything in English (code, comments, notebooks, user interaction)
- Notebooks: each phase has one numbered notebook, self-contained with explanations
- Keep notebooks/utils/ helpers generic and reusable across phases
- Follow existing patterns in `src/` (type hints, docstrings, pathlib)

## Commands

- `ollama serve` - Start Ollama (usually already running)
- `python scripts/scrape_docs.py` - Re-scrape docs from GitHub
- Notebooks run from the `notebooks/` directory (PROJECT_ROOT = parent)

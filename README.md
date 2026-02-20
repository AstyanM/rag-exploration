# RAG From Scratch

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-green)
![Ollama](https://img.shields.io/badge/Ollama-local-purple)
![Chainlit](https://img.shields.io/badge/Chainlit-1.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete Retrieval-Augmented Generation pipeline built from scratch, systematically exploring every component through **10 benchmark notebooks** and assembling the best techniques into a **production Chainlit chat application**.
Corpus: **LangChain Python documentation** (~1,500 chunks). Models: **Mistral 7B** + **mxbai-embed-large** via Ollama.
Zero cloud dependency - everything runs 100% locally.

---

## Motivation

Most RAG tutorials demonstrate a single "hello world" pipeline: load docs, chunk, embed, retrieve, generate. This project goes deeper - it treats RAG as a **system design problem**, systematically benchmarking every component and quantifying the impact of each decision on end-to-end quality.

| Phase                 | What was explored           | Components tested                                       | Key finding                                            |
| --------------------- | --------------------------- | ------------------------------------------------------- | ------------------------------------------------------ |
| **Chunking**          | 5 splitting strategies      | Fixed, recursive, token, markdown, semantic             | Recursive (1000/200) best speed/quality balance        |
| **Embeddings**        | 4 models across 2 providers | MiniLM, nomic, BGE, mxbai-embed-large                   | mxbai-embed-large: 2x better semantic separation       |
| **Retrieval**         | 8+ strategies               | Similarity, MMR, BM25, hybrid, multi-query              | Hybrid (BM25+dense) captures keyword + semantic        |
| **Query Translation** | 5 LLM-based techniques      | HyDE, RAG Fusion, step-back, multi-query, decomposition | HyDE +6.7% MRR but 141x latency - not worth it locally |
| **Routing**           | 2 routing approaches        | Logical (LLM), semantic (centroids)                     | Routing hurts on imbalanced corpora                    |
| **Reranking**         | 2 reranking methods         | Cross-encoder, LLM-as-judge                             | Cross-encoder: +20% faithfulness for 78x latency       |
| **Advanced RAG**      | 3 LangGraph patterns        | CRAG, Self-RAG, Adaptive RAG                            | Only CRAG works with 7B models                         |
| **Evaluation**        | RAGAS end-to-end            | Naive vs hybrid+reranked vs HyDE                        | Hybrid+reranked: 0.87 avg score                        |

The final pipeline achieves **0.874 average RAGAS score** (faithfulness 0.92, relevancy 0.85, precision 0.95, recall 0.78), all targets passed, running entirely on consumer hardware.

---

## Features

- **10 exploration notebooks** - each phase compares multiple approaches on the same 25-question benchmark with expert-written ground truth
- **Production Chainlit chat app** - streaming answers, source display, conversation memory, 4 RAG modes
- **Hybrid retrieval** - BM25 keyword matching + dense vector search merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** - ms-marco-MiniLM re-scores candidates for +20% faithfulness
- **RAGAS evaluation** - faithfulness, answer relevancy, context precision, context recall
- **Advanced RAG patterns** - CRAG, Self-RAG, Adaptive RAG implemented with LangGraph
- **Comparison mode** - side-by-side RAG vs Direct LLM answers in the chat UI
- **Conversation memory** - LLM-based follow-up reformulation with 3-turn history
- **Configurable pipeline** - swap any component via YAML without code changes
- **JSON conversation persistence** - resume previous chat threads
- **25-question benchmark** - 4 categories (conceptual, technical, how-to, error-related) with ground truth
- **30+ configurations benchmarked** - every decision backed by quantitative data

---

## Architecture Overview

```
GitHub (langchain-ai/docs)
         |
         v
┌────────────────────────────────────────────────────┐
│              Data Ingestion Pipeline               │
│                                                    │
│  scraper.py ──> loaders.py ──> cleaners.py         │
│                                (8-step pipeline)   │
│  1,463 raw docs ──> 130 core docs (filter noise)   │
│                         |                          │
│                    chunkers.py                     │
│              (recursive, 1000/200)                 │
│                    ~1,500 chunks                   │
└────────────────────┬───────────────────────────────┘
                     |
         ┌───────────┴───────────┐
         │   ChromaDB + Ollama   │
         │  mxbai-embed-large    │
         │     (1024 dims)       │
         └───────────┬───────────┘
                     |
┌────────────────────┴───────────────────────────────┐
│              Retrieval Pipeline                    │
│                                                    │
│  ┌──────────┐  ┌──────────┐                        │
│  │  Dense   │  │   BM25   │   Hybrid (RRF)         │
│  │ (vector) │  │(keyword) │──> Ensemble            │
│  └────┬─────┘  └────┬─────┘       |                │
│       └──────┬───────┘             v               │
│              |            Cross-encoder Reranker   │
│              |           (ms-marco-MiniLM-L-6-v2)  │
│              └─────────────────┐                   │
│                                v                   │
│                         Top-5 documents            │
└────────────────────────────┬───────────────────────┘
                             |
                    ┌────────┴────────┐
                    │   Mistral 7B    │
                    │    (Ollama)     │
                    │  temp=0.0       │
                    │  context=4096   │
                    └────────┬────────┘
                             |
                    ┌────────┴────────┐
                    │    Chainlit     │
                    │  Streaming UI   │
                    │  Sources panel  │
                    │  Settings       │
                    │  Memory         │
                    └─────────────────┘
```

**No external API required** - Ollama runs Mistral 7B and mxbai-embed-large locally. ChromaDB persists on disk. The entire pipeline runs on a single machine.

---

## Tech Stack

| Layer         | Technology                           | Details                                            |
| ------------- | ------------------------------------ | -------------------------------------------------- |
| LLM           | Mistral 7B via Ollama                | 4096 token context, temperature=0.0                |
| Embeddings    | mxbai-embed-large via Ollama         | 1024 dimensions, highest semantic separation       |
| Vector store  | ChromaDB                             | Persistent local storage, cosine similarity        |
| Sparse search | BM25 (rank-bm25)                     | Keyword-based retrieval via Reciprocal Rank Fusion |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint (query, doc) relevance scoring               |
| Framework     | LangChain + LangGraph                | Orchestration, chains, stateful graph patterns     |
| Evaluation    | RAGAS                                | Faithfulness, relevancy, precision, recall         |
| Frontend      | Chainlit                             | Streaming chat UI, settings panel, source display  |
| Configuration | YAML                                 | Centralized pipeline config + model registry       |
| Notebooks     | Jupyter                              | 10 self-contained exploration phases               |

---

## Project Structure

```
rag-exploration/
├── Makefile                          # Build & run commands
├── requirements.txt                  # Python dependencies
├── configs/
│   ├── default.yaml                  # Pipeline configuration
│   └── models.yaml                   # Model registry (4 embeddings, 2 rerankers)
│
├── notebooks/                        # Exploration phase (10 notebooks)
│   ├── 01_indexing_basics.ipynb      # Scraping, cleaning, baseline indexing
│   ├── 02_chunking_strategies.ipynb  # 5 strategies compared
│   ├── 03_embeddings_comparison.ipynb # 4 models compared
│   ├── 04_retrieval_methods.ipynb    # 8+ strategies compared
│   ├── 05_query_translation.ipynb    # HyDE, RAG Fusion, step-back, etc.
│   ├── 06_routing.ipynb              # Logical + semantic routing
│   ├── 07_reranking.ipynb            # Cross-encoder + LLM-as-judge
│   ├── 08_advanced_rag.ipynb         # CRAG, Self-RAG, Adaptive RAG (LangGraph)
│   ├── 09_evaluation_ragas.ipynb     # RAGAS evaluation of 3 configs
│   ├── 10_full_pipeline.ipynb        # Final assembly + full benchmark
│   └── utils/                        # Shared helpers
│       ├── display.py                # Radar charts, comparison tables
│       └── metrics.py                # MRR, nDCG, Precision@k, Recall@k
│
├── src/                              # Production application
│   ├── app.py                        # Chainlit chat interface
│   ├── pipeline.py                   # Unified RAG pipeline (sync + async streaming)
│   ├── data_layer.py                 # JSON-based conversation persistence
│   ├── config.py                     # YAML config loader
│   ├── ingestion/
│   │   ├── scraper.py                # GitHub docs scraper
│   │   ├── loaders.py                # Document loaders
│   │   ├── cleaners.py               # 8-step text cleaning pipeline
│   │   └── chunkers.py              # 5 chunking strategies
│   ├── embeddings/
│   │   └── models.py                 # Model registry + factory
│   ├── retrieval/
│   │   ├── dense.py                  # Vector similarity / MMR
│   │   ├── sparse.py                 # BM25
│   │   ├── hybrid.py                 # Ensemble (RRF)
│   │   ├── factory.py                # Config-driven retriever builder
│   │   └── reranker.py               # Cross-encoder + LLM-as-judge
│   ├── chains/
│   │   ├── query_translation.py      # HyDE, RAG Fusion, multi-query, etc.
│   │   ├── routing.py                # Logical + semantic routing
│   │   └── advanced.py               # CRAG, Self-RAG, Adaptive RAG (LangGraph)
│   └── evaluation/
│       └── evaluator.py              # RAGAS wrapper + sample collection
│
├── data/
│   ├── raw/                          # Scraped JSON docs (1,463 pages)
│   └── evaluation/                   # 25 benchmark questions + ground truth
│
├── results/                          # Experiment outputs (10 JSON files)
├── scripts/
│   └── scrape_docs.py                # Documentation scraper CLI
├── public/                           # Chainlit custom CSS
├── chainlit.md                       # Chat welcome page
└── vectorstore/
    └── chroma_db/                    # ChromaDB persistent storage
```

---

## Prerequisites

| Requirement          | Version          | Install                                         |
| -------------------- | ---------------- | ----------------------------------------------- |
| **Python**           | 3.13+            | [python.org](https://www.python.org/downloads/) |
| **Ollama**           | latest           | [ollama.com](https://ollama.com/)               |
| **GPU** _(optional)_ | NVIDIA with CUDA | Accelerates Mistral 7B inference                |

> **Note**: A GPU is **not required**. Mistral 7B runs on CPU (slower inference, ~6s/query vs ~3s on GPU). The embedding model (mxbai-embed-large) and cross-encoder reranker run on CPU by default.

---

## Installation

### Quick setup

```bash
# 1. Clone
git clone https://github.com/AstyanM/rag-exploration.git
cd rag-exploration

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# 3. Pull Ollama models
ollama pull mistral:7b
ollama pull mxbai-embed-large

# 4. Scrape the corpus (or use pre-scraped data if included)
python scripts/scrape_docs.py
```

The Makefile provides shortcuts for common operations:

```bash
make setup          # Create venv + install deps
make scrape         # Scrape LangChain docs
make scrape-sample  # Scrape 50 pages (for testing)
make app            # Launch Chainlit chat (port 8000)
make clean          # Remove generated data
make all            # Full pipeline: scrape -> index -> evaluate -> app
```

---

## Usage

### Launch the chat app

```bash
make app
# or directly:
chainlit run src/app.py --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

### Run the notebooks

Notebooks run from the `notebooks/` directory. Each is self-contained and numbered by phase:

```bash
cd notebooks
jupyter notebook
```

### RAG modes

The Chainlit app offers 4 modes, selectable from the settings panel:

| Mode                | Retrieval             | Reranking     | Use case                       |
| ------------------- | --------------------- | ------------- | ------------------------------ |
| **Simple**          | Dense similarity only | None          | Fast baseline                  |
| **Hybrid**          | BM25 + dense (RRF)    | None          | Keyword + semantic             |
| **Hybrid + Rerank** | BM25 + dense (RRF)    | Cross-encoder | Best quality (default)         |
| **Direct LLM**      | None (bypasses RAG)   | None          | Compare with/without retrieval |

### Chat features

| Feature        | Description                                           |
| -------------- | ----------------------------------------------------- |
| **Streaming**  | Token-by-token generation with real-time display      |
| **Sources**    | Expandable side panels showing retrieved chunks       |
| **Timing**     | Footer with retrieval, generation, and total latency  |
| **Memory**     | Follow-up questions reformulated using 3-turn history |
| **Comparison** | Side-by-side RAG vs Direct LLM answers                |
| **History**    | Conversations persisted as JSON, resumable threads    |

---

## Configuration

All pipeline settings live in `configs/default.yaml`.

| Setting                     | Default                                | Description                                             |
| --------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `llm.model`                 | `mistral:7b`                           | Ollama LLM model                                        |
| `llm.temperature`           | `0.0`                                  | Generation temperature (0 = deterministic)              |
| `llm.num_ctx`               | `4096`                                 | Context window size                                     |
| `embeddings.model`          | `mxbai-embed-large`                    | Embedding model (1024 dims)                             |
| `chunking.strategy`         | `recursive`                            | Chunking method                                         |
| `chunking.chunk_size`       | `1000`                                 | Target chunk size in characters                         |
| `chunking.chunk_overlap`    | `200`                                  | Overlap between chunks                                  |
| `retrieval.strategy`        | `hybrid`                               | Retrieval method: `similarity`, `mmr`, `bm25`, `hybrid` |
| `retrieval.hybrid.weights`  | `[0.5, 0.5]`                           | Dense/sparse weight balance                             |
| `retrieval.final_k`         | `5`                                    | Number of results returned                              |
| `reranking.enabled`         | `true`                                 | Enable cross-encoder reranking                          |
| `reranking.model`           | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model                                          |
| `query_translation.enabled` | `false`                                | Enable query transformation (HyDE, etc.)                |
| `routing.enabled`           | `false`                                | Enable query routing to sub-indexes                     |

The model registry (`configs/models.yaml`) defines all available embedding models:

| Model         | Provider              | Dimensions | Notes                  |
| ------------- | --------------------- | ---------- | ---------------------- |
| `nomic_embed` | Ollama                | 768        | Good balance, fast     |
| `minilm`      | sentence-transformers | 384        | Fastest, lightweight   |
| `bge_small`   | sentence-transformers | 384        | Strong MTEB scores     |
| `mxbai_large` | Ollama                | 1024       | Best quality (default) |

---

## Benchmark Results

All experiments use the same **25-question benchmark** with expert-written ground truth answers across 4 categories: conceptual (6), technical (6), how-to (7), error-related (6).

### Final pipeline (Phase 10)

| Metric                | Score     | Target |
| --------------------- | --------- | ------ |
| **Faithfulness**      | 0.919     | 0.85   |
| **Answer Relevancy**  | 0.846     | 0.75   |
| **Context Precision** | 0.954     | 0.75   |
| **Context Recall**    | 0.777     | 0.70   |
| **Average**           | **0.874** | -      |

Mean latency: 3.5s (retrieval: 313ms, generation: 3,208ms). P95: 6.9s.

### Phase-by-phase highlights

**Chunking** (Phase 2) - 5 strategies on full corpus:

| Strategy                 | Chunks       | MRR       | Speed       |
| ------------------------ | ------------ | --------- | ----------- |
| Fixed (1000/200)         | 14,576       | 0.484     | Fast        |
| **Recursive (1000/200)** | **15,592**   | **0.467** | **Fastest** |
| Token (256 tokens)       | 15,565       | 0.453     | Moderate    |
| Markdown                 | 18,817       | 0.397     | Moderate    |
| Semantic                 | 592 (sample) | N/A       | Very slow   |

**Embeddings** (Phase 3) - 4 models:

| Model                 | Dims     | Semantic Separation | Throughput    |
| --------------------- | -------- | ------------------- | ------------- |
| all-MiniLM-L6-v2      | 384      | 0.131               | 102 docs/s    |
| nomic-embed-text      | 768      | 0.100               | 174 docs/s    |
| BAAI/bge-small        | 384      | 0.181               | 27 docs/s     |
| **mxbai-embed-large** | **1024** | **0.207**           | **73 docs/s** |

**Retrieval** (Phase 4) - 8+ strategies on 25 questions:

| Strategy             | MRR       | Latency   |
| -------------------- | --------- | --------- |
| BM25                 | 0.303     | 2 ms      |
| Hybrid (0.5/0.5)     | 0.545     | 35 ms     |
| Similarity           | 0.597     | 51 ms     |
| **MMR (lambda=0.9)** | **0.605** | **76 ms** |
| Multi-Query          | 0.508     | 5,019 ms  |

**Query Translation** (Phase 5) - 5 techniques:

| Technique             | MRR       | vs Baseline | Latency      |
| --------------------- | --------- | ----------- | ------------ |
| Similarity (baseline) | 0.597     | -           | 19 ms        |
| **HyDE**              | **0.637** | **+6.7%**   | **2,690 ms** |
| RAG Fusion            | 0.628     | +5.2%       | 1,400 ms     |
| Step-Back             | 0.597     | +0.0%       | 495 ms       |
| Multi-Query           | 0.543     | -8.9%       | 1,387 ms     |
| Decomposition         | 0.535     | -10.4%      | 1,024 ms     |

**Reranking** (Phase 7) - retrieve 20, rerank to 5:

| Technique             | MRR       | Latency  |
| --------------------- | --------- | -------- |
| No reranking          | 0.597     | 24 ms    |
| Cross-encoder         | 0.575     | 1,883 ms |
| LLM-as-judge (sample) | **0.900** | 8,178 ms |

**Advanced RAG** (Phase 8) - LangGraph patterns:

| Pattern      | MRR      | Latency      | Works with 7B? |
| ------------ | -------- | ------------ | -------------- |
| Baseline     | 0.75     | 18 ms        | -              |
| **CRAG**     | **1.00** | **8,067 ms** | **Yes**        |
| Self-RAG     | 0.50     | 5,540 ms     | No             |
| Adaptive RAG | 0.75     | 3,872 ms     | No             |

**RAGAS Evaluation** (Phase 9) - 3 pipeline configs:

| Config                | Faithfulness | Relevancy | Avg Score |
| --------------------- | ------------ | --------- | --------- |
| Naive                 | 0.747        | 0.603     | 0.782     |
| **Hybrid + Reranked** | **0.947**    | **0.793** | **0.871** |
| HyDE                  | 0.869        | 0.666     | 0.799     |

---

## Key Lessons Learned

**What works locally:**

- **Hybrid retrieval** - BM25 + dense captures both keyword and semantic matches
- **Cross-encoder reranking** - the single most impactful upgrade (+20% faithfulness)
- **CRAG** - the only advanced pattern that works with 7B models (simple yes/no grading)

**What fails locally (Mistral 7B):**

- **Multi-Query / Decomposition** - the model generates imprecise query variants (-9% MRR)
- **Self-RAG / Adaptive RAG** - require frontier-level meta-reasoning (GPT-4, Claude)
- **Query routing** - hurts on imbalanced corpora regardless of algorithm

**Broader insights:**

1. The embedding model matters more than the retrieval strategy
2. Cross-encoder reranking has disproportionate impact on generation quality
3. Simple baselines are hard to beat - dense similarity (MRR 0.60) outperforms most complex techniques
4. Retrieval metric improvements don't always translate to generation quality improvements (and vice versa)
5. Corpus quality dominates pipeline complexity - technical questions are limited by content, not algorithms
6. Generation is the latency bottleneck (~91% of total time), not retrieval

---

## Data Pipeline

### Corpus

The LangChain Python documentation is scraped from the `langchain-ai/docs` GitHub repository.

| Stage          | Count  | Notes                                                    |
| -------------- | ------ | -------------------------------------------------------- |
| Raw documents  | 1,463  | Full scrape including integrations                       |
| After cleaning | 1,445  | 8-step pipeline: frontmatter, base64, JSX, whitespace... |
| Core documents | ~130   | Filtered out /python/integrations/ (90% noise)           |
| Final chunks   | ~1,500 | Recursive splitting, 1000 chars, 200 overlap             |

### Benchmark dataset

25 questions across 4 categories with expert-written ground truth:

| Category      | Count | Example                                                       |
| ------------- | ----- | ------------------------------------------------------------- |
| Conceptual    | 6     | "What is RAG and how does it work?"                           |
| Technical     | 6     | "What parameters does RecursiveCharacterTextSplitter accept?" |
| How-to        | 7     | "How do I create a RAG chain with LangChain?"                 |
| Error-related | 6     | "How do I fix INVALID_TOOL_RESULTS errors?"                   |

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Author

- **Martin Astyan** - [GitHub](https://github.com/AstyanM)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

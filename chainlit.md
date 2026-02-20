# LangChain RAG Explorer

Ask questions about the **LangChain Python framework** and get answers
backed by retrieved documentation.

## How it works

1. Your question is searched against ~1,500 documentation chunks
2. The most relevant chunks are retrieved (and optionally reranked)
3. Mistral 7B generates an answer using only the retrieved context

## Settings

Use the settings panel (gear icon) to change:

- **RAG Mode** - Simple (dense), Hybrid (BM25 + dense), Hybrid + Rerank, or Direct LLM
- **Number of results** - How many documents to retrieve (1-15)
- **Show sources** - Display retrieved source documents alongside answers
- **Conversation memory** - Enable follow-up questions that reference previous answers
- **Comparison mode** - Show RAG and Direct LLM answers side by side for the same question

## Example questions

- "How do I create a retriever in LangChain?"
- "What parameters does RecursiveCharacterTextSplitter accept?"
- "How do I use Ollama with LangChain?"
- "What is the difference between stuff and map-reduce chains?"

## Features

- **Streaming** - Answers are streamed token by token
- **Response time** - Retrieval and generation timing shown after each answer
- **Comparison mode** - See both RAG and Direct LLM answers for the same question
- **Conversation history** - Previous conversations are saved and can be resumed

## Details

Everything runs **100% locally** - no external API calls.
LLM: Mistral 7B via Ollama | Embeddings: mxbai-embed-large | Vector store: ChromaDB

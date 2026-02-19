"""
Document loaders for the RAG pipeline.

Loads scraped documents from JSON files back into LangChain Document objects.
"""

import json
from pathlib import Path

from langchain_core.documents import Document


def load_scraped_documents(path: str = "./data/raw/langchain_docs.json") -> list[Document]:
    """
    Load previously scraped documents from JSON into LangChain Document objects.

    Args:
        path: Path to the JSON file produced by the scraper.

    Returns:
        List of LangChain Document objects.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Scraped documents not found at {path}. "
            "Run the scraper first: python scripts/scrape_docs.py"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    documents = [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"],
        )
        for doc in raw_docs
    ]

    print(f"Loaded {len(documents)} documents from {path}")
    return documents


def load_section_documents(
    section: str,
    raw_dir: str = "./data/raw",
) -> list[Document]:
    """Load documents for a specific section (tutorials, how_to, concepts, api_reference)."""
    path = Path(raw_dir) / f"langchain_docs_{section}.json"
    return load_scraped_documents(str(path))

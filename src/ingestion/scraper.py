"""
LangChain documentation scraper.

Scrapes the LangChain Python documentation from python.langchain.com,
extracts main content, and saves documents as serialized JSON for
offline processing.
"""

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)


def bs4_extractor(html: str) -> str:
    """Extract main text content from HTML, removing navigation and boilerplate."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Try to find the main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
    if main is None:
        main = soup

    return main.get_text(separator="\n", strip=True)


def extract_title(html: str) -> str:
    """Extract the page title from HTML."""
    soup = BeautifulSoup(html, "lxml")
    # Try h1 first, then <title>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    return "Untitled"


def classify_section(url: str) -> str:
    """Classify a URL into a documentation section."""
    path = urlparse(url).path.lower()
    if "/tutorials/" in path:
        return "tutorials"
    if "/how_to/" in path or "/how-to/" in path:
        return "how_to"
    if "/concepts/" in path:
        return "concepts"
    if "/api_reference/" in path or "/api/" in path:
        return "api_reference"
    if "/integrations/" in path:
        return "integrations"
    return "other"


def classify_content_type(text: str) -> str:
    """Heuristic classification of content type based on text patterns."""
    code_pattern_count = len(re.findall(r"```|import |def |class |from .+ import", text))
    total_lines = max(text.count("\n"), 1)
    code_ratio = code_pattern_count / total_lines

    if code_ratio > 0.15:
        return "code-heavy"
    if any(kw in text.lower() for kw in ["step 1", "step 2", "first,", "next,", "then,"]):
        return "procedural"
    if any(kw in text.lower() for kw in ["parameters", "returns", "raises", "type:", "args:"]):
        return "reference"
    return "narrative"


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Fetch all URLs from the sitemap XML."""
    logger.info("Fetching sitemap from %s", sitemap_url)
    resp = requests.get(sitemap_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "lxml-xml")
    urls = [loc.text.strip() for loc in soup.find_all("loc")]
    logger.info("Found %d URLs in sitemap", len(urls))
    return urls


def filter_urls(urls: list[str], base_url: str, sections: list[dict] | None = None) -> list[str]:
    """Filter URLs to keep only relevant documentation pages."""
    filtered = []
    base_parsed = urlparse(base_url)

    for url in urls:
        parsed = urlparse(url)
        # Must be same host
        if parsed.netloc != base_parsed.netloc:
            continue
        # Skip non-doc pages
        path = parsed.path.lower()
        if any(skip in path for skip in [
            "/blog/", "/changelog/", "/_static/", "/search", ".xml", ".json",
        ]):
            continue
        # Must be under /docs/ or /api_reference/
        if "/docs/" in path or "/api_reference/" in path:
            filtered.append(url)

    logger.info("Filtered to %d documentation URLs", len(filtered))
    return filtered


def scrape_page(url: str, timeout: int = 30) -> dict | None:
    """Scrape a single page and return a document dict."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        content = bs4_extractor(html)
        if not content or len(content.strip()) < 50:
            logger.debug("Skipping %s — content too short", url)
            return None

        title = extract_title(html)
        section = classify_section(url)
        content_type = classify_content_type(content)

        return {
            "page_content": content,
            "metadata": {
                "source": url,
                "title": title,
                "section": section,
                "content_type": content_type,
                "content_length": len(content),
            },
        }
    except requests.RequestException as e:
        logger.warning("Failed to scrape %s: %s", url, e)
        return None


def scrape_langchain_docs(
    sitemap_url: str = "https://python.langchain.com/sitemap.xml",
    base_url: str = "https://python.langchain.com",
    output_dir: str = "./data/raw",
    timeout: int = 30,
    delay: float = 0.5,
    max_pages: int | None = None,
) -> list[dict]:
    """
    Scrape the LangChain documentation.

    Args:
        sitemap_url: URL of the sitemap XML.
        base_url: Base URL to filter pages.
        output_dir: Directory to save scraped documents.
        timeout: HTTP request timeout in seconds.
        delay: Delay between requests in seconds (be polite).
        max_pages: Maximum number of pages to scrape (None for all).

    Returns:
        List of document dicts with page_content and metadata.
    """
    # Fetch and filter URLs
    all_urls = fetch_sitemap_urls(sitemap_url)
    doc_urls = filter_urls(all_urls, base_url)

    if max_pages is not None:
        doc_urls = doc_urls[:max_pages]
        logger.info("Limited to %d pages", max_pages)

    # Scrape each page
    documents = []
    for url in tqdm(doc_urls, desc="Scraping pages"):
        doc = scrape_page(url, timeout=timeout)
        if doc is not None:
            documents.append(doc)
        time.sleep(delay)

    logger.info("Successfully scraped %d documents", len(documents))

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all documents as a single JSON file
    all_docs_path = output_path / "langchain_docs.json"
    with open(all_docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info("Saved all documents to %s", all_docs_path)

    # Save per-section files for easier inspection
    by_section: dict[str, list] = {}
    for doc in documents:
        section = doc["metadata"]["section"]
        by_section.setdefault(section, []).append(doc)

    for section, docs in by_section.items():
        section_path = output_path / f"langchain_docs_{section}.json"
        with open(section_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d %s documents to %s", len(docs), section, section_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Scraping Summary")
    print(f"{'='*60}")
    print(f"Total URLs in sitemap:    {len(all_urls)}")
    print(f"Filtered documentation:   {len(doc_urls)}")
    print(f"Successfully scraped:     {len(documents)}")
    print(f"{'─'*60}")
    for section, docs in sorted(by_section.items()):
        print(f"  {section:20s}: {len(docs):4d} pages")
    print(f"{'─'*60}")

    total_chars = sum(len(d["page_content"]) for d in documents)
    avg_chars = total_chars // len(documents) if documents else 0
    print(f"Total content:            {total_chars:,} characters")
    print(f"Average page size:        {avg_chars:,} characters")
    print(f"Output directory:         {output_path.resolve()}")
    print(f"{'='*60}\n")

    return documents

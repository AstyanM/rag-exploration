"""
LangChain documentation scraper.

Downloads MDX documentation files directly from the langchain-ai/docs
GitHub repository. This approach is more reliable than web scraping
since the site is a Mintlify SPA that renders content client-side.
"""

import base64
import json
import logging
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REPO = "langchain-ai/docs"
DOCS_BASE_PATH = "src/oss"
SITE_BASE_URL = "https://python.langchain.com/oss"


def classify_section(file_path: str) -> str:
    """Classify a file path into a documentation section."""
    path = file_path.lower()
    if "/langchain/" in path:
        return "langchain"
    if "/langgraph/" in path:
        return "langgraph"
    if "/concepts/" in path:
        return "concepts"
    if "/reference/" in path:
        return "reference"
    if "/contributing/" in path:
        return "contributing"
    if "/deepagents/" in path:
        return "deepagents"
    if "/integrations/" in path or "/python/integrations/" in path:
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


def extract_title_from_mdx(content: str) -> str:
    """Extract the title from an MDX file's frontmatter or first heading."""
    # Try frontmatter title
    fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).split("\n"):
            if line.strip().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
                return title
            if line.strip().startswith("sidebarTitle:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
                return title

    # Try first markdown heading
    heading_match = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()

    return "Untitled"


def file_path_to_url(file_path: str) -> str:
    """Convert a repo file path to the corresponding documentation URL."""
    # src/oss/langchain/overview.mdx -> /oss/python/langchain/overview
    relative = file_path.replace(DOCS_BASE_PATH + "/", "")
    relative = re.sub(r"\.mdx$", "", relative)
    return f"{SITE_BASE_URL}/{relative}"


def fetch_file_tree(
    include_integrations: bool = False,
    github_token: str | None = None,
) -> list[dict]:
    """Fetch the complete file tree from the GitHub API."""
    logger.info("Fetching file tree from GitHub...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    resp = requests.get(
        f"{GITHUB_API}/repos/{REPO}/git/trees/main?recursive=1",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("truncated"):
        logger.warning("GitHub tree was truncated — some files may be missing")

    tree = data.get("tree", [])

    # Filter for MDX files under src/oss/
    mdx_files = []
    for item in tree:
        path = item["path"]
        if not path.startswith(f"{DOCS_BASE_PATH}/") or not path.endswith(".mdx"):
            continue
        # Skip JavaScript docs
        if "/javascript/" in path:
            continue
        # Optionally skip integration pages (there are ~1300 of them)
        if not include_integrations and "/integrations/" in path:
            continue
        mdx_files.append(item)

    logger.info("Found %d MDX files to download", len(mdx_files))
    return mdx_files


def download_file_content(
    file_path: str,
    github_token: str | None = None,
) -> str | None:
    """Download a single file's content from GitHub."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    resp = requests.get(
        f"{GITHUB_API}/repos/{REPO}/contents/{file_path}",
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 403:
        # Rate limit hit
        logger.warning("GitHub API rate limit hit. Consider using a token.")
        return None
    resp.raise_for_status()

    data = resp.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8")
    return data.get("content", "")


def download_raw_file(
    file_path: str,
    github_token: str | None = None,
) -> str | None:
    """Download a file using the raw content URL (doesn't count against API rate limit)."""
    url = f"https://raw.githubusercontent.com/{REPO}/main/{file_path}"
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", file_path, e)
        return None


def scrape_langchain_docs(
    output_dir: str = "./data/raw",
    include_integrations: bool = False,
    github_token: str | None = None,
    max_pages: int | None = None,
    delay: float = 0.1,
    **kwargs,
) -> list[dict]:
    """
    Download LangChain documentation from the GitHub repository.

    Args:
        output_dir: Directory to save scraped documents.
        include_integrations: Whether to include integration docs (~1300 pages).
        github_token: GitHub personal access token (for higher rate limits).
        max_pages: Maximum number of pages to download (None for all).
        delay: Delay between requests in seconds.

    Returns:
        List of document dicts with page_content and metadata.
    """
    # Fetch file tree
    mdx_files = fetch_file_tree(
        include_integrations=include_integrations,
        github_token=github_token,
    )

    if max_pages is not None:
        mdx_files = mdx_files[:max_pages]
        logger.info("Limited to %d pages", max_pages)

    # Download each file
    documents = []
    failed = 0
    for item in tqdm(mdx_files, desc="Downloading docs"):
        file_path = item["path"]

        content = download_raw_file(file_path, github_token=github_token)
        if content is None or len(content.strip()) < 50:
            failed += 1
            continue

        title = extract_title_from_mdx(content)
        section = classify_section(file_path)
        content_type = classify_content_type(content)
        url = file_path_to_url(file_path)

        documents.append({
            "page_content": content,
            "metadata": {
                "source": url,
                "github_path": file_path,
                "title": title,
                "section": section,
                "content_type": content_type,
                "content_length": len(content),
            },
        })

        time.sleep(delay)

    logger.info("Successfully downloaded %d documents (%d failed)", len(documents), failed)

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all documents as a single JSON file
    all_docs_path = output_path / "langchain_docs.json"
    with open(all_docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info("Saved all documents to %s", all_docs_path)

    # Save per-section files
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
    print("Scraping Summary")
    print(f"{'='*60}")
    print(f"Total MDX files found:    {len(mdx_files)}")
    print(f"Successfully downloaded:  {len(documents)}")
    print(f"Failed:                   {failed}")
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

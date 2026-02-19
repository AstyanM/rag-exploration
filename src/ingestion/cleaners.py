"""
Document cleaning utilities for the RAG pipeline.

Strips MDX/JSX markup, base64 blobs, frontmatter, and other noise
from scraped documentation before chunking.
"""

import re
from langchain_core.documents import Document


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by ---."""
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)


def strip_mdx_jsx_tags(text: str) -> str:
    """Remove JSX/MDX component tags but keep their text content.

    Handles self-closing tags (<Icon />), opening/closing pairs
    (<Tip>...</Tip>), and HTML tags mixed in the markdown.
    """
    # Self-closing tags: <Icon icon="check" />, <br />, etc.
    text = re.sub(r"<[A-Za-z][^>]*/\s*>", "", text)
    # Opening and closing tags: <Tip>, </Tip>, <CodeGroup>, </CodeGroup>
    # Keep inner content by only removing the tags themselves
    text = re.sub(r"</?\s*[A-Za-z][A-Za-z0-9]*(?:\s[^>]*)?>", "", text)
    return text


def strip_base64_blobs(text: str, min_length: int = 200) -> str:
    """Remove long base64-encoded strings (images, fonts, binary data).

    Matches contiguous runs of base64 characters (A-Z, a-z, 0-9, +, /, =)
    that are at least `min_length` characters long.
    """
    # Long contiguous base64 strings
    text = re.sub(
        rf"[A-Za-z0-9+/=]{{{min_length},}}",
        " [binary content removed] ",
        text,
    )
    # Shorter base64 strings in quoted lists (e.g. 'ZGIyYzA4NzU...',)
    # These are base64-encoded UUIDs/IDs that appear in arrays
    text = re.sub(
        r"'[A-Za-z0-9+/=]{20,}'",
        "'[id removed]'",
        text,
    )
    return text


def strip_mdx_imports(text: str) -> str:
    """Remove MDX/ESM import lines (e.g. import X from '/snippets/...')."""
    return re.sub(r"^import\s+\w+.*?;\s*$", "", text, flags=re.MULTILINE)


def strip_js_tab_blocks(text: str) -> str:
    """Remove :::js blocks entirely (keep only Python content).

    The LangChain docs ship both Python and JS code in tab blocks
    delimited by :::python / :::js markers. We strip the JS blocks
    and unwrap the Python blocks (remove markers, keep content).
    """
    # Remove :::js ... ::: blocks (including the markers)
    text = re.sub(r":::js\s*\n.*?:::", "", text, flags=re.DOTALL)
    # Unwrap :::python ... ::: blocks (keep inner content)
    text = re.sub(r":::python\s*\n", "", text)
    # Remove remaining standalone ::: markers
    text = re.sub(r"^:::\s*$", "", text, flags=re.MULTILINE)
    return text


def strip_js_code_fences(text: str) -> str:
    """Remove JavaScript/TypeScript code fences, keep Python ones.

    Pages about frontend integration contain raw JS/TS code blocks
    (```javascript, ```typescript, ```tsx, ```jsx) that are unrelated
    to Python usage. Strip them entirely including their content.
    """
    js_langs = r"(?:javascript|typescript|tsx|jsx|js|ts)"
    text = re.sub(
        rf"```{js_langs}[^\n]*\n.*?```",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text


def strip_template_variables(text: str) -> str:
    """Remove JavaScript template literal expressions like ${variable}."""
    return re.sub(r"\$\{[^}]*\}", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trailing spaces."""
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()


def clean_document(doc: Document) -> Document:
    """Apply all cleaning steps to a single document.

    Returns a new Document with cleaned page_content.
    Metadata is preserved and a 'cleaned' flag is added.
    """
    text = doc.page_content

    text = strip_frontmatter(text)
    text = strip_mdx_imports(text)
    text = strip_js_tab_blocks(text)
    text = strip_js_code_fences(text)
    text = strip_mdx_jsx_tags(text)
    text = strip_base64_blobs(text)
    text = strip_template_variables(text)
    text = normalize_whitespace(text)

    metadata = {**doc.metadata, "cleaned": True}
    return Document(page_content=text, metadata=metadata)


def clean_corpus(
    docs: list[Document],
    min_content_length: int = 50,
) -> tuple[list[Document], dict]:
    """Clean a list of documents and filter out near-empty results.

    Args:
        docs: Raw documents to clean.
        min_content_length: Minimum character count after cleaning.
            Documents shorter than this are discarded.

    Returns:
        Tuple of (cleaned documents, stats dict).
    """
    cleaned = []
    dropped = 0

    for doc in docs:
        clean_doc = clean_document(doc)
        if len(clean_doc.page_content) >= min_content_length:
            cleaned.append(clean_doc)
        else:
            dropped += 1

    stats = {
        "original_count": len(docs),
        "cleaned_count": len(cleaned),
        "dropped_count": dropped,
        "original_total_chars": sum(len(d.page_content) for d in docs),
        "cleaned_total_chars": sum(len(d.page_content) for d in cleaned),
    }

    return cleaned, stats

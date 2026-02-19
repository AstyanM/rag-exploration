"""
Visualization and display helpers for RAG exploration notebooks.
"""

from collections import Counter

import pandas as pd
from IPython.display import HTML, display
from langchain_core.documents import Document


def display_doc_preview(doc: Document, max_chars: int = 500) -> None:
    """Display a single document with metadata and content preview."""
    meta = doc.metadata
    content = doc.page_content[:max_chars]
    if len(doc.page_content) > max_chars:
        content += "..."

    html = f"""
    <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0;
                background:#fafafa; font-family:monospace; font-size:13px;">
        <div style="margin-bottom:8px;">
            <b>{meta.get('title', 'Untitled')}</b>
            <span style="color:#888; margin-left:12px;">[{meta.get('section', '?')}]</span>
            <span style="color:#888; margin-left:12px;">{meta.get('content_type', '?')}</span>
        </div>
        <div style="color:#555; font-size:11px; margin-bottom:6px;">
            {meta.get('source', 'N/A')}
        </div>
        <div style="white-space:pre-wrap; font-size:12px; color:#333; max-height:200px; overflow-y:auto;">
{content}
        </div>
        <div style="color:#888; font-size:11px; margin-top:6px;">
            {len(doc.page_content):,} characters
        </div>
    </div>
    """
    display(HTML(html))


def display_retrieval_results(
    query: str,
    results: list[tuple[Document, float]],
    max_content_chars: int = 300,
) -> None:
    """Display retrieval results with scores and content previews."""
    html = f'<h4 style="margin-bottom:4px;">Query: <i>"{query}"</i></h4>'

    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content[:max_content_chars]
        if len(doc.page_content) > max_content_chars:
            content += "..."

        score_color = "#2ecc71" if score > 0.7 else "#f39c12" if score > 0.4 else "#e74c3c"
        html += f"""
        <div style="border-left:4px solid {score_color}; padding:8px 12px; margin:6px 0;
                    background:#fafafa; font-size:12px;">
            <b>#{i}</b> — Score: <b style="color:{score_color}">{score:.4f}</b>
            — <i>{doc.metadata.get('title', 'Untitled')}</i>
            <span style="color:#888;">[{doc.metadata.get('section', '?')}]</span>
            <div style="white-space:pre-wrap; color:#555; margin-top:4px; font-size:11px;">
{content}
            </div>
        </div>
        """

    display(HTML(html))


def corpus_summary_table(docs: list[Document]) -> pd.DataFrame:
    """Build a summary DataFrame of the corpus."""
    records = []
    for doc in docs:
        records.append({
            "title": doc.metadata.get("title", "Untitled"),
            "section": doc.metadata.get("section", "unknown"),
            "content_type": doc.metadata.get("content_type", "unknown"),
            "chars": len(doc.page_content),
        })
    return pd.DataFrame(records)


def section_breakdown(docs: list[Document]) -> pd.DataFrame:
    """Return a per-section summary."""
    df = corpus_summary_table(docs)
    return (
        df.groupby("section")
        .agg(
            count=("chars", "size"),
            total_chars=("chars", "sum"),
            avg_chars=("chars", "mean"),
            min_chars=("chars", "min"),
            max_chars=("chars", "max"),
        )
        .sort_values("count", ascending=False)
    )


def chunk_stats_table(chunks: list[Document]) -> pd.DataFrame:
    """Build a summary table of chunk sizes."""
    sizes = [len(c.page_content) for c in chunks]
    return pd.DataFrame({
        "metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "chars": [
            len(sizes),
            pd.Series(sizes).mean(),
            pd.Series(sizes).std(),
            min(sizes),
            pd.Series(sizes).quantile(0.25),
            pd.Series(sizes).median(),
            pd.Series(sizes).quantile(0.75),
            max(sizes),
        ],
    }).set_index("metric").round(0).astype(int)

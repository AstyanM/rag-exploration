#!/usr/bin/env python3
"""
Standalone script to scrape the LangChain documentation.

Usage:
    python scripts/scrape_docs.py                      # Scrape all pages
    python scripts/scrape_docs.py --max-pages 50       # Scrape first 50 pages (for testing)
    python scripts/scrape_docs.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.ingestion.scraper import scrape_langchain_docs


def main():
    parser = argparse.ArgumentParser(description="Scrape LangChain documentation")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (None for all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = load_config(args.config)
    scraping_config = config.get("scraping", {})

    # Determine parameters
    sitemap_url = scraping_config.get("sitemap_url", "https://python.langchain.com/sitemap.xml")
    base_url = scraping_config.get("base_url", "https://python.langchain.com")
    output_dir = args.output_dir or scraping_config.get("output_dir", "./data/raw")
    timeout = scraping_config.get("timeout", 30)

    print(f"Scraping LangChain documentation...")
    print(f"  Sitemap:    {sitemap_url}")
    print(f"  Output:     {output_dir}")
    print(f"  Max pages:  {args.max_pages or 'all'}")
    print(f"  Delay:      {args.delay}s between requests")
    print()

    documents = scrape_langchain_docs(
        sitemap_url=sitemap_url,
        base_url=base_url,
        output_dir=output_dir,
        timeout=timeout,
        delay=args.delay,
        max_pages=args.max_pages,
    )

    print(f"Done! Scraped {len(documents)} documents to {output_dir}/")


if __name__ == "__main__":
    main()

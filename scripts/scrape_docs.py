#!/usr/bin/env python3
"""
Standalone script to download the LangChain documentation from GitHub.

Usage:
    python scripts/scrape_docs.py                          # Download core docs (~146 files)
    python scripts/scrape_docs.py --max-pages 20           # Download first 20 files (for testing)
    python scripts/scrape_docs.py --include-integrations    # Include integration pages (~1300 more)
    python scripts/scrape_docs.py --github-token ghp_xxx   # Use token for higher rate limits
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
    parser = argparse.ArgumentParser(description="Download LangChain documentation from GitHub")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of files to download (None for all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--include-integrations",
        action="store_true",
        help="Include integration docs (~1300 extra pages)",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="GitHub personal access token (for higher API rate limits)",
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
    output_dir = args.output_dir or scraping_config.get("output_dir", "./data/raw")

    print("Downloading LangChain documentation from GitHub...")
    print(f"  Repository:     langchain-ai/docs")
    print(f"  Output:         {output_dir}")
    print(f"  Max pages:      {args.max_pages or 'all'}")
    print(f"  Integrations:   {'yes' if args.include_integrations else 'no (core docs only)'}")
    print(f"  Delay:          {args.delay}s between requests")
    print()

    documents = scrape_langchain_docs(
        output_dir=output_dir,
        include_integrations=args.include_integrations,
        github_token=args.github_token,
        max_pages=args.max_pages,
        delay=args.delay,
    )

    print(f"Done! Downloaded {len(documents)} documents to {output_dir}/")


if __name__ == "__main__":
    main()

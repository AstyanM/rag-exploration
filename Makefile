.PHONY: setup scrape index evaluate app clean

# Setup environment
setup:
	python -m venv .venv
	.venv/Scripts/activate && pip install -r requirements.txt

# Scrape LangChain documentation
scrape:
	python scripts/scrape_docs.py

# Scrape a small sample (for testing)
scrape-sample:
	python scripts/scrape_docs.py --max-pages 50

# Build/rebuild vector index
index:
	python scripts/index_documents.py --config configs/default.yaml

# Run full evaluation
evaluate:
	python scripts/run_evaluation.py --config configs/default.yaml

# Launch Chainlit app
app:
	chainlit run src/app.py --port 8000

# Clean generated data
clean:
	rm -rf vectorstore/chroma_db/*
	rm -rf data/processed/*
	rm -rf results/*

# Full pipeline: scrape -> index -> evaluate -> app
all: scrape index evaluate app

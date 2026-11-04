PY ?= python

.PHONY: install install-embeddings data text text-llm demo ask eval eval-fake test mcp app all clean

install:
	$(PY) -m pip install -r requirements.txt

install-embeddings:     ## sentence-transformers (torch); optional
	$(PY) -m pip install -r requirements-optional.txt

data:                   ## synthetic DuckDB database + labelled set
	$(PY) -m src.data.generate

text:                   ## clustering + classifier comparison -> reports/text_results_*.json
	$(PY) -m src.text.pipeline

text-llm:               ## same, plus LLM classifier and cluster labels (needs ANTHROPIC_API_KEY)
	$(PY) -m src.text.pipeline --llm

demo:                   ## scripted agent run, no API key needed
	$(PY) -m src.cli demo

ask:                    ## make ask Q="Which channel has the best ROAS?"
	$(PY) -m src.cli ask "$(Q)"

eval:                   ## golden questions against the real model (needs ANTHROPIC_API_KEY)
	$(PY) -m eval.run_eval

eval-fake:              ## same questions through a scripted replay; checks plumbing only
	$(PY) -m eval.run_eval --fake

test:
	$(PY) -m pytest

mcp:
	$(PY) -m src.mcp_server

app:
	streamlit run app/streamlit_app.py

all: data text eval-fake demo test

clean:
	rm -rf data/campaigns.duckdb data/feedback_truth.csv data/feedback_index.joblib traces .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +

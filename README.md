# Bilingual Speech Research Copilot

A CSCI 599 course project: a single-turn research assistant for bilingual child speech development. A LangGraph supervisor coordinates Research, Data, and Summary agents over a 15-paper local corpus. FastAPI streams answers, citations, and execution traces to a plain HTML/JavaScript frontend.

## Architecture

- Research retrieves Chroma-indexed passages and can use a shared Tavily MCP search adapter.
- Data performs deterministic metadata queries through an in-memory, read-only SQLite tool.
- Summary synthesizes evidence and marks partial results when citation checks flag unsupported claims.
- OpenAI, Anthropic, and Groq model clients have separate routing and usage records.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/evaluation.md`](docs/evaluation.md). The corpus catalog/manifest and a questions-only evaluation file are included; the evaluation report documents the historical results. Original paper PDFs, generated vector indexes, raw traces, and local credentials are excluded from this public repository. The manifest identifies public sources; redistribution rights for each PDF must be checked separately.

## Evaluation snapshot

On five fixed local-RAG questions, RAGAS mean faithfulness was 0.8764 (n=5). This small course evaluation is not a general reliability benchmark. The report also records provider comparisons, costs, limitations, and one Cloud Run cold request versus three warm requests; one warm result was partial. Detailed results are in [`docs/evaluation.md`](docs/evaluation.md).

## Run locally

Use Python 3.11 or newer for the application. Copy `.env.example` to `.env` and fill in your own provider credentials and model names. The evaluation reference excerpts, corpus PDFs and Chroma index are not included; see [`docs/corpus_reading_list.md`](docs/corpus_reading_list.md) and `scripts/redownload_corpus.py` for source references and rebuilding guidance.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Run offline Python tests with `python -m unittest discover -s tests -p 'test_*.py' -q` and frontend tests with `node tests/test_frontend.mjs`. Evaluation scripts may make paid model calls only when passed explicit run flags; inspect their help first.

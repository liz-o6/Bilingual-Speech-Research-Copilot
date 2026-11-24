# RAG migration and usage

## Scope and provenance

Adapted from the user's Healthcare_Research_Copilot:

| Original module | HW2 module / adaptation |
| --- | --- |
| `app/agent/tools/retriever_toos.py` | `src/rag/loaders.py`, `ingest.py`, `retriever.py`: PDF/TXT/DOCX, recursive splitting, OpenAI embeddings, Chroma |
| `app/agent/nodes/local_db.py` | SHA256 change detection; commit only after successful verified ingestion |
| `app/agent/nodes/summary.py` | `src/tools/sources.py`: request-local source registry, grounding prompt |
| `app/agent/nodes/citation_verification.py` | Claim/ID extraction, existence check and optional injected semantic judge |
| `app/agent/cores/error_codes.py` | `src/rag/errors.py`: structured exception with code/retryable fields; bounded transient retries |

Old project files, credentials, databases and medical corpus were not copied or modified. FastAPI/LangGraph orchestration is intentionally deferred to the next implementation stage. Chroma uses its public client API directly so snapshot copying can preserve embeddings without paying to regenerate unchanged files. The original recursive splitter and OpenAI embedding integration remain.

## Install and run

Run from the `hw2` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY locally in .env, then put your papers under data/corpus/raw/.
python -m src.rag ingest
python -m src.rag search "What experimental tasks were used?" --k 5
python -m unittest discover -s tests -p 'test_*.py' -v
```

Default paths resolve relative to the project, not the current working directory. Explicit CLI paths resolve relative to the caller. A single file or any combination of files/directories is supported; directories are scanned recursively. The corpus root is the identity boundary; symlinks escaping it are rejected.

```bash
python -m src.rag ingest data/corpus/raw/paper.pdf
python -m src.rag ingest --catalog data/corpus/catalog.json
```

Optional catalog JSON maps corpus-relative filenames to verified metadata:

```json
{
  "paper.pdf": {
    "doc_id": "paper-001",
    "title": "Human-verified paper title",
    "source_url": "https://example.org/replace-with-real-source",
    "corpus_version": "v1"
  }
}
```

This is a format example, not a real research source. Unspecified metadata is not inferred. Existing catalog fields are retained when a later invocation omits the catalog. A supplied entry replaces that entry's metadata. Without catalog metadata, title falls back to filename, source_url is empty, and version is `unversioned`. Final corpus bibliographic manifest, authors/license and verified participant metadata still need preparation.

## Interfaces for agents

```python
from src.rag.retriever import Retriever
from src.tools.sources import SourceRegistry, read_document, check_citations, verify_citations

retriever = Retriever()
hits = retriever.search("naming task", k=5)  # async equivalent: await retriever.asearch(...)
registry = SourceRegistry()  # one registry per orchestration request, shared across workers
registry.register(hits)
prompt = registry.prompt("Compare the experimental tasks")
# Pass prompt to the future provider router/generation model.
# checks = check_citations(answer, registry)
# semantic_checks = await verify_citations(answer, registry, routed_chat_model)
# pages = read_document(hits[0]["doc_id"], page=2)
```

Each hit includes text, stable doc_id/chunk_id/source_id, title, corpus-relative source filename, source_url, SHA256, corpus_version, section_id/section_title, page, start_index, distance, and index_snapshot. Distance is cosine distance (lower is closer), not confidence. `[S1]` is a request-local presentation ID mapped to the stable chunk ID; do not create separate registries that reset IDs for each worker.

PDF pages are **1-based physical PDF page numbers**, not printed journal pagination. TXT and DOCX expose page=null. Section IDs currently identify page blocks or full-text blocks, not inferred Introduction/Methods headings. No OCR or table reconstruction is included. Empty/scanned-only documents fail clearly. Full-text reads return indexed text, not a re-read of a potentially modified original file. The frontend source-serving route is not implemented yet.

`check_citations` checks only that IDs exist: valid IDs remain `unverified`. `verify_citations` injects a LangChain structured-output model to assess support; omitted, duplicated or failed judge results remain unverified. This is not a guarantee of factual correctness or a detector for uncited claims, and does not replace RAGAS. Generation/provider routing is not implemented in this stage.

## Consistency and failure handling

- SHA256, supplied metadata and chunk settings determine whether a file needs reprocessing.
- Deterministic chunk IDs include relative file identity, version, chunk position/text and settings.
- An offline writer lock serializes local ingestions. Build a new Chroma collection, copy unchanged vectors, embed changed files, verify the complete ID set, then atomically replace `active.json`.
- Failed loads, embedding calls, writes or pointer replacement leave the old active manifest/hash intact. Queries pin one snapshot per Retriever instance; instantiate a new Retriever to see a committed update.
- Previous collections are retained for existing readers. This trades disk space for safe rollback/read consistency; old collection cleanup is manual and only safe when no readers use them. Normal repeated unchanged ingestion creates no collection.
- Ingestion upserts specified files. Missing/deleted files are **not automatically pruned**. Rebuild into a new `--storage` directory to remove files or change embedding model/chunk settings. Do not silently mix embedding spaces.
- Retry only transient embedding errors (connection/timeouts, 408/429/selected 5xx), at most three attempts with backoff. Parser/configuration errors are not retried. Raw Chroma/filesystem failures propagate as exceptions; callers must catch failures instead of assuming success.
- This lock/snapshot design is for offline local ingestion on macOS/Linux, not concurrent distributed writers or GCSFuse atomicity. Conversation persistence is a separate future concern.

## Verification and remaining work

Tests use real persistent Chroma with deterministic **test-only** vectors, temporary TXT/PDF/DOCX fixtures and simulated faults. No API keys or live API calls are used. They cover file/directory inputs, unchanged skip, modified-file replacement, unchanged-document preservation, failure rollback/retry, metadata and page preservation, configuration guards, source registration, full-text reads, and semantic-judge missing/failure behavior.

Pinned dependencies were read from the existing project's installed environment. Tests ran with that Python 3.14 environment; a clean Python 3.13 install/container has not yet been verified. No retrieval-quality or RAGAS results are claimed. Next: finalize the actual corpus/catalog, perform real OpenAI ingestion/query smoke tests, review retrieval relevance, then connect agents and routed generation.

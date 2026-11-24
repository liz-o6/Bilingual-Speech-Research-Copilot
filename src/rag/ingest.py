"""Offline snapshot ingestion. The active pointer changes only after verified writes."""
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .loaders import discover_files, file_sha256, load_file
from .errors import RagError, retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "data/corpus/raw"
DEFAULT_STORAGE = PROJECT_ROOT / "storage/chroma"


def get_embeddings(model="text-embedding-3-small"):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model, max_retries=0)


def client_for(storage):
    import chromadb
    return chromadb.PersistentClient(path=str(Path(storage).resolve() / "db"))


def read_snapshot(storage):
    path = Path(storage) / "active.json"
    if not path.exists():
        raise RagError("INDEX_NOT_FOUND", "Run ingestion before retrieval")
    return json.loads(path.read_text())


def ingest(paths, *, corpus_root=DEFAULT_CORPUS, storage=DEFAULT_STORAGE,
           embeddings=None, embedding_model="text-embedding-3-small",
           chunk_size=1000, chunk_overlap=200, catalog=None):
    """Upsert supplied files; omitted files remain. Model/chunk config changes require a new storage directory.

    catalog maps corpus-relative filenames to verified bibliographic metadata.
    Embeddings may be injected in tests; embedding_model must identify that embedding space.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require 0 <= chunk_overlap < chunk_size")
    paths = discover_files(paths, corpus_root)
    if not paths:
        raise RagError("NO_DOCUMENTS", "No PDF/TXT/DOCX files found")
    storage = Path(storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    with (storage / ".ingest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ingest(paths, Path(corpus_root).resolve(), storage, embeddings, embedding_model,
                       chunk_size, chunk_overlap, catalog or {})


def _ingest(paths, root, storage, embeddings, model, size, overlap, catalog):
    config = {"embedding_model": model, "chunk_size": size, "chunk_overlap": overlap, "loader_version": 1}
    old = read_snapshot(storage) if (storage / "active.json").exists() else None
    if old and old["config"] != config:
        raise RagError("INDEX_CONFIG_MISMATCH", "Use a new storage directory to rebuild with changed embedding/chunk settings")
    records = dict(old["documents"]) if old else {}
    changed = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        metadata = catalog.get(rel, records.get(rel, {}).get("fingerprint", {}).get("catalog", {}))
        fingerprint = {"sha256": file_sha256(path), "catalog": metadata}
        if records.get(rel, {}).get("fingerprint") != fingerprint:
            changed.append((path, rel, fingerprint))
    if not changed:
        return {"changed": 0, "skipped": len(paths), "collection": old["collection"]}
    split = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    chunks = []
    for path, rel, fingerprint in changed:
        pages = load_file(path, root, fingerprint["catalog"])
        if file_sha256(path) != fingerprint["sha256"] or pages[0].metadata["sha256"] != fingerprint["sha256"]:
            raise RagError("SOURCE_CHANGED", "Source changed while loading; retry ingestion")
        pieces = split.split_documents(pages)
        for i, chunk in enumerate(pieces):
            identity = json.dumps([rel, fingerprint, config, i, chunk.page_content], sort_keys=True)
            chunk.metadata["chunk_id"] = "chunk_" + hashlib.sha256(identity.encode()).hexdigest()
        if not pieces:
            raise RagError("EMPTY_DOCUMENT", rel)
        records[rel] = {"fingerprint": fingerprint, "doc_id": pages[0].metadata["doc_id"],
                        "chunk_ids": [c.metadata["chunk_id"] for c in pieces],
                        "pages": [{"text": p.page_content, "metadata": p.metadata} for p in pages]}
        chunks.extend(pieces)
    ids = [r["doc_id"] for r in records.values()]
    if len(ids) != len(set(ids)):
        raise RagError("DUPLICATE_DOC_ID", "Each corpus file must have a unique doc_id")
    client = client_for(storage)
    name = "corpus_" + uuid.uuid4().hex
    target = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    committed = False
    temp = storage / (name + ".json.tmp")
    try:
        if old:
            previous = client.get_collection(old["collection"])
            unchanged = [cid for rel, record in old["documents"].items()
                         if rel not in {item[1] for item in changed} for cid in record["chunk_ids"]]
            for offset in range(0, len(unchanged), 128):
                data = previous.get(ids=unchanged[offset:offset+128], include=["documents", "metadatas", "embeddings"])
                target.upsert(ids=data["ids"], documents=data["documents"], metadatas=data["metadatas"], embeddings=data["embeddings"])
        embedder = embeddings if embeddings is not None else get_embeddings(model)
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset:offset+64]
            vectors = retry(lambda: embedder.embed_documents([c.page_content for c in batch]))
            target.upsert(ids=[c.metadata["chunk_id"] for c in batch], documents=[c.page_content for c in batch],
                          metadatas=[c.metadata for c in batch], embeddings=vectors)
        expected = {cid for r in records.values() for cid in r["chunk_ids"]}
        if set(target.get(include=[])["ids"]) != expected:
            raise RagError("INDEX_VERIFY_FAILED", "Stored chunk IDs do not match manifest")
        snapshot = {"schema_version": 1, "collection": name, "config": config, "documents": records}
        with temp.open("w") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, storage / "active.json")
        committed = True
        return {"changed": len(changed), "skipped": len(paths)-len(changed), "chunks": len(expected), "collection": name}
    finally:
        if not committed:
            client.delete_collection(name)
        temp.unlink(missing_ok=True)
    # Previous snapshots deliberately remain available to readers already using them.

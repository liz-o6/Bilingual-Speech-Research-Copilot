"""Structured retrieval preserves source metadata and snapshot identity."""

import asyncio
from .ingest import DEFAULT_STORAGE, client_for, get_embeddings, read_snapshot
from .errors import RagError, retry


class Retriever:
    def __init__(
        self,
        storage=DEFAULT_STORAGE,
        *,
        embeddings=None,
        traces=None,
        embedding_model="text-embedding-3-small"
    ):

        self.snapshot = read_snapshot(storage)
        if self.snapshot["config"]["embedding_model"] != embedding_model:
            raise RagError(
                "EMBEDDING_MISMATCH",
                "Query and ingestion must use the same embedding model",
            )
        self.collection = client_for(storage).get_collection(
            self.snapshot["collection"]
        )
        self.embeddings = (
            embeddings if embeddings is not None else get_embeddings(embedding_model)
        )
        if embeddings is None and traces is not None:
            from src.observability.embeddings import track_embeddings
            track_embeddings(self.embeddings, traces, 'rag_query')

    def search(self, query, k=5, doc_id=None):
        if not query.strip() or k < 1:
            raise ValueError("Nonempty query and positive k required")
        vector = retry(lambda: self.embeddings.embed_query(query))
        args = {
            "query_embeddings": [vector],
            "n_results": min(k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_id:
            args["where"] = {"doc_id": doc_id}
        result = self.collection.query(**args)
        return [
            {
                "text": text,
                **meta,
                "page": meta.get("page") or None,
                "distance": distance,
                "source_id": cid,
                "index_snapshot": self.snapshot["collection"],
            }
            for cid, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    async def asearch(self, query, k=5, doc_id=None):
        return await asyncio.to_thread(self.search, query, k, doc_id)

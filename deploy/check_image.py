"""Validate runtime files and copied Chroma vectors without API calls."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.rag.ingest import client_for, read_snapshot
from src.tools.metadata import MetadataQuery, query_metadata

snapshot = read_snapshot(root / 'storage/chroma')
collection = client_for(root / 'storage/chroma').get_collection(snapshot['collection'])
assert collection.count() > 0, 'Empty Chroma collection'
# Open the vector segment as well as SQLite; this checks cross-platform index reads.
vector = collection.get(limit=1, include=['embeddings'])['embeddings'][0]
assert collection.query(query_embeddings=[list(vector)], n_results=1)['ids'][0]
assert (root / 'frontend/index.html').is_file()
assert (root / 'data/corpus/manifest.jsonl').is_file()
query_metadata(MetadataQuery(operation='count'), root / 'data/corpus/manifest.jsonl')
assert not (root / '.env').exists(), 'Do not include .env in the image'
print(f'Runtime files verified; {collection.count()} chunks; no API calls.')

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from .ingest import DEFAULT_CORPUS, DEFAULT_STORAGE, PROJECT_ROOT, ingest
from .retriever import Retriever
from .errors import RagError
from src.tools.sources import SourceRegistry

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Offline corpus ingestion and structured retrieval")
    parser.add_argument("command", choices=["ingest", "search"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--catalog", type=Path, help="JSON object keyed by corpus-relative file path")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            result = ingest(args.values or [args.corpus_root], corpus_root=args.corpus_root,
                storage=args.storage, embedding_model=args.model,
                catalog=json.loads(args.catalog.read_text()) if args.catalog else None)
        else:
            registry = SourceRegistry()
            result = registry.register(Retriever(args.storage, embedding_model=args.model).search(" ".join(args.values), args.k))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except RagError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False))
        raise SystemExit(1)

if __name__ == "__main__":
    main()

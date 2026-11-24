"""PDF/TXT/DOCX loading; file and directory inputs share one explicit corpus root."""
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from .errors import RagError

SUPPORTED = {".pdf", ".txt", ".docx"}

def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()


def discover_files(paths, corpus_root):
    root = Path(corpus_root).resolve()
    found = set()
    for value in paths:
        path = Path(value).resolve()
        if not path.exists():
            raise RagError("INPUT_NOT_FOUND", str(path))
        candidates = path.rglob("*") if path.is_dir() else [path]
        for file in candidates:
            if not file.is_file() or file.suffix.lower() not in SUPPORTED:
                continue
            file = file.resolve()
            if not file.is_relative_to(root):
                raise RagError("OUTSIDE_CORPUS", str(file))
            found.add(file)
    return sorted(found)


def load_file(path, corpus_root, catalog=None):
    path, root = Path(path).resolve(), Path(corpus_root).resolve()
    relative = path.relative_to(root).as_posix()
    catalog = catalog or {}
    doc_id = catalog.get("doc_id") or "doc_" + hashlib.sha256(relative.encode()).hexdigest()[:24]
    metadata = {"doc_id": doc_id, "title": catalog.get("title") or path.stem,
                "source": relative, "source_url": catalog.get("source_url") or "",
                "corpus_version": catalog.get("corpus_version") or "unversioned",
                "sha256": file_sha256(path)}
    try:
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
        elif path.suffix.lower() == ".txt":
            pages = [(None, path.read_text(encoding="utf-8-sig"))]
        elif path.suffix.lower() == ".docx":
            import docx2txt
            pages = [(None, docx2txt.process(str(path)))]
        else:
            raise ValueError("Unsupported document format")
    except Exception as exc:
        raise RagError("DOCUMENT_LOAD_FAILED", f"Cannot load {relative} ({type(exc).__name__})") from exc
    docs = []
    for page, text in pages:
        if text.strip():
            # Page blocks are explicit locators, not inferred scientific section names.
            locator = f"page-{page}" if page is not None else "full-text"
            docs.append(Document(page_content=text, metadata={**metadata, "page": page or 0,
                "section_id": locator, "section_title": f"Page {page}" if page else "Full text"}))
    if not docs:
        raise RagError("EMPTY_DOCUMENT", f"No extractable text in {relative}; scanned PDFs need OCR")
    return docs

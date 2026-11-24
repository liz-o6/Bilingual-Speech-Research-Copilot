"""Read full result pages from the snapshot without model or embedding calls."""
import hashlib
import re

from src.tools.sources import read_document


def read_result_pages(hits, reader=read_document, max_pages=4):
    documents = []
    for hit in hits:
        doc_id = hit.get('doc_id')
        if doc_id and doc_id not in documents:
            documents.append(doc_id)
    # Read one or two papers, not the entire corpus.
    documents = documents[:2]
    groups = []
    warnings = []
    for doc_id in documents:
        try:
            pages = reader(doc_id)
            chosen = []
            for index, page in enumerate(pages):
                if re.search(r'(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:results(?:\s+and\s+discussion)?|findings|结果)\s*$', page['text']):
                    # Include the results page and two following pages for context.
                    chosen = pages[index:index + 3]
                    break
            if not chosen:
                # If no results heading is found, read the hit page and the next page.
                hit_page = next((h.get('page') for h in hits if h.get('doc_id') == doc_id), None)
                for index, page in enumerate(pages):
                    if page['metadata'].get('page') == hit_page:
                        chosen = pages[index:index + 2]
                        break
            group = []
            for page in chosen:
                if not page['text'].strip():
                    continue
                metadata = page['metadata']
                identity = doc_id + ':' + str(metadata.get('page')) + ':' + page['text']
                group.append({**metadata, 'text': page['text'],
                              'chunk_id': 'page:' + hashlib.sha256(identity.encode()).hexdigest(),
                              'evidence_type': 'full_page'})
            groups.append(group)
        except Exception as exc:
            warnings.append(f'{doc_id}: page reading failed ({type(exc).__name__})')
    selected = []
    # Select two consecutive pages per paper before adding a third.
    for group in groups:
        selected.extend(group[:2])
    for group in groups:
        if len(selected) < max_pages:
            selected.extend(group[2:3])
    return selected[:max_pages], warnings

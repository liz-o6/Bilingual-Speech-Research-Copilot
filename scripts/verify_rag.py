"""Live, paid API smoke test. Run from hw2: python -m scripts.verify_rag.
Not a RAGAS evaluation. Never uses fake embeddings or fabricated results.
"""
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.rag.ingest import DEFAULT_CORPUS, DEFAULT_STORAGE, PROJECT_ROOT, ingest, read_snapshot
from src.rag.retriever import Retriever
from src.rag.answer import generate_cited_answer
from src.tools.sources import SourceRegistry, check_citations, verify_citations

CASES = [
    ("In the study 'Bilingual Speech Sound Development During the Preschool Years', how many children participated, and how far apart were the two assessments?", "bcs-001"),
    ("In 'Speech Perception and Production by Sequential Bilingual Children', which two languages did the bilingual children speak, and what speech contrast was studied?", "bcs-002"),
    ("In 'Interaction in Bilingual Phonological Acquisition: Evidence from Phonetic Inventories', which language pair was investigated and what aspect of children's phonology was compared?", "bcs-009"),
]

async def main():
    load_dotenv(PROJECT_ROOT / '.env')
    required = [name for name in ('OPENAI_API_KEY', 'OPENAI_MODEL') if not os.getenv(name)]
    if required:
        print('Configuration required in hw2/.env: ' + ', '.join(required))
        return 2
    model = os.environ['OPENAI_MODEL']
    embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    catalog = json.loads((PROJECT_ROOT / 'data/corpus/catalog.json').read_text())
    actual = {p.name for p in DEFAULT_CORPUS.glob('*.pdf')}
    if set(catalog) != actual:
        raise ValueError('Catalog filenames must exactly match the PDF corpus')
    output = PROJECT_ROOT / 'artifacts/evaluation/rag_acceptance.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'model': model,
              'embedding_model': embedding_model, 'status': 'running', 'cases': [],
              'scope': 'Live smoke test, not RAGAS or exhaustive citation/answer evaluation'}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    save()
    try:
        print('Ingesting real PDF corpus with OpenAI embeddings...', flush=True)
        report['ingestion'] = ingest([DEFAULT_CORPUS], catalog=catalog, embedding_model=embedding_model)
        second = ingest([DEFAULT_CORPUS], catalog=catalog, embedding_model=embedding_model)
        report['repeat_ingestion_unchanged'] = second['changed'] == 0
        snapshot = read_snapshot(DEFAULT_STORAGE)
        report['document_count'] = len(snapshot['documents'])
        report['snapshot'] = snapshot['collection']
        retriever = Retriever(embedding_model=embedding_model)
        llm = ChatOpenAI(model=model, max_retries=2, timeout=90)
        for question, expected_doc in CASES:
            start = perf_counter()
            hits = retriever.search(question, k=8)
            # Long quoted paper titles can dominate similarity: also retrieve the
            # actual question within the top-ranked paper, without using expected_doc.
            focused = re.sub(r"'[^']+'", '', question)
            if hits and focused != question:
                extra = retriever.search('Participants and study methods: ' + focused,
                                         k=5, doc_id=hits[0]['doc_id'])
                hits = list({h['chunk_id']: h for h in hits + extra}.values())
            retrieval_seconds = perf_counter() - start
            registry = SourceRegistry()
            registry.register(hits)
            provenance = []
            for hit in hits:
                record = snapshot['documents'][hit['source']]
                pages = [p for p in record['pages'] if p['metadata']['page'] == hit['page']]
                provenance.append(hit['chunk_id'] in record['chunk_ids'] and bool(pages)
                                  and hit['text'] in pages[0]['text']
                                  and hit['doc_id'] == record['doc_id']
                                  and bool(hit['source_url']))
            generated = await generate_cited_answer(question, registry, llm)
            answer = generated['answer']
            existence = check_citations(answer, registry)
            semantic = generated['citation_checks']
            passed = (expected_doc in {h['doc_id'] for h in hits} and bool(hits)
                      and all(provenance) and bool(existence)
                      and all(c['status'] != 'invalid' for c in existence)
                      and bool(semantic) and all(c['status'] == 'supported' for c in semantic))
            report['cases'].append({'question': question, 'expected_doc_id': expected_doc,
                                   'retrieval_seconds': retrieval_seconds,
                                   'total_seconds_including_verifier': perf_counter() - start,
                                   'answer': answer, 'sources': registry.sources,
                                   'provenance_checks': provenance, 'citation_checks': semantic,
                                   'generation_attempts': generated['attempts'],
                                   'automated_checks_passed': passed})
            save()
            print(expected_doc, 'PASS' if passed else 'REVIEW REQUIRED', flush=True)
        passed = all(c['automated_checks_passed'] for c in report['cases']) and report['repeat_ingestion_unchanged'] and report['document_count'] == 15
        report['status'] = 'automated_checks_passed' if passed else 'review_required'
        report['manual_review'] = 'Pending: compare answer claims with the saved evidence; automated judge is not proof of correctness.'
        save()
        print('Saved evidence and answers:', output)
        return 0 if passed else 1
    except Exception as exc:
        # Avoid logging provider errors containing request details or credentials.
        report['status'] = 'failed'
        report['error_type'] = type(exc).__name__
        save()
        print('Acceptance failed:', type(exc).__name__)
        return 1

if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))

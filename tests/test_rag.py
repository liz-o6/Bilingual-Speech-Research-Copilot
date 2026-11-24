import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from src.rag.ingest import ingest, read_snapshot, client_for
from src.rag.loaders import discover_files, load_file
from src.rag.retriever import Retriever
from src.rag.errors import RagError, retry
from src.tools.sources import SourceRegistry, check_citations, read_document


class TestEmbeddings:
    """Deterministic vectors for plumbing tests, NOT a retrieval-quality evaluator."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [(v + 1) / 256 for v in digest[:8]]


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'corpus'
        self.root.mkdir()
        self.storage = Path(self.temp.name) / 'index'
        self.file = self.root / 'study.txt'
        self.file.write_text('Bilingual children completed a naming task. Sample size was 24. ' * 10)
        self.kw = dict(corpus_root=self.root, storage=self.storage, embeddings=TestEmbeddings(),
                       embedding_model='test-only', chunk_size=100, chunk_overlap=20)

    def ingest(self, paths=None, **changes):
        return ingest(paths or [self.root], **{**self.kw, **changes})

    def test_file_directory_dedup_and_skip(self):
        self.assertEqual(discover_files([self.root, self.file], self.root), [self.file.resolve()])
        first = self.ingest([self.file])
        before = read_snapshot(self.storage)
        self.assertEqual(self.ingest()['changed'], 0)
        self.assertEqual(read_snapshot(self.storage), before)
        self.assertEqual(client_for(self.storage).get_collection(first['collection']).count(), first['chunks'])

    def test_modified_replaces_chunks_keeps_other_documents(self):
        second = self.root / 'other.txt'
        second.write_text('Other evidence from another study.')
        self.ingest()
        old = read_snapshot(self.storage)
        self.file.write_text('Updated evidence with a different sample.')
        self.ingest([self.file])
        new = read_snapshot(self.storage)
        actual = set(client_for(self.storage).get_collection(new['collection']).get(include=[])['ids'])
        self.assertFalse(actual.intersection(old['documents']['study.txt']['chunk_ids']))
        self.assertTrue(set(old['documents']['other.txt']['chunk_ids']).issubset(actual))
        self.assertEqual(old['documents']['study.txt']['doc_id'], new['documents']['study.txt']['doc_id'])

    def test_embedding_failure_preserves_hash_and_can_retry(self):
        self.ingest()
        old = read_snapshot(self.storage)
        self.file.write_text('Changed text after original ingestion.')
        class Broken(TestEmbeddings):
            def embed_documents(self, texts):
                raise ValueError('bad credentials or configuration')
        with self.assertRaises(RagError):
            self.ingest(embeddings=Broken())
        self.assertEqual(read_snapshot(self.storage), old)
        self.assertEqual(self.ingest()['changed'], 1)

    def test_commit_failure_preserves_old_index(self):
        self.ingest()
        old = read_snapshot(self.storage)
        self.file.write_text('New content')
        with patch('src.rag.ingest.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                self.ingest()
        self.assertEqual(read_snapshot(self.storage), old)
        self.assertEqual(self.ingest()['changed'], 1)

    def test_metadata_registry_fulltext_and_citations(self):
        self.ingest(catalog={'study.txt': {'doc_id': 'paper-01', 'title': 'Verified title', 'source_url': 'https://example.org/paper', 'corpus_version': 'v1'}})
        retriever = Retriever(self.storage, embeddings=TestEmbeddings(), embedding_model='test-only')
        hits = retriever.search('naming task', k=3)
        self.assertEqual(hits[0]['doc_id'], 'paper-01')
        self.assertIsNone(hits[0]['page'])
        self.assertEqual(hits[0]['source_url'], 'https://example.org/paper')
        self.assertTrue(hits[0]['chunk_id'])
        registry = SourceRegistry()
        registry.register(hits)
        registry.register(hits)
        self.assertEqual(len(registry.sources), len(hits))
        checks = check_citations('Claim [S1]. Another [S999].', registry)
        self.assertEqual([x['status'] for x in checks], ['unverified', 'invalid'])
        self.assertIn('[S1]', registry.prompt('Question'))
        self.assertEqual(read_document('paper-01', storage=self.storage)[0]['text'], self.file.read_text())
        self.assertEqual(retriever.search('question', doc_id='missing'), [])

    def test_config_and_invalid_inputs(self):
        self.ingest()
        with self.assertRaises(RagError):
            self.ingest(chunk_size=200)
        with self.assertRaises(RagError):
            Retriever(self.storage, embeddings=TestEmbeddings(), embedding_model='wrong')
        with self.assertRaises(RagError):
            discover_files([self.root / 'missing.pdf'], self.root)
        self.file.write_text('')
        with self.assertRaises(RagError):
            self.ingest()

    def test_pdf_page_metadata_and_docx(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        for content in ['First page evidence', 'Second page evidence']:
            page = writer.add_blank_page(612, 792)
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 10 700 Td ({content}) Tj ET'.encode())
            page[NameObject('/Contents')] = stream
        pdf = self.root / 'paper.pdf'
        with pdf.open('wb') as output:
            writer.write(output)
        pages = load_file(pdf, self.root)
        self.assertEqual([p.metadata['page'] for p in pages], [1, 2])
        self.assertIn('Second page', pages[1].page_content)
        docx = self.root / 'paper.docx'
        with ZipFile(docx, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>DOCX evidence</w:t></w:r></w:p></w:body></w:document>')
        self.assertEqual(load_file(docx, self.root)[0].page_content, 'DOCX evidence')
        self.ingest()
        hits = Retriever(self.storage, embeddings=TestEmbeddings(), embedding_model='test-only').search('page', k=20, doc_id=pages[0].metadata['doc_id'])
        self.assertEqual({hit['page'] for hit in hits}, {1, 2})

    def test_catalog_retained_when_not_resupplied(self):
        self.ingest(catalog={'study.txt': {'doc_id': 'stable-id', 'title': 'Verified title'}})
        self.assertEqual(self.ingest()['changed'], 0)
        self.file.write_text('Changed text')
        self.ingest()
        self.assertEqual(read_snapshot(self.storage)['documents']['study.txt']['doc_id'], 'stable-id')

    def test_semantic_verifier_missing_and_failed_output(self):
        import asyncio
        from src.tools.sources import verify_citations
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'c1', 'text': 'Evidence'}])
        class Judge:
            def with_structured_output(self, schema):
                self.schema = schema
                return self
            async def ainvoke(self, prompt):
                return self.schema(checks=[])
        result = asyncio.run(verify_citations('Claim [S1]', registry, Judge()))
        self.assertEqual(result[0]['status'], 'unverified')
        class BrokenJudge(Judge):
            async def ainvoke(self, prompt):
                raise TimeoutError()
        result = asyncio.run(verify_citations('Claim [S1]', registry, BrokenJudge()))
        self.assertEqual(result[0]['status'], 'unverified')

    def test_retry_only_transient(self):
        calls = []
        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError()
            return 'ok'
        with patch('src.rag.errors.time.sleep'):
            self.assertEqual(retry(operation), 'ok')
        self.assertEqual(len(calls), 3)

if __name__ == '__main__':
    unittest.main()

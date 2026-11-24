import unittest
from src.rag.reading import read_result_pages


class ReadingTests(unittest.TestCase):
    def reader(self, doc_id):
        return [{'text': text, 'metadata': {'doc_id': doc_id, 'page': i + 1}}
                for i, text in enumerate(['Introduction', '4. Results\nObserved result begins',
                                          'Result continues here', '5. Discussion\nLimitations'])]

    def test_locates_results_and_reads_continuation(self):
        pages, warnings = read_result_pages([{'doc_id': 'p1', 'page': 1}], self.reader)
        self.assertEqual([p['page'] for p in pages], [2, 3, 4])
        self.assertEqual(warnings, [])
        self.assertTrue(all(p['chunk_id'].startswith('page:') for p in pages))

    def test_two_papers_bounded_and_no_heading_fallback(self):
        pages, _ = read_result_pages([{'doc_id': 'p1'}, {'doc_id': 'p2'}], self.reader)
        self.assertEqual([(p['doc_id'], p['page']) for p in pages],
                         [('p1', 2), ('p1', 3), ('p2', 2), ('p2', 3)])
        def reader(doc):
            return [{'text': 'No heading', 'metadata': {'doc_id': doc, 'page': n}} for n in [1, 2, 3]]
        pages, _ = read_result_pages([{'doc_id': 'p1', 'page': 2}], reader)
        self.assertEqual([p['page'] for p in pages], [2, 3])

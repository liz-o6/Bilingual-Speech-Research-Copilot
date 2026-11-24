import unittest
from src.rag.selection import is_reference_list, select_evidence


class SelectionTests(unittest.TestCase):
    def test_bibliography_and_normal_citations(self):
        references = '\n'.join(f'Kehoe, M. ({year}). Language study. Journal, 38, 75–81.' for year in (2011, 2015, 2020))
        self.assertTrue(is_reference_list(references))
        body = 'Smith (2020) found improvements. Jones (2021) reported differences.\nResults: accuracy improved over time, but not for all segments.'
        self.assertFalse(is_reference_list(body))
        hits = [{'text': references}, {'text': body, 'chunk_id': 'body'},
                {'text': body, 'chunk_id': 'duplicate'}, {'text': 'A different result.', 'chunk_id': 'other'}]
        self.assertEqual([h['chunk_id'] for h in select_evidence(hits)], ['body', 'other'])

    def test_does_not_fill_with_references_or_exceed_four(self):
        self.assertEqual(select_evidence([{'text': 'References\nSmith, J. (2020). Title.'}]), [])
        hits = [{'text': f'Distinct finding number {i} about topic {i}', 'chunk_id': str(i)} for i in range(8)]
        self.assertEqual(len(select_evidence(hits)), 4)

    def test_vancouver_references(self):
        text = 'Smith A. A paper. Journal. 2004; 35:5–15.\nJones B. Another. 2007; 45(4):353.\nHarada T. VOT. 2008; 36(4):649–663.'
        self.assertTrue(is_reference_list(text))
        self.assertFalse(is_reference_list('Results differed (Smith, 2004; Jones, 2008). Accuracy improved.'))

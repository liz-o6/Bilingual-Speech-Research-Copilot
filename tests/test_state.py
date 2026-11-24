import unittest

from src.orchestration.state import State, ResearchResult
from src.tools.sources import SourceRegistry


class StateTests(unittest.TestCase):
    def test_results_do_not_share_sources(self):
        first = ResearchResult(status='success')
        second = ResearchResult(status='success')
        first.sources['S1'] = {'text': 'Evidence'}
        self.assertEqual(second.sources, {})
        self.assertIsNone(State(data_task='', query='A new question', research_task='', reason='').research_result)

    def test_research_roundtrip_preserves_existing_sources(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id':'doc-1:p2:0', 'doc_id':'doc-1',
                            'text':'Evidence', 'page':2, 'distance':0.15}])
        state = State(data_task='', query='What is the evidence?', research_task='Search the corpus', reason='',
                      research_result=ResearchResult(status='success',
                          answer='Evidence [S1]', sources=registry.sources))
        restored = State.model_validate_json(state.model_dump_json())
        self.assertEqual(restored.research_result.sources, registry.sources)
        self.assertEqual(restored.research_result.answer, 'Evidence [S1]')
        self.assertIsNone(restored.final_answer)

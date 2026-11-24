import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from src.rag.answer import generate_cited_answer
from src.tools.sources import SourceRegistry

class AnswerTests(unittest.TestCase):
    def test_failed_support_stops_without_repair(self):
        reg=SourceRegistry();reg.register([{'chunk_id':'c1','text':'Evidence'}])
        llm=SimpleNamespace(ainvoke=AsyncMock(side_effect=[SimpleNamespace(content='Draft [S1]',usage_metadata={}),SimpleNamespace(content='Revised [S1]',usage_metadata={})]))
        with patch('src.rag.answer.verify_citations',new=AsyncMock(side_effect=[[{'status':'unsupported'}],[{'status':'supported'}]])):
            result=asyncio.run(generate_cited_answer('Question',reg,llm))
        self.assertTrue(result['review_required'])
        self.assertEqual(len(result['attempts']),1)
        self.assertEqual(result['attempts'][0]['answer'],'Draft [S1]')

    def test_persistent_failure_remains_review_required(self):
        reg=SourceRegistry();reg.register([{'chunk_id':'c1','text':'Evidence'}])
        llm=SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='Draft [S1]',usage_metadata={})))
        with patch('src.rag.answer.verify_citations',new=AsyncMock(return_value=[{'status':'unverified'}])):
            result=asyncio.run(generate_cited_answer('Question',reg,llm))
        self.assertTrue(result['review_required'])
        self.assertEqual(llm.ainvoke.await_count,1)

class CitationBatchTests(unittest.TestCase):
    def test_batches_isolate_failure_and_filter_sources(self):
        import json
        from src.tools.sources import verify_citations
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'c1', 'text': 'Evidence'},
                           {'chunk_id': 'c2', 'text': 'Other evidence'}])

        class Judge:
            def __init__(self):
                self.batches = []
                self.sources = []

            def with_structured_output(self, schema, method=None):
                self.schema = schema
                self.method = method
                return self

            async def ainvoke(self, prompt):
                pairs_text, sources_text = prompt.split('\nPairs: ')[1].split('\nSources: ')
                pairs = json.loads(pairs_text)
                self.batches.append(pairs)
                self.sources.append(json.loads(sources_text))
                if len(self.batches) == 2:
                    raise ValueError('Simulated parsing failure')
                output = []
                for pair in pairs:
                    output.append(dict(pair, status='supported', reason='Direct evidence'))
                return self.schema(checks=output)

        report = '\n'.join(f'Claim {i} [S1]' for i in range(6))
        report += '\nFinal claim [S2]\nUnknown claim [S99]'
        judge = Judge()
        checks = asyncio.run(verify_citations(report, registry, judge, method='json_schema'))
        self.assertEqual([len(batch) for batch in judge.batches], [3, 3, 1])
        self.assertEqual([check['status'] for check in checks],
                         ['supported'] * 3 + ['unverified'] * 3 + ['supported', 'invalid'])
        self.assertEqual(judge.method, 'json_schema')
        self.assertEqual([s['citation_id'] for s in judge.sources[0]], ['S1'])
        self.assertEqual([s['citation_id'] for s in judge.sources[2]], ['S2'])
        self.assertEqual(checks[6]['claim'], 'Final claim')

class CombinedCitationTests(unittest.TestCase):
    def test_combined_labels_are_all_checked_and_unknown_is_invalid(self):
        from src.tools.sources import extract_cited_claims, check_citations
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'Evidence'}, {'chunk_id': 'b', 'text': 'Evidence B'}])
        report = 'Finding [S1, S2] [S1]\nOther [S2，S99]\nMetadata [M_bcs-001; S1]'
        self.assertEqual(extract_cited_claims(report), [
            ('Finding', 'S1'), ('Finding', 'S2'), ('Other', 'S2'), ('Other', 'S99'),
            ('Metadata', 'M_bcs-001'), ('Metadata', 'S1')])
        checks = check_citations(report, registry)
        self.assertEqual(checks[3]['status'], 'invalid')

    def test_conditional_rejection_requires_revision(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'If X occurs, it could support Y.'}])
        llm = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            SimpleNamespace(content='The study found Y [S1]', usage_metadata={}),
            SimpleNamespace(content='The authors hypothesize that if X occurs, it could support Y [S1]', usage_metadata={})]))
        with patch('src.rag.answer.verify_citations', new=AsyncMock(side_effect=[
            [{'status': 'unsupported', 'reason': 'Conditional hypothesis, not an observed result'}],
            [{'status': 'supported', 'reason': 'Preserves conditional interpretation'}]])):
            result = asyncio.run(generate_cited_answer('Explain the evidence', registry, llm))
        self.assertEqual(len(result['attempts']), 1)
        self.assertIn('The study found', result['answer'])
        self.assertTrue(result['review_required'])

    def test_joint_evidence_reaches_verifier_as_one_claim(self):
        import json
        from src.tools.sources import verify_citations
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'At Time 2 there was no significant'},
                           {'chunk_id': 'b', 'text': 'difference between groups.'}])
        class Judge:
            def with_structured_output(self, schema):
                self.schema = schema
                return self
            async def ainvoke(self, prompt):
                pairs, sources = prompt.split('\nPairs: ')[1].split('\nSources: ')
                self.pairs = json.loads(pairs)
                self.sources = json.loads(sources)
                return self.schema(checks=[dict(self.pairs[0], status='supported', reason='Sentence spans both pages')])
        judge = Judge()
        checks = asyncio.run(verify_citations('No difference at Time 2 [S1, S2]', registry, judge))
        self.assertEqual(len(judge.pairs), 1)
        self.assertEqual(judge.pairs[0]['citation_id'], 'S1,S2')
        self.assertEqual(len(judge.sources), 2)
        self.assertEqual(checks[0]['status'], 'supported')
        checks = asyncio.run(verify_citations('Claim [S1, S99]', registry, judge))
        self.assertEqual(checks[0]['status'], 'invalid')

    def test_truncated_answer_cannot_pass_even_with_supported_citations(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'Evidence'}])
        response = SimpleNamespace(content='Incomplete [S1]', usage_metadata={},
                                   response_metadata={'stop_reason': 'max_tokens'})
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        with patch('src.rag.answer.verify_citations', new=AsyncMock(side_effect=[
            [{'status': 'supported'}], [{'status': 'supported'}]])):
            result = asyncio.run(generate_cited_answer('Question', registry, llm))
        self.assertTrue(result['review_required'])
        self.assertEqual(result['attempts'][-1]['finish_reason'], 'max_tokens')


class LowCostAnswerTests(unittest.TestCase):
    def test_deferred_check_never_calls_judge_or_claims_supported(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'Evidence'}])
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='Finding [S1]', usage_metadata={})))
        with patch('src.rag.answer.verify_citations', new=AsyncMock()) as judge:
            result = asyncio.run(generate_cited_answer('Question', registry, llm, verify=False))
        judge.assert_not_awaited()
        self.assertTrue(result['verification_deferred'])
        self.assertFalse(result['review_required'])
        self.assertEqual(result['citation_checks'][0]['status'], 'unverified')
        llm.ainvoke.assert_awaited_once()

    def test_empty_thinking_only_output_is_incomplete_without_retry(self):
        response = SimpleNamespace(content='', usage_metadata={'output_tokens': 2048},
                                   response_metadata={'stop_reason': 'max_tokens'})
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=response))
        with patch('src.rag.answer.verify_citations', new=AsyncMock()) as judge:
            result = asyncio.run(generate_cited_answer('Question', SourceRegistry(), llm))
        judge.assert_not_awaited()
        llm.ainvoke.assert_awaited_once()
        self.assertTrue(result['review_required'])
        self.assertEqual(result['answer'], '')

    def test_final_verification_requests_one_batch(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'a', 'text': 'Evidence'}])
        llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='Finding [S1]', usage_metadata={})))
        with patch('src.rag.answer.verify_citations', new=AsyncMock(return_value=[{'status': 'supported'}])) as judge:
            result = asyncio.run(generate_cited_answer('Question', registry, llm))
        judge.assert_awaited_once()
        self.assertIsNone(judge.await_args.kwargs['batch_size'])
        self.assertFalse(result['review_required'])

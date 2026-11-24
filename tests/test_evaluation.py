import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch
from src.eval.common import DATASET, read, save, cost
from src.observability.timing import timed
from src.api.service import ChatRun, ChatService, current_run


class EvaluationTests(unittest.TestCase):
    @unittest.skipUnless(DATASET.exists(), "Original reference excerpts are excluded from the public release")
    def test_fixed_dataset_has_five_references(self):
        cases=read(DATASET)
        self.assertEqual(len(cases),5)
        self.assertEqual(len({c['id'] for c in cases}),5)
        for case in cases:
            self.assertTrue(case['reference_answer'])
            self.assertTrue(case['reference_sources'][0]['text'])
            self.assertIn('Do not search the web',case['question'])

    def test_timing_includes_failure_and_missing_is_not_zero(self):
        run=ChatRun('Q')
        self.assertIsNone(run.metrics()['retrieval_ms'])
        with patch('src.observability.timing.perf_counter',side_effect=[10,10.25]):
            with self.assertRaises(ValueError):
                with timed(run.traces,'rag_retrieval','Research'):raise ValueError()
        self.assertEqual(run.metrics()['retrieval_ms'],250)
        self.assertEqual(run.traces[0]['status'],'failed')

    def test_unknown_prices_and_cached_usage_are_not_free(self):
        calls=[dict(provider='p',model='m',usage={'input_tokens':100,'output_tokens':20})]
        self.assertIsNone(cost(calls,{})['generation_cost_usd'])
        prices={'p/m':{'input_per_million':1,'output_per_million':2}}
        self.assertAlmostEqual(cost(calls,prices)['generation_cost_usd'],.00014)
        calls[0]['usage']['input_token_details']={'cache_read':100}
        self.assertIsNone(cost(calls,prices)['generation_cost_usd'])

    def test_evaluation_captures_ordered_actual_context_not_references(self):
        class Graph:
            async def ainvoke(self,state):
                run=current_run.get()
                run.traces.append({'event':'research_evidence','sources':{'S2':{'text':'Actual second'},'S1':{'text':'Actual first'}}})
                state.update(next_agent='end',final_answer='Answer')
                return state
        async def run(capture):
            return [e async for e in ChatService(Graph(),capture_evaluation=capture).events('Q')]
        events=asyncio.run(run(True))
        payload=next(e['data'] for e in events if e['event']=='evaluation')
        self.assertEqual([s['text'] for s in payload['contexts']],['Actual second','Actual first'])
        self.assertFalse(any(e['event']=='evaluation' for e in asyncio.run(run(False))))

    def test_openai_cached_input_is_not_double_counted(self):
        calls = [dict(provider='openai', model='m', usage={
            'input_tokens': 1000, 'output_tokens': 100,
            'input_token_details': {'cache_read': 800}})]
        prices = {'openai/m': {'input_per_million': .2,
                              'output_per_million': 1.25,
                              'cached_input_per_million': .02}}
        self.assertAlmostEqual(cost(calls, prices)['generation_cost_usd'], .000181)

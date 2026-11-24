import asyncio
import unittest
from unittest.mock import patch, AsyncMock, call
from langchain_core.messages import AIMessageChunk
from src.eval.compare_providers import make_messages, collect_one, pause_before_request


class ProviderComparisonTests(unittest.TestCase):
    def test_only_actual_context_is_sent(self):
        record = {'question': 'Q', 'retrieved_contexts': ['Evidence'],
                  'sources': [{'text': 'Evidence', 'title': 'Paper', 'page': 2}],
                  'reference_answer': 'SECRET REFERENCE'}
        messages = make_messages(record)
        self.assertIn('[S1]', messages[1][1])
        self.assertIn('Evidence', messages[1][1])
        self.assertNotIn('SECRET REFERENCE', str(messages))

    def test_stream_failure_retains_partial_answer_without_retry(self):
        class Model:
            async def astream(self, messages):
                yield AIMessageChunk(content='Partial')
                raise TimeoutError()
        with patch('src.eval.compare_providers.create_llm', return_value=Model()) as factory:
            result = asyncio.run(collect_one('openai', [], 500))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['answer'], 'Partial')
        self.assertIsNotNone(result['ttft_ms'])
        self.assertEqual(factory.call_args.kwargs['max_retries'], 0)

    def test_groq_requests_stream_usage(self):
        class Model:
            def bind(self, **kwargs):
                self.options = kwargs
                return self
            async def astream(self, messages):
                yield AIMessageChunk(content='Answer', response_metadata={'finish_reason': 'stop'})
        model = Model()
        with patch('src.eval.compare_providers.create_llm', return_value=model):
            result = asyncio.run(collect_one('groq', [], 500))
        self.assertEqual(model.options, {'stream_options': {'include_usage': True}})
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['status'], 'success')

    def test_pacing_is_split_into_short_waits(self):
        with patch('src.eval.compare_providers.asyncio.sleep', new_callable=AsyncMock) as sleep:
            asyncio.run(pause_before_request(65))
        self.assertEqual(sleep.await_args_list, [call(30), call(30), call(5)])

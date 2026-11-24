import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from src.api.main import create_app
from src.api.service import ChatService, current_run
from src.orchestration.state import ResearchResult
from src.rag.answer import generate_cited_answer
from src.tools.sources import SourceRegistry


class FakeGraph:
    def __init__(self, status="success", fail=False):
        self.status = status
        self.fail = fail
        self.calls = 0
        self.cancelled = False

    async def ainvoke(self, state):
        self.calls += 1
        self.assert_fresh = state['research_result'] is None
        if self.fail:
            raise RuntimeError('secret key /Users/private')
        run = current_run.get()
        run.traces.emit_token(state['query'])
        await asyncio.sleep(0)
        run.traces.append({'event': 'llm_call', 'agent': 'Research', 'provider': 'fake',
                          'model': 'offline', 'status': 'success',
                          'usage': {'input_tokens': 3, 'output_tokens': 2}})
        state.update(next_agent='end', agent_plan=['research'], plan_index=0,
                     final_answer=state['query'] + ' [S1]',
                     research_result=ResearchResult(status=self.status, answer='Answer [S1]', sources={
                         'S1': {'title': 'Paper', 'source': '/Users/private/file.pdf',
                                'source_url': 'file:///Users/private/file.pdf', 'text': 'Evidence', 'page': 1}}))
        return state


class ApiTests(unittest.TestCase):
    def test_health_validation_and_json(self):
        graph = FakeGraph()
        with TestClient(create_app(ChatService(graph))) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertIn('Speech Research', client.get('/').text)
            self.assertEqual(client.get('/static/js/app.js').status_code, 200)
            self.assertEqual(client.get('/static/../.env').status_code, 404)
            self.assertEqual(client.get('/static/%2e%2e/.env').status_code, 404)
            for body in ({'query': ' '}, {'query': 'x' * 4001}, {'query': 'x', 'api_key': 'secret'}, {}):
                self.assertEqual(client.post('/api/chat', json=body).status_code, 422)
            self.assertEqual(graph.calls, 0)
            response = client.post('/api/chat/result', json={'query': '中文问题'})
        result = response.json()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], '中文问题 [S1]')
        self.assertEqual(result['metrics']['input_tokens'], 3)
        self.assertNotIn('/Users/private', response.text)
        self.assertTrue(graph.assert_fresh)

    def test_sse_sequence_and_one_done(self):
        with TestClient(create_app(ChatService(FakeGraph()))) as client:
            response = client.post('/api/chat', json={'query': '中文\nquestion'})
        self.assertIn('text/event-stream', response.headers['content-type'])
        frames = response.text.strip().split('\n\n')
        events = [(f.splitlines()[0][7:], json.loads(f.splitlines()[1][6:])) for f in frames]
        self.assertEqual([d['seq'] for _, d in events], list(range(1, len(events) + 1)))
        self.assertEqual(len({d['request_id'] for _, d in events}), 1)
        self.assertEqual([name for name, _ in events].count('done'), 1)
        self.assertIn('token', [name for name, _ in events])
        self.assertEqual(events[-1][0], 'done')

    def test_partial_and_errors_are_not_success(self):
        for graph, expected in [(FakeGraph(status='partial'), 'partial'), (FakeGraph(fail=True), 'failed')]:
            with TestClient(create_app(ChatService(graph))) as client:
                response = client.post('/api/chat/result', json={'query': 'Test'})
            self.assertEqual(response.json()['status'], expected)
            self.assertIsNotNone(response.json()['error'])
            self.assertNotIn('secret key', response.text)


class AsyncApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_do_not_share_state_or_trace(self):
        service = ChatService(FakeGraph())
        async def collect(query):
            return [e async for e in service.events(query)]
        a, b = await asyncio.gather(collect('AAA'), collect('BBB'))
        self.assertNotEqual(a[0]['data']['request_id'], b[0]['data']['request_id'])
        self.assertNotIn('BBB', json.dumps(a))
        self.assertNotIn('AAA', json.dumps(b))

    async def test_closing_event_stream_cancels_graph(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        class SlowGraph:
            async def ainvoke(self, state):
                started.set()
                try:
                    await asyncio.sleep(30)
                finally:
                    cancelled.set()
        events = ChatService(SlowGraph()).events('Test')
        await anext(events)
        await started.wait()
        await events.aclose()
        self.assertTrue(cancelled.is_set())

    async def test_timeout_is_terminal(self):
        class SlowGraph:
            async def ainvoke(self, state):
                await asyncio.sleep(30)
        events = [e async for e in ChatService(SlowGraph(), timeout_seconds=0.01).events('Test')]
        self.assertEqual(events[-2]['event'], 'error')
        self.assertEqual(events[-1]['data']['status'], 'failed')

    async def test_text_tokens_stream_but_verifier_does_not(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id': 'x', 'text': 'Evidence'}])
        async def stream(prompt):
            yield AIMessageChunk(content='Finding ')
            yield AIMessageChunk(content='[S1]', response_metadata={'stop_reason': 'end_turn'})
        model = SimpleNamespace(astream=stream)
        tokens = []
        with patch('src.rag.answer.verify_citations', new=AsyncMock(return_value=[{'status': 'supported'}])) as judge:
            result = await generate_cited_answer('Question', registry, model, on_token=tokens.append)
        self.assertEqual(tokens, ['Finding ', '[S1]'])
        self.assertEqual(result['answer'], 'Finding [S1]')
        judge.assert_awaited_once()

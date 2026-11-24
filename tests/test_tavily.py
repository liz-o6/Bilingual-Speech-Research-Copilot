import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from mcp.types import CallToolResult, TextContent
from src.mcp.tavily import parse_results
from src.agents.research import research
from src.orchestration.state import State


class TavilyTests(unittest.IsolatedAsyncioTestCase):
    def test_mcp_results_keep_provenance_and_stable_identity(self):
        raw = '{"results":[{"url":"https://example.org/paper","title":"Paper","content":"Evidence"}]}'
        result = CallToolResult(content=[TextContent(type='text',text=raw)])
        hits = parse_results(result)
        self.assertEqual(hits[0]['source_type'],'web')
        self.assertEqual(hits[0]['source_url'],'https://example.org/paper')
        self.assertEqual(hits[0]['chunk_id'],parse_results(result)[0]['chunk_id'])
        self.assertIn('accessed_at',hits[0])
        with self.assertRaises(ValueError):
            parse_results(CallToolResult(isError=True,content=[]))
        with self.assertRaises(ValueError):
            parse_results(CallToolResult(content=[TextContent(type='text',text='Invalid output')]))

    async def test_live_sources_work_without_local_hits(self):
        state = State(data_task='', query='Latest research',research_task='Search recent papers',reason='',research_use_web=True)
        retriever = SimpleNamespace(asearch=AsyncMock(return_value=[]))
        hits = [{'chunk_id':'web:1','text':'Recent finding','source_url':'https://example.org','source_type':'web'}]
        with patch('src.agents.research.search_web',new=AsyncMock(return_value=hits)) as web:
            with patch('src.agents.research.generate_cited_answer',new=AsyncMock(return_value={
                'answer':'Recent finding [S1]','review_required':False})):
                result = await research(state,llm=object(),retriever=retriever)
        web.assert_awaited_once()
        self.assertEqual(result.status,'success')
        self.assertEqual(result.sources['S1']['source_type'],'web')

    async def test_web_failure_preserves_local_evidence_as_partial(self):
        state = State(data_task='', query='Latest research',research_task='Search',reason='',research_use_web=True)
        retriever = SimpleNamespace(asearch=AsyncMock(return_value=[{'chunk_id':'local','text':'Local evidence'}]))
        with patch('src.agents.research.search_web',new=AsyncMock(side_effect=RuntimeError('secret-url'))):
            with patch('src.agents.research.generate_cited_answer',new=AsyncMock(return_value={
                'answer':'Local evidence [S1]','review_required':False})):
                result = await research(state,llm=object(),retriever=retriever)
        self.assertEqual(result.status,'partial')
        self.assertEqual(result.sources['S1']['chunk_id'],'local')
        self.assertNotIn('secret-url',result.model_dump_json())

    async def test_static_query_skips_web(self):
        state = State(data_task='', query='Paper question',research_task='Search corpus',reason='')
        retriever = SimpleNamespace(asearch=AsyncMock(return_value=[]))
        with patch('src.agents.research.search_web',new=AsyncMock()) as web:
            await research(state,llm=object(),retriever=retriever)
        web.assert_not_awaited()

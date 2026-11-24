import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage
from src.agents.data import data
from src.tools.metadata import MetadataQuery, query_metadata
from src.orchestration.state import State
from src.orchestration.supervisor import Supervisor, ResearchTask, ResearchReview, run_workflow


class DataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'papers.jsonl'
        papers = [
            {'doc_id':'p1','title':'Bilingual speech','authors':['Smith'],'year':2020,'journal':'J1'},
            {'doc_id':'p2','title':'Speech learning','authors':['Jones'],'year':2020,'journal':'J2'},
            {'doc_id':'p3','title':'Phonology','authors':['Smith'],'year':None,'journal':'J1'},
        ]
        self.path.write_text('\n'.join(json.dumps(p) for p in papers))

    def tool(self, query):
        return query_metadata(query,self.path)

    def state(self):
        return State(query='How many papers?',research_task='',data_task='Count local papers',reason='')

    def llm(self, query):
        llm = Mock()
        llm.with_structured_output.return_value = SimpleNamespace(ainvoke=AsyncMock(return_value=query))
        llm.ainvoke = AsyncMock(return_value=AIMessage(content='The tool found 3 papers.'))
        return llm

    def test_counts_filters_groups_and_nulls(self):
        result = self.tool(MetadataQuery(operation='count'))
        self.assertEqual(result['rows'][0]['count'],3)
        self.assertEqual(len(result['sources']),3)
        self.assertEqual(self.tool(MetadataQuery(operation='count',year_from=2020))['rows'][0]['count'],2)
        result = self.tool(MetadataQuery(operation='group_count'))
        self.assertEqual([(r['year'],r['count']) for r in result['rows']],[(None,1),(2020,2)])
        result = self.tool(MetadataQuery(author_contains='smith',journal_contains='j1'))
        self.assertEqual(len(result['rows']),2)
        self.assertEqual(self.tool(MetadataQuery(operation='count',title_contains="' OR 1=1 --"))['rows'][0]['count'],0)
        self.assertEqual(self.tool(MetadataQuery(operation='count',title_contains='%'))['rows'][0]['count'],0)
        with self.assertRaises(ValueError): self.tool(MetadataQuery(year_from=2025,year_to=2020))

    async def test_agent_keeps_tool_rows_and_handles_unsupported(self):
        llm = self.llm(MetadataQuery(operation='count'))
        result = await data(self.state(),llm=llm,metadata_tool=self.tool)
        self.assertEqual(result.rows[0]['count'],3)
        self.assertEqual(result.status,'success')
        llm.ainvoke.assert_not_awaited()
        llm.with_structured_output.return_value.ainvoke.assert_awaited_once()
        llm = self.llm(MetadataQuery(operation='unsupported',unsupported_reason='Sample sizes not curated'))
        result = await data(self.state(),llm=llm,metadata_tool=self.tool)
        self.assertEqual(result.status,'partial')
        self.assertEqual(result.error, 'Sample sizes not curated')
        llm.ainvoke.assert_not_awaited()

    async def test_explanation_failure_preserves_computed_results(self):
        llm = self.llm(MetadataQuery(operation='list'))
        llm.ainvoke.side_effect = RuntimeError('secret')
        result = await data(self.state(),llm=llm,metadata_tool=self.tool)
        self.assertEqual(result.status,'partial')
        self.assertEqual(len(result.rows),3)
        self.assertNotIn('secret',result.model_dump_json())

    async def test_supervisor_routes_to_data_and_accepts(self):
        llm = Mock()
        llm.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(next_agent='data',data_task='Count papers',reason='Metadata question'))),
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchReview(action='accept',reason='Matches tool counts'))),
        ]
        data_llm = self.llm(MetadataQuery(operation='count'))
        async def worker(state):
            return await data(state,llm=data_llm,metadata_tool=self.tool)
        research = AsyncMock()
        original = self.state()
        result = await run_workflow(original,Supervisor(llm),research,worker)
        research.assert_not_awaited()
        self.assertEqual(result.data_result.rows[0]['count'],3)
        self.assertIn('3 papers',result.final_answer)
        self.assertIsNone(original.data_result)

    async def test_unsupported_records_decision_and_preserves_reason(self):
        traces = []
        result = await data(self.state(), llm=self.llm(MetadataQuery(
            operation='unsupported', unsupported_reason='No participant metadata')),
            traces=traces, metadata_tool=self.tool)
        self.assertEqual(result.error, 'No participant metadata')
        self.assertEqual(traces[0]['event'], 'data_plan')
        self.assertEqual(traces[0]['operation'], 'unsupported')
        self.assertEqual(traces[0]['task'], self.state().data_task)

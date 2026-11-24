import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from src.orchestration.graph import build_graph, run_workflow
from src.orchestration.state import State, ResearchResult, DataResult, SummaryResult
from src.orchestration.supervisor import Supervisor, ResearchTask


class GraphTests(unittest.IsolatedAsyncioTestCase):
    def supervisor(self, plan):
        llm = Mock()
        llm.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(
                agent_plan=plan, next_agent=plan[0], reason='Test',
                research_task='Find evidence', data_task='Count papers', summary_task='Combine results'))),
            SimpleNamespace(ainvoke=AsyncMock()),
        ]
        return Supervisor(llm)

    def state(self):
        return State(query='Test', research_task='', data_task='', reason='')

    async def test_three_agents_share_results_in_order(self):
        calls = []

        async def data(state):
            calls.append('data')
            return DataResult(status='success', rows=[{'count': 8}], explanation='8 papers')

        async def research(state):
            calls.append('research')
            self.assertEqual(state.data_result.rows, [{'count': 8}])
            return ResearchResult(status='success', answer='Evidence [S1]', sources={'S1': {'text': 'Evidence'}})

        async def summary(state):
            calls.append('summary')
            self.assertEqual(state.summary_mode, 'shared')
            self.assertEqual(state.data_result.rows, [{'count': 8}])
            self.assertIn('S1', state.research_result.sources)
            return SummaryResult(status='success', answer='8 papers; Evidence [S1]')

        original = self.state()
        graph = build_graph(self.supervisor(['data', 'research', 'summary']), research, data, summary)
        result = await run_workflow(graph, original)
        self.assertEqual(calls, ['data', 'research', 'summary'])
        self.assertEqual(result.final_answer, '8 papers; Evidence [S1]')
        self.assertEqual(result.step_count, 4)
        self.assertIsNone(original.data_result)

    async def test_partial_stops_without_retry(self):
        worker = AsyncMock(return_value=ResearchResult(status='partial', answer='Unchecked'))
        graph = build_graph(self.supervisor(['research']), worker, AsyncMock(), AsyncMock())
        result = await run_workflow(graph, self.state())
        self.assertEqual(worker.await_count, 1)
        self.assertEqual(result.repair_count, 0)
        self.assertEqual(result.next_agent, 'end')
        self.assertNotEqual(result.final_answer, 'Unchecked')

    async def test_exception_returns_failed_result(self):
        worker = AsyncMock(side_effect=RuntimeError('secret'))
        graph = build_graph(self.supervisor(['research']), worker, AsyncMock(), AsyncMock())
        result = await run_workflow(graph, self.state())
        self.assertEqual(result.research_result.status, 'failed')
        self.assertEqual(result.next_agent, 'end')
        self.assertNotIn('secret', result.model_dump_json())

    async def test_ended_state_does_not_run_agents(self):
        state = self.state()
        state.next_agent = 'end'
        state.final_answer = 'Done'
        node = AsyncMock()
        graph = build_graph(node, node, node, node)
        result = await run_workflow(graph, state)
        node.assert_not_awaited()
        self.assertEqual(result.final_answer, 'Done')

    async def test_reuse_graph_starts_with_fresh_state(self):
        worker = AsyncMock(return_value=ResearchResult(status='success', answer='Done'))
        graph = build_graph(self.supervisor(['research']), worker, AsyncMock(), AsyncMock())
        for question in ['First', 'Second']:
            state = self.state()
            state.query = question
            result = await run_workflow(graph, state)
            self.assertEqual(result.step_count, 2)
            self.assertEqual(result.final_answer, 'Done')
        self.assertEqual(worker.await_count, 2)
        for call in worker.await_args_list:
            self.assertIsNone(call.args[0].research_result)

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from src.orchestration.state import State, ResearchResult
from src.orchestration.supervisor import Supervisor, ResearchTask, ResearchReview, delegate_to_research


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setup_supervisor(self, reviews):
        planner = AsyncMock(return_value=ResearchTask(research_task='Find evidence.',reason='Need sources'))
        reviewer = AsyncMock(side_effect=reviews)
        llm = Mock()
        llm.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=planner), SimpleNamespace(ainvoke=reviewer)]
        return Supervisor(llm, max_worker_calls=2),planner,reviewer

    def state(self):
        return State(data_task='', query='Question',research_task='',reason='')

    async def test_accepts_original_answer_and_keeps_input_unchanged(self):
        node,_,_ = self.setup_supervisor([ResearchReview(action='accept',reason='Supported')])
        worker = AsyncMock(return_value=ResearchResult(status='success',answer='Answer [S1]'))
        original = self.state()
        result = await delegate_to_research(original,worker,node)
        self.assertEqual(result.final_answer,'Answer [S1]')
        self.assertEqual(result.next_agent,'end')
        self.assertEqual(result.step_count,2)
        self.assertIsNone(original.research_result)
        worker.assert_awaited_once()

    async def test_partial_stops_without_paid_followup(self):
        node,_,reviewer = self.setup_supervisor([
            ResearchReview(action='research',reason='Missing sample size',research_task='Find sample size.'),
            ResearchReview(action='accept',reason='Now supported')])
        worker = AsyncMock(side_effect=[ResearchResult(status='partial',answer='Missing evidence'),
                                       ResearchResult(status='success',answer='Twenty children [S1]')])
        result = await delegate_to_research(self.state(),worker,node)
        self.assertEqual(worker.await_count,1)
        self.assertNotEqual(result.final_answer,'Twenty children [S1]')
        self.assertEqual(result.step_count,2)

    async def test_repeated_research_stops_at_limit(self):
        again = ResearchReview(action='research',reason='Need more',research_task='Search again')
        node,_,_ = self.setup_supervisor([again,again])
        worker = AsyncMock(return_value=ResearchResult(status='partial',answer='Unchecked draft'))
        result = await delegate_to_research(self.state(),worker,node)
        self.assertEqual(worker.await_count,1)
        self.assertEqual(result.next_agent,'end')
        self.assertIn('Partial results',result.reason)
        self.assertIn('Unchecked draft',result.final_answer)
        self.assertEqual(result.research_result.status, 'partial')

    async def test_partial_cannot_be_accepted(self):
        node,_,_ = self.setup_supervisor([ResearchReview(action='accept',reason='Accept')])
        worker = AsyncMock(return_value=ResearchResult(status='partial',answer='Unchecked'))
        result = await delegate_to_research(self.state(),worker,node)
        self.assertNotEqual(result.final_answer,'Unchecked')

    async def test_failures_stop_safely(self):
        node,planner,_ = self.setup_supervisor([])
        planner.side_effect = RuntimeError('secret')
        worker = AsyncMock()
        result = await delegate_to_research(self.state(),worker,node)
        worker.assert_not_awaited()
        self.assertNotIn('secret',result.model_dump_json())
        node,_,reviewer = self.setup_supervisor([])
        worker.side_effect = RuntimeError('secret')
        result = await delegate_to_research(self.state(),worker,node)
        reviewer.assert_not_awaited()
        self.assertEqual(result.research_result.status,'failed')
        self.assertNotIn('secret',result.model_dump_json())

    async def test_review_failure_and_explicit_stop(self):
        for decision in [RuntimeError('secret'), ResearchReview(action='stop',reason='No usable evidence')]:
            node,_,_ = self.setup_supervisor([decision])
            worker = AsyncMock(return_value=ResearchResult(status='partial'))
            result = await delegate_to_research(self.state(),worker,node)
            self.assertEqual(result.next_agent,'end')
            self.assertNotIn('secret',result.model_dump_json())

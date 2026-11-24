import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import test_workflow
from src.orchestration.state import ResearchResult, DataResult, SummaryResult


class DebugWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_records_all_workers_without_network(self):
        async def workflow(agents, state):
            self.assertIsNone(state.research_result)
            self.assertEqual(state.step_count, 0)
            await agents['supervisor'](state)
            state.research_result = await agents['research_agent'](state)
            state.data_result = await agents['data_agent'](state)
            state.summary_result = await agents['summary_agent'](state)
            state.final_answer = 'Done'
            return state

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(test_workflow, '__file__', str(Path(folder) / 'test_workflow.py')), \
                 patch('builtins.input', side_effect=[' ', 'First question', 'Second question', ' EXIT ']), \
                 patch('builtins.print'), \
                 patch.object(test_workflow, 'Supervisor', return_value=AsyncMock(return_value={})), \
                 patch.object(test_workflow, 'research_agent', new=AsyncMock(return_value=ResearchResult(status='success', answer='Research answer'))), \
                 patch.object(test_workflow, 'data_agent', new=AsyncMock(return_value=DataResult(status='success', explanation='Data answer'))), \
                 patch.object(test_workflow, 'summary_agent', new=AsyncMock(return_value=SummaryResult(status='success', answer='Summary answer'))), \
                 patch.object(test_workflow, 'build_graph', side_effect=lambda **agents: agents) as builder, \
                 patch.object(test_workflow, 'run_workflow', side_effect=workflow):
                await test_workflow.main()
            builder.assert_called_once()
            files = sorted(Path(folder).glob('artifacts/debug/*/workflow.json'))
            self.assertEqual(len(files), 2)
            reports = [json.loads(file.read_text()) for file in files]
            self.assertEqual([r['question'] for r in reports], ['First question', 'Second question'])
            self.assertEqual(len(reports[0]['events']), len(reports[1]['events']))
            self.assertNotIn('First question', files[1].read_text())
            report = reports[0]
            events = [item['event'] for item in report['events']]
            for role in ('supervisor', 'research', 'data', 'summary'):
                self.assertIn(role + '_input', events)
                self.assertIn(role + '_output', events)
            self.assertIn('Research answer', files[0].with_suffix('.md').read_text())

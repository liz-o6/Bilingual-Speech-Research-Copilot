import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
from src.tools.sources import SourceRegistry, verify_citations
from src.rag.answer import annotate_review, review_message
from src.orchestration.state import State, ResearchResult, SummaryResult, DataResult
from src.orchestration.supervisor import Supervisor
from src.api.service import result_status, worker, ChatRun, current_run


class PartialDisplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_ids_do_not_require_echoed_claims(self):
        registry = SourceRegistry()
        registry.register([{'chunk_id':'a', 'text':'Evidence'}])
        class Judge:
            def with_structured_output(self, schema):
                self.schema = schema
                return self
            async def ainvoke(self, prompt):
                pairs = json.loads(prompt.split('\nPairs: ')[1].split('\nSources: ')[0])
                # Only an ID and judgment: no repeated punctuation/text needed.
                return self.schema(checks=[{'check_id':p['check_id'], 'status':'supported', 'reason':'Faithful paraphrase'} for p in reversed(pairs)])
        result = await verify_citations('Finding，one [S1]\nFinding two! [S1]', registry, Judge())
        self.assertEqual([c['status'] for c in result], ['supported','supported'])
        self.assertEqual(result[0]['claim'], 'Finding，one')

    async def test_partial_research_advances_and_summary_is_retained(self):
        supervisor = Supervisor(Mock())
        state = State(query='Count then research then summary', research_task='Find', data_task='Count', reason='',
                      agent_plan=['data','research','summary'], plan_index=1, step_count=2, next_agent='research',
                      data_result=DataResult(status='success',rows=[{'count':8}],explanation='8 papers'),
                      research_result=ResearchResult(status='partial',answer='Finding [S1]',error='Web unavailable'))
        update = await supervisor.review(state)
        self.assertEqual(update['next_agent'], 'summary')
        state = state.model_copy(update=update)
        state.summary_result = SummaryResult(status='partial',answer='【Unverified】Finding [S1]',error='C1 Verification incomplete')
        update = await supervisor.review(state)
        self.assertIn('8 papers', update['final_answer'])
        self.assertIn('Finding [S1]', update['final_answer'])
        self.assertNotIn('Unable to confirm', update['final_answer'])
        state = state.model_copy(update=update)
        self.assertEqual(result_status(state), 'partial')

    async def test_checks_reach_trace(self):
        run = ChatRun('Question')
        token = current_run.set(run)
        try:
            check = {'check_id':'C1','claim':'Finding','status':'unsupported','reason':'Wrong population'}
            agent = AsyncMock(return_value=SummaryResult(status='partial',answer='Finding',citation_checks=[check]))
            await worker(agent,'summary',State(query='Q',research_task='',data_task='',reason=''))
            await run.queue.get()
            event = await run.queue.get()
            self.assertEqual(event['data']['citation_checks'],[check])
        finally:
            current_run.reset(token)

    def test_paragraph_warnings_preserve_supported_findings(self):
        checks = [{'claim':'Good.', 'status':'supported'},
                  {'claim':'Bad.', 'status':'unsupported','reason':'Wrong population','check_id':'C2'},
                  {'claim':'Unknown.', 'status':'unverified','reason':'Judge unavailable','check_id':'C3'}]
        text = annotate_review('Good. [S1]\nBad. [S2]\nUnknown. [S3]',checks)
        self.assertIn('\nGood. [S1]',text)
        self.assertIn('【Insufficient evidence; not a confirmed finding】Bad.',text)
        self.assertIn('【Unverified】Unknown.',text)
        self.assertIn('C2：Insufficient evidence',review_message(checks))
        self.assertIn('C3：Verification incomplete',review_message(checks))

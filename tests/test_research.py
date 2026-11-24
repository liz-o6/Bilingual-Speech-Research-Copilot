import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage
from src.agents.research import research
from src.orchestration.state import State
from src.orchestration.supervisor import Supervisor, ResearchTask, ResearchReview, delegate_to_research


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def state(self):
        return State(data_task='', query='How many children participated?',
                     research_task='Find the participant count in the paper.', reason='Need evidence')

    def retriever(self):
        return SimpleNamespace(asearch=AsyncMock(return_value=[{
            'chunk_id':'paper-1:p2', 'doc_id':'paper-1', 'page':2,
            'text':'Twenty children participated.', 'title':'Paper 1',
        }]))

    def llm(self):
        # Exercise real RAG answer generation with a fake model, not a fake worker.
        model = Mock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content=[{
            'type':'text', 'text':'Twenty children participated. [S1]',
        }]))
        check = SimpleNamespace(check_id='C1', status='supported', reason='Direct evidence',
                                model_dump=lambda: {'status':'supported','reason':'Direct evidence'})
        model.with_structured_output.return_value = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(checks=[check])))
        return model

    async def test_supervisor_connects_to_real_research_function(self):
        planner = Mock()
        planner.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(
                research_task='Find the participant count.', reason='Need evidence'))),
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchReview(
                action='accept', reason='The answer is supported'))),
        ]
        retriever, llm = self.retriever(), self.llm()

        async def worker(state):
            return await research(state,llm=llm,retriever=retriever)

        original = State(data_task='', query='How many children?',research_task='',reason='')
        result = await delegate_to_research(original,worker,Supervisor(planner))
        self.assertEqual(result.research_result.status,'success')
        self.assertIn('[S1]',result.research_result.answer)
        self.assertEqual(result.research_result.sources['S1']['page'],2)
        self.assertEqual(retriever.asearch.call_args.args[0], original.query)
        self.assertEqual(retriever.asearch.call_args.kwargs['k'], 20)
        self.assertEqual(result.final_answer, result.research_result.answer)
        self.assertIsNone(original.research_result)

    async def test_no_evidence_does_not_call_model(self):
        retriever = SimpleNamespace(asearch=AsyncMock(return_value=[]))
        llm = self.llm()
        result = await research(self.state(),llm=llm,retriever=retriever)
        self.assertEqual(result.status,'partial')
        llm.ainvoke.assert_not_awaited()

    async def test_failed_verification_returns_partial(self):
        llm = self.llm()
        llm.with_structured_output.return_value.ainvoke.side_effect = RuntimeError('unavailable')
        result = await research(self.state(),llm=llm,retriever=self.retriever())
        self.assertEqual(result.status,'partial')
        self.assertIn('S1',result.sources)
        self.assertEqual(llm.ainvoke.await_count,1)

    async def test_provider_failure_is_safe_and_preserves_sources(self):
        llm = self.llm()
        llm.ainvoke.side_effect = RuntimeError('secret-key')
        result = await research(self.state(),llm=llm,retriever=self.retriever())
        self.assertEqual(result.status,'failed')
        self.assertIn('S1',result.sources)
        self.assertNotIn('secret-key',result.model_dump_json())
        self.assertEqual(result.error,
                         'Research failed during generating answer or checking citations (RuntimeError).')

    async def test_empty_task_stops_before_retrieval(self):
        state = self.state()
        state.research_task = ' '
        retriever = self.retriever()
        result = await research(state,llm=self.llm(),retriever=retriever)
        self.assertEqual(result.status,'failed')
        retriever.asearch.assert_not_awaited()

    async def test_followup_keeps_source_labels(self):
        from src.orchestration.state import ResearchResult
        state = self.state()
        state.research_result = ResearchResult(status='partial',sources={
            'S1': {'chunk_id':'old-chunk','text':'Old evidence','citation_id':'S1','source_id':'old-chunk'}})
        with patch('src.agents.research.generate_cited_answer',new=AsyncMock(return_value={
            'review_required':False,'answer':'New evidence [S2]'})):
            result = await research(state,llm=self.llm(),retriever=self.retriever())
        self.assertEqual(result.sources['S1']['chunk_id'],'old-chunk')
        self.assertEqual(result.sources['S2']['chunk_id'],'paper-1:p2')
        self.assertEqual(list(state.research_result.sources),['S1'])

    async def test_retrieval_failure_identifies_stage(self):
        retriever = self.retriever()
        retriever.asearch.side_effect = TimeoutError('private request details')
        llm = self.llm()
        result = await research(self.state(), llm=llm, retriever=retriever)
        self.assertEqual(result.error,
                         'Research failed during retrieving local evidence (TimeoutError).')
        self.assertNotIn('private request details', result.model_dump_json())
        llm.ainvoke.assert_not_awaited()

    async def test_http_code_from_wrapped_error_is_safe(self):
        cause = RuntimeError('secret-key and private response')
        cause.code = 503
        error = RuntimeError('private wrapper')
        error.__cause__ = cause
        llm = self.llm()
        llm.ainvoke.side_effect = error
        result = await research(self.state(), llm=llm, retriever=self.retriever())
        self.assertIn('RuntimeError, HTTP 503', result.error)
        self.assertNotIn('secret-key', result.model_dump_json())
        self.assertNotIn('private', result.model_dump_json())

    def test_http_code_locations_and_invalid_values(self):
        from src.agents.research import error_details
        for attribute in ('code', 'status_code'):
            error = RuntimeError('hidden')
            setattr(error, attribute, 429)
            self.assertEqual(error_details(error), 'RuntimeError, HTTP 429')
        error = RuntimeError('hidden')
        error.response = SimpleNamespace(status=502)
        self.assertEqual(error_details(error), 'RuntimeError, HTTP 502')
        error.response = None
        error.code = 'secret-key'
        self.assertEqual(error_details(error), 'RuntimeError')

    @patch("src.llm.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_rate_limit_diagnostics_preserve_only_safe_fields(self, sleep):
        cause = RuntimeError('secret-key')
        cause.status_code = 429
        cause.body = {'error': {'type': 'tokens', 'message':
            'Rate limit for private-model in organization private-org on tokens per minute (TPM): '
            'Limit 8000, Used 7000, Requested 2000. Please try again in 12.5s. secret-key'}}
        cause.response = SimpleNamespace(headers={'retry-after': '13', 'private': 'secret-key'})
        error = RuntimeError('private wrapper')
        error.__cause__ = cause
        llm = self.llm()
        llm.ainvoke.side_effect = error
        result = await research(self.state(), llm=llm, retriever=self.retriever())
        self.assertIn('HTTP 429, limit_type=TPM, limit=8000, used=7000, requested=2000, retry_after=13s', result.error)
        self.assertNotIn('private', result.error)
        self.assertNotIn('secret-key', result.error)

    def test_rate_limit_missing_or_unknown_body_is_not_guessed(self):
        from src.agents.research import error_details
        error = RuntimeError('private raw response')
        error.status_code = 429
        error.body = {'error': {'message': 'Unknown secret-key'}}
        self.assertEqual(error_details(error), 'RuntimeError, HTTP 429')
        error.body = {'type': 'requests', 'message':
                      'Requests per day (RPD): Limit 1000, Used 1000, Requested 1. Please try again in 2m3.5s.'}
        self.assertIn('limit_type=RPD', error_details(error))
        self.assertIn('retry_after=2m3.5s', error_details(error))

    async def test_followup_uses_doc_filter_and_only_current_evidence(self):
        from src.orchestration.state import ResearchResult
        state = self.state()
        state.research_task = 'Find results in bcs-003 and bcs-004'
        state.research_result = ResearchResult(status='partial', sources={
            'S1': {'chunk_id': 'old', 'text': 'Old evidence'}})
        retriever = self.retriever()
        with patch('src.agents.research.generate_cited_answer', new=AsyncMock(return_value={
            'answer': 'New evidence [S2]', 'review_required': False})) as generate:
            result = await research(state, llm=self.llm(), retriever=retriever)
        self.assertEqual([c.kwargs['doc_id'] for c in retriever.asearch.call_args_list], ['bcs-003', 'bcs-004'])
        active = generate.call_args.args[1]
        self.assertEqual(list(active.sources), ['S2'])
        self.assertEqual(list(result.sources), ['S1', 'S2'])

    async def test_default_claude_route_and_native_verification(self):
        llm = self.llm()
        with patch('src.agents.research.create_llm', return_value=llm) as factory:
            result = await research(self.state(), retriever=self.retriever())
        self.assertEqual(result.status, 'success')
        self.assertEqual(factory.call_args.args[0], 'anthropic')
        self.assertEqual(factory.call_args.kwargs['max_tokens'], 2048)
        self.assertEqual(llm.with_structured_output.call_args.kwargs['method'], 'json_schema')

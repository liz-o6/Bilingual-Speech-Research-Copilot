import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src.agents.summary import summary, choose_provider
from src.orchestration.state import State, DocumentRequest, ResearchResult, DataResult
from src.orchestration.supervisor import Supervisor, ResearchTask, ResearchReview, run_workflow
from src.tools.sources import SourceRegistry, check_citations


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    def state(self, documents=None):
        return State(query='Summarize the evidence',research_task='',data_task='',reason='',
                     summary_task='Summarize conclusions and limitations',
                     summary_documents=documents or [DocumentRequest(doc_id='p1',page=2)])

    def reader(self, doc_id, page=None):
        return [{'text':'The study reports an association, not causality.',
                 'metadata':{'doc_id':doc_id,'title':doc_id,'page':page or 1,'source_url':'https://example.org/'+doc_id}}]

    def generated(self):
        return {'answer':'A supported summary [S1]', 'review_required':False,
                'citation_checks':[{'status':'supported'}]}

    def test_routing_thresholds_and_existing_metadata_citations(self):
        self.assertEqual(choose_provider(1,12000)[0],'groq')
        self.assertEqual(choose_provider(1,12001)[0],'anthropic')
        self.assertEqual(choose_provider(2,50)[0],'anthropic')
        reg=SourceRegistry()
        reg.restore({'M_bcs-001':{'text':'Metadata'}})
        self.assertEqual(check_citations('A claim [M_bcs-001]',reg)[0]['status'],'unverified')

    async def test_single_page_groq_and_cross_paper_claude(self):
        for documents, expected in [([DocumentRequest(doc_id='p1',page=2)],'groq'),
                                    ([DocumentRequest(doc_id='p1'),DocumentRequest(doc_id='p2')],'anthropic')]:
            reader=Mock(side_effect=self.reader)
            with patch('src.agents.summary.create_llm',return_value=object()) as factory:
                with patch('src.agents.summary.generate_cited_answer',new=AsyncMock(return_value=self.generated())) as generate:
                    result=await summary(self.state(documents),document_reader=reader)
            self.assertEqual(result.status,'success')
            self.assertEqual(result.provider,expected)
            self.assertEqual(factory.call_args.args[0],expected)
            self.assertEqual(generate.call_args.kwargs['verification_method'], 'json_schema' if expected=='anthropic' else 'function_calling')
            self.assertEqual(reader.call_args_list[0].kwargs['page'],documents[0].page)

    async def test_source_labels_and_data_results_survive(self):
        state=self.state()
        state.research_result=ResearchResult(status='success',sources={
            'S1':{'chunk_id':'chunk1','doc_id':'p1','text':'Original passage','citation_id':'S1','source_id':'chunk1'}})
        state.data_result=DataResult(status='success',rows=[{'count':1,'unit':'papers'}],explanation='Do not use this as evidence',
                                    sources={'M_p1':{'doc_id':'p1','text':'Metadata'}})
        with patch('src.agents.summary.generate_cited_answer',new=AsyncMock(return_value=self.generated())):
            result=await summary(state,llm=object(),document_reader=self.reader)
        self.assertEqual(result.sources['S1'],state.research_result.sources['S1'])
        self.assertIn('M_p1',result.sources)
        self.assertIn('document_page',[s.get('source_type') for s in result.sources.values()])
        self.assertIn('data_result',[s.get('source_type') for s in result.sources.values()])
        self.assertEqual(list(state.research_result.sources),['S1'])

    async def test_shared_mcp_and_failure(self):
        for failure in [False,True]:
            state=self.state(); state.summary_use_web=True
            web=AsyncMock(return_value=[{'chunk_id':'web:1','source_type':'web','source_url':'https://example.org/new','text':'Current evidence'}])
            if failure: web.side_effect=RuntimeError('secret key')
            with patch('src.agents.summary.generate_cited_answer',new=AsyncMock(return_value=self.generated())):
                result=await summary(state,llm=object(),document_reader=self.reader,web_search=web)
            self.assertEqual(web.call_args.kwargs['agent'],'Summary')
            self.assertEqual(result.status,'partial' if failure else 'success')
            self.assertNotIn('secret key',result.model_dump_json())

    async def test_missing_pages_and_context_limit(self):
        result=await summary(self.state(),llm=object(),document_reader=Mock(return_value=[]))
        self.assertEqual(result.status,'partial')
        with patch('src.agents.summary.MAX_CONTEXT_CHARS',10):
            with patch('src.agents.summary.generate_cited_answer',new=AsyncMock()) as generate:
                result=await summary(self.state(),llm=object(),document_reader=self.reader)
            generate.assert_not_awaited()
            self.assertEqual(result.status,'partial')

    async def test_supervisor_summary_handoff(self):
        llm=Mock()
        llm.with_structured_output.side_effect=[
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(next_agent='summary',summary_task='Summarize page 2',
                summary_documents=[DocumentRequest(doc_id='p1',page=2)],reason='Page summary'))),
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchReview(action='accept',reason='Supported')))]
        async def worker(state):
            return await summary(state,llm=object(),document_reader=self.reader)
        research=AsyncMock(); data=AsyncMock()
        with patch('src.agents.summary.generate_cited_answer',new=AsyncMock(return_value=self.generated())):
            result=await run_workflow(self.state(),Supervisor(llm),research,data,worker)
        self.assertEqual(result.final_answer,self.generated()['answer'])
        research.assert_not_awaited(); data.assert_not_awaited()

    async def test_native_claude_citation_schema_is_used(self):
        from langchain_core.messages import AIMessage
        state=self.state([DocumentRequest(doc_id='p1'),DocumentRequest(doc_id='p2')])
        model=Mock()
        model.ainvoke=AsyncMock(return_value=AIMessage(content='A supported summary [S1]'))
        check=SimpleNamespace(check_id='C1', status='supported', reason='Supported',
                              model_dump=lambda: {'status':'supported','reason':'Supported'})
        model.with_structured_output.return_value=SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(checks=[check])))
        result=await summary(state,llm=model,document_reader=self.reader)
        self.assertEqual(result.status,'success')
        self.assertEqual(model.with_structured_output.call_args.kwargs['method'],'json_schema')

    async def test_three_agents_can_run_before_final_summary(self):
        from src.orchestration.state import SummaryResult
        llm=Mock()
        llm.with_structured_output.side_effect=[
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(agent_plan=['research','data','summary'], research_task='Find evidence', data_task='Count papers', summary_task='Combine findings and count',reason='Need papers'))),
            SimpleNamespace(ainvoke=AsyncMock(side_effect=[
                ResearchReview(action='data',data_task='Count papers',reason='Need count'),
                ResearchReview(action='summary',summary_task='Combine findings and count',reason='Need synthesis'),
                ResearchReview(action='accept',reason='Complete')]))]
        research=AsyncMock(return_value=ResearchResult(status='success',answer='Evidence'))
        data=AsyncMock(return_value=DataResult(status='success',explanation='Count',rows=[{'count':2}]))
        async def summary_worker(state):
            self.assertEqual(state.research_result.answer,'Evidence')
            self.assertEqual(state.data_result.rows[0]['count'],2)
            return SummaryResult(status='success',answer='Combined result')
        result=await run_workflow(self.state(),Supervisor(llm),research,data,summary_worker)
        self.assertEqual(result.final_answer,'Combined result')
        self.assertEqual(result.step_count,4)

    async def test_data_then_research_then_summary_preserves_tasks(self):
        import json
        from src.orchestration.state import SummaryResult
        llm = Mock()
        reviewer = AsyncMock(side_effect=[
            ResearchReview(action='research', research_task='Find evidence', reason='Research is pending'),
            ResearchReview(action='summary', summary_task='Combine count and findings', reason='Synthesis is pending'),
            ResearchReview(action='accept', reason='All requested steps completed')])
        llm.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(
                next_agent='data', agent_plan=['data','research','summary'], research_task='Find evidence', summary_task='Combine count and findings', data_task='Count papers since 2020', reason='First step'))),
            SimpleNamespace(ainvoke=reviewer)]
        order = []
        async def data_worker(state):
            order.append('data')
            return DataResult(status='success', explanation='8 papers', rows=[{'count': 8}])
        async def research_worker(state):
            order.append('research')
            self.assertEqual(state.data_result.rows, [{'count': 8}])
            return ResearchResult(status='success', answer='Findings')
        async def summary_worker(state):
            order.append('summary')
            self.assertEqual(state.data_task, 'Count papers since 2020')
            self.assertEqual(state.research_task, 'Find evidence')
            return SummaryResult(status='success', answer='8 papers and findings')
        result = await run_workflow(self.state(), Supervisor(llm), research_worker, data_worker, summary_worker)
        self.assertEqual(order, ['data', 'research', 'summary'])
        self.assertEqual(result.final_answer, '8 papers and findings')
        reviewer.assert_not_awaited()

    async def test_synthesis_does_not_implicitly_read_full_papers(self):
        state = self.state()
        state.summary_documents = []
        state.research_result = ResearchResult(status='success', answer='Finding [S1]', sources={
            'S1': {'chunk_id': 'a', 'doc_id': 'p1', 'text': 'Evidence'}})
        reader = Mock(side_effect=AssertionError('Should use shared sources'))
        with patch('src.agents.summary.generate_cited_answer', new=AsyncMock(return_value=self.generated())):
            result = await summary(state, llm=object(), document_reader=reader)
        reader.assert_not_called()
        self.assertEqual(result.status, 'success')

    async def test_shared_mode_ignores_erroneous_fulltext_list(self):
        state = self.state([DocumentRequest(doc_id=f'bcs-{i:03}') for i in range(1, 10)])
        state.summary_mode = 'shared'
        state.research_result = ResearchResult(status='success', answer='Finding [S1]', sources={
            'S1': {'chunk_id': 'a', 'doc_id': 'p1', 'text': 'Evidence'}})
        reader = Mock(side_effect=AssertionError('Must not read extra full papers'))
        with patch('src.agents.summary.generate_cited_answer', new=AsyncMock(return_value=self.generated())):
            result = await summary(state, llm=object(), document_reader=reader)
        reader.assert_not_called()
        self.assertEqual(result.status, 'success')
        self.assertEqual(len(result.sources), 1)

    async def test_simple_research_overrides_inflated_plan(self):
        llm = Mock()
        llm.with_structured_output.side_effect = [
            SimpleNamespace(ainvoke=AsyncMock(return_value=ResearchTask(
                agent_plan=['data', 'research', 'summary'], next_agent='data',
                data_task='Unneeded list', research_task='Find evidence', summary_task='Unneeded synthesis',
                summary_documents=[DocumentRequest(doc_id='bcs-001')], reason='Overplanning'))),
            SimpleNamespace(ainvoke=AsyncMock())]
        state = self.state()
        state.query = '请从本地论文正文检索双语儿童语音发展的相关证据，概括两个发现，并给出引用，不联网。'
        worker = AsyncMock(return_value=ResearchResult(status='success', answer='Two findings'))
        data_worker = AsyncMock()
        summary_worker = AsyncMock()
        result = await run_workflow(state, Supervisor(llm), worker, data_worker, summary_worker)
        self.assertEqual(result.agent_plan, ['research'])
        self.assertEqual(result.final_answer, 'Two findings')
        data_worker.assert_not_awaited()
        summary_worker.assert_not_awaited()

    async def test_nonretryable_summary_stops_without_reexecution(self):
        from src.orchestration.state import SummaryResult
        state = self.state()
        state.agent_plan = ['summary']
        state.next_agent = 'summary'
        state.step_count = 1
        state.summary_result = SummaryResult(status='partial', retryable=False, error='Context is too large.')
        updates = await Supervisor(Mock()).review(state)
        self.assertEqual(updates['next_agent'], 'end')
        self.assertNotIn('repair_count', updates)

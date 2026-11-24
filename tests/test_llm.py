"""Offline checks for native output preservation and metadata-only callbacks."""
import asyncio
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.llm import create_llm, LLMCallLogger


class LLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_output_preserved_through_binding(self):
        traces = []
        message = AIMessage(content=[{'type':'text','text':'Hello'},
                                     {'type':'thinking','thinking':'private'}],
                            tool_calls=[{'name':'lookup','args':{'id':1},'id':'call-1'}],
                            usage_metadata={'input_tokens':2,'output_tokens':1,'total_tokens':3})
        model = GenericFakeChatModel(messages=iter([message]), callbacks=[
            LLMCallLogger(traces,'google','test','Research','test reason')])
        result = await model.bind(stop=['END']).ainvoke('secret prompt')
        self.assertEqual(result.content, message.content)
        self.assertEqual(result.tool_calls, message.tool_calls)
        self.assertEqual(traces[0]['usage']['total_tokens'],3)
        self.assertEqual(traces[0]['status'],'success')
        self.assertNotIn('secret',repr(traces))
        self.assertNotIn('private',repr(traces))

    async def test_stream_and_parser_keep_logging(self):
        traces = []
        model = GenericFakeChatModel(messages=iter(['hello world']), callbacks=[
            LLMCallLogger(traces,'openai','test','Summary','test reason')])
        chain = model | StrOutputParser()
        text = ''.join([part async for part in chain.astream('test')])
        self.assertEqual(text,'hello world')
        self.assertEqual(len(traces),1)
        self.assertEqual(traces[0]['status'],'success')
        self.assertIsNotNone(traces[0]['ttft_seconds'])

    async def test_native_error_propagates_without_logging_body(self):
        traces = []
        class Failing(GenericFakeChatModel):
            def _generate(self,*args,**kwargs): raise ValueError('secret provider response')
        model = Failing(messages=iter([]),callbacks=[
            LLMCallLogger(traces,'anthropic','test','Data','test reason')])
        with self.assertRaisesRegex(ValueError,'secret provider response'):
            await model.ainvoke('test')
        self.assertEqual(traces[0]['error_type'],'ValueError')
        self.assertNotIn('secret',repr(traces))

    async def test_overlapping_calls_and_cancellation(self):
        traces = []
        logger = LLMCallLogger(traces,'openai','test','Data','test reason')
        first, second = uuid4(), uuid4()
        for run_id in [first,second]: logger.on_chat_model_start({},[],run_id=run_id)
        logger.on_llm_error(asyncio.CancelledError(),run_id=second)
        logger.on_llm_error(TimeoutError('secret'),run_id=first)
        self.assertEqual([t['call_id'] for t in traces],[str(second),str(first)])
        self.assertEqual([t['status'] for t in traces],['cancelled','error'])
        self.assertFalse(logger.active)

    async def test_factory_returns_native_models_without_network(self):
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI
        env = {'OPENAI_API_KEY':'test','OPENAI_MODEL':'test',
               'ANTHROPIC_API_KEY':'test','ANTHROPIC_MODEL':'test',
               'GOOGLE_API_KEY':'test','GEMINI_MODEL':'test'}
        with patch('src.llm.client.load_dotenv'), patch.dict('os.environ',env):
            for provider,cls in [('openai',ChatOpenAI),('anthropic',ChatAnthropic),('google',ChatGoogleGenerativeAI)]:
                traces = []
                model = create_llm(provider,traces=traces)
                self.assertIsInstance(model,cls)
                if provider == "anthropic":
                    self.assertEqual(model._get_request_payload([("human", "Hello")])["thinking"], {"type": "disabled"})
                self.assertIs(model.callbacks[0].traces,traces)
            with patch.dict('os.environ',{'GOOGLE_API_KEY':''}):
                with self.assertRaisesRegex(ValueError,'GOOGLE_API_KEY'):
                    create_llm('google')

    async def test_groq_request_disables_thinking_and_uses_tool_schema(self):
        import json
        import httpx
        from pydantic import BaseModel

        class Result(BaseModel):
            answer: str

        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'qwen/qwen3.8-27b',
                'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': 'call1', 'type': 'function', 'function': {
                            'name': 'Result', 'arguments': '{"answer":"OK"}'}}]}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            })

        env = {'GROQ_API_KEY': 'test-groq', 'GROQ_MODEL': 'qwen/qwen3.8-27b'}
        with patch('src.llm.client.load_dotenv'), patch.dict('os.environ', env):
            model = create_llm('groq', traces=[])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model.root_async_client = model.root_async_client.with_options(http_client=client)
            model.async_client = model.root_async_client.chat.completions
            output = await model.with_structured_output(Result, method='function_calling').ainvoke('Return OK')
        self.assertEqual(output.answer, 'OK')
        self.assertEqual(requests[0].url.host, 'api.groq.com')
        body = json.loads(requests[0].content)
        self.assertEqual(body['model'], 'qwen/qwen3.8-27b')
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertEqual(body['tools'][0]['function']['name'], 'Result')
        self.assertEqual(model.callbacks[0].context['provider'], 'groq')

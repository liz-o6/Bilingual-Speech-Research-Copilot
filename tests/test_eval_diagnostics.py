import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.exceptions import OutputParserException
from src.eval.diagnostics import JudgeDiagnostics, error_details


class DiagnosticTests(unittest.TestCase):
    def test_parser_message_and_raw_output_are_retained_with_redaction(self):
        with patch.dict('os.environ', {'TEST_API_KEY':'private-key-12345'}):
            error=OutputParserException('missing verdict private-key-12345',llm_output='{"unexpected":"private-key-12345"}')
            result=error_details(error)
            self.assertIn('missing verdict',result['message'])
            self.assertIn('unexpected',result['parser_output'])
            self.assertNotIn('private-key-12345',str(result))

    def test_outputs_include_call_id_and_finish_reason(self):
        records=[]
        callback=JudgeDiagnostics(records)
        callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content='{"statements":[]}',response_metadata={'finish_reason':'stop'}))]]),run_id='test-id')
        self.assertEqual(records[0]['call_id'],'test-id')
        self.assertEqual(records[0]['finish_reason'],'stop')
        self.assertEqual(records[0]['output'],'{"statements":[]}')

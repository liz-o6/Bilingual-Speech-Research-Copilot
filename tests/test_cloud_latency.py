import json
import unittest
import httpx
from src.eval.cloud_latency import measure


def frame(event, seq, **data):
    return f'event: {event}\r\ndata: ' + json.dumps(dict(request_id='r1', seq=seq, **data)) + '\r\n\r\n'


class CloudLatencyTests(unittest.TestCase):
    def run_stream(self, body):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = measure(client, 'https://example.test/api/chat', 'Q')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].method, 'POST')
        return result

    def test_complete_stream_and_metrics(self):
        r = self.run_stream(': heartbeat\r\n\r\n' + frame('trace', 1, type='request_start') +
                            frame('token', 2, text='Hi') + frame('answer', 3, text='Final') +
                            frame('done', 4, status='partial', metrics={'total_ms': 42}))
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(r['answer'], 'Final')
        self.assertEqual(r['metrics']['total_ms'], 42)
        self.assertIsNotNone(r['client_ttft_ms'])
        self.assertEqual(r['classification'], 'unverified')

    def test_no_token_is_unknown(self):
        r = self.run_stream(frame('done', 1, status='completed', metrics={}))
        self.assertIsNone(r['client_ttft_ms'])

    def test_truncated_done_does_not_count(self):
        r = self.run_stream(frame('done', 1, status='completed').rstrip())
        self.assertEqual(r['error_type'], 'MissingDoneEvent')
        self.assertEqual(r['status'], 'incomplete')

    def test_duplicate_sequence_is_rejected(self):
        r = self.run_stream(frame('token', 1, text='x') + frame('done', 1, status='completed'))
        self.assertEqual(r['status'], 'failed')

"""Measure the deployed SSE endpoint; no health checks or automatic retries."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from .common import ROOT, save

DEFAULT_URL = 'https://hw2-research-copilot-2pfqauay5a-uw.a.run.app'
QUESTION = ("Please retrieve evidence from local papers on bilingual children's speech development, "
            "summarize two findings, and cite sources. Do not search the web.")


def sse_events(response):
    event, data = 'message', []
    for line in response.iter_lines():
        if line == '':
            if data:
                yield {'event': event, 'data': json.loads('\n'.join(data))}
            event, data = 'message', []
        elif line.startswith(':'):
            continue
        else:
            field, separator, value = line.partition(':')
            if value.startswith(' '):
                value = value[1:]
            if field == 'event':
                event = value
            elif field == 'data':
                data.append(value)
    # An unfinished final frame is not a complete event.


def measure(client, url, question):
    record = dict(started_at=datetime.now(timezone.utc).isoformat(), question=question,
                  endpoint=url, classification='unverified', classification_evidence=None,
                  request_id=None, status='incomplete', http_status=None, answer='',
                  client_ttft_ms=None, client_total_ms=None, metrics={}, events=[], error_type=None)
    started = perf_counter()
    last_seq = 0
    try:
        with client.stream('POST', url, json={'query': question},
                           headers={'Accept': 'text/event-stream'}) as response:
            record['http_status'] = response.status_code
            response.raise_for_status()
            if 'text/event-stream' not in response.headers.get('content-type', ''):
                raise ValueError('Expected SSE')
            for event in sse_events(response):
                elapsed = (perf_counter() - started) * 1000
                data = event['data']
                request_id, seq = data.get('request_id'), data.get('seq')
                if not request_id or not isinstance(seq, int) or seq <= last_seq:
                    raise ValueError('Invalid event identity or sequence')
                if record['request_id'] and record['request_id'] != request_id:
                    raise ValueError('Request ID changed')
                record['request_id'], last_seq = request_id, seq
                record['events'].append({**event, 'received_ms': elapsed})
                if event['event'] == 'token' and data.get('text') and record['client_ttft_ms'] is None:
                    record['client_ttft_ms'] = elapsed
                if event['event'] == 'answer':
                    record['answer'] = data.get('text', '')
                if event['event'] == 'done':
                    record['status'] = data.get('status', 'unknown')
                    record['metrics'] = data.get('metrics', {})
                    record['client_total_ms'] = elapsed
                    break
            else:
                record['error_type'] = 'MissingDoneEvent'
    except Exception as exc:
        record['error_type'] = type(exc).__name__
        record['status'] = 'failed'
    if record['client_total_ms'] is None:
        record['client_total_ms'] = (perf_counter() - started) * 1000
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    return record


def write_report(folder, records):
    lines = ['# Cloud SSE latency measurements', '',
             'Phase labels describe the intended experiment, not verified instance state. '
             'Classify cold/warm only after checking Cloud Run instance-start logs and UTC request times. '
             'No homepage/health request or retry was issued. Every request uses a new HTTP client '
             'so connection reuse does not favor later requests. Timings include client connection/network overhead. '
             'TTFT measures first visible candidate text; missing tokens are N/A. '
             'Total ends at done; failed/incomplete records instead measure time to error/EOF. '
             'Pre-request client construction is excluded. Server workflow time excludes container startup. '
             'Client minus server time is not a pure cold-start measurement.', '',
             '| Attempt | Intended phase | Status | Client total (s) | Client TTFT (s) | Retrieval (s) | LLM (s) | Server total (s) | Server TTFT (s) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def seconds(value):
        return 'N/A' if value is None else f'{value/1000:.3f}'
    for r in records:
        m = r['metrics']
        values = [r['client_total_ms'], r['client_ttft_ms'], m.get('retrieval_ms'),
                  m.get('llm_ms'), m.get('total_ms'), m.get('server_ttft_ms')]
        lines.append(f"| {r['attempt']} | {r['intended_phase']} | {r['status']} | " +
                     ' | '.join(seconds(v) for v in values) + ' |')
    lines += ['', 'Each JSON retains UTC timestamps, request ID, final answer, all SSE events, '
              'provider/model/usage traces, and server metrics (including query embedding tokens). '
              'Keep failures; do not silently replace them with successful retries. '
              'This script does not estimate costs or authenticate to Google Cloud. '
              'Attach instance/revision startup evidence before reporting cold/warm averages.']
    (folder / 'latency.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=DEFAULT_URL, help='Service base URL, not /api/chat')
    parser.add_argument('--query', default=QUESTION)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--first-phase', choices=['cold-candidate', 'warm-candidate'], default='cold-candidate')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        parser.error('--url must be a service base URL without credentials, path, or query')
    if not 1 <= args.count <= 10 or not 1 <= len(args.query.strip()) <= 4000:
        parser.error('Use 1–10 requests and a query of 1–4000 characters')
    url = args.url.rstrip('/') + '/api/chat'
    print(f'{args.count} sequential POST requests to {url}; no retries or preflight requests.')
    if not args.run:
        print('Preview only. Wait for scale-to-zero if measuring cold start, then add --run. Calls may incur fees.')
        return
    folder = args.output or ROOT / 'artifacts/evaluation/cloud' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'config.json', dict(url=url, query=args.query, count=args.count,
                                    first_phase=args.first_phase, connection_policy='new client per request'))
    records = []
    for i in range(1, args.count + 1):
        print(f'Request {i}/{args.count}...', flush=True)
        with httpx.Client(timeout=httpx.Timeout(330, connect=60), follow_redirects=False) as client:
            result = measure(client, url, args.query.strip())
        result.update(attempt=i, intended_phase=args.first_phase if i == 1 else 'warm-candidate')
        save(folder / f'request-{i:02}.json', result)
        records.append(result)
        write_report(folder, records)
        print(f"  {result['status']}: {result['client_total_ms']/1000:.2f}s; ID={result['request_id']}", flush=True)
        if result['error_type']:
            print(f"  {result['error_type']}; stopping to avoid repeated failed paid calls.")
            break
    print(folder / 'latency.md')


if __name__ == '__main__':
    main()

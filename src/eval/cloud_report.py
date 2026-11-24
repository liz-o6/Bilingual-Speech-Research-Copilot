"""Summarize saved cloud timings and manually reviewed startup evidence."""
from statistics import mean
from .common import read


def cloud_section(base):
    batch = 'cloud/20261027T215158613801Z'
    rows = [read(base / batch / f'request-{i:02}.json') for i in range(1, 5)]
    fields = ['retrieval_ms', 'llm_ms', 'total_ms', 'server_ttft_ms']

    def values(row):
        return [row['client_total_ms'], row['client_ttft_ms']] + [row['metrics'][k] for k in fields]

    lines = ['## Cloud Run cold and warm measurements', '',
             'Deployed service: https://hw2-research-copilot-2pfqauay5a-uw.a.run.app. The project owner reported that all three default UI questions returned answers. The measurements below are four sequential calls to `/api/chat` using the same local-RAG question, with no health check, homepage request, or retry before them. Each call used a new HTTP client.', '',
             f'Raw records: `artifacts/evaluation/{batch}/request-01.json` through `request-04.json`. Original records retain their initial unverified classification; this report adds classification based on the owner-provided Cloud Run log screenshots supplied in the conversation. The temporary screenshot files were no longer readable when this report was generated; retain the originals with the submission.', '',
             'The first request appears at 2026-10-27 21:51:58.901 UTC (14:51:58.901 PDT). The logs show “Starting new instance” due to AUTOSCALING at 21:51:58.932, application startup complete at 21:52:08.298, and the startup probe succeeding at 21:52:08.311. This supports classifying request 1 as cold. Requests 2–4 follow immediately with no additional instance-start event shown and are classified as warm based on that timeline. Instance IDs are not visible in these screenshots.', '',
             '| Request | Phase | Status | Client total (s) | Client TTFT (s) | Retrieval (s) | LLM (s) | Server total (s) | Server TTFT (s) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for i, row in enumerate(rows, 1):
        lines.append(f"| {i} | {'Cold' if i == 1 else 'Warm'} | {row['status']} | " + ' | '.join(f'{v/1000:.3f}' for v in values(row)) + ' |')
    averages = [mean(values(row)[i] for row in rows[1:]) for i in range(6)]
    lines += ['| **Warm mean (n=3)** | Warm | 2 completed, 1 partial | ' + ' | '.join(f'{v/1000:.3f}' for v in averages) + ' |', '',
              'All four requests returned HTTP 200 and a terminal SSE event. Request 2 retained a partial answer because citation review flagged a claim; it is included in the timing mean rather than discarded. Transport success does not establish answer correctness. The cloud runs were not separately scored with RAGAS.', '',
              'Cloud Run request-log durations were 38.326, 7.667, 8.633, and 7.035 seconds. These differ from client timing because measurement boundaries differ. The observed instance-start-to-application-ready interval is about 9.366 seconds; neither the entire 38.819-second request nor the client/server difference should be described as pure container startup. Initialization and orchestration overhead are also included in total time.', '',
              'Client TTFT ends at the first visible candidate text, before verification; client total ends at the final SSE done event. Server workflow time excludes container startup. Retrieval includes query embedding and local retrieval work; LLM time sums recorded provider calls. One cold observation and three warm observations demonstrate the distinction but are not a latency distribution or SLA.', '',
              'Log evidence: owner-provided screenshots dated 2026-10-27 at 3:02:40 PM and 3:02:59 PM (startup timeline and expanded request log). First platform request trace: `12df821f9c3d02086c987a1fb9d0b0a`. These are screenshot observations, not an exported instance-ID correlation.', '']
    return lines

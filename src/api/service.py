"""Reuse the graph with separate state, traces, and SSE queues per request."""
import asyncio
from contextlib import suppress
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4
from urllib.parse import urlsplit

from src.agents.data import data
from src.agents.research import research
from src.agents.summary import summary
from src.orchestration.graph import build_graph, run_workflow
from src.orchestration.state import State
from src.orchestration.supervisor import Supervisor

# Keep logs request-local because async requests may overlap.
current_run = ContextVar("api_current_run")


def public_trace(item):
    """Expose public fields only; hide reasoning, full prompts, and raw errors."""
    kind = item.get("event")
    if kind == "operation_timing":
        return {"type": kind, **{k: item[k] for k in ("operation", "agent", "status", "duration_ms")}}
    if kind == "llm_call":
        keys = ("agent", "provider", "model", "reason", "status", "usage", "duration_seconds", "ttft_seconds")
        return {"type": "llm_route", **{k: item.get(k) for k in keys}}
    if kind in ("mcp_tool_call", "data_query", "llm_route"):
        keys = ("agent", "provider", "reason", "status", "server", "tool", "operation", "matched_count", "result_count", "duration_seconds")
        return {"type": kind, **{k: item[k] for k in keys if k in item}}
    if kind == "data_plan":
        return {"type": "data_plan", "agent": "Data", "task": item.get("task"),
                "operation": item.get("operation"), "filters": item.get("filters")}
    if kind == "research_evidence":
        return {"type": "rag_retrieval", "agent": "Research",
                "source_ids": item.get("source_ids", []), "candidate_count": item.get("candidate_count")}
    return None


class TraceLog(list):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def append(self, item):
        super().append(item)
        if item.get("event") == "research_evidence":
            self.run.evidence = list(item["sources"].values())
        visible = public_trace(item)
        if visible is not None:
            self.run.emit("trace", visible)

    def emit_token(self, text):
        if text:
            if self.run.first_token_ms is None:
                self.run.first_token_ms = (perf_counter() - self.run.started) * 1000
            self.run.emit("token", {"text": text, "provisional": True})


class ChatRun:
    def __init__(self, query):
        self.request_id = str(uuid4())
        self.query = query
        self.queue = asyncio.Queue()
        self.seq = 0
        self.started = perf_counter()
        self.first_token_ms = None
        self.traces = TraceLog(self)
        self.supervisor = None
        self.evidence = []

    def emit(self, event, data):
        self.seq += 1
        self.queue.put_nowait({"event": event, "data": {
            **data, "request_id": self.request_id, "seq": self.seq}})

    def metrics(self):
        calls = [t for t in self.traces if t.get("event") == "llm_call"]
        usage = [t["usage"] for t in calls if t.get("usage") is not None]
        return {"total_ms": round((perf_counter() - self.started) * 1000, 2),
                "server_ttft_ms": self.first_token_ms,
                "llm_calls": len(calls), "usage_complete": len(usage) == len(calls),
                "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
                "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
                "llm_ms": round(sum(t.get("duration_seconds", 0) for t in calls) * 1000, 2),
                "retrieval_ms": self.operation_ms("rag_retrieval"),
                "embedding_calls": len([t for t in self.traces if t.get('event') == 'embedding_call']),
                "embedding_input_tokens": sum((t.get('usage') or {}).get('input_tokens', 0)
                                              for t in self.traces if t.get('event') == 'embedding_call'),
                "embedding_usage_complete": all(t.get('usage') is not None for t in self.traces
                                                if t.get('event') == 'embedding_call')}

    def operation_ms(self, operation):
        records = [t for t in self.traces if t.get("event") == "operation_timing"
                   and t.get("operation") == operation]
        return sum(t['duration_ms'] for t in records) if records else None


async def supervisor_node(state):
    run = current_run.get()
    if run.supervisor is None:
        run.supervisor = Supervisor(traces=run.traces)
    run.emit("trace", {"type": "agent_start", "agent": "supervisor"})
    updates = await run.supervisor(state)
    run.emit("trace", {"type": "agent_route", "agent": "supervisor",
                       "next_agent": updates.get("next_agent"), "reason": updates.get("reason"),
                       "plan": updates.get("agent_plan", state.agent_plan)})
    return updates


async def worker(agent, name, state):
    run = current_run.get()
    run.emit("trace", {"type": "agent_start", "agent": name,
                       "task": getattr(state, name + "_task", "")})
    started = perf_counter()
    try:
        result = await agent(state, traces=run.traces)
        run.emit("trace", {"type": "agent_result", "agent": name, "status": result.status, "error": result.error,
                           "duration_ms": round((perf_counter() - started) * 1000, 2),
                           "citation_checks": getattr(result, "citation_checks", [])})
        return result
    except Exception:
        run.emit("trace", {"type": "agent_result", "agent": name, "status": "failed"})
        raise


async def research_node(state):
    return await worker(research, "research", state)


async def data_node(state):
    return await worker(data, "data", state)


async def summary_node(state):
    return await worker(summary, "summary", state)


def result_status(state):
    results = [state.data_result, state.research_result, state.summary_result]
    results = [r for r in results if r is not None]
    if not results or any(r.status == "failed" for r in results):
        return "failed"
    if any(r.status == "partial" for r in results):
        return "partial"
    if state.plan_index < len(state.agent_plan) - 1:
        return "partial"
    return "completed"


def citations(state):
    # Use the final Summary registry for shared source labels.
    sources = {}
    for result in (state.data_result, state.research_result, state.summary_result):
        if result is not None:
            sources.update(result.sources)
    cards = []
    for label, source in sources.items():
        url = source.get("source_url", "")
        if urlsplit(url).scheme not in ("http", "https"):
            url = ""
        cards.append({"citation_id": label, "source_id": source.get("source_id") or source.get("chunk_id") or label,
                      "doc_id": source.get("doc_id"), "title": source.get("title", ""),
                      "page": source.get("page"), "section": source.get("section_id"),
                      "url": url, "excerpt": source.get("text", "")})
    return cards


class ChatService:
    def __init__(self, graph=None, timeout_seconds=180, capture_evaluation=False):
        self.capture_evaluation = capture_evaluation
        self.graph = graph if graph is not None else build_graph(
            supervisor_node, research_node, data_node, summary_node)
        self.timeout_seconds = timeout_seconds

    async def produce(self, run):
        token = current_run.set(run)
        status = "failed"
        try:
            async with asyncio.timeout(self.timeout_seconds):
                state = State(query=run.query, research_task="", data_task="", reason="")
                result = await run_workflow(self.graph, state)
                status = result_status(result)
                if status != "completed":
                    run.emit("error", {"code": "WORKFLOW_INCOMPLETE", "message": result.reason,
                                       "recoverable": False})
                for card in citations(result):
                    run.emit("citation", card)
                run.emit("answer", {"text": result.final_answer or "No confirmed answer was generated.",
                                    "status": status, "replace": True})
        except TimeoutError:
            run.emit("error", {"code": "WORKFLOW_TIMEOUT", "message": "The request timed out; execution has stopped.", "recoverable": False})
        except Exception:
            run.emit("error", {"code": "WORKFLOW_FAILED", "message": "The request failed. Check the local configuration or server logs.", "recoverable": False})
        finally:
            if self.capture_evaluation:
                # Local evaluation runner only; normal browser API does not send this payload.
                run.emit("evaluation", {"contexts": run.evidence,
                                       "calls": [t for t in run.traces if t.get("event") == "llm_call"],
                                       "embedding_calls": [t for t in run.traces if t.get("event") == "embedding_call"]})
            current_run.reset(token)
            run.emit("done", {"status": status, "metrics": run.metrics()})

    async def events(self, query):
        run = ChatRun(query)
        run.emit("trace", {"type": "request_start", "status": "running"})
        task = asyncio.create_task(self.produce(run))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(run.queue.get(), timeout=15)
                except TimeoutError:
                    yield {"event": "heartbeat", "data": {}}
                    continue
                yield event
                if event["event"] == "done":
                    break
        finally:
            # Cancel upstream work when the SSE stream closes.
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

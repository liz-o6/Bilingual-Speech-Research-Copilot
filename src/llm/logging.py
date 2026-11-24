"""Metadata-only callbacks; never store prompts, responses or error messages."""
import asyncio
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler


class LLMCallLogger(BaseCallbackHandler):
    # Keep state changes inline, including for async model calls.
    run_inline = True

    def __init__(self, traces, provider, model, agent, reason):
        self.traces = traces
        self.context = dict(provider=provider, model=model, agent=agent, reason=reason)
        self.active = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.active[run_id] = (perf_counter(), dict(
            event='llm_call', call_id=str(run_id), **self.context,
            status='running', usage=None, ttft_seconds=None))

    def on_llm_new_token(self, token, *, run_id, chunk=None, **kwargs):
        if run_id not in self.active:
            return
        start, event = self.active[run_id]
        # Observe visible text only, without modifying the original message.
        text = chunk.message.text if chunk is not None and hasattr(chunk, 'message') else token
        if isinstance(text, str) and text and event['ttft_seconds'] is None:
            event['ttft_seconds'] = perf_counter() - start

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = None
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], 'message', None)
            usage = getattr(message, 'usage_metadata', None)
        self._finish(run_id, status='success', usage=usage)

    def on_llm_error(self, error, *, run_id, **kwargs):
        status = 'cancelled' if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else 'error'
        self._finish(run_id, status=status, error_type=type(error).__name__)

    def _finish(self, run_id, **fields):
        entry = self.active.pop(run_id, None)
        if entry is not None:
            start, event = entry
            event.update(fields, duration_seconds=perf_counter() - start)
            self.traces.append(event)

"""Record provider-reported usage at each embedding HTTP request, not estimates."""
from time import perf_counter


def record(traces, model, purpose, started, response=None, error=None):
    if traces is None:
        return
    payload = response.model_dump() if hasattr(response, 'model_dump') else response
    usage = payload.get('usage') if isinstance(payload, dict) else None
    tokens = usage.get('prompt_tokens', usage.get('total_tokens')) if usage else None
    traces.append({'event': 'embedding_call', 'provider': 'openai', 'model': model,
                   'purpose': purpose, 'status': 'error' if error else 'success',
                   'duration_ms': (perf_counter() - started) * 1000,
                   'usage': {'input_tokens': tokens, 'output_tokens': 0} if tokens is not None else None,
                   'usage_source': 'api_response' if tokens is not None else 'unavailable',
                   'error_type': type(error).__name__ if error else None})


class RecordedClient:
    def __init__(self, client, traces, model, purpose):
        self.client, self.traces, self.model, self.purpose = client, traces, model, purpose

    def create(self, **kwargs):
        started = perf_counter()
        try:
            response = self.client.create(**kwargs)
        except BaseException as exc:
            record(self.traces, self.model, self.purpose, started, error=exc)
            raise
        record(self.traces, self.model, self.purpose, started, response)
        return response


class RecordedAsyncClient(RecordedClient):
    async def create(self, **kwargs):
        started = perf_counter()
        try:
            response = await self.client.create(**kwargs)
        except BaseException as exc:
            record(self.traces, self.model, self.purpose, started, error=exc)
            raise
        record(self.traces, self.model, self.purpose, started, response)
        return response


def track_embeddings(embedder, traces, purpose):
    # Preserve LangChain's batching, chunking, tokenization and vector processing.
    embedder.client = RecordedClient(embedder.client, traces, embedder.model, purpose)
    embedder.async_client = RecordedAsyncClient(embedder.async_client, traces, embedder.model, purpose)
    return embedder

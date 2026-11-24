"""Structured failures and bounded retries, adapted from the old ToolError pattern."""
import time

class RagError(RuntimeError):
    def __init__(self, code, message, *, retryable=False):
        super().__init__(message)
        self.code, self.retryable = code, retryable

    def to_dict(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


def retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or status in (408, 429, 500, 502, 503, 504) or type(exc).__name__ in ("APITimeoutError", "APIConnectionError")
            if not transient or attempt == attempts - 1:
                raise RagError("RAG_OPERATION_FAILED", f"Operation failed ({type(exc).__name__})", retryable=transient) from exc
            time.sleep(0.5 * 2 ** attempt)

"""Local evaluation diagnostics; not attached to the production API logger."""
import os
import re
from langchain_core.callbacks import BaseCallbackHandler


def redact(text):
    text = str(text)
    for name, value in os.environ.items():
        if value and len(value) >= 8 and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            text = text.replace(value, '[REDACTED]')
    return re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', text)


def error_details(error):
    details = {'type': type(error).__name__, 'message': redact(error)}
    # LangChain parser exceptions may carry the precise unparseable output.
    if getattr(error, 'llm_output', None) is not None:
        details['parser_output'] = redact(error.llm_output)
    cause = error.__cause__
    if cause is not None:
        details['cause'] = {'type': type(cause).__name__, 'message': redact(cause)}
    return details


class JudgeDiagnostics(BaseCallbackHandler):
    run_inline = True

    def __init__(self, records):
        self.records = records

    def on_llm_end(self, response, *, run_id, **kwargs):
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, 'message', None)
                metadata = getattr(message, 'response_metadata', {}) or {}
                info = generation.generation_info or {}
                self.records.append({'call_id': str(run_id),
                    'output': redact(generation.text),
                    'finish_reason': metadata.get('finish_reason') or metadata.get('stop_reason') or info.get('finish_reason')})

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.records.append({'call_id': str(run_id), 'error': error_details(error)})

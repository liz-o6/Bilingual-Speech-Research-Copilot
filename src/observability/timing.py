"""Measure elapsed time without model calls."""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(traces, operation, agent):
    started = perf_counter()
    status = 'success'
    try:
        yield
    except BaseException:
        status = 'failed'
        raise
    finally:
        if traces is not None:
            traces.append({'event': 'operation_timing', 'operation': operation,
                           'agent': agent, 'status': status,
                           'duration_ms': (perf_counter() - started) * 1000})

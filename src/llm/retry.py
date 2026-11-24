"""Wait after HTTP 429 without rerunning the whole agent."""
import asyncio
import re
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone


def is_rate_limit(exc):
    current = exc
    for _ in range(3):
        response = getattr(current, "response", None)
        codes = [getattr(current, "status_code", None),
                 getattr(current, "code", None),
                 getattr(response, "status_code", None),
                 getattr(response, "status", None)]
        if 429 in codes:
            return True
        current = current.__cause__ or current.__context__
        if current is None:
            break
    return False


def retry_delay(exc):
    """Use the server delay or default to 13 seconds."""
    current = exc
    delays = []
    for _ in range(3):
        response = getattr(current, 'response', None)
        headers = getattr(response, 'headers', {}) or {}
        value = headers.get('retry-after', '')
        if isinstance(value, str) and re.fullmatch(r'\d+(?:\.\d+)?', value):
            delays.append(float(value))
        elif value:
            try:
                date = parsedate_to_datetime(value)
                delays.append(max(0, (date - datetime.now(timezone.utc)).total_seconds()))
            except (ValueError, TypeError, OverflowError):
                pass
        body = getattr(current, 'body', None)
        if isinstance(body, dict):
            body = body.get('error', body)
            if isinstance(body, dict):
                message = body.get('message', '')
                if isinstance(message, str):
                    match = re.search(r'try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)(?!\w)', message, re.I)
                    if match:
                        seconds = 0
                        units = {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}
                        for number, unit in re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)', match.group(1).lower()):
                            seconds += float(number) * units[unit]
                        delays.append(seconds)
        current = current.__cause__ or current.__context__
        if current is None:
            break
    return max(delays) if delays else 13


async def invoke_with_retry(model, messages):
    """Retry HTTP 429 up to twice, with a default 13-second delay."""
    for attempt in range(3):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == 2:
                raise
            delay = retry_delay(exc)
            # Return the error instead of waiting for long quota resets.
            if delay > 60:
                raise
            await asyncio.sleep(delay)

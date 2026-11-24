import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, call
from src.llm.retry import invoke_with_retry


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_429_waits_13_seconds_then_succeeds(self):
        error = RuntimeError('rate limited')
        error.status_code = 429
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[error, error, 'OK']))
        with patch('src.llm.retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
            result = await invoke_with_retry(model, 'question')
        self.assertEqual(result, 'OK')
        self.assertEqual(sleep.await_args_list, [call(13), call(13)])
        self.assertEqual(model.ainvoke.await_count, 3)

    async def test_exhausted_retries_raise_original_error(self):
        cause = RuntimeError('rate limited')
        cause.code = 429
        error = RuntimeError('wrapped')
        error.__cause__ = cause
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
        with patch('src.llm.retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
            with self.assertRaises(RuntimeError) as caught:
                await invoke_with_retry(model, 'question')
        self.assertIs(caught.exception, error)
        self.assertEqual(model.ainvoke.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_other_errors_do_not_retry(self):
        error = RuntimeError('service unavailable')
        error.status_code = 503
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
        with patch('src.llm.retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
            with self.assertRaises(RuntimeError):
                await invoke_with_retry(model, 'question')
        sleep.assert_not_awaited()
        self.assertEqual(model.ainvoke.await_count, 1)

    async def test_server_delay_and_long_limit(self):
        error = RuntimeError('limited')
        error.status_code = 429
        error.body = {'message': 'Please try again in 14s.'}
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[error, 'OK']))
        with patch('src.llm.retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
            self.assertEqual(await invoke_with_retry(model, 'question'), 'OK')
            sleep.assert_awaited_once_with(14)
        error.body = {'message': 'Please try again in 2m3s.'}
        model.ainvoke.side_effect = error
        with patch('src.llm.retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
            with self.assertRaises(RuntimeError):
                await invoke_with_retry(model, 'question')
            sleep.assert_not_awaited()

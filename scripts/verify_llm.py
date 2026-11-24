"""Opt-in live API smoke test, four short requests/provider, no automatic retry.
Run from hw2: .venv/bin/python -m scripts.verify_llm --live
"""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import ToolMessage
from src.llm import create_llm
from src.llm.client import ROOT, VARIABLES
import os

class Probe(BaseModel):
    status: Literal['ok']
    value: int

TOOL = {'name': 'add_numbers', 'description': 'Add two integers.', 'parameters': {
    'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b']}}

async def verify(provider, traces):
    result = {'provider': provider, 'checks': {}}
    try:
        llm = create_llm(provider, traces=traces, agent='smoke_test',
                         reason='explicit provider capability verification', max_tokens=512)
        result['model'] = os.environ[VARIABLES[provider][1]]
        response = await llm.ainvoke('Reply with exactly OK.')
        result['generation_text'] = response.text
        result['checks']['generation'] = response.text.strip().upper() == 'OK'
        schema_options = {'method': 'json_schema'} if provider == 'anthropic' else {}
        parsed = await llm.with_structured_output(Probe, **schema_options).ainvoke(
            'Return status ok and value 7 in the requested schema.')
        result['checks']['structured_output'] = parsed.status == 'ok' and parsed.value == 7
        bound = llm.bind_tools([TOOL])
        response = await bound.ainvoke('Use add_numbers to add 2 and 3.')
        calls = response.tool_calls
        valid = len(calls) == 1 and calls[0]['name'] == 'add_numbers' and calls[0]['args'] == {'a': 2, 'b': 3}
        result['checks']['tool_call'] = valid
        if not valid:
            result['status'] = 'failed'
            return result
        # Real local tool execution and a streamed follow-up prove message roundtrip.
        call = calls[0]
        value = call['args']['a'] + call['args']['b']
        messages = [('human', 'Use add_numbers to add 2 and 3. Report just the result.'),
                    response, ToolMessage(content=str(value), tool_call_id=call['id'])]
        chunks = []
        async for chunk in llm.bind_tools([TOOL]).astream(messages):
            chunks.append(chunk.text)
        result['checks']['tool_result_stream'] = ''.join(chunks).strip().strip('.') == '5'
        result['status'] = 'passed' if all(result['checks'].values()) else 'failed'
    except Exception as exc:
        # Type only: SDK validation exceptions may echo credentials/configuration.
        result.update(status='failed', error_code=type(exc).__name__)
    return result

async def main(args):
    traces = []
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Live capability smoke test; not agent routing or RAGAS evaluation',
              'providers': [], 'traces': traces}
    for provider in args.providers:
        result = await verify(provider, traces)
        report['providers'].append(result)
        print(json.dumps(result), flush=True)
    report['status'] = 'passed' if all(p['status'] == 'passed' for p in report['providers']) else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Report:', args.output)
    return 0 if report['status'] == 'passed' else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Allow paid API calls')
    parser.add_argument('--providers', nargs='+', choices=['openai', 'anthropic', 'google'],
                        default=['openai', 'anthropic', 'google'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/evaluation/llm_connections.json')
    args = parser.parse_args()
    if not args.live:
        parser.error('Pass --live to explicitly enable real API requests')
    raise SystemExit(asyncio.run(main(args)))

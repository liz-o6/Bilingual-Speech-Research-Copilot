"""Tavily Remote MCP, adapted from hw1/integrations/tavily.py."""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from time import perf_counter
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.llm.client import ROOT

SERVER_URL = 'https://mcp.tavily.com/mcp/'
SEARCH_NAMES = ['tavily-search', 'tavily_search']


def parse_results(result):
    """Convert MCP search output into the same source shape as local RAG hits."""
    if result.isError:
        raise ValueError('Tavily returned a tool error')
    data = result.structuredContent
    if not isinstance(data, dict) or 'results' not in data:
        data = None
        for block in result.content:
            if block.type == 'text':
                try:
                    candidate = json.loads(block.text)
                except (ValueError, TypeError):
                    continue
                if isinstance(candidate, dict) and 'results' in candidate:
                    data = candidate
                    break
    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise ValueError('Unrecognized Tavily search result')

    hits = []
    for item in data['results']:
        url = item.get('url', '')
        content = item.get('content', '')
        if not isinstance(url, str) or not url.startswith(('https://', 'http://')):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        # Include content in identity so changed web evidence gets a new source ID.
        identity = sha256((url + '\n' + content).encode()).hexdigest()
        hits.append({
            'chunk_id': 'web:' + identity, 'source_type': 'web',
            'title': item.get('title', ''), 'source_url': url, 'text': content,
            'accessed_at': datetime.now(timezone.utc).isoformat(),
        })
    return hits


async def search_web(query, traces=None, agent='Research'):
    """Discover and call the real remote search tool once; no local MCP server."""
    start = perf_counter()
    event = {'event': 'mcp_tool_call', 'agent': agent, 'server': 'tavily',
             'tool': None, 'status': 'running',
             'reason': 'Supervisor requested current or external evidence.'}
    try:
        load_dotenv(ROOT / '.env', override=False)
        key = os.getenv('TAVILY_API_KEY', '').strip()
        if not key:
            raise ValueError('Missing TAVILY_API_KEY')
        if not query.strip():
            raise ValueError('Empty search query')
        url = SERVER_URL + '?' + urlencode({'tavilyApiKey': key})
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=60)) as client:
            async with streamable_http_client(url, http_client=client) as streams:
                read, write, _ = streams
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)) as session:
                    await session.initialize()
                    discovered = await session.list_tools()
                    tool_name = None
                    for tool in discovered.tools:
                        if tool.name in SEARCH_NAMES:
                            tool_name = tool.name
                            break
                    if tool_name is None:
                        raise ValueError('Search tool not found')
                    event['tool'] = tool_name
                    response = await session.call_tool(tool_name, {
                        'query': query, 'search_depth': 'basic', 'max_results': 3,
                    })
                    hits = parse_results(response)
        event['status'] = 'success'
        event['result_count'] = len(hits)
        return hits
    except asyncio.CancelledError:
        event['status'] = 'cancelled'
        raise
    except Exception:
        event['status'] = 'error'
        # Never expose the URL containing the API key or raw SDK exceptions.
        raise RuntimeError('Tavily Remote MCP search is unavailable.') from None
    finally:
        event['duration_seconds'] = perf_counter() - start
        if traces is not None:
            traces.append(event)

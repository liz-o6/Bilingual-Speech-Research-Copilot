import {createSSEParser} from './sse.js';

export async function ask(query, signal, onEvent) {
  const response = await fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query}), signal
  });
  if (!response.ok) throw new Error(`Request failed (HTTP ${response.status}). Check your input or the service status.`);
  if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body) {
    throw new Error('The server did not return an SSE stream.');
  }
  let done = false, requestId = null, seq = 0;
  const parser = createSSEParser(event => {
    const data = event.data;
    if (!data || typeof data.request_id !== 'string' || !Number.isInteger(data.seq)) throw new Error('Invalid response event format.');
    if (requestId === null) requestId = data.request_id;
    if (requestId !== data.request_id || data.seq <= seq) throw new Error('Response events arrived out of order.');
    seq = data.seq;
    if (done) return;
    onEvent(event);
    if (event.event === 'done') done = true;
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) { parser.push(decoder.decode(), true); break; }
      parser.push(decoder.decode(chunk.value, {stream: true}));
    }
    if (!done) throw new Error('Connection interrupted before completion. The draft remains unverified.');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

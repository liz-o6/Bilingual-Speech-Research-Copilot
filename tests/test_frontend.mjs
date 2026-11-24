import assert from 'node:assert/strict';
import {createSSEParser} from '../frontend/js/sse.js';
import {ask} from '../frontend/js/api.js';

// Feed one UTF-8 byte at a time to test split characters and CRLF.
const events = [];
const parser = createSSEParser(event => events.push(event));
const decoder = new TextDecoder();
const stream = ': heartbeat\r\nevent: token\r\ndata: {"text":"论文"}\r\n\r\nevent: done\ndata: {\ndata: "status":"completed"}\n\n';
for (const byte of new TextEncoder().encode(stream)) {
  parser.push(decoder.decode(new Uint8Array([byte]), {stream:true}));
}
parser.push(decoder.decode(), true);
assert.deepEqual(events, [{event:'token',data:{text:'论文'}},{event:'done',data:{status:'completed'}}]);
const incomplete = [];
createSSEParser(e => incomplete.push(e)).push('data: {"text":"残缺"}\n', true);
assert.equal(incomplete.length, 0);
assert.throws(() => createSSEParser(() => {}).push('data: invalid\n\n'));

function frame(event, seq, extra = {}) {
  return `event: ${event}\ndata: ${JSON.stringify({request_id:'test',seq,...extra})}\n\n`;
}
function mockFetch(body, status = 200) {
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/chat');
    assert.deepEqual(JSON.parse(options.body), {query:'测试'});
    return new Response(body, {status,headers:{'content-type':'text/event-stream'}});
  };
}
const received = [];
mockFetch(frame('answer',1,{text:'回答'}) + frame('done',2,{status:'completed'}));
await ask('测试', new AbortController().signal, e => received.push(e));
assert.equal(received.length,2);
mockFetch(frame('token',1,{text:'草稿'}));
await assert.rejects(ask('测试',null,()=>{}), /interrupted/);
mockFetch(frame('trace',2) + frame('done',1));
await assert.rejects(ask('测试',null,()=>{}), /out of order/);
mockFetch('',422);
await assert.rejects(ask('测试',null,()=>{}), /422/);
console.log('Frontend offline checks passed: UTF-8, SSE framing, final event, interruption, sequence, HTTP error.');

// Render server text with textContent, never as model-generated HTML.
const el = id => document.getElementById(id);
function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}
const statusNames = {idle:'Ready', running:'Working', completed:'Completed', partial:'Partial', failed:'Failed', stopped:'Stopped'};
export function status(state) {
  el('status').textContent = statusNames[state] || state;
  el('status').dataset.state = state;
}
export function notice(message = '') {
  el('notice').textContent = message;
  el('notice').hidden = !message;
}
export function busy(value) {
  el('submit').disabled = value;
  el('stop').disabled = !value;
  el('query').disabled = value;
  document.querySelectorAll('[data-example]').forEach(button => button.disabled = value);
}
export function showAnswer(text, provisional = false) {
  el('answer').replaceChildren();
  el('answer-note').textContent = provisional ? 'Streaming draft · Pending verification, not the final result' : 'Result returned. Read alongside its completion status and citations.';
  // Support bold text and citation buttons, not arbitrary links or HTML.
  const parts = text.split(/(\*\*[^*]+\*\*|\[(?:S\d+|M_[A-Za-z0-9_-]+)(?:\s*[,，;；]\s*(?:S\d+|M_[A-Za-z0-9_-]+))*\])/g);
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) el('answer').append(node('strong', part.slice(2,-2)));
    else if (/^\[(?:S\d+|M_)/.test(part)) {
      for (const label of part.match(/S\d+|M_[A-Za-z0-9_-]+/g)) {
        const button = node('button', `[${label}]`, 'cite');
        button.type = 'button'; button.title = `View source ${label}`;
        button.onclick = () => {
          const source = document.getElementById('source-' + label);
          if (!source) { notice('This source has not arrived or is missing from this response.'); return; }
          source.scrollIntoView({behavior: 'smooth', block: 'center'});
          source.querySelector('details').open = true;
          source.classList.add('highlight');
          setTimeout(() => source.classList.remove('highlight'), 1500);
        };
        el('answer').append(button);
      }
    } else el('answer').append(document.createTextNode(part));
  }
  el('copy').disabled = !text;
}
export function trace(data) {
  el('trace-empty').hidden = true;
  const names = {request_start:'Request started',agent_start:'Agent started',agent_result:'Agent result',agent_route:'Agent routing',llm_route:'Model routing',rag_retrieval:'Evidence retrieval',data_query:'Data query',data_plan:'Query plan',mcp_tool_call:'Web tool'};
  const item = node('li', undefined, 'trace');
  item.append(node('div', `${data.agent || 'System'} · ${names[data.type] || data.type}`, 'trace-title'));
  const description = [data.next_agent ? `→ ${data.next_agent}` : '', [data.provider, data.model].filter(Boolean).join(" → "),
    data.status, data.reason || data.error, data.tool, data.matched_count !== undefined ? `${data.matched_count} papers matched` : ''].filter(Boolean).join(' · ');
  if (description) item.append(node('p', description, 'trace-sub'));
  const details = node('details'); details.append(node('summary', 'View details'), node('pre', JSON.stringify(data, null, 2)));
  item.append(details); el('traces').append(item);
  el('trace-count').textContent = `${el('traces').children.length} events`;
}
export function citation(data) {
  const id = 'source-' + data.citation_id;
  const card = node('article', undefined, 'source'); card.id = id;
  const head = node('div', undefined, 'source-head');
  head.append(node('span', data.citation_id, 'source-label'), node('h3', data.title || 'Source'));
  card.append(head, node('p', [data.doc_id, data.page ? `PDF page ${data.page}` : '', data.section].filter(Boolean).join(' · ') || 'Computed data / Metadata source', 'source-meta'));
  if (data.source_id) card.append(node('p', `Source / chunk ID: ${data.source_id}`, 'source-meta source-id'));
  try {
    const url = new URL(data.url);
    if (['http:', 'https:'].includes(url.protocol)) {
      const link = node('a', 'Open original source ↗'); link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; card.append(link);
    }
  } catch { /* Local sources without links can still show excerpts. */ }
  const details = node('details'); details.append(node('summary', 'View evidence excerpt'), node('pre', data.excerpt || 'No excerpt available'));
  card.append(details);
  const old = document.getElementById(id);
  if (old) old.replaceWith(card); else el('sources').append(card);
  el('source-count').textContent = el('sources').children.length;
  el('source-empty').hidden = el('sources').children.length > 0;
}
export function metrics(data, clientTTFT) {
  const seconds = value => value == null ? '—' : `${(value / 1000).toFixed(1)} s`;
  const rows = [['Total time', seconds(data.total_ms)], ['RAG retrieval', seconds(data.retrieval_ms)], ['LLM time (sum)', seconds(data.llm_ms)], ['Model calls', data.llm_calls ?? '—'],
    ['Input / output tokens', `${data.input_tokens ?? '—'} / ${data.output_tokens ?? '—'}${data.usage_complete === false ? ' (incomplete)' : ''}`],
    ['Client TTFT', seconds(clientTTFT)]];
  el('metrics').replaceChildren();
  for (const [label, value] of rows) {const group = node('div'); group.append(node('dt', label), node('dd', String(value))); el('metrics').append(group);}
  el('metrics').hidden = false;
}
export function clear() {
  ['answer','traces','sources','metrics'].forEach(id => el(id).replaceChildren());
  el('answer').append(node('p', 'Your answer will appear here after you submit a question.', 'empty-small'));
  el('answer-note').textContent = 'A draft streams while generating, then updates to the final result.';
  el('metrics').hidden = true; el('trace-empty').hidden = false; el('source-empty').hidden = false;
  el('trace-count').textContent = '0 events'; el('source-count').textContent = '0'; el('copy').disabled = true;
  notice(); status('idle');
}

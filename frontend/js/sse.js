// Parse SSE by line; a network read may contain only part of an event.
export function createSSEParser(onEvent) {
  let buffer = '', name = '', data = [];
  function line(text) {
    if (text === '') {
      if (data.length) onEvent({event: name || 'message', data: JSON.parse(data.join('\n'))});
      name = ''; data = [];
      return;
    }
    if (text.startsWith(':')) return;
    const colon = text.indexOf(':');
    const field = colon < 0 ? text : text.slice(0, colon);
    let value = colon < 0 ? '' : text.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') name = value;
    if (field === 'data') data.push(value);
  }
  return {
    push(text, final = false) {
      buffer += text;
      while (true) {
        const match = /\r\n|\r|\n/.exec(buffer);
        if (!match) break;
        if (!final && match[0] === '\r' && match.index === buffer.length - 1) break;
        line(buffer.slice(0, match.index));
        buffer = buffer.slice(match.index + match[0].length);
      }
      // Discard incomplete frames; the caller also checks for done.
      if (final) buffer = '';
    }
  };
}

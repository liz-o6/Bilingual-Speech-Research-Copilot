import {ask} from './api.js';
import * as view from './render.js';

const query = document.getElementById('query');
let controller = null, version = 0, answer = '';

document.querySelectorAll('[data-example]').forEach(button => {
  button.onclick = () => { query.value = button.dataset.example; query.focus(); };
});
document.getElementById('question-form').onsubmit = async event => {
  event.preventDefault();
  if (controller) return;
  const question = query.value.trim();
  if (!question) {query.focus(); return;}
  const turn = ++version;
  controller = new AbortController();
  const signal = controller.signal;
  answer = ''; let clientTTFT = null, finalReceived = false, errorMessage = ''; 
  const started = performance.now();
  view.clear(); view.busy(true); view.status('running');
  try {
    await ask(question, signal, ({event, data}) => {
      if (turn !== version) return;
      if (event === 'trace') view.trace(data);
      if (event === 'citation') view.citation(data);
      if (event === 'token' && !finalReceived) {
        if (clientTTFT === null) clientTTFT = performance.now() - started;
        answer += data.text; view.showAnswer(answer, true);
      }
      if (event === 'answer') {
        finalReceived = true; answer = data.text; view.showAnswer(answer);
      }
      if (event === 'error') {
        errorMessage = data.message || 'The request failed.';
        view.notice(errorMessage);
        if (!finalReceived) { answer = ''; view.showAnswer('No confirmed final answer is available yet.'); }
      }
      if (event === 'done') {
        view.status(data.status); view.metrics(data.metrics || {}, clientTTFT);
        if (data.status !== 'completed') view.notice(errorMessage || 'This result is partial. See the answer and trace for details.');
        if (!finalReceived && data.status === 'completed') throw new Error('The response ended without a final answer.');
      }
    });
  } catch (error) {
    if (turn !== version) return;
    view.status(signal.aborted ? 'stopped' : 'failed');
    view.notice(signal.aborted ? 'Stopped. The draft is unverified; model requests already received may still be billed.' : error.message);
    if (!finalReceived && answer) view.showAnswer(answer, true);
  } finally {
    if (turn === version) { controller = null; view.busy(false); }
  }
};
document.getElementById('stop').onclick = () => controller?.abort();
document.getElementById('reset').onclick = () => {
  ++version; controller?.abort(); controller = null; answer = '';
  query.value = ''; view.clear(); view.busy(false); query.focus();
};
document.getElementById('copy').onclick = async () => {
  try {await navigator.clipboard.writeText(answer); document.getElementById('copy').textContent = 'Copied';}
  catch {view.notice('Clipboard access was denied. Select the answer and copy it manually.');}
  setTimeout(() => document.getElementById('copy').textContent = 'Copy', 1500);
};

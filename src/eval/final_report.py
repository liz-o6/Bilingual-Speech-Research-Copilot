"""Rebuild the five-question coursework report from saved files; no API calls."""
from collections import defaultdict
from statistics import mean
from .common import ROOT, read, save, cost, embedding_cost
from .comparison_report import comparison_section
from .cloud_report import cloud_section


def main():
    base = ROOT / 'artifacts/evaluation'
    prices = read(ROOT / 'data/evaluation/prices.2026-10-27.json')
    metrics = ['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']
    rows = []
    providers = defaultdict(lambda: {'calls': 0, 'input': 0, 'output': 0, 'usd': 0, 'seconds': 0})
    for number in range(1, 6):
        case = f'rag-{number:02}'
        batch = '2' if number < 3 else '3'
        run_path = f'run{batch}/{case}.json'
        score_path = f'scores{batch}/{case}.json'
        run, score = read(base / run_path), read(base / score_path)
        assert run['status'] == 'completed' and not score['errors']
        assert score['embedding_usage_complete'] and run['metrics']['usage_complete']
        for call in run['calls'] + score['calls']:
            assert call['usage'] is not None
            if call['provider'] == 'anthropic':
                assert call['usage']['input_tokens'] <= 100000
        generation = cost(run['calls'], prices)['generation_cost_usd']
        query_embedding = embedding_cost(run['embedding_calls'], prices)
        judge = cost(score['calls'], prices)['generation_cost_usd']
        eval_embedding = embedding_cost(score['embedding_calls'], prices)
        assert None not in (generation, query_embedding, judge, eval_embedding)
        rows.append(dict(id=case, run=run_path, scoring=score_path, question=run['question'],
                         scores=score['scores'], metrics=run['metrics'], generation_usd=generation,
                         query_embedding_usd=query_embedding, query_usd=generation + query_embedding,
                         judge_usd=judge, evaluation_embedding_usd=eval_embedding,
                         evaluation_usd=judge + eval_embedding))
        for call in run['calls']:
            key = call['provider'] + '/' + call['model']
            entry = providers[key]
            entry['calls'] += 1
            entry['input'] += call['usage']['input_tokens']
            entry['output'] += call['usage']['output_tokens']
            entry['usd'] += cost([call], prices)['generation_cost_usd']
            entry['seconds'] += call['duration_seconds']
    out = base / 'final_local_5'
    save(out / 'summary.json', {'pricing_date': '2026-10-27', 'rows': rows, 'providers': dict(providers)})
    lines = ['# Evaluation: Local RAG Quality, Performance, and Cost', '',
             'Measured on 2026-10-27. Five saved workflow runs completed; all 20 RAGAS scores are available. Costs below are token-based estimates in USD, not reconciled invoices. No model calls were made to produce this report.', '',
             '## Dataset and procedure', '',
             'The fixed English question set is `data/evaluation/local_rag_5.json`. Reference answers and source-page text were prepared from local PDFs. Scoring uses the actual contexts supplied to Research, not reference contexts injected into generation. This is a small coursework regression set, not a held-out benchmark.', '',
             'RAGAS 0.3.9; judge `gpt-5.4-nano`; embedding `text-embedding-3-small`; answer-relevance strictness 1; judge output limit 4096. Questions 1–2 use run2/scores2 and questions 3–5 use run3/scores3. Question 2 faithfulness was retried after a parsing failure. JSON mode was enabled for that retry and for questions 3–5; questions 1 and the other question-2 metrics used the earlier format. Thus judge formatting was not identical across all measurements. Original artifacts, attempts, hashes, and configurations are preserved.', '',
             'Each question was collected once in the selected runs. Source hashes differ between run2 and run3 after evaluation-diagnostics changes; see their run_config.json files. Earlier run1 and development experiments are excluded from these totals.', '']
    for row in rows:
        lines.append(f"- **{row['id']}**: {row['question']}")
    lines += ['', '## RAGAS results', '', '| Question | Faithfulness | Answer relevance | Context precision | Context recall |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['id'] + ' | ' + ' | '.join(f"{row['scores'][k]:.4f}" for k in metrics) + ' |')
    lines.append('| **Mean (5/5 each)** | ' + ' | '.join(f"{mean(r['scores'][k] for r in rows):.4f}" for k in metrics) + ' |')
    lines += ['', 'Completion does not mean every claim is correct. Question 1 has the lowest faithfulness; question 2 has the lowest context precision. Questions 1 and 5 have recall 0.5, indicating incomplete reference coverage under the judge. These results are reported without discarding low scores. Five examples and one judge configuration do not establish general reliability.', '',
              '## Local latency', '',
              'All values below are seconds. Retrieval includes query embedding, vector search, selection, and evidence expansion. LLM time sums provider-call durations. Total time measures the service workflow and excludes graph construction; it includes other orchestration overhead. Server TTFT is the first visible candidate text, before citation verification, not the first verified final answer. These are local service measurements, not browser/network TTFT or Cloud Run cold/warm measurements.', '',
              '| Question | Retrieval | LLM total | Workflow total | Server TTFT | LLM calls |', '|---|---:|---:|---:|---:|---:|']
    timing = ['retrieval_ms','llm_ms','total_ms','server_ttft_ms']
    for row in rows:
        lines.append('| ' + row['id'] + ' | ' + ' | '.join(f"{row['metrics'][k]/1000:.3f}" for k in timing) + f" | {row['metrics']['llm_calls']} |")
    lines.append('| **Mean** | ' + ' | '.join(f"{mean(r['metrics'][k] for r in rows)/1000:.3f}" for k in timing) + ' | 3 |')
    lines += ['', '## Pricing basis', '',
              'Rates are USD per million tokens, dated 2026-10-27. Generation rates below are transcribed from the user-provided pricing screenshots; they are not a claim of independent invoice verification. The screenshots show Haiku rates for prompts up to 100K tokens; every recorded Anthropic request is below that threshold.', '',
              '| Model | Input | Output | Cached input | Source |', '|---|---:|---:|---:|---|',
              '| gpt-5.4-nano | 0.20 | 1.25 | 0.02 | User screenshot, 11:21:50 AM |',
              '| claude-haiku-5-5 | 0.10 | 0.50 | 0.01 | User screenshot, 11:18:27 AM |',
              '| text-embedding-3-small | 0.02 | N/A | N/A | [Official model documentation](https://developers.openai.com/api/docs/models/text-embedding-3-small) |', '',
              'OpenAI cost = ((input tokens − cached input tokens) × input rate + cached input tokens × cached rate + output tokens × output rate) / 1,000,000. There are no Anthropic cache reads/writes in these selected records, so its cost uses ordinary input/output rates. RAGAS question 2 includes 3,840 cached input tokens, billed at the supplied cached rate. Embedding cost = input tokens × 0.02 / 1,000,000.', '',
              '## Per-question API cost', '',
              'Query cost includes the Supervisor, Research answer generation, citation verification, and query embedding. RAGAS cost is separate and includes all recorded judge calls (including the question-2 failed scoring attempt and retry) plus evaluation embeddings. API-successful calls that produced unparseable scoring output still incur cost. All required usage fields are present.', '',
              '| Question | Generation | Query embedding | **Query total** | RAGAS judge | RAGAS embedding | **RAGAS total** |', '|---|---:|---:|---:|---:|---:|---:|']
    keys = ['generation_usd','query_embedding_usd','query_usd','judge_usd','evaluation_embedding_usd','evaluation_usd']
    for row in rows:
        lines.append('| ' + row['id'] + ' | ' + ' | '.join(f"${row[k]:.8f}" for k in keys) + ' |')
    lines.append('| **Total** | ' + ' | '.join(f"${sum(r[k] for r in rows):.8f}" for k in keys) + ' |')
    query = sum(r['query_usd'] for r in rows)
    evaluation = sum(r['evaluation_usd'] for r in rows)
    lines += ['', f'Five-query cost: **${query:.8f}**; mean query cost: **${query/5:.8f}**. RAGAS scoring cost: **${evaluation:.8f}**. Combined selected-run cost: **${query+evaluation:.8f}**.', '',
              'Excluded: earlier development/reruns outside the selected files, one-time corpus ingestion embeddings, hosting, taxes, and other account/project usage. These estimates do not explain the entire account balance. The Groq dashboard monthly amount is not allocated to these five questions.', '',
              '## Observed generation-provider usage', '',
              '| Provider/model | Calls | Input tokens | Output tokens | Cumulative call time (s) | Cost (USD) |', '|---|---:|---:|---:|---:|---:|']
    for key, value in providers.items():
        lines.append(f"| {key} | {value['calls']} | {value['input']} | {value['output']} | {value['seconds']:.3f} | {value['usd']:.8f} |")
    lines += ['| groq/qwen/qwen3.8-27b | 0 | 0 | 0 | 0 | 0 |', '',
              'Groq cost is zero here because these questions did not call Groq, not because its plan is assumed free. A generic Llama pricing range cannot price this Qwen model. OpenAI performs planning and Anthropic performs synthesis/verification, so the table is a workload breakdown, not a controlled quality/speed comparison. RAGAS judge calls are excluded from this generation-provider table.', '',
              '## Submission evidence checklist', '',
              '- Quantitative measurements are complete: five-question RAGAS, local timing and cost, three-provider comparison, and cloud cold/warm timing. Retain the startup screenshots with the report.',
              '- Groq comparison cost is N/A because it could not be isolated from account-wide usage; add an attributable cost if exact pricing or billing becomes available.',
              '- Include existing UI/multi-agent traces and evidence that both Research and Summary use the shared Tavily MCP adapter. Local RAG measurements alone do not establish that acceptance requirement; verify saved evidence before deciding whether a targeted test is needed.', '',
              '## Reproduction and provenance', '',
              'Run `python -m src.eval.final_report` from hw2 to rebuild this document and the machine-readable summary without API calls. Prices: `data/evaluation/prices.2026-10-27.json`. Summary: `artifacts/evaluation/final_local_5/summary.json`. Raw source files:', '']
    for row in rows:
        lines.append(f"- `{row['run']}` and `{row['scoring']}` under `artifacts/evaluation/`.")
    position = lines.index('## Submission evidence checklist')
    lines[position:position] = comparison_section(base, prices) + cloud_section(base)
    path = ROOT / 'docs/evaluation.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)
    print(f'Query={query:.8f}; RAGAS={evaluation:.8f}; combined={query+evaluation:.8f}')


if __name__ == '__main__':
    main()

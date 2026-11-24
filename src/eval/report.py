"""Create CSV and Markdown from saved results without API calls."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from .common import read, cost, embedding_cost


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--scores');p.add_argument('--prices');p.add_argument('--output',required=True)
    args=p.parse_args(); rows=[]; metrics=['faithfulness','answer_relevance','context_precision','context_recall']
    prices=read(args.prices) if args.prices else {}
    for path in sorted(Path(args.input).glob('rag-*.json')):
        record=read(path);m=record['metrics']
        row=dict(id=record['id'],status=record['status'],environment=record['environment'],
                 total_ms=m.get('total_ms'),retrieval_ms=m.get('retrieval_ms'),llm_ms=m.get('llm_ms'),
                 ttft_ms=m.get('server_ttft_ms'),llm_calls=m.get('llm_calls'),
                 input_tokens=m.get('input_tokens'),output_tokens=m.get('output_tokens'),usage_complete=m.get('usage_complete'),
                 models='; '.join(sorted({c['provider']+'/'+c['model'] for c in record['calls']})))
        row.update(cost(record['calls'],prices))
        row['query_embedding_tokens'] = sum((c.get('usage') or {}).get('input_tokens',0) for c in record.get('embedding_calls',[])) if 'embedding_calls' in record else None
        row['query_embedding_cost_usd'] = embedding_cost(record.get('embedding_calls'), prices)
        row['query_api_cost_usd'] = (row['generation_cost_usd'] + row['query_embedding_cost_usd']
                                   if row['generation_cost_usd'] is not None and row['query_embedding_cost_usd'] is not None else None)
        scores={}
        evaluation = read(Path(args.scores)/path.name) if args.scores and (Path(args.scores)/path.name).exists() else {}
        scores = evaluation.get('scores', {})
        row['evaluation_embedding_cost_usd'] = embedding_cost(evaluation.get('embedding_calls'),prices) if evaluation.get('embedding_usage_complete') else None
        row['evaluation_judge_cost_usd'] = cost(evaluation['calls'], prices)['generation_cost_usd'] if 'calls' in evaluation else None
        row['evaluation_api_cost_usd'] = (row['evaluation_embedding_cost_usd'] + row['evaluation_judge_cost_usd']
                                        if row['evaluation_embedding_cost_usd'] is not None and row['evaluation_judge_cost_usd'] is not None else None)
        for metric in metrics:row[metric]=scores.get(metric)
        rows.append(row)
    if not rows:raise ValueError('No saved records')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_question.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Local RAG evaluation','',f'Recorded questions: {len(rows)}. No Cloud Run cold/warm measurements yet.',
           'Missing scores/costs are unknown, not zero. Partial/failed rows remain in the CSV.',
           'Reference answers were drafted from local PDFs; review them before the formal run.',
           '', '| Metric | Mean | Valid / recorded |','|---|---:|---:|']
    for key in metrics+['total_ms','retrieval_ms','llm_ms','ttft_ms']:
        values=[r[key] for r in rows if r[key] is not None]
        lines.append(f'| {key} | {mean(values):.4f} | {len(values)} / {len(rows)} |' if values else f'| {key} | N/A | 0 / {len(rows)} |')
    lines += ['', 'Generation cost excludes embeddings. Query API cost combines generation and query embeddings. Evaluation API cost is separate. Hosting and one-time ingestion are excluded.',
              'Model names here describe actual routing, not a controlled provider comparison. Use matched inputs for a provider shootout.']
    (out/'evaluation.md').write_text('\n'.join(lines)+'\n')
    print(out/'evaluation.md')

if __name__ == '__main__':main()

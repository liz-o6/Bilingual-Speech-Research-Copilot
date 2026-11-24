"""Score saved workflow outputs only. Run using .venv-eval, not the app venv."""
import argparse
import asyncio
import math
import os
from pathlib import Path
from .common import ROOT, read, save, fingerprint


async def score(args):
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig
    from src.llm.logging import LLMCallLogger
    load_dotenv(ROOT / '.env', override=False)
    model = os.getenv('EVAL_MODEL') or os.environ['OPENAI_MODEL']
    embedding_model = os.getenv('EVAL_EMBEDDING_MODEL', 'text-embedding-3-small')
    from .diagnostics import JudgeDiagnostics, error_details
    diagnostics = []
    traces = []
    judge = ChatOpenAI(model=model, api_key=os.environ['OPENAI_API_KEY'], temperature=0,
                      max_tokens=args.max_tokens, max_retries=0,
                      model_kwargs={"response_format": {"type": "json_object"}} if args.json_mode else {},
                      callbacks=[LLMCallLogger(traces,'openai',model,'RAGAS','Offline saved-answer evaluation'),
                                 JudgeDiagnostics(diagnostics)])
    wrapper = LangchainLLMWrapper(judge, run_config=RunConfig(max_retries=1,max_workers=1,timeout=120))
    from src.observability.embeddings import track_embeddings
    embedding_traces = []
    embedder = track_embeddings(OpenAIEmbeddings(model=embedding_model, max_retries=0),
                               embedding_traces, 'ragas_evaluation')
    embeddings = LangchainEmbeddingsWrapper(embedder)
    metrics = {'faithfulness': Faithfulness(llm=wrapper),
               'answer_relevance': ResponseRelevancy(llm=wrapper,embeddings=embeddings,strictness=1),
               'context_precision': LLMContextPrecisionWithReference(llm=wrapper),
               'context_recall': LLMContextRecall(llm=wrapper)}
    for path in sorted(Path(args.input).glob('rag-*.json')):
        row = read(path)
        if args.ids and row['id'] not in args.ids: continue
        output = Path(args.output) / path.name
        signature = fingerprint({'record':row,'judge':model,'embedding':embedding_model,'ragas':'0.3.9','strictness':1})
        result = read(output) if output.exists() else dict(id=row['id'],signature=signature,judge_model=model,
            embedding_model=embedding_model,ragas_version='0.3.9',relevance_strictness=1,
            scores={},errors={},calls=[],embedding_calls=[],embedding_usage_complete=True,embedding_cost_usd=None)
        result.setdefault('embedding_usage_complete', False)  # Legacy runs did not record embedding usage.
        if result['signature'] != signature: raise ValueError('Evaluation inputs changed; use a new output directory.')
        if not row['answer'] or not row['retrieved_contexts']:
            result['errors']['input'] = 'No answer or contexts; metrics not computed. Do not count as zero.'
            save(output,result); continue
        sample = SingleTurnSample(user_input=row['question'], response=row['answer'],
                                  reference=row['reference_answer'], retrieved_contexts=row['retrieved_contexts'])
        for name,metric in metrics.items():
            if args.metrics and name not in args.metrics: continue
            if not should_run(result, name, args.retry_failed): continue
            # Preserve failed attempts and their already-recorded token usage.
            result.setdefault('attempts', []).append({
                'metric': name, 'previous_score': result['scores'].get(name),
                'previous_error': result['errors'].get(name),
                'max_tokens': args.max_tokens, 'json_mode': args.json_mode, 'status': 'started'})
            save(output, result)
            diagnostics.clear()
            traces.clear()
            embedding_traces.clear()
            try:
                value = float(await metric.single_turn_ascore(sample))
                result['errors'].pop(name, None)
                result['scores'][name] = value if math.isfinite(value) else None
                if not math.isfinite(value): result['errors'][name] = 'Metric returned non-finite score'
            except Exception as exc:
                result['scores'][name] = None
                result['errors'][name] = type(exc).__name__
                result['attempts'][-1]['error_details'] = error_details(exc)
            result['attempts'][-1].update(status='failed' if name in result['errors'] else 'completed', score=result['scores'].get(name), error=result['errors'].get(name))
            result['attempts'][-1]['judge_outputs'] = list(diagnostics)
            result['calls'].extend(traces)
            result.setdefault('embedding_calls', []).extend(embedding_traces)
            if any(c['usage'] is None for c in embedding_traces):
                result['embedding_usage_complete'] = False
            save(output,result)
            print(row['id'],name,result['scores'].get(name))


def should_run(result, name, retry_failed=False):
    saved = name in result['scores'] or name in result['errors']
    failed = name in result['errors'] or result['scores'].get(name) is None
    return not saved or (retry_failed and failed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ids',nargs='+');p.add_argument('--run',action='store_true')
    p.add_argument('--metrics', nargs='+', choices=['faithfulness','answer_relevance','context_precision','context_recall'])
    p.add_argument('--retry-failed', action='store_true', help='Retry selected missing/failed scores; never overwrite successful scores')
    p.add_argument('--json-mode', action='store_true', help='Ask a compatible judge API for a JSON object; schema validation still applies')
    p.add_argument('--max-tokens', type=int, default=2048, help='Judge output limit per call')
    args=p.parse_args()
    if args.max_tokens < 1: p.error('--max-tokens must be positive')
    if not args.run:
        print('Preview only. Add --run to call the RAGAS judge and embeddings. No workflow calls are made.')
        return
    asyncio.run(score(args))

if __name__ == '__main__':main()

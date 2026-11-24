"""Run the real workflow once per selected question; checkpoint every result."""
import argparse
import asyncio
from datetime import datetime, timezone
from time import perf_counter
from .common import ROOT, DATASET, read, save, fingerprint


async def collect(args):
    from src.api.service import ChatService
    cases = read(args.dataset)
    selected = cases if not args.ids else [c for c in cases if c['id'] in args.ids]
    if not selected or args.ids and set(args.ids) - {c['id'] for c in cases}:
        raise ValueError('Unknown or empty case selection')
    from pathlib import Path
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    config = {'models': {k: os.getenv(k) for k in
              ('OPENAI_MODEL','ANTHROPIC_MODEL','GROQ_MODEL','OPENAI_EMBEDDING_MODEL')},
              'source_hash': fingerprint({str(p.relative_to(ROOT)): p.read_text()
                             for p in (ROOT / 'src').rglob('*.py')})}
    manifest = Path(args.output) / 'run_config.json'
    if manifest.exists() and read(manifest) != config:
        raise ValueError('Code/model configuration changed. Use a new output directory.')
    save(manifest, config)
    service = ChatService(capture_evaluation=True)
    snapshot = ROOT / 'storage/chroma/active.json' 
    snapshot_hash = fingerprint(read(snapshot)) if snapshot.exists() else None
    for case in selected:
        path = Path(args.output) / (case['id'] + '.json')
        if path.exists():
            old = read(path)
            if old['case_hash'] != fingerprint(case) or old['snapshot_hash'] != snapshot_hash:
                raise ValueError('Inputs changed. Use a new output directory.')
            print('Skip saved:', case['id'])
            continue
        started = perf_counter()
        events = [event async for event in service.events(case['question'])]
        record = dict(id=case['id'], question=case['question'], reference_answer=case['reference_answer'],
                      case_hash=fingerprint(case), snapshot_hash=snapshot_hash,
                      environment='local', recorded_at=datetime.now(timezone.utc).isoformat(),
                      client_total_ms=(perf_counter()-started)*1000, events=events)
        record.update(answer='', retrieved_contexts=[], calls=[], metrics={}, status='failed')
        for event in events:
            data = event['data']
            if event['event'] == 'answer': record['answer'] = data['text']
            if event['event'] == 'evaluation':
                record['sources'] = data['contexts']
                record['retrieved_contexts'] = [s['text'] for s in data['contexts']]
                record['calls'] = data['calls']
                record['embedding_calls'] = data['embedding_calls']
            if event['event'] == 'done':
                record['metrics'], record['status'] = data['metrics'], data['status']
        save(path, record)
        print(case['id'], record['status'], 'saved to', path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default=str(DATASET)); p.add_argument('--output',required=True)
    p.add_argument('--ids',nargs='+'); p.add_argument('--run',action='store_true',help='Run paid API calls')
    args=p.parse_args()
    if not args.run:
        for c in read(args.dataset): print(c['id'], c['question'])
        print('Preview only. Add --run to invoke model APIs. Existing records are skipped.')
        return
    asyncio.run(collect(args))

if __name__ == '__main__': main()

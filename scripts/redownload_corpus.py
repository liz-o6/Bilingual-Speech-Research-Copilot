"""Restore manifest PDFs without overwriting originals; record elapsed time."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Enable network downloads')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/corpus/restored')
    args = parser.parse_args()
    records = [json.loads(line) for line in (ROOT / 'data/corpus/manifest.jsonl').read_text().splitlines() if line.strip()]
    if not args.run:
        for row in records:
            print(row['doc_id'], row['pmcid'], row['download_url'])
        print('Preview only. Use --run to download into', args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'download_report.json'
    if report_path.exists():
        parser.error('Use a fresh output directory to preserve the previous report.')
    started = time.perf_counter()
    results = []
    for row in records:
        target = args.output / Path(row['local_file']).name
        result = {'doc_id': row['doc_id'], 'url': row['download_url']}
        tick = time.perf_counter()
        try:
            if target.exists():
                raise ValueError('Target exists; refusing to overwrite')
            request = Request(row['download_url'], headers={'User-Agent': 'CourseworkCorpusRestore/1.0'})
            with urlopen(request, timeout=60) as response:
                content = response.read()
            if not content.startswith(b'%PDF-'):
                raise ValueError('Response is not a PDF; use the manifest fulltext_url to download manually')
            digest = hashlib.sha256(content).hexdigest()
            result['sha256'] = digest
            result['matches_original'] = digest == row['sha256']
            with target.open('xb') as output:
                output.write(content)
            result['status'] = 'restored' if result['matches_original'] else 'downloaded_different_bytes_review_required'
        except Exception as exc:
            result.update(status='failed', error=str(exc))
        result['seconds'] = time.perf_counter() - tick
        results.append(result)
        print(row['doc_id'], result['status'])
        time.sleep(1)
    report = {'corpus_version': records[0]['corpus_version'], 'elapsed_seconds': time.perf_counter() - started,
              'results': results, 'exact_restore_complete': all(r['status'] == 'restored' for r in results)}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print('Elapsed seconds:', report['elapsed_seconds'])
    if not report['exact_restore_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

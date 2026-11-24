"""Small file and accounting helpers. Unknown prices/usage remain unknown."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'data/evaluation/local_rag_5.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cost(calls, prices):
    total = 0
    missing = []
    for call in calls:
        key = call['provider'] + '/' + call['model']
        usage, rate = call.get('usage'), prices.get(key)
        if usage is None or not rate or rate.get('input_per_million') is None or rate.get('output_per_million') is None:
            missing.append(key)
            continue
        details = usage.get('input_token_details') or {}
        cached = details.get('cache_read', 0)
        if any(details.get(k, 0) for k in ('cache_read', 'cache_creation')):
            # OpenAI includes cached input in input_tokens. Other providers need
            # their own accounting rule; keep those costs unknown for now.
            if (call['provider'] != 'openai' or details.get('cache_creation', 0)
                    or rate.get('cached_input_per_million') is None):
                missing.append(key + ': cache pricing required')
                continue
        total += ((usage['input_tokens'] - cached) * rate['input_per_million'] +
                  cached * rate.get('cached_input_per_million', 0) +
                  usage['output_tokens'] * rate['output_per_million']) / 1_000_000
    return {'known_generation_cost_usd': total,
            'generation_cost_usd': None if missing else total,
            'missing_prices_or_usage': missing,
            'excludes': ['query embeddings', 'RAGAS judge and embeddings', 'hosting']}


def embedding_cost(calls, prices):
    if calls is None:
        return None  # Old run: not instrumented, rather than no charge.
    total = 0
    for call in calls:
        usage = call.get('usage')
        rate = prices.get(call['provider'] + '/' + call['model'], {}).get('input_per_million')
        if usage is None or rate is None:
            return None
        total += usage['input_tokens'] * rate / 1_000_000
    return total

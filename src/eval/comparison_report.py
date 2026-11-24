"""Saved round-2 measurements and qualitative review; no model calls."""
from statistics import mean
from .common import read, cost


def comparison_section(base, prices):
    folder = base / 'provider_comparison2'
    records = [read(folder / f'rag-{n:02}-{p}.json')
               for p in ('openai', 'anthropic', 'groq') for n in range(1, 4)]
    config = read(folder / 'config.json')
    assert config['max_tokens'] == 800 and config['groq_interval'] == 65
    assert all(r['signature'] == config['signature'] for r in records)
    assert all(r['status'] == 'success' and r['finish_reason'] in ('stop', 'end_turn') for r in records)
    lines = ['', '## Matched-input provider comparison (round 2)', '',
             'The first three questions were reused with exactly the same saved evidence and shared prompt for each provider. '
             'Each provider received one streaming request per question (nine calls). The prompt requested 2–3 bullets and at most 120 English words; '
             'the output cap was 800 tokens. This is a requested length, not an enforced word-count constraint. '
             'No retrieval, embedding, Supervisor, citation-judge, or RAGAS calls were made during this comparison. '
             'Provider order rotated across questions. Provider defaults otherwise applied; Groq reasoning and Anthropic thinking were disabled. '
             'Identical text does not imply identical token counts because tokenizers differ.', '',
             '| Provider/model | API success | Normal completion | Mean TTFT (s) | Mean generation time (s) | Mean cost/query (USD) |',
             '|---|---:|---:|---:|---:|---:|']
    known = 0
    for provider in ('openai', 'anthropic', 'groq'):
        group = [r for r in records if r['provider'] == provider]
        amounts = [cost(r['calls'], prices)['generation_cost_usd'] for r in group]
        price = 'N/A'
        if all(x is not None for x in amounts):
            price = f'{mean(amounts):.8f}'
            known += sum(amounts)
        lines.append(f"| {provider}/{group[0]['model']} | 3/3 | 3/3 | "
                     f"{mean(r['ttft_ms'] for r in group)/1000:.3f} | "
                     f"{mean(r['total_ms'] for r in group)/1000:.3f} | {price} |")
    lines += ['', '| Question | Provider | TTFT (s) | Generation (s) | Pre-request wait (s) | Cost (USD) |',
              '|---|---|---:|---:|---:|---:|']
    for r in sorted(records, key=lambda r: (r['id'], r['provider'])):
        amount = cost(r['calls'], prices)['generation_cost_usd']
        price = 'N/A' if amount is None else f'{amount:.8f}'
        lines.append(f"| {r['id']} | {r['provider']} | {r['ttft_ms']/1000:.3f} | "
                     f"{r['total_ms']/1000:.3f} | {r['pre_request_wait_seconds']:.3f} | {price} |")
    wait = sum(r['pre_request_wait_seconds'] for r in records)
    lines += ['', f'All nine calls succeeded and ended normally: no truncation or rate-limit errors were recorded. '
              f'Groq requests were paced at least 65 seconds apart, with **{wait:.3f} seconds** of cumulative pre-request waiting. '
              'These waits are excluded from the generation/TTFT columns; consequently, Groq’s lower generation latency does not demonstrate unrestricted throughput or the same latency for a queued user request. '
              'Single-trial timings also include local network variability.', '',
              f'The known round-2 comparison cost subtotal is **${known:.8f}** (OpenAI and Anthropic only; not a complete three-provider total). '
              'Groq cost: The dashboard showed approximately $0.23 on October 26 and $0.02 on October 27, as reported by the project owner. These amounts include development and testing beyond this comparison. The cost attributable to the three Groq comparison requests could not be isolated, so it is reported as N/A. Token usage is recorded for reproducibility. The approximately $0.25 account-wide amount is not added to the comparison subtotal. '
              'These comparison costs are additional to the five-question workflow/RAGAS totals above; round 1 is excluded.', '',
              '### Qualitative evidence review', '',
              'The following is an assistant-assisted qualitative inspection of saved answers against the supplied excerpts, not another RAGAS run or an independent human-rated benchmark. '
              'Review criteria are relevance, evidence support, preservation of qualifications, and citation placement. '
              'API success and normal completion are not factual-quality scores; no numerical quality ranking is inferred.', '',
              '| Case | Observation | Evidence and implication |', '|---|---|---|',
              '| OpenAI, rag-01 | Ambiguous statistical qualification | The answer contrasts Spanish shared singletons that “improved” with unshared singletons that did not reach significance. S2 reports shared p=.014 with adjusted alpha=.0063, so the shared change is not significant either. The numerical increase must be distinguished from statistical significance. |',
              '| Claude, rag-03 | Citation placement error | Combined English/Spanish PCC-R AUC=.99 is correct, but the answer attaches S1; the combination result is in S3. S1 contains the individual-language PCC-R values. |',
              '| Qwen, rag-03 | Measure conflation and overstatement | Spanish AUC=1.00 refers to the multivariate combination, not Spanish PCC-R alone (.99). The answer mixes these measures and says “significantly increased” although the supplied passage does not establish a significance test for that improvement. |',
              '| All providers, rag-02 | Main requested findings covered | Answers capture the exposure-dependent input-diversity pattern and lack of credible bilingual/monolingual difference. Claude explicitly distinguishes proposed mechanisms from tested findings. This observation is not a claim that every sentence has passed a separate verifier. |', '',
              '### Interpretation and model allocation', '',
              'For this small sample, Haiku was cheaper and faster on average than Nano under the supplied prices; paced Qwen had the fastest generation. '
              'The current allocation remains a reasonable coursework choice: Nano for planning, Haiku for evidence synthesis, and Qwen for simpler explanations with deterministic tools performing statistics. '
              'However, this generator-only experiment did not directly evaluate Supervisor routing or Data-agent tool use, and does not establish an overall best model. '
              'The factual and citation issues above support retaining source inspection and clearly labeling uncertain claims.', '',
              'Round 1 remains preserved in `artifacts/evaluation/provider_comparison1/`: Claude hit its 500-token cap in two cases, and Groq had two rate-limit failures. '
              'Round 2 changed both the common length instruction/cap and Groq pacing. Its better completion rate is consistent with those adjustments, but the experiment does not isolate their causal effects. '
              'The two rounds must not be pooled or cherry-picked as repeated trials under identical settings.', '',
              'Provenance: `artifacts/evaluation/provider_comparison2/config.json`, the nine `rag-XX-provider.json` records, and `comparison.md`. '
              'The saved config contains the exact prompts, evidence, model IDs, and signature; references were retained for review, not sent to models.', '']
    return lines

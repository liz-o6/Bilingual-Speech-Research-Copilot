"""Generate once and verify once without automatic rewriting."""
from src.llm.retry import invoke_with_retry
from src.tools.sources import check_citations, verify_citations, extract_cited_claims


async def generate_cited_answer(question, registry, llm, verification_method=None, verify=True, on_token=None):
    prompt = registry.prompt(question) + (
        '\nKeep the answer short: at most three factual paragraphs, with inline citations. '
        'Include essential qualifications in the same paragraph. Do not add a Sources section. '
        'If evidence is insufficient, say so. Do not invent missing findings.')
    if on_token is None:
        response = await invoke_with_retry(llm, prompt)
    else:
        # Stream candidate answer text, not verification JSON.
        # Do not retry failed streams and duplicate output or charges.
        response = None
        async for chunk in llm.astream(prompt):
            if chunk.text:
                on_token(chunk.text)
            response = chunk if response is None else response + chunk
        if response is None:
            raise ValueError("Empty model stream")
    answer = response.content if isinstance(response.content, str) else response.text
    metadata = getattr(response, 'response_metadata', {}) or {}
    finish_reason = metadata.get('stop_reason') or metadata.get('finish_reason')
    checks = check_citations(answer, registry)
    truncated = finish_reason in ('max_tokens', 'length')
    valid_ids = bool(checks) and all(c['status'] != 'invalid' for c in checks)
    # Check IDs for intermediate Research; defer semantic checks to Summary.
    accepted = valid_ids and bool(answer.strip()) and not truncated
    if verify and accepted:
        checks = await verify_citations(answer, registry, llm,
                                        method=verification_method, batch_size=None)
        accepted = bool(checks) and all(c['status'] == 'supported' for c in checks)
    if not answer.strip() or truncated:
        checks.append({'claim': '[incomplete output]', 'citation_id': '', 'status': 'unverified',
                       'reason': 'No visible answer or output limit reached; no automatic retry.'})
    displayed_answer = annotate_review(answer, checks) if verify and not accepted else answer
    return {'answer': displayed_answer, 'citation_checks': checks, 'review_required': not accepted,
            'verification_deferred': not verify,
            'attempts': [{'answer': answer, 'citation_checks': checks,
                          'answer_usage': response.usage_metadata, 'finish_reason': finish_reason}]}


def review_message(checks):
    """Distinguish insufficient evidence from incomplete verification."""
    names = {'unsupported': 'Insufficient evidence', 'unverified': 'Verification incomplete', 'invalid': 'Unknown citation ID'}
    messages = []
    for check in checks:
        if check.get('status') == 'supported':
            continue
        label = check.get('check_id') or check.get('citation_id') or 'Output'
        messages.append(f"{label}：{names.get(check.get('status'), 'Pending verification')}（{check.get('reason', 'No reason provided')}）")
    return '；'.join(messages) or 'No verifiable citations; the answer is retained for inspection.'


def annotate_review(answer, checks):
    """Keep the text and label uncertain passages explicitly."""
    lines = []
    for line in answer.splitlines():
        claims = {claim for claim, _ in extract_cited_claims(line)}
        issues = [c for c in checks if c.get('claim') in claims
                  and c.get('status') != 'supported']
        if issues:
            label = 'Insufficient evidence; not a confirmed finding' if any(c['status'] in ('unsupported', 'invalid') for c in issues) else 'Unverified'
            line = f'【{label}】' + line
        lines.append(line)
    text = '\n'.join(lines)
    if text.strip():
        text = '【Partial result: generated content is retained below; unverified content is not a confirmed finding.】\n\n' + text
    return text

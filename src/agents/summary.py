"""Summary: read full papers/pages, preserve evidence, and synthesize conclusions."""
import json
from hashlib import sha256

from src.llm import create_llm
from src.mcp.tavily import search_web
from src.orchestration.state import DocumentRequest, SummaryResult
from src.rag.answer import generate_cited_answer, review_message
from src.tools.sources import SourceRegistry, read_document, extract_cited_claims

LONG_TEXT_CHARS = 12000
MAX_CONTEXT_CHARS = 240000


def choose_provider(document_count, text_length):
    if document_count > 1:
        return 'anthropic', 'Cross-paper synthesis requires comparing multiple documents.'
    if text_length > LONG_TEXT_CHARS:
        return 'anthropic', 'Source text exceeds 12000 characters; use long-text synthesis.'
    return 'groq', 'Short single-document or evidence summary; prioritize cost.'


async def summary(state, llm=None, document_reader=None, web_search=None, traces=None):
    """Read State and return a separate SummaryResult."""
    if not state.summary_task.strip():
        return SummaryResult(status='failed', error='Summary has no assigned task.')
    if document_reader is None:
        document_reader = read_document
    if web_search is None:
        web_search = search_web

    registry = SourceRegistry()
    warnings = []
    provider = ''
    reason = ''
    try:
        # 1. Keep source IDs; other agents' explanations are not primary evidence.
        if state.summary_result is not None and state.summary_mode == "documents":
            registry.restore(state.summary_result.sources)
        if state.research_result is not None:
            sources = state.research_result.sources
            if state.summary_mode == "shared":
                labels = {label for _, label in extract_cited_claims(state.research_result.answer)}
                sources = {label: source for label, source in sources.items() if label in labels}
            registry.restore(sources)
        if state.data_result is not None:
            if state.summary_mode != "shared":
                registry.restore(state.data_result.sources)
            # Use tool-computed statistics, not model explanations, as evidence.
            if state.data_result.rows:
                text = json.dumps({'task': state.data_task, 'rows': state.data_result.rows,
                                   'scope': 'Local manifest only; counts refer to papers.'}, ensure_ascii=False)
                identity = sha256(text.encode()).hexdigest()
                registry.register([{
                    'chunk_id': 'data-result:' + identity, 'text': text,
                    'title': 'Computed local metadata query results',
                    'source_type': 'data_result', 'source_url': '',
                }])

        # 2. Reuse shared evidence; read extra documents only when requested.
        documents = []
        if state.summary_mode == "documents":
            documents = list(state.summary_documents)

        for document in documents:
            try:
                pages = document_reader(document.doc_id, page=document.page)
                if not pages:
                    warnings.append('A requested document/page returned no text.')
                usable_pages = 0
                for page in pages:
                    metadata = page['metadata']
                    text = page['text']
                    if not text.strip():
                        continue
                    usable_pages += 1
                    identity = sha256((document.doc_id + str(metadata.get('page')) + text).encode()).hexdigest()
                    source = dict(metadata)
                    source['chunk_id'] = 'fulltext:' + identity
                    source['text'] = text
                    source['source_type'] = 'document_page'
                    registry.register([source])
                if pages and usable_pages == 0:
                    warnings.append('A requested document/page contained only empty text.')
            except Exception:
                warnings.append('A requested document/page could not be read.')

        # 3. Use shared MCP search only when current information is needed.
        if state.summary_use_web and state.summary_mode == "documents":
            try:
                query = state.query + '\n' + state.summary_task
                hits = await web_search(query, traces=traces, agent='Summary')
                registry.register(hits)
                if not hits:
                    warnings.append('Web search returned no usable evidence.')
            except Exception:
                warnings.append('Live web search is unavailable.')

        if not registry.sources:
            return SummaryResult(status='partial', error='No usable sources. ' + ' '.join(warnings))

        # 4. Choose a model by document count and text length.
        document_ids = set()
        text_length = 0
        for source in registry.sources.values():
            if source.get('doc_id'):
                document_ids.add(source['doc_id'])
            elif source.get('source_type') == 'web':
                document_ids.add(source['source_url'])
            text_length += len(source.get('text', ''))
        provider, reason = choose_provider(len(document_ids), text_length)
        if traces is not None:
            traces.append({'event':'llm_route', 'agent':'Summary', 'provider':provider,
                           'reason':reason, 'document_count':len(document_ids),
                           'source_characters':text_length})

        # Ask for a smaller scope instead of silently truncating full text.
        if len(registry.context()) > MAX_CONTEXT_CHARS:
            return SummaryResult(status='partial', retryable=False, sources=registry.sources,
                                 provider=provider, route_reason=reason,
                                 error='Context is too large. Start a new request with fewer papers or specific pages.')
        if llm is None:
            llm = create_llm(provider, traces=traces, agent='Summary', reason=reason, max_tokens=2048)

        # 5. Synthesize and check citations using Claude's native JSON schema.
        task = 'User question: ' + state.query + '\nSummary task: ' + state.summary_task
        if state.research_result is not None:
            task += '\nResearch draft (not independent evidence): ' + state.research_result.answer
        task += ('\nGive a concise final answer: one short paragraph for the computed count, '
                 'then at most two short research findings with essential limitations and citations. '
                 'Do not enumerate all metadata records, years or statistics unless the user requests them. '
                 'Use Data rows for counts; a Research remark about missing statistics cannot override Data rows. '
                 'Synthesize conclusions from the supplied evidence. Preserve citation IDs, '
                 'distinguish agreements, conflicts and limitations. Do not infer causality, '
                 'invent missing study data or recompute the supplied statistics. '
                 'Use the language of the user question. Only claim to have read the supplied pages.')
        method = "function_calling"
        if provider == 'anthropic':
            method = 'json_schema'
        result = await generate_cited_answer(task, registry, llm, verification_method=method,
                                             on_token=getattr(traces, "emit_token", None))
        status = 'success'
        if result['review_required']:
            warnings.append(review_message(result['citation_checks']))
        if warnings:
            status = 'partial'
        error = None
        if warnings:
            error = ' '.join(warnings)
        return SummaryResult(status=status, answer=result['answer'], sources=registry.sources,
                             citation_checks=result['citation_checks'], provider=provider,
                             route_reason=reason, error=error)
    except Exception:
        return SummaryResult(status='failed', sources=registry.sources,
                             provider=provider, route_reason=reason,
                             error='Summary could not complete synthesis or citation verification.')

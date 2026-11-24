from src.observability.timing import timed
"""Search local papers or the web and answer with citations."""

import os
import re

from dotenv import load_dotenv

from src.mcp.tavily import search_web
from src.llm import create_llm
from src.llm.client import ROOT
from src.orchestration.state import ResearchResult
from src.rag.answer import generate_cited_answer, review_message
from src.rag.retriever import Retriever
from src.rag.selection import select_evidence
from src.rag.reading import read_result_pages
from src.tools.sources import extract_cited_claims
from src.tools.sources import SourceRegistry


def rate_limit_details(exc):
    """Extract allowed fields from parsed SDK errors, never the full message."""
    fields = {}
    current = exc
    for _ in range(3):
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                message = error.get("message", "")
                if isinstance(message, str):
                    for category in ("TPM", "TPD", "RPM", "RPD"):
                        if re.search(r"\b" + category + r"\b", message, re.I):
                            fields["limit_type"] = category
                    if "limit_type" not in fields:
                        if error.get("type") == "tokens":
                            fields["limit_type"] = "tokens (window unknown)"
                        elif error.get("type") == "requests":
                            fields["limit_type"] = "requests (window unknown)"
                    for label in ("Limit", "Used", "Requested"):
                        match = re.search(r"\b" + label + r"\s*:?\s*(\d+(?:\.\d+)?)\b", message, re.I)
                        if match:
                            fields[label.lower()] = match.group(1)
                    match = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)(?!\w)", message, re.I)
                    if match:
                        fields["retry_after"] = match.group(1)
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            # Accept numeric seconds only; do not expose other header content.
            value = headers.get("retry-after", "")
            if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?", value):
                fields["retry_after"] = value + "s"
        current = current.__cause__ or current.__context__
        if current is None:
            break
    return ", ".join(f"{name}={value}" for name, value in fields.items())


def error_details(exc):
    """Return error type, HTTP status, and filtered limit details."""
    details = type(exc).__name__
    current = exc
    # Inspect at most three levels to avoid exception-chain loops.
    for _ in range(3):
        response = getattr(current, "response", None)
        codes = [
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(response, "status_code", None),
            getattr(response, "status", None),
        ]
        for code in codes:
            if isinstance(code, int) and 100 <= code <= 599:
                details += f", HTTP {code}"
                if code == 429:
                    limits = rate_limit_details(exc)
                    if limits:
                        details += ", " + limits
                return details
        current = current.__cause__ or current.__context__
        if current is None:
            break
    return details


async def research(state, llm=None, retriever=None, traces=None, document_reader=None):
    """Return ResearchResult without changing the input State."""

    # 1. Check that a task was assigned.
    if not state.research_task.strip():
        return ResearchResult(status="failed", error="Research has no assigned task.")

    # Use a separate registry while preserving this request's source IDs.
    registry = SourceRegistry()

    stage = "restoring sources"
    try:
        if state.research_result is not None:
            registry.restore(state.research_result.sources)
        if state.summary_result is not None:
            registry.restore(state.summary_result.sources)

        # 2. Reuse the embedding configuration and Chroma index.
        stage = "initializing retriever"
        load_dotenv(ROOT / ".env", override=False)
        if retriever is None:
            embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
            retriever = Retriever(embedding_model=embedding_model, traces=traces)

        # 3. Keep the original question and task to preserve titles and scope.
        question = "Assigned Research question: " + state.research_task
        question += "\nResearch task: " + state.research_task
        question += (
            "\nComplete only the Research task. Do not answer counts assigned to Data or perform final synthesis. "
            "The original user language is indicated here: " + state.query + "\n"
            "For a request for two findings, return exactly two numbered paragraphs, each 2-3 short "
            "sentences with inline source IDs and essential limitations. Use the user's language. "
            "Do not append methods lists, p-value lists, quotations or author/year audits unless requested. "
            "A source ID already identifies the paper and page. Prefer observed results, not hypotheses "
            "or clinical advice. Preserve uncertainty and study attribution. If a finding lacks evidence, "
            "briefly state that gap instead of padding the answer."
        )
        stage = "retrieving local evidence"
        search_query = state.research_query.strip() or state.query
        # Filter by explicit document IDs instead of relying on embeddings.
        doc_ids = list(dict.fromkeys(re.findall(
            r"\bbcs-\d{3}\b", state.research_task + " " + search_query
        )))
        with timed(traces, "rag_retrieval", "Research"):
            candidates = []
            if doc_ids:
                groups = []
                for doc_id in doc_ids:
                    group = await retriever.asearch(search_query, k=20, doc_id=doc_id)
                    groups.append(select_evidence(group, k=4))
                # Alternate papers so one paper does not fill all four slots.
                for position in range(4):
                    for group in groups:
                        if position < len(group):
                            candidates.append(group[position])
            else:
                candidates = await retriever.asearch(search_query, k=20)
            hits = select_evidence(candidates, k=4)

            stage = "reading result pages"
            if document_reader is None:
                page_hits, reading_warnings = read_result_pages(hits)
            else:
                page_hits, reading_warnings = read_result_pages(hits, reader=document_reader)
            if page_hits:
                hits = page_hits
            if traces is not None:
                traces.append({'event': 'research_page_reading',
                               'pages': [(h.get('doc_id'), h.get('page')) for h in page_hits],
                               'warnings': reading_warnings})

        stage = "registering local sources"
        registry.register(hits)

        # 4. Use shared Tavily Remote MCP when current information is needed.
        web_error = None
        web_hits = []
        if state.research_use_web:
            try:
                web_hits = await search_web(question, traces=traces, agent="Research")
                registry.register(web_hits)
                if not web_hits:
                    web_error = "Web search returned no usable evidence."
            except Exception as exc:
                web_error = (f"Live web search failed ({error_details(exc)}); "
                             "only local or earlier evidence is available.")

        # Keep existing IDs but generate from the current evidence.
        current_hits = list(hits)
        if web_hits:
            current_hits = hits[:2] + web_hits[:2]
        active_ids = {hit['chunk_id'] for hit in current_hits}
        active_registry = SourceRegistry()
        for label, source in registry.sources.items():
            if source.get('chunk_id') in active_ids:
                active_registry.restore({label: source})


        if not active_registry.sources:
            return ResearchResult(
                status="partial", answer="No usable evidence was retrieved.",
                sources=registry.sources, error=web_error,
            )

        # Keep up to one previously cited page during follow-up retrieval.
        if state.research_result is not None and state.research_result.status == "success":
            for claim, label in extract_cited_claims(state.research_result.answer):
                if label in registry.sources and label not in active_registry.sources:
                    if len(active_registry.sources) < 4:
                        active_registry.restore({label: registry.sources[label]})
                    break
            question += ("\nPrevious draft (not independent evidence; retain only claims supported "
                         "by the CURRENT evidence): " + state.research_result.answer)

        if traces is not None:
            traces.append({'event': 'research_evidence', 'source_ids': list(active_registry.sources),
                           'candidate_count': len(candidates), 'doc_ids': doc_ids,
                           'sources': active_registry.sources})

        # 5. Use the shared client and Claude for synthesis and citation checks.
        stage = "initializing Anthropic model"
        if llm is None:
            llm = create_llm(
                "anthropic",
                traces=traces,
                agent="Research",
                reason="Paper evidence synthesis and distinction between findings and hypotheses",
                max_tokens=2048,
            )

        # 6. Defer semantic checks to the final Summary when planned.
        stage = "generating answer or checking citations"
        final_stage = "summary" not in state.agent_plan[state.plan_index + 1:]
        on_token = getattr(traces, "emit_token", None) if final_stage else None
        result = await generate_cited_answer(
            question, active_registry, llm, verification_method="json_schema",
            verify=final_stage, on_token=on_token
        )

        if traces is not None:
            traces.append({'event': 'research_verification',
                           'attempts': result.get('attempts', []),
                           'citation_checks': result.get('citation_checks', []),
                           'verification_deferred': result.get('verification_deferred', False),
                           'review_required': result['review_required']})

        if web_error is not None:
            return ResearchResult(
                status="partial", answer=result["answer"],
                sources=registry.sources, error=web_error, citation_checks=result.get("citation_checks", []),
            )

        if result["review_required"]:
            return ResearchResult(
                status="partial",
                answer=result["answer"],
                sources=registry.sources,
                error=review_message(result.get("citation_checks", [])),
                citation_checks=result.get("citation_checks", []),
            )

        return ResearchResult(
            status="success",
            citation_checks=result.get("citation_checks", []),
            answer=result["answer"],
            sources=registry.sources,
        )

    except Exception as exc:
        # Keep sources while hiding API keys and raw error messages.
        return ResearchResult(
            status="failed",
            sources=registry.sources,
            error=f"Research failed during {stage} ({error_details(exc)}).",
        )

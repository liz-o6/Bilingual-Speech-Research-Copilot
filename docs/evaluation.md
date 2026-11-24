# Evaluation: Local RAG Quality, Performance, and Cost

Measured on 2026-10-27. Five saved workflow runs completed; all 20 RAGAS scores are available. Costs below are token-based estimates in USD, not reconciled invoices. No model calls were made to produce this report.

## Dataset and procedure

The fixed English question set is `data/evaluation/local_rag_5.json`. Reference answers and source-page text were prepared from local PDFs. Scoring uses the actual contexts supplied to Research, not reference contexts injected into generation. This is a small coursework regression set, not a held-out benchmark.

RAGAS 0.3.9; judge `gpt-5.4-nano`; embedding `text-embedding-3-small`; answer-relevance strictness 1; judge output limit 4096. Questions 1–2 use run2/scores2 and questions 3–5 use run3/scores3. Question 2 faithfulness was retried after a parsing failure. JSON mode was enabled for that retry and for questions 3–5; questions 1 and the other question-2 metrics used the earlier format. Thus judge formatting was not identical across all measurements. Original artifacts, attempts, hashes, and configurations are preserved.

Each question was collected once in the selected runs. Source hashes differ between run2 and run3 after evaluation-diagnostics changes; see their run_config.json files. Earlier run1 and development experiments are excluded from these totals.

- **rag-01**: In the longitudinal Head Start study of Spanish-English bilingual preschoolers, how did speech accuracy and error rates change over one year, and did all sound categories improve? Use Research to retrieve local paper evidence, cite sources, and answer in English. Do not search the web.
- **rag-02**: In the study of English vowel perception in Spanish-English bilingual preschoolers, how did input diversity interact with English exposure, and how did bilingual vowel perception compare with monolingual perception? Use Research to retrieve local paper evidence, cite sources, and answer in English. Do not search the web.
- **rag-03**: For bilingual Spanish-English preschoolers, how did combining phonological measures within and across languages affect diagnostic accuracy, and was the same benefit found for monolingual children? Use Research to retrieve local paper evidence, cite sources, and answer in English. Do not search the web.
- **rag-04**: In the study of initial consonant deletion in Spanish-English bilingual children with speech sound disorders, what quantitative group differences were found and what did the qualitative analysis suggest? Use Research to retrieve local paper evidence, cite sources, and answer in English. Do not search the web.
- **rag-05**: In the community language exposure study of Spanish-English bilingual and functional English monolingual children, how did English voiced and voiceless stop VOT differ between groups, and what did similar pre-voicing suggest? Use Research to retrieve local paper evidence, cite sources, and answer in English. Do not search the web.

## RAGAS results

| Question | Faithfulness | Answer relevance | Context precision | Context recall |
|---|---:|---:|---:|---:|
| rag-01 | 0.6250 | 0.7582 | 1.0000 | 0.5000 |
| rag-02 | 1.0000 | 0.5985 | 0.3333 | 1.0000 |
| rag-03 | 1.0000 | 0.7312 | 0.5833 | 1.0000 |
| rag-04 | 0.8621 | 0.7347 | 0.8333 | 0.7500 |
| rag-05 | 0.8947 | 0.7112 | 1.0000 | 0.5000 |
| **Mean (5/5 each)** | 0.8764 | 0.7068 | 0.7500 | 0.7500 |

Completion does not mean every claim is correct. Question 1 has the lowest faithfulness; question 2 has the lowest context precision. Questions 1 and 5 have recall 0.5, indicating incomplete reference coverage under the judge. These results are reported without discarding low scores. Five examples and one judge configuration do not establish general reliability.

## Local latency

All values below are seconds. Retrieval includes query embedding, vector search, selection, and evidence expansion. LLM time sums provider-call durations. Total time measures the service workflow and excludes graph construction; it includes other orchestration overhead. Server TTFT is the first visible candidate text, before citation verification, not the first verified final answer. These are local service measurements, not browser/network TTFT or Cloud Run cold/warm measurements.

| Question | Retrieval | LLM total | Workflow total | Server TTFT | LLM calls |
|---|---:|---:|---:|---:|---:|
| rag-01 | 1.077 | 12.564 | 15.061 | 8.761 | 3 |
| rag-02 | 0.237 | 5.864 | 6.142 | 2.922 | 3 |
| rag-03 | 0.734 | 7.983 | 10.359 | 5.615 | 3 |
| rag-04 | 0.333 | 7.976 | 8.356 | 3.503 | 3 |
| rag-05 | 0.473 | 7.212 | 7.734 | 3.660 | 3 |
| **Mean** | 0.571 | 8.320 | 9.530 | 4.892 | 3 |

## Pricing basis

Rates are USD per million tokens, dated 2026-10-27. Generation rates below are transcribed from the user-provided pricing screenshots; they are not a claim of independent invoice verification. The screenshots show Haiku rates for prompts up to 100K tokens; every recorded Anthropic request is below that threshold.

| Model | Input | Output | Cached input | Source |
|---|---:|---:|---:|---|
| gpt-5.4-nano | 0.20 | 1.25 | 0.02 | User screenshot, 11:21:50 AM |
| claude-haiku-5-5 | 0.10 | 0.50 | 0.01 | User screenshot, 11:18:27 AM |
| text-embedding-3-small | 0.02 | N/A | N/A | [Official model documentation](https://developers.openai.com/api/docs/models/text-embedding-3-small) |

OpenAI cost = ((input tokens − cached input tokens) × input rate + cached input tokens × cached rate + output tokens × output rate) / 1,000,000. There are no Anthropic cache reads/writes in these selected records, so its cost uses ordinary input/output rates. RAGAS question 2 includes 3,840 cached input tokens, billed at the supplied cached rate. Embedding cost = input tokens × 0.02 / 1,000,000.

## Per-question API cost

Query cost includes the Supervisor, Research answer generation, citation verification, and query embedding. RAGAS cost is separate and includes all recorded judge calls (including the question-2 failed scoring attempt and retry) plus evaluation embeddings. API-successful calls that produced unparseable scoring output still incur cost. All required usage fields are present.

| Question | Generation | Query embedding | **Query total** | RAGAS judge | RAGAS embedding | **RAGAS total** |
|---|---:|---:|---:|---:|---:|---:|
| rag-01 | $0.00286115 | $0.00000038 | $0.00286153 | $0.00912530 | $0.00000174 | $0.00912704 |
| rag-02 | $0.00215955 | $0.00000052 | $0.00216007 | $0.01048550 | $0.00000190 | $0.01048740 |
| rag-03 | $0.00214795 | $0.00000042 | $0.00214837 | $0.00548825 | $0.00000160 | $0.00548985 |
| rag-04 | $0.00217395 | $0.00000034 | $0.00217429 | $0.00661445 | $0.00000176 | $0.00661621 |
| rag-05 | $0.00175645 | $0.00000044 | $0.00175689 | $0.00596255 | $0.00000224 | $0.00596479 |
| **Total** | $0.01109905 | $0.00000210 | $0.01110115 | $0.03767605 | $0.00000924 | $0.03768529 |

Five-query cost: **$0.01110115**; mean query cost: **$0.00222023**. RAGAS scoring cost: **$0.03768529**. Combined selected-run cost: **$0.04878644**.

Excluded: earlier development/reruns outside the selected files, one-time corpus ingestion embeddings, hosting, taxes, and other account/project usage. These estimates do not explain the entire account balance. The Groq dashboard monthly amount is not allocated to these five questions.

## Observed generation-provider usage

| Provider/model | Calls | Input tokens | Output tokens | Cumulative call time (s) | Cost (USD) |
|---|---:|---:|---:|---:|---:|
| openai/gpt-5.4-nano | 5 | 7621 | 1017 | 12.363 | 0.00279545 |
| anthropic/claude-haiku-5-5 | 10 | 67891 | 3029 | 29.236 | 0.00830360 |
| groq/qwen/qwen3.8-27b | 0 | 0 | 0 | 0 | 0 |

Groq cost is zero here because these questions did not call Groq, not because its plan is assumed free. A generic Llama pricing range cannot price this Qwen model. OpenAI performs planning and Anthropic performs synthesis/verification, so the table is a workload breakdown, not a controlled quality/speed comparison. RAGAS judge calls are excluded from this generation-provider table.


## Matched-input provider comparison (round 2)

The first three questions were reused with exactly the same saved evidence and shared prompt for each provider. Each provider received one streaming request per question (nine calls). The prompt requested 2–3 bullets and at most 120 English words; the output cap was 800 tokens. This is a requested length, not an enforced word-count constraint. No retrieval, embedding, Supervisor, citation-judge, or RAGAS calls were made during this comparison. Provider order rotated across questions. Provider defaults otherwise applied; Groq reasoning and Anthropic thinking were disabled. Identical text does not imply identical token counts because tokenizers differ.

| Provider/model | API success | Normal completion | Mean TTFT (s) | Mean generation time (s) | Mean cost/query (USD) |
|---|---:|---:|---:|---:|---:|
| openai/gpt-5.4-nano | 3/3 | 3/3 | 1.377 | 2.562 | 0.00103350 |
| anthropic/claude-haiku-5-5 | 3/3 | 3/3 | 0.914 | 1.926 | 0.00069463 |
| groq/qwen/qwen3.8-27b | 3/3 | 3/3 | 0.573 | 0.833 | N/A |

| Question | Provider | TTFT (s) | Generation (s) | Pre-request wait (s) | Cost (USD) |
|---|---|---:|---:|---:|---:|
| rag-01 | anthropic | 0.920 | 1.989 | 0.000 | 0.00086190 |
| rag-01 | groq | 0.660 | 0.911 | 58.124 | N/A |
| rag-01 | openai | 1.819 | 3.454 | 0.000 | 0.00135810 |
| rag-02 | anthropic | 0.836 | 2.237 | 0.000 | 0.00066040 |
| rag-02 | groq | 0.421 | 0.738 | 61.828 | N/A |
| rag-02 | openai | 1.220 | 2.247 | 0.000 | 0.00085875 |
| rag-03 | anthropic | 0.986 | 1.551 | 0.000 | 0.00056160 |
| rag-03 | groq | 0.637 | 0.852 | 61.982 | N/A |
| rag-03 | openai | 1.093 | 1.985 | 0.000 | 0.00088365 |

All nine calls succeeded and ended normally: no truncation or rate-limit errors were recorded. Groq requests were paced at least 65 seconds apart, with **181.935 seconds** of cumulative pre-request waiting. These waits are excluded from the generation/TTFT columns; consequently, Groq’s lower generation latency does not demonstrate unrestricted throughput or the same latency for a queued user request. Single-trial timings also include local network variability.

The known round-2 comparison cost subtotal is **$0.00518440** (OpenAI and Anthropic only; not a complete three-provider total). Groq cost: The dashboard showed approximately $0.23 on October 26 and $0.02 on October 27, as reported by the project owner. These amounts include development and testing beyond this comparison. The cost attributable to the three Groq comparison requests could not be isolated, so it is reported as N/A. Token usage is recorded for reproducibility. The approximately $0.25 account-wide amount is not added to the comparison subtotal. These comparison costs are additional to the five-question workflow/RAGAS totals above; round 1 is excluded.

### Qualitative evidence review

The following is an assistant-assisted qualitative inspection of saved answers against the supplied excerpts, not another RAGAS run or an independent human-rated benchmark. Review criteria are relevance, evidence support, preservation of qualifications, and citation placement. API success and normal completion are not factual-quality scores; no numerical quality ranking is inferred.

| Case | Observation | Evidence and implication |
|---|---|---|
| OpenAI, rag-01 | Ambiguous statistical qualification | The answer contrasts Spanish shared singletons that “improved” with unshared singletons that did not reach significance. S2 reports shared p=.014 with adjusted alpha=.0063, so the shared change is not significant either. The numerical increase must be distinguished from statistical significance. |
| Claude, rag-03 | Citation placement error | Combined English/Spanish PCC-R AUC=.99 is correct, but the answer attaches S1; the combination result is in S3. S1 contains the individual-language PCC-R values. |
| Qwen, rag-03 | Measure conflation and overstatement | Spanish AUC=1.00 refers to the multivariate combination, not Spanish PCC-R alone (.99). The answer mixes these measures and says “significantly increased” although the supplied passage does not establish a significance test for that improvement. |
| All providers, rag-02 | Main requested findings covered | Answers capture the exposure-dependent input-diversity pattern and lack of credible bilingual/monolingual difference. Claude explicitly distinguishes proposed mechanisms from tested findings. This observation is not a claim that every sentence has passed a separate verifier. |

### Interpretation and model allocation

For this small sample, Haiku was cheaper and faster on average than Nano under the supplied prices; paced Qwen had the fastest generation. The current allocation remains a reasonable coursework choice: Nano for planning, Haiku for evidence synthesis, and Qwen for simpler explanations with deterministic tools performing statistics. However, this generator-only experiment did not directly evaluate Supervisor routing or Data-agent tool use, and does not establish an overall best model. The factual and citation issues above support retaining source inspection and clearly labeling uncertain claims.

Round 1 remains preserved in `artifacts/evaluation/provider_comparison1/`: Claude hit its 500-token cap in two cases, and Groq had two rate-limit failures. Round 2 changed both the common length instruction/cap and Groq pacing. Its better completion rate is consistent with those adjustments, but the experiment does not isolate their causal effects. The two rounds must not be pooled or cherry-picked as repeated trials under identical settings.

Provenance: `artifacts/evaluation/provider_comparison2/config.json`, the nine `rag-XX-provider.json` records, and `comparison.md`. The saved config contains the exact prompts, evidence, model IDs, and signature; references were retained for review, not sent to models.

## Cloud Run cold and warm measurements

Deployed service: https://hw2-research-copilot-2pfqauay5a-uw.a.run.app. The project owner reported that all three default UI questions returned answers. The measurements below are four sequential calls to `/api/chat` using the same local-RAG question, with no health check, homepage request, or retry before them. Each call used a new HTTP client.

Raw records: `artifacts/evaluation/cloud/20261027T215158613801Z/request-01.json` through `request-04.json`. Original records retain their initial unverified classification; this report adds classification based on the owner-provided Cloud Run log screenshots supplied in the conversation. The temporary screenshot files were no longer readable when this report was generated; retain the originals with the submission.

The first request appears at 2026-10-27 21:51:58.901 UTC (14:51:58.901 PDT). The logs show “Starting new instance” due to AUTOSCALING at 21:51:58.932, application startup complete at 21:52:08.298, and the startup probe succeeding at 21:52:08.311. This supports classifying request 1 as cold. Requests 2–4 follow immediately with no additional instance-start event shown and are classified as warm based on that timeline. Instance IDs are not visible in these screenshots.

| Request | Phase | Status | Client total (s) | Client TTFT (s) | Retrieval (s) | LLM (s) | Server total (s) | Server TTFT (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Cold | completed | 38.819 | 35.076 | 2.167 | 6.788 | 29.034 | 25.307 |
| 2 | Warm | partial | 7.800 | 4.057 | 0.542 | 7.040 | 7.663 | 3.940 |
| 3 | Warm | completed | 8.826 | 3.962 | 0.248 | 8.307 | 8.629 | 3.841 |
| 4 | Warm | completed | 7.157 | 3.313 | 0.690 | 6.272 | 7.033 | 3.201 |
| **Warm mean (n=3)** | Warm | 2 completed, 1 partial | 7.928 | 3.777 | 0.494 | 7.206 | 7.775 | 3.661 |

All four requests returned HTTP 200 and a terminal SSE event. Request 2 retained a partial answer because citation review flagged a claim; it is included in the timing mean rather than discarded. Transport success does not establish answer correctness. The cloud runs were not separately scored with RAGAS.

Cloud Run request-log durations were 38.326, 7.667, 8.633, and 7.035 seconds. These differ from client timing because measurement boundaries differ. The observed instance-start-to-application-ready interval is about 9.366 seconds; neither the entire 38.819-second request nor the client/server difference should be described as pure container startup. Initialization and orchestration overhead are also included in total time.

Client TTFT ends at the first visible candidate text, before verification; client total ends at the final SSE done event. Server workflow time excludes container startup. Retrieval includes query embedding and local retrieval work; LLM time sums recorded provider calls. One cold observation and three warm observations demonstrate the distinction but are not a latency distribution or SLA.

Log evidence: owner-provided screenshots dated 2026-10-27 at 3:02:40 PM and 3:02:59 PM (startup timeline and expanded request log). First platform request trace: `12df821f9c3d02086c987a1fb9d0b0a`. These are screenshot observations, not an exported instance-ID correlation.

## Submission evidence checklist

- Quantitative measurements are complete: five-question RAGAS, local timing and cost, three-provider comparison, and cloud cold/warm timing. Retain the startup screenshots with the report.
- Groq comparison cost is N/A because it could not be isolated from account-wide usage; add an attributable cost if exact pricing or billing becomes available.
- Include existing UI/multi-agent traces and evidence that both Research and Summary use the shared Tavily MCP adapter. Local RAG measurements alone do not establish that acceptance requirement; verify saved evidence before deciding whether a targeted test is needed.

## Reproduction and provenance

Run `python -m src.eval.final_report` from hw2 to rebuild this document and the machine-readable summary without API calls. Prices: `data/evaluation/prices.2026-10-27.json`. Summary: `artifacts/evaluation/final_local_5/summary.json`. Raw source files:

- `run2/rag-01.json` and `scores2/rag-01.json` under `artifacts/evaluation/`.
- `run2/rag-02.json` and `scores2/rag-02.json` under `artifacts/evaluation/`.
- `run3/rag-03.json` and `scores3/rag-03.json` under `artifacts/evaluation/`.
- `run3/rag-04.json` and `scores3/rag-04.json` under `artifacts/evaluation/`.
- `run3/rag-05.json` and `scores3/rag-05.json` under `artifacts/evaluation/`.

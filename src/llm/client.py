"""Create native LangChain models with shared configuration and logging."""
import os
from pathlib import Path

from dotenv import load_dotenv
from .logging import LLMCallLogger

ROOT = Path(__file__).resolve().parents[2]
VARIABLES = {
    'openai': ('OPENAI_API_KEY', 'OPENAI_MODEL'),
    'anthropic': ('ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL'),
    'groq': ('GROQ_API_KEY', 'GROQ_MODEL'),
    'google': ('GOOGLE_API_KEY', 'GEMINI_MODEL'),
}


def create_llm(provider, *, traces=None, agent='unspecified',
               reason='explicit provider selection', timeout=60,
               max_tokens=1024, max_retries=0):
    """Return a native model; agents own prompts, tools, schemas and errors.

    Pass a request-local list as traces to collect safe call metadata.
    SDK timeouts/retries are connection settings, not a workflow deadline.
    """
    if provider not in VARIABLES:
        raise ValueError(f'Unsupported provider: {provider}')
    load_dotenv(ROOT / '.env', override=False)
    key_var, model_var = VARIABLES[provider]
    missing = [name for name in (key_var, model_var) if not os.getenv(name, '').strip()]
    if missing:
        raise ValueError('Missing configuration: ' + ', '.join(missing))
    if timeout <= 0 or max_tokens <= 0 or not 0 <= max_retries <= 2:
        raise ValueError('Require positive timeout/tokens and 0..2 retries')
    model = os.environ[model_var]
    options = dict(model=model, api_key=os.environ[key_var], timeout=timeout,
                   max_tokens=max_tokens, max_retries=max_retries)
    if traces is not None:
        options['callbacks'] = [LLMCallLogger(traces, provider, model, agent, reason)]
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**options, stream_usage=True)
    if provider == 'groq':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            **options, base_url='https://api.groq.com/openai/v1',
            reasoning_effort='none', use_responses_api=False, stream_usage=False,
        )
    if provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**options, thinking={"type": "disabled"})
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(**options, vertexai=False)

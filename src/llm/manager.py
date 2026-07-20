from config.settings import (
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    llm_ready,
)

from src.llm.base import BaseLLMProvider
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.openai_provider import OpenAIProvider


_PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def get_llm_provider() -> BaseLLMProvider | None:
    """
    Return the selected LLM provider.

    Returns None when LLM usage is disabled or the selected provider
    is not ready.
    """
    if not LLM_ENABLED:
        return None

    if not llm_ready():
        return None

    provider_class = _PROVIDER_CLASSES.get(
        LLM_PROVIDER
    )

    if provider_class is None:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        )

    return provider_class()


def generate_text(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str | None:
    """
    Generate text with the configured LLM provider.

    Returns None when the LLM is unavailable or when generation fails.
    """
    provider = get_llm_provider()

    if provider is None:
        return None

    selected_max_tokens = (
        max_tokens
        if max_tokens is not None
        else LLM_MAX_TOKENS
    )

    selected_temperature = (
        temperature
        if temperature is not None
        else LLM_TEMPERATURE
    )

    try:
        response = provider.generate(
            prompt=prompt,
            max_tokens=selected_max_tokens,
            temperature=selected_temperature,
        )

        cleaned_response = response.strip()

        return cleaned_response or None

    except Exception:
        return None


def get_llm_runtime_info() -> dict[str, str | bool]:
    """
    Return a safe runtime summary without exposing API keys.
    """
    return {
        "enabled": LLM_ENABLED,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "ready": llm_ready(),
    }
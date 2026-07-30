from __future__ import annotations

from importlib import import_module
from typing import Type

from config.settings import (
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    llm_ready,
)

from src.llm.base import BaseLLMProvider


_PROVIDER_PATHS = {
    "anthropic": (
        "src.llm.providers.anthropic_provider",
        "AnthropicProvider",
    ),
    "openai": (
        "src.llm.providers.openai_provider",
        "OpenAIProvider",
    ),
    "gemini": (
        "src.llm.providers.gemini_provider",
        "GeminiProvider",
    ),
}


def _load_provider_class(
    provider_name: str,
) -> Type[BaseLLMProvider]:
    """
    Import only the selected LLM provider at runtime.

    This prevents an uninstalled provider SDK from breaking the whole
    application when another provider is selected.
    """
    normalized_provider = provider_name.lower().strip()

    provider_config = _PROVIDER_PATHS.get(
        normalized_provider
    )

    if provider_config is None:
        supported = ", ".join(
            sorted(_PROVIDER_PATHS)
        )

        raise ValueError(
            f"Unsupported LLM provider: {provider_name}. "
            f"Supported providers: {supported}"
        )

    module_path, class_name = provider_config

    try:
        provider_module = import_module(
            module_path
        )

        provider_class = getattr(
            provider_module,
            class_name,
        )

    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"The SDK required for provider "
            f"'{normalized_provider}' is not installed. "
            f"Install the selected provider dependency."
        ) from exc

    except AttributeError as exc:
        raise RuntimeError(
            f"Provider class '{class_name}' could not be found "
            f"in module '{module_path}'."
        ) from exc

    return provider_class


def get_llm_provider() -> BaseLLMProvider | None:
    """
    Return an instance of the selected LLM provider.

    Returns None when LLM usage is disabled or the selected provider
    does not have a valid configuration.
    """
    if not LLM_ENABLED:
        return None

    if not llm_ready():
        return None

    provider_class = _load_provider_class(
        LLM_PROVIDER
    )

    return provider_class()


def generate_text(
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str | None:
    """
    Generate text using only the configured LLM provider.

    Returns None when the provider is disabled, not configured,
    unavailable, or when generation fails.
    """
    try:
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

        response = provider.generate(
            prompt=prompt,
            max_tokens=selected_max_tokens,
            temperature=selected_temperature,
        )

        if not response:
            return None

        cleaned_response = response.strip()

        return cleaned_response or None

    except Exception:
        return None


def get_llm_runtime_info() -> dict[str, str | bool]:
    """
    Return a safe LLM runtime summary without exposing credentials.
    """
    provider_name = LLM_PROVIDER.lower().strip()

    provider_supported = (
        provider_name in _PROVIDER_PATHS
    )

    return {
        "enabled": LLM_ENABLED,
        "provider": provider_name,
        "model": LLM_MODEL,
        "ready": (
            LLM_ENABLED
            and provider_supported
            and llm_ready()
        ),
        "provider_supported": provider_supported,
    }
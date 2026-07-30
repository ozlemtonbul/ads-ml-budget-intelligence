from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# Environment helpers
# ============================================================

def _get_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)

    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid number. "
            f"Received: {value}"
        ) from exc


def _get_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)

    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer. "
            f"Received: {value}"
        ) from exc


def _get_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _has_value(
    value: str | None,
) -> bool:
    return (
        value is not None
        and value.strip() != ""
    )


def require_env(
    value: str | None,
    name: str,
) -> str:
    if not _has_value(value):
        raise ValueError(
            f"Missing environment variable: {name}"
        )

    return value.strip()


# ============================================================
# Google Ads
# ============================================================

GOOGLE_ADS_DEVELOPER_TOKEN = os.getenv(
    "GOOGLE_ADS_DEVELOPER_TOKEN"
)

GOOGLE_ADS_CLIENT_ID = os.getenv(
    "GOOGLE_ADS_CLIENT_ID"
)

GOOGLE_ADS_CLIENT_SECRET = os.getenv(
    "GOOGLE_ADS_CLIENT_SECRET"
)

GOOGLE_ADS_REFRESH_TOKEN = os.getenv(
    "GOOGLE_ADS_REFRESH_TOKEN"
)

GOOGLE_ADS_LOGIN_CUSTOMER_ID = os.getenv(
    "GOOGLE_ADS_LOGIN_CUSTOMER_ID"
)

GOOGLE_ADS_CUSTOMER_ID = os.getenv(
    "GOOGLE_ADS_CUSTOMER_ID"
)


def google_ads_ready() -> bool:
    required_values = [
        GOOGLE_ADS_DEVELOPER_TOKEN,
        GOOGLE_ADS_CLIENT_ID,
        GOOGLE_ADS_CLIENT_SECRET,
        GOOGLE_ADS_REFRESH_TOKEN,
        GOOGLE_ADS_LOGIN_CUSTOMER_ID,
        GOOGLE_ADS_CUSTOMER_ID,
    ]

    return all(
        _has_value(value)
        for value in required_values
    )


# ============================================================
# Google Analytics 4
# ============================================================

GA4_PROPERTY_ID = os.getenv(
    "GA4_PROPERTY_ID"
)

GA4_SERVICE_ACCOUNT_FILE = os.getenv(
    "GA4_SERVICE_ACCOUNT_FILE"
)


def ga4_ready() -> bool:
    required_values = [
        GA4_PROPERTY_ID,
        GA4_SERVICE_ACCOUNT_FILE,
    ]

    return all(
        _has_value(value)
        for value in required_values
    )


# ============================================================
# Date range
# ============================================================

SUPPORTED_DATE_MODES = {
    "custom",
    "yesterday",
    "last_30_days",
    "last_60_days",
}

DATE_MODE = os.getenv(
    "DATE_MODE",
    "last_60_days",
).strip().lower()

DATE_FROM = os.getenv(
    "DATE_FROM"
)

DATE_TO = os.getenv(
    "DATE_TO"
)

if DATE_MODE not in SUPPORTED_DATE_MODES:
    raise ValueError(
        "DATE_MODE must be one of: "
        f"{', '.join(sorted(SUPPORTED_DATE_MODES))}. "
        f"Received: {DATE_MODE}"
    )


# ============================================================
# Business target
# ============================================================

TARGET_ROAS = _get_float(
    "TARGET_ROAS",
    3.0,
)


# ============================================================
# Large Language Model
# ============================================================

SUPPORTED_LLM_PROVIDERS = {
    "anthropic",
    "openai",
    "gemini",
}

SUPPORTED_LLM_PROVIDER_SETTINGS = {
    *SUPPORTED_LLM_PROVIDERS,
    "auto",
}

DEFAULT_LLM_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.1",
    "gemini": "gemini-2.5-pro",
}


LLM_ENABLED = _get_bool(
    "LLM_ENABLED",
    False,
)

CONFIGURED_LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "auto",
).strip().lower()

if (
    CONFIGURED_LLM_PROVIDER
    not in SUPPORTED_LLM_PROVIDER_SETTINGS
):
    raise ValueError(
        "LLM_PROVIDER must be one of: "
        f"{', '.join(sorted(SUPPORTED_LLM_PROVIDER_SETTINGS))}. "
        f"Received: {CONFIGURED_LLM_PROVIDER}"
    )

CONFIGURED_LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "",
).strip()

LLM_LANGUAGE = os.getenv(
    "LLM_LANGUAGE",
    os.getenv(
        "LLM_LANG",
        "tr",
    ),
).strip().lower()

LLM_MAX_CAMPAIGNS = _get_int(
    "LLM_MAX_CAMPAIGNS",
    20,
)

LLM_MAX_TOKENS = _get_int(
    "LLM_MAX_TOKENS",
    1200,
)

LLM_TEMPERATURE = _get_float(
    "LLM_TEMPERATURE",
    0.2,
)


# ============================================================
# LLM API keys
# ============================================================

ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


def get_llm_provider_keys() -> dict[str, str | None]:
    """
    Return LLM API keys for internal readiness checks.

    API key values must never be displayed by the dashboard.
    """

    return {
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
        "gemini": GEMINI_API_KEY,
    }


def resolve_llm_provider(
    configured_provider: str,
    provider_keys: dict[str, str | None],
) -> str:
    """
    Resolve the active LLM provider.

    When LLM_PROVIDER=auto:
    - One configured key selects its provider.
    - No configured key keeps deterministic mode active.
    - Multiple configured keys require an explicit provider.
    """

    normalized_provider = (
        configured_provider
        .strip()
        .lower()
    )

    if normalized_provider != "auto":
        return normalized_provider

    configured_providers = [
        provider
        for provider, api_key in provider_keys.items()
        if _has_value(api_key)
    ]

    if len(configured_providers) == 1:
        return configured_providers[0]

    return "auto"


LLM_PROVIDER = resolve_llm_provider(
    CONFIGURED_LLM_PROVIDER,
    get_llm_provider_keys(),
)

LLM_MODEL = (
    CONFIGURED_LLM_MODEL
    or DEFAULT_LLM_MODELS.get(
        LLM_PROVIDER,
        "",
    )
)


def anthropic_ready() -> bool:
    return _has_value(
        ANTHROPIC_API_KEY
    )


def openai_ready() -> bool:
    return _has_value(
        OPENAI_API_KEY
    )


def gemini_ready() -> bool:
    return _has_value(
        GEMINI_API_KEY
    )


def selected_llm_api_key() -> str | None:
    return get_llm_provider_keys().get(
        LLM_PROVIDER
    )


def llm_ready() -> bool:
    """
    Return whether live LLM generation is available.

    Deterministic analytics remain available when False.
    """

    if not LLM_ENABLED:
        return False

    if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
        return False

    if LLM_MODEL == "":
        return False

    return _has_value(
        selected_llm_api_key()
    )


def llm_status() -> dict[str, str | bool | int]:
    provider_keys = get_llm_provider_keys()

    configured_key_count = sum(
        1
        for api_key in provider_keys.values()
        if _has_value(api_key)
    )

    return {
        "enabled": LLM_ENABLED,
        "configured_provider": (
            CONFIGURED_LLM_PROVIDER
        ),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "api_key_configured": _has_value(
            selected_llm_api_key()
        ),
        "configured_key_count": (
            configured_key_count
        ),
        "ready": llm_ready(),
    }


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.getenv(
    "VICCO_OUTPUT_DIR",
    os.getenv(
        "OUTPUT_DIR",
        "./outputs",
    ),
).strip()


# ============================================================
# PostgreSQL
# ============================================================

POSTGRES_ENABLED = _get_bool(
    "POSTGRES_ENABLED",
    False,
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER"
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD"
)

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST"
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "5432",
).strip()

POSTGRES_DB = os.getenv(
    "POSTGRES_DB"
)

POSTGRES_IF_EXISTS = os.getenv(
    "POSTGRES_IF_EXISTS",
    "replace",
).strip().lower()

SUPPORTED_POSTGRES_IF_EXISTS = {
    "replace",
    "append",
    "fail",
}

if (
    POSTGRES_IF_EXISTS
    not in SUPPORTED_POSTGRES_IF_EXISTS
):
    raise ValueError(
        "POSTGRES_IF_EXISTS must be one of: "
        f"{', '.join(sorted(SUPPORTED_POSTGRES_IF_EXISTS))}. "
        f"Received: {POSTGRES_IF_EXISTS}"
    )
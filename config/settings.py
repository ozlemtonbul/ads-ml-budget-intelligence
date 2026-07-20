import os

from dotenv import load_dotenv


load_dotenv()


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid number. Received: {value}"
        ) from exc


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer. Received: {value}"
        ) from exc


def _get_bool(name: str, default: bool = False) -> bool:
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


def require_env(
    value: str | None,
    name: str,
) -> str:
    if value is None or value.strip() == "":
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
        value is not None
        and value.strip() != ""
        for value in required_values
    )


# ============================================================
# GA4
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
        value is not None
        and value.strip() != ""
        for value in required_values
    )


# ============================================================
# Date range
# ============================================================

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

LLM_ENABLED = _get_bool(
    "LLM_ENABLED",
    True,
)

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "anthropic",
).strip().lower()

if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ValueError(
        "LLM_PROVIDER must be one of: "
        f"{', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}. "
        f"Received: {LLM_PROVIDER}"
    )

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "",
).strip()

LLM_LANGUAGE = os.getenv(
    "LLM_LANGUAGE",
    os.getenv(
        "LLM_LANG",
        "en",
    ),
).strip()

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


def _has_value(value: str | None) -> bool:
    return (
        value is not None
        and value.strip() != ""
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
    provider_keys = {
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
        "gemini": GEMINI_API_KEY,
    }

    return provider_keys.get(
        LLM_PROVIDER
    )


def llm_ready() -> bool:
    if not LLM_ENABLED:
        return False

    if LLM_MODEL == "":
        return False

    return _has_value(
        selected_llm_api_key()
    )


def llm_status() -> dict[str, str | bool]:
    return {
        "enabled": LLM_ENABLED,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "api_key_configured": _has_value(
            selected_llm_api_key()
        ),
        "ready": llm_ready(),
    }


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = os.getenv(
    "VICCO_OUTPUT_DIR",
    "./outputs",
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
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
# Anthropic
# ============================================================

ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY"
)


def anthropic_ready() -> bool:
    return (
        ANTHROPIC_API_KEY is not None
        and ANTHROPIC_API_KEY.strip() != ""
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
# LLM
# ============================================================

LLM_LANG = os.getenv(
    "LLM_LANG",
    "en",
).strip()

LLM_MAX_CAMPAIGNS = _get_int(
    "LLM_MAX_CAMPAIGNS",
    20,
)


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
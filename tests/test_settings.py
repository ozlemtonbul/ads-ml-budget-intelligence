import pytest

import config.settings as settings


# ============================================================
# require_env
# ============================================================

def test_require_env_returns_trimmed_value():
    result = settings.require_env(
        "  example-value  ",
        "TEST_VALUE",
    )

    assert result == "example-value"


def test_require_env_raises_for_none():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: TEST_VALUE",
    ):
        settings.require_env(
            None,
            "TEST_VALUE",
        )


def test_require_env_raises_for_empty_string():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: TEST_VALUE",
    ):
        settings.require_env(
            "",
            "TEST_VALUE",
        )


def test_require_env_raises_for_whitespace():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: TEST_VALUE",
    ):
        settings.require_env(
            "   ",
            "TEST_VALUE",
        )


# ============================================================
# _get_float
# ============================================================

def test_get_float_returns_default_when_missing(
    monkeypatch,
):
    monkeypatch.delenv(
        "TEST_FLOAT",
        raising=False,
    )

    result = settings._get_float(
        "TEST_FLOAT",
        3.5,
    )

    assert result == 3.5


def test_get_float_returns_default_when_empty(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_FLOAT",
        "   ",
    )

    result = settings._get_float(
        "TEST_FLOAT",
        4.5,
    )

    assert result == 4.5


def test_get_float_converts_valid_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_FLOAT",
        "7.25",
    )

    result = settings._get_float(
        "TEST_FLOAT",
        3.0,
    )

    assert result == 7.25


def test_get_float_raises_for_invalid_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_FLOAT",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="TEST_FLOAT must be a valid number",
    ):
        settings._get_float(
            "TEST_FLOAT",
            3.0,
        )


# ============================================================
# _get_int
# ============================================================

def test_get_int_returns_default_when_missing(
    monkeypatch,
):
    monkeypatch.delenv(
        "TEST_INT",
        raising=False,
    )

    result = settings._get_int(
        "TEST_INT",
        20,
    )

    assert result == 20


def test_get_int_returns_default_when_empty(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_INT",
        "",
    )

    result = settings._get_int(
        "TEST_INT",
        10,
    )

    assert result == 10


def test_get_int_converts_valid_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_INT",
        "35",
    )

    result = settings._get_int(
        "TEST_INT",
        20,
    )

    assert result == 35


def test_get_int_raises_for_decimal_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_INT",
        "4.5",
    )

    with pytest.raises(
        ValueError,
        match="TEST_INT must be a valid integer",
    ):
        settings._get_int(
            "TEST_INT",
            20,
        )


def test_get_int_raises_for_invalid_value(
    monkeypatch,
):
    monkeypatch.setenv(
        "TEST_INT",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="TEST_INT must be a valid integer",
    ):
        settings._get_int(
            "TEST_INT",
            20,
        )


# ============================================================
# _get_bool
# ============================================================

@pytest.mark.parametrize(
    "value",
    [
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "y",
        "Y",
        "on",
        "ON",
        " true ",
    ],
)
def test_get_bool_returns_true_for_supported_values(
    monkeypatch,
    value,
):
    monkeypatch.setenv(
        "TEST_BOOL",
        value,
    )

    result = settings._get_bool(
        "TEST_BOOL",
        False,
    )

    assert result is True


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "false",
        "no",
        "n",
        "off",
        "",
        "random",
    ],
)
def test_get_bool_returns_false_for_other_values(
    monkeypatch,
    value,
):
    monkeypatch.setenv(
        "TEST_BOOL",
        value,
    )

    result = settings._get_bool(
        "TEST_BOOL",
        True,
    )

    assert result is False


def test_get_bool_returns_false_default_when_missing(
    monkeypatch,
):
    monkeypatch.delenv(
        "TEST_BOOL",
        raising=False,
    )

    result = settings._get_bool(
        "TEST_BOOL",
        False,
    )

    assert result is False


def test_get_bool_returns_true_default_when_missing(
    monkeypatch,
):
    monkeypatch.delenv(
        "TEST_BOOL",
        raising=False,
    )

    result = settings._get_bool(
        "TEST_BOOL",
        True,
    )

    assert result is True


# ============================================================
# Google Ads readiness
# ============================================================

def test_google_ads_ready_returns_true_when_all_values_exist(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "developer-token",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CLIENT_ID",
        "client-id",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CLIENT_SECRET",
        "client-secret",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_REFRESH_TOKEN",
        "refresh-token",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        "1234567890",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CUSTOMER_ID",
        "9876543210",
    )

    assert settings.google_ads_ready() is True


def test_google_ads_ready_returns_false_when_value_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "developer-token",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CLIENT_ID",
        "client-id",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CLIENT_SECRET",
        "",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_REFRESH_TOKEN",
        "refresh-token",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        "1234567890",
    )
    monkeypatch.setattr(
        settings,
        "GOOGLE_ADS_CUSTOMER_ID",
        "9876543210",
    )

    assert settings.google_ads_ready() is False


# ============================================================
# GA4 readiness
# ============================================================

def test_ga4_ready_returns_true_when_values_exist(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "GA4_PROPERTY_ID",
        "123456789",
    )
    monkeypatch.setattr(
        settings,
        "GA4_SERVICE_ACCOUNT_FILE",
        "credentials/ga4_service_account.json",
    )

    assert settings.ga4_ready() is True


def test_ga4_ready_returns_false_when_property_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "GA4_PROPERTY_ID",
        "",
    )
    monkeypatch.setattr(
        settings,
        "GA4_SERVICE_ACCOUNT_FILE",
        "credentials/ga4_service_account.json",
    )

    assert settings.ga4_ready() is False


def test_ga4_ready_returns_false_when_file_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "GA4_PROPERTY_ID",
        "123456789",
    )
    monkeypatch.setattr(
        settings,
        "GA4_SERVICE_ACCOUNT_FILE",
        None,
    )

    assert settings.ga4_ready() is False


# ============================================================
# Anthropic readiness
# ============================================================

def test_anthropic_ready_returns_true_when_key_exists(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "ANTHROPIC_API_KEY",
        "test-api-key",
    )

    assert settings.anthropic_ready() is True


def test_anthropic_ready_returns_false_when_key_empty(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "ANTHROPIC_API_KEY",
        "   ",
    )

    assert settings.anthropic_ready() is False


def test_anthropic_ready_returns_false_when_key_none(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "ANTHROPIC_API_KEY",
        None,
    )

    assert settings.anthropic_ready() is False


# ============================================================
# Loaded configuration values
# ============================================================

def test_target_roas_is_float():
    assert isinstance(
        settings.TARGET_ROAS,
        float,
    )


def test_llm_max_campaigns_is_integer():
    assert isinstance(
        settings.LLM_MAX_CAMPAIGNS,
        int,
    )


def test_postgres_enabled_is_boolean():
    assert isinstance(
        settings.POSTGRES_ENABLED,
        bool,
    )


def test_postgres_port_is_string():
    assert isinstance(
        settings.POSTGRES_PORT,
        str,
    )


def test_date_mode_is_lowercase():
    assert settings.DATE_MODE == settings.DATE_MODE.lower()


def test_postgres_if_exists_is_lowercase():
    assert (
        settings.POSTGRES_IF_EXISTS
        == settings.POSTGRES_IF_EXISTS.lower()
    )
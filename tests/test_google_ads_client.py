from unittest.mock import MagicMock, patch

import pytest

import config.google_ads_client as google_ads_client


def set_valid_credentials(monkeypatch):
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "developer-token",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_ID",
        "client-id",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_SECRET",
        "client-secret",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_REFRESH_TOKEN",
        "refresh-token",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        "1234567890",
    )


@patch(
    "config.google_ads_client.GoogleAdsClient.load_from_dict",
)
def test_get_google_ads_client_returns_created_client(
    mock_load_from_dict,
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    expected_client = MagicMock()
    mock_load_from_dict.return_value = expected_client

    result = google_ads_client.get_google_ads_client()

    assert result is expected_client


@patch(
    "config.google_ads_client.GoogleAdsClient.load_from_dict",
)
def test_get_google_ads_client_builds_expected_config(
    mock_load_from_dict,
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    mock_load_from_dict.return_value = MagicMock()

    google_ads_client.get_google_ads_client()

    mock_load_from_dict.assert_called_once_with(
        {
            "developer_token": "developer-token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token",
            "login_customer_id": "1234567890",
            "use_proto_plus": True,
        }
    )


def test_get_google_ads_client_raises_when_developer_token_missing(
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "",
    )

    with pytest.raises(
        ValueError,
        match="Missing environment variable: GOOGLE_ADS_DEVELOPER_TOKEN",
    ):
        google_ads_client.get_google_ads_client()


def test_get_google_ads_client_raises_when_client_id_missing(
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_ID",
        None,
    )

    with pytest.raises(
        ValueError,
        match="Missing environment variable: GOOGLE_ADS_CLIENT_ID",
    ):
        google_ads_client.get_google_ads_client()


def test_get_google_ads_client_raises_when_client_secret_missing(
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_SECRET",
        "   ",
    )

    with pytest.raises(
        ValueError,
        match="Missing environment variable: GOOGLE_ADS_CLIENT_SECRET",
    ):
        google_ads_client.get_google_ads_client()


def test_get_google_ads_client_raises_when_refresh_token_missing(
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_REFRESH_TOKEN",
        "",
    )

    with pytest.raises(
        ValueError,
        match="Missing environment variable: GOOGLE_ADS_REFRESH_TOKEN",
    ):
        google_ads_client.get_google_ads_client()


def test_get_google_ads_client_raises_when_login_customer_id_missing(
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing environment variable: "
            "GOOGLE_ADS_LOGIN_CUSTOMER_ID"
        ),
    ):
        google_ads_client.get_google_ads_client()


@patch(
    "config.google_ads_client.GoogleAdsClient.load_from_dict",
)
def test_get_google_ads_client_wraps_initialization_error(
    mock_load_from_dict,
    monkeypatch,
):
    set_valid_credentials(monkeypatch)

    mock_load_from_dict.side_effect = RuntimeError(
        "Underlying Google Ads error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to initialize Google Ads client",
    ) as exc_info:
        google_ads_client.get_google_ads_client()

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    assert str(exc_info.value.__cause__) == (
        "Underlying Google Ads error"
    )


@patch(
    "config.google_ads_client.GoogleAdsClient.load_from_dict",
)
def test_get_google_ads_client_trims_credentials(
    mock_load_from_dict,
    monkeypatch,
):
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "  developer-token  ",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_ID",
        "  client-id  ",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_CLIENT_SECRET",
        "  client-secret  ",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_REFRESH_TOKEN",
        "  refresh-token  ",
    )
    monkeypatch.setattr(
        google_ads_client,
        "GOOGLE_ADS_LOGIN_CUSTOMER_ID",
        "  1234567890  ",
    )

    mock_load_from_dict.return_value = MagicMock()

    google_ads_client.get_google_ads_client()

    mock_load_from_dict.assert_called_once_with(
        {
            "developer_token": "developer-token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token",
            "login_customer_id": "1234567890",
            "use_proto_plus": True,
        }
    )
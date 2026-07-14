from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.extract.ads_extractor import (
    AdsExtractor,
    extract_category,
    extract_product,
    fetch_ads_purchase_only,
)


def build_google_ads_row(
    date: str = "2026-01-15",
    campaign_id: int = 101,
    campaign_name: str = "Brand Campaign",
    channel: str = "SEARCH",
    ad_group_id: int = 201,
    ad_group_name: str = "Shoes | Kids",
    impressions: int = 1000,
    clicks: int = 100,
    conversions: float = 10.0,
    conversion_value: float = 1000.0,
    cost_micros: int = 200_000_000,
):
    return SimpleNamespace(
        segments=SimpleNamespace(
            date=date,
        ),
        campaign=SimpleNamespace(
            id=campaign_id,
            name=campaign_name,
            advertising_channel_type=channel,
        ),
        ad_group=SimpleNamespace(
            id=ad_group_id,
            name=ad_group_name,
        ),
        metrics=SimpleNamespace(
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            conversions_value=conversion_value,
            cost_micros=cost_micros,
        ),
    )


def test_extract_category_with_separator():
    result = extract_category("Shoes | Kids")

    assert result == "Shoes"


def test_extract_category_without_separator():
    result = extract_category("Boots Winter Collection")

    assert result == "Boots"


def test_extract_category_returns_unknown_for_empty_value():
    assert extract_category("") == "UNKNOWN"
    assert extract_category(None) == "UNKNOWN"


def test_extract_product_with_separator():
    result = extract_product("Shoes | Kids")

    assert result == "Kids"


def test_extract_product_without_separator():
    result = extract_product("Winter Boots")

    assert result == "Winter Boots"


def test_extract_product_returns_unknown_for_empty_value():
    assert extract_product("") == "UNKNOWN"
    assert extract_product(None) == "UNKNOWN"


@patch(
    "src.extract.ads_extractor.GOOGLE_ADS_CUSTOMER_ID",
    "1234567890",
)
@patch(
    "src.extract.ads_extractor.get_google_ads_client",
)
def test_fetch_ads_purchase_only_returns_expected_dataframe(
    mock_get_client,
):
    mock_client = MagicMock()
    mock_service = MagicMock()

    mock_get_client.return_value = mock_client
    mock_client.get_service.return_value = mock_service

    mock_service.search.return_value = [
        build_google_ads_row()
    ]

    result = fetch_ads_purchase_only(
        "2026-01-01",
        "2026-01-31",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert result.loc[0, "CampaignId"] == 101
    assert result.loc[0, "Campaign"] == "Brand Campaign"
    assert result.loc[0, "Channel"] == "SEARCH"
    assert result.loc[0, "AdGroupId"] == 201
    assert result.loc[0, "AdGroup"] == "Shoes | Kids"

    assert result.loc[0, "Impressions"] == 1000
    assert result.loc[0, "Clicks"] == 100
    assert result.loc[0, "Conversions"] == 10.0
    assert result.loc[0, "ConversionValue"] == 1000.0
    assert result.loc[0, "Spend"] == 200.0

    assert result.loc[0, "Category"] == "Shoes"
    assert result.loc[0, "ProductGroup"] == "Kids"

    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    mock_client.get_service.assert_called_once_with(
        "GoogleAdsService"
    )

    mock_service.search.assert_called_once()

    search_call = mock_service.search.call_args

    assert search_call.kwargs["customer_id"] == "1234567890"
    assert "2026-01-01" in search_call.kwargs["query"]
    assert "2026-01-31" in search_call.kwargs["query"]


@patch(
    "src.extract.ads_extractor.GOOGLE_ADS_CUSTOMER_ID",
    "1234567890",
)
@patch(
    "src.extract.ads_extractor.get_google_ads_client",
)
def test_fetch_ads_purchase_only_returns_empty_dataframe(
    mock_get_client,
):
    mock_client = MagicMock()
    mock_service = MagicMock()

    mock_get_client.return_value = mock_client
    mock_client.get_service.return_value = mock_service
    mock_service.search.return_value = []

    result = fetch_ads_purchase_only(
        "2026-01-01",
        "2026-01-31",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


@patch(
    "src.extract.ads_extractor.GOOGLE_ADS_CUSTOMER_ID",
    "",
)
def test_fetch_ads_purchase_only_raises_for_missing_customer_id():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: GOOGLE_ADS_CUSTOMER_ID",
    ):
        fetch_ads_purchase_only(
            "2026-01-01",
            "2026-01-31",
        )


@patch(
    "src.extract.ads_extractor.GoogleAdsException",
    RuntimeError,
)
@patch(
    "src.extract.ads_extractor.GOOGLE_ADS_CUSTOMER_ID",
    "1234567890",
)
@patch(
    "src.extract.ads_extractor.get_google_ads_client",
)
def test_fetch_ads_purchase_only_reraises_google_ads_error(
    mock_get_client,
):
    mock_client = MagicMock()
    mock_service = MagicMock()

    mock_get_client.return_value = mock_client
    mock_client.get_service.return_value = mock_service

    mock_service.search.side_effect = RuntimeError(
        "Google Ads API failed."
    )

    with pytest.raises(
        RuntimeError,
        match="Google Ads API failed.",
    ):
        fetch_ads_purchase_only(
            "2026-01-01",
            "2026-01-31",
        )


@patch(
    "src.extract.ads_extractor.get_google_ads_client",
)
def test_ads_extractor_get_client_caches_client(
    mock_get_client,
):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    extractor = AdsExtractor()

    first_result = extractor.get_client()
    second_result = extractor.get_client()

    assert first_result is mock_client
    assert second_result is mock_client

    mock_get_client.assert_called_once()


@patch(
    "src.extract.ads_extractor.fetch_ads_purchase_only",
)
def test_ads_extractor_fetch_purchase_data_delegates(
    mock_fetch,
):
    expected_df = pd.DataFrame(
        {
            "CampaignId": [1],
        }
    )

    mock_fetch.return_value = expected_df

    extractor = AdsExtractor()

    result = extractor.fetch_purchase_data(
        "2026-01-01",
        "2026-01-31",
    )

    pd.testing.assert_frame_equal(
        result,
        expected_df,
    )

    mock_fetch.assert_called_once_with(
        "2026-01-01",
        "2026-01-31",
    )
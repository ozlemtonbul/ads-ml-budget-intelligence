from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.extract.ga4_extractor import GA4Extractor


def build_dimension(value):
    return SimpleNamespace(value=value)


def build_metric(value):
    return SimpleNamespace(value=str(value))


def build_ga4_row(
    date="20260115",
    campaign="Brand Campaign",
    source_medium="google / cpc",
    sessions=100,
    total_users=80,
    engaged_sessions=60,
    purchases=10,
    purchase_revenue=1000,
):
    return SimpleNamespace(
        dimension_values=[
            build_dimension(date),
            build_dimension(campaign),
            build_dimension(source_medium),
        ],
        metric_values=[
            build_metric(sessions),
            build_metric(total_users),
            build_metric(engaged_sessions),
            build_metric(purchases),
            build_metric(purchase_revenue),
        ],
    )


@patch(
    "src.extract.ga4_extractor.GA4_PROPERTY_ID",
    "",
)
def test_ga4_extractor_raises_when_property_id_missing():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: GA4_PROPERTY_ID",
    ):
        GA4Extractor()


@patch(
    "src.extract.ga4_extractor.GA4_SERVICE_ACCOUNT_FILE",
    "",
)
@patch(
    "src.extract.ga4_extractor.GA4_PROPERTY_ID",
    "123456789",
)
def test_ga4_extractor_raises_when_service_account_path_missing():
    with pytest.raises(
        ValueError,
        match="Missing environment variable: GA4_SERVICE_ACCOUNT_FILE",
    ):
        GA4Extractor()


@patch(
    "src.extract.ga4_extractor.Path.exists",
    return_value=False,
)
@patch(
    "src.extract.ga4_extractor.GA4_SERVICE_ACCOUNT_FILE",
    "credentials/missing.json",
)
@patch(
    "src.extract.ga4_extractor.GA4_PROPERTY_ID",
    "123456789",
)
def test_ga4_extractor_raises_when_service_account_file_not_found(
    mock_exists,
):
    with pytest.raises(
        FileNotFoundError,
        match="GA4 service account file was not found",
    ):
        GA4Extractor()

    mock_exists.assert_called_once()


@patch(
    "src.extract.ga4_extractor.BetaAnalyticsDataClient",
)
@patch(
    "src.extract.ga4_extractor.service_account.Credentials.from_service_account_file",
)
@patch(
    "src.extract.ga4_extractor.Path.exists",
    return_value=True,
)
@patch(
    "src.extract.ga4_extractor.GA4_SERVICE_ACCOUNT_FILE",
    "credentials/ga4_service_account.json",
)
@patch(
    "src.extract.ga4_extractor.GA4_PROPERTY_ID",
    "123456789",
)
def test_ga4_extractor_initializes_correctly(
    mock_exists,
    mock_from_file,
    mock_client_class,
):
    mock_credentials = MagicMock()
    mock_client = MagicMock()

    mock_from_file.return_value = mock_credentials
    mock_client_class.return_value = mock_client

    extractor = GA4Extractor()

    assert extractor.credentials is mock_credentials
    assert extractor.client is mock_client
    assert extractor.property_name == "properties/123456789"

    mock_from_file.assert_called_once_with(
        str(Path("credentials/ga4_service_account.json"))
    )

    mock_client_class.assert_called_once_with(
        credentials=mock_credentials
    )


def build_initialized_extractor():
    extractor = GA4Extractor.__new__(GA4Extractor)
    extractor.client = MagicMock()
    extractor.property_name = "properties/123456789"
    return extractor


def test_fetch_campaign_performance_returns_expected_dataframe():
    extractor = build_initialized_extractor()

    response = SimpleNamespace(
        rows=[
            build_ga4_row(),
        ]
    )

    extractor.client.run_report.return_value = response

    result = extractor.fetch_campaign_performance(
        "2026-01-01",
        "2026-01-31",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert result.loc[0, "Campaign"] == "Brand Campaign"
    assert result.loc[0, "SourceMedium"] == "google / cpc"
    assert result.loc[0, "Sessions"] == 100.0
    assert result.loc[0, "TotalUsers"] == 80.0
    assert result.loc[0, "EngagedSessions"] == 60.0
    assert result.loc[0, "Purchases"] == 10.0
    assert result.loc[0, "PurchaseRevenue"] == 1000.0

    assert result.loc[0, "GA4ConversionRate"] == 0.1
    assert result.loc[0, "GA4RevenuePerSession"] == 10.0
    assert result.loc[0, "EngagementRate"] == 0.6

    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    extractor.client.run_report.assert_called_once()


def test_fetch_campaign_performance_uses_unknown_for_blank_dimensions():
    extractor = build_initialized_extractor()

    response = SimpleNamespace(
        rows=[
            build_ga4_row(
                campaign="",
                source_medium="",
            ),
        ]
    )

    extractor.client.run_report.return_value = response

    result = extractor.fetch_campaign_performance(
        "2026-01-01",
        "2026-01-31",
    )

    assert result.loc[0, "Campaign"] == "UNKNOWN"
    assert result.loc[0, "SourceMedium"] == "UNKNOWN"


def test_fetch_campaign_performance_handles_zero_sessions():
    extractor = build_initialized_extractor()

    response = SimpleNamespace(
        rows=[
            build_ga4_row(
                sessions=0,
                engaged_sessions=0,
                purchases=0,
                purchase_revenue=0,
            ),
        ]
    )

    extractor.client.run_report.return_value = response

    result = extractor.fetch_campaign_performance(
        "2026-01-01",
        "2026-01-31",
    )

    assert result.loc[0, "GA4ConversionRate"] == 0
    assert result.loc[0, "GA4RevenuePerSession"] == 0
    assert result.loc[0, "EngagementRate"] == 0


def test_fetch_campaign_performance_returns_empty_dataframe():
    extractor = build_initialized_extractor()

    extractor.client.run_report.return_value = SimpleNamespace(
        rows=[]
    )

    result = extractor.fetch_campaign_performance(
        "2026-01-01",
        "2026-01-31",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_fetch_campaign_performance_reraises_api_error():
    extractor = build_initialized_extractor()

    extractor.client.run_report.side_effect = RuntimeError(
        "GA4 API failed."
    )

    with pytest.raises(
        RuntimeError,
        match="GA4 API failed.",
    ):
        extractor.fetch_campaign_performance(
            "2026-01-01",
            "2026-01-31",
        )


def test_fetch_campaign_performance_builds_correct_request():
    extractor = build_initialized_extractor()

    extractor.client.run_report.return_value = SimpleNamespace(
        rows=[]
    )

    extractor.fetch_campaign_performance(
        "2026-02-01",
        "2026-02-28",
    )

    request = extractor.client.run_report.call_args.args[0]

    assert request.property == "properties/123456789"

    assert [
        dimension.name
        for dimension in request.dimensions
    ] == [
        "date",
        "sessionCampaignName",
        "sessionSourceMedium",
    ]

    assert [
        metric.name
        for metric in request.metrics
    ] == [
        "sessions",
        "totalUsers",
        "engagedSessions",
        "ecommercePurchases",
        "purchaseRevenue",
    ]

    assert request.date_ranges[0].start_date == "2026-02-01"
    assert request.date_ranges[0].end_date == "2026-02-28"
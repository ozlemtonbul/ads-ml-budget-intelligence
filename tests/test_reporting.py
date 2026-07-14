import pandas as pd

from src.features.reporting import (
    build_category_summary,
    build_daily_weekly_monthly_outputs,
    build_holiday_impact_summary,
    build_product_summary,
    build_zero_activity_report,
)


def sample_ads_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-08",
                "2026-01-09",
            ],
            "CampaignId": [1, 1, 2, 2],
            "Campaign": [
                "Brand Campaign",
                "Brand Campaign",
                "Shopping Campaign",
                "Shopping Campaign",
            ],
            "Channel": [
                "SEARCH",
                "SEARCH",
                "SHOPPING",
                "SHOPPING",
            ],
            "Category": [
                "Shoes",
                "Shoes",
                "Boots",
                "Boots",
            ],
            "ProductGroup": [
                "Kids",
                "Kids",
                "Winter",
                "Winter",
            ],
            "Spend": [100.0, 120.0, 200.0, 180.0],
            "Clicks": [50, 60, 80, 70],
            "Impressions": [1000, 1200, 1600, 1400],
            "Conversions": [5.0, 6.0, 8.0, 7.0],
            "ConversionValue": [500.0, 600.0, 900.0, 800.0],
        }
    )


def test_build_category_summary():
    ads_raw = sample_ads_data()
    holiday_map = {
        "2026-01-01": "New Year",
    }

    result = build_category_summary(
        ads_raw,
        holiday_map,
    )

    assert len(result) == 2
    assert set(result["Category"]) == {"Shoes", "Boots"}
    assert "ROAS" in result.columns
    assert "Profit" in result.columns

    shoes_row = result.loc[
        result["Category"] == "Shoes"
    ].iloc[0]

    assert shoes_row["Spend"] == 220.0
    assert shoes_row["ConversionValue"] == 1100.0
    assert shoes_row["ROAS"] == 5.0


def test_build_product_summary():
    ads_raw = sample_ads_data()
    holiday_map = {}

    result = build_product_summary(
        ads_raw,
        holiday_map,
    )

    assert len(result) == 2
    assert "ProductGroup" in result.columns
    assert "Campaign" in result.columns
    assert "ROAS" in result.columns

    boots_row = result.loc[
        result["Category"] == "Boots"
    ].iloc[0]

    assert boots_row["Spend"] == 380.0
    assert boots_row["ConversionValue"] == 1700.0


def test_build_holiday_impact_summary():
    ads_raw = sample_ads_data()
    holiday_map = {
        "2026-01-01": "New Year",
    }

    result = build_holiday_impact_summary(
        ads_raw,
        holiday_map,
    )

    assert "PeriodLabel" in result.columns
    assert "AvgDailySpend" in result.columns
    assert "AvgDailyRevenue" in result.columns
    assert "ROAS" in result.columns

    assert "Holiday" in result["PeriodLabel"].tolist()
    assert "Normal Day" in result["PeriodLabel"].tolist()


def test_build_daily_weekly_monthly_outputs():
    ads_raw = sample_ads_data()
    holiday_map = {
        "2026-01-01": "New Year",
    }

    daily_df, weekly_df, monthly_df = (
        build_daily_weekly_monthly_outputs(
            ads_raw,
            holiday_map,
        )
    )

    assert len(daily_df) == 4
    assert len(weekly_df) >= 2
    assert len(monthly_df) == 2

    assert "Date" in daily_df.columns
    assert "Week" in weekly_df.columns
    assert "Month" in monthly_df.columns

    assert "ROAS" in daily_df.columns
    assert "Profit" in weekly_df.columns
    assert "CTR" in monthly_df.columns


def test_build_zero_activity_report():
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 2, 3],
            "Campaign": [
                "No Impressions",
                "No Clicks",
                "Active Campaign",
            ],
            "Channel": [
                "SEARCH",
                "SEARCH",
                "SHOPPING",
            ],
            "Impressions": [0, 1000, 1000],
            "Clicks": [0, 0, 100],
            "Spend": [0.0, 100.0, 200.0],
            "Conversions": [0.0, 0.0, 10.0],
        }
    )

    result = build_zero_activity_report(
        ads_raw
    )

    assert len(result) == 2
    assert set(result["IssueType"]) == {
        "Zero Impressions",
        "Zero Clicks",
    }

    assert (
        result["Recommendation"]
        == "Review campaign targeting, bids, and ad approvals."
    ).all()


def test_reporting_functions_return_empty_for_empty_input():
    empty_df = pd.DataFrame()
    holiday_map = {}

    assert build_category_summary(
        empty_df,
        holiday_map,
    ).empty

    assert build_product_summary(
        empty_df,
        holiday_map,
    ).empty

    assert build_holiday_impact_summary(
        empty_df,
        holiday_map,
    ).empty

    daily_df, weekly_df, monthly_df = (
        build_daily_weekly_monthly_outputs(
            empty_df,
            holiday_map,
        )
    )

    assert daily_df.empty
    assert weekly_df.empty
    assert monthly_df.empty

    assert build_zero_activity_report(
        empty_df
    ).empty
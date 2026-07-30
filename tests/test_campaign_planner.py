from datetime import timedelta

import pandas as pd

from src.recommendations.campaign_planner import (
    PROPOSAL_COLUMNS,
    build_new_campaign_proposals,
)


def build_sample_ads_data() -> pd.DataFrame:
    rows = []

    for day in range(1, 31):
        current_date = pd.Timestamp(
            year=2026,
            month=1,
            day=day,
        )

        rows.extend(
            [
                {
                    "Date": current_date,
                    "Campaign": "brand-ayakkabi",
                    "Channel": "SEARCH",
                    "Category": "Ayakkabı",
                    "ProductGroup": "Spor Ayakkabı",
                    "Spend": 100.0,
                    "Conversions": 5.0,
                    "ConversionValue": 500.0,
                },
                {
                    "Date": current_date,
                    "Campaign": "generic-ayakkabi",
                    "Channel": "SEARCH",
                    "Category": "Ayakkabı",
                    "ProductGroup": "Spor Ayakkabı",
                    "Spend": 60.0,
                    "Conversions": 2.0,
                    "ConversionValue": 180.0,
                },
                {
                    "Date": current_date,
                    "Campaign": "pmax-cocuk",
                    "Channel": "PERFORMANCE_MAX",
                    "Category": "Çocuk",
                    "ProductGroup": "Çocuk Ayakkabısı",
                    "Spend": 80.0,
                    "Conversions": 3.0,
                    "ConversionValue": 320.0,
                },
            ]
        )

    return pd.DataFrame(rows)


def test_empty_dataframe_returns_empty_proposals():
    result = build_new_campaign_proposals(
        ads_df=pd.DataFrame(),
        target_roas=3.0,
    )

    assert result.empty
    assert result.columns.tolist() == PROPOSAL_COLUMNS


def test_missing_required_columns_returns_empty_proposals():
    ads_df = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Campaign": ["brand"],
        }
    )

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
    )

    assert result.empty
    assert result.columns.tolist() == PROPOSAL_COLUMNS


def test_build_new_campaign_proposals_returns_expected_columns():
    ads_df = build_sample_ads_data()

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
        max_proposals=5,
    )

    assert not result.empty
    assert result.columns.tolist() == PROPOSAL_COLUMNS
    assert len(result) <= 5


def test_proposal_budget_limits_are_consistent():
    ads_df = build_sample_ads_data()

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
    )

    assert not result.empty

    assert (
        result["MinimumDailyBudget"]
        <= result["SuggestedDailyBudget"]
    ).all()

    assert (
        result["SuggestedDailyBudget"]
        <= result["MaximumDailyBudget"]
    ).all()

    assert (
        result["SuggestedDailyBudget"] > 0
    ).all()


def test_proposal_dates_start_after_analysis_period():
    ads_df = build_sample_ads_data()

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
        test_duration_days=14,
    )

    assert not result.empty

    analysis_end = pd.to_datetime(
        ads_df["Date"],
        errors="coerce",
    ).max().date()

    expected_start = analysis_end + timedelta(days=1)
    expected_end = expected_start + timedelta(days=13)

    proposal_start_dates = pd.to_datetime(
        result["ProposedStartDate"],
        errors="coerce",
    ).dt.date

    proposal_end_dates = pd.to_datetime(
        result["ProposedEndDate"],
        errors="coerce",
    ).dt.date

    assert proposal_start_dates.notna().all()
    assert proposal_end_dates.notna().all()

    assert (
        proposal_start_dates == expected_start
    ).all()

    assert (
        proposal_end_dates == expected_end
    ).all()

    durations = (
        pd.to_datetime(
            result["ProposedEndDate"],
            errors="coerce",
        )
        - pd.to_datetime(
            result["ProposedStartDate"],
            errors="coerce",
        )
    ).dt.days + 1

    assert (durations == 14).all()


def test_expected_financial_values_are_consistent():
    ads_df = build_sample_ads_data()

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
    )

    assert not result.empty

    expected_revenue = (
        result["SuggestedDailyBudget"]
        * result["ExpectedROAS"]
    )

    expected_profit = (
        result["ExpectedDailyRevenue"]
        - result["SuggestedDailyBudget"]
    )

    assert (
        result["ExpectedDailyRevenue"]
        - expected_revenue
    ).abs().max() < 0.10

    assert (
        result["ExpectedDailyProfit"]
        - expected_profit
    ).abs().max() < 0.10


def test_holiday_strategy_detects_upcoming_holiday():
    ads_df = build_sample_ads_data()

    holiday_map = {
        "2026-02-20": "Ramazan Bayramı",
    }

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
        holiday_map=holiday_map,
    )

    assert not result.empty

    assert result["HolidayStrategy"].str.contains(
        "Ramazan Bayramı",
        regex=False,
    ).all()


def test_proposal_names_are_unique():
    ads_df = build_sample_ads_data()

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
    )

    assert not result.empty
    assert result["ProposalId"].is_unique
    assert result["ProposedCampaignName"].is_unique


def test_insufficient_history_does_not_create_proposal():
    ads_df = pd.DataFrame(
        [
            {
                "Date": "2026-01-01",
                "Campaign": "short-test",
                "Channel": "SEARCH",
                "Category": "Ayakkabı",
                "ProductGroup": "Spor Ayakkabı",
                "Spend": 100.0,
                "Conversions": 5.0,
                "ConversionValue": 500.0,
            }
        ]
    )

    result = build_new_campaign_proposals(
        ads_df=ads_df,
        target_roas=3.0,
    )

    assert result.empty
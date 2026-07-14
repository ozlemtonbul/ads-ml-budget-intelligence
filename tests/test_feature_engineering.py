import pandas as pd
import pytest

from src.features.feature_engineering import (
    add_calendar_context_features,
    add_lag_features,
    add_time_features,
    build_holiday_map,
    compute_kpis,
    compute_roas_target_gap,
    get_feature_columns,
    get_latest_campaign_state,
    get_turkey_public_holidays,
    prepare_training_data,
)


def test_compute_kpis():
    df = pd.DataFrame(
        {
            "Impressions": [1000],
            "Clicks": [100],
            "Spend": [200.0],
            "Conversions": [10.0],
            "ConversionValue": [1000.0],
        }
    )

    result = compute_kpis(df)

    assert result.loc[0, "CTR"] == 0.1
    assert result.loc[0, "CPC"] == 2.0
    assert result.loc[0, "ConvRate"] == 0.1
    assert result.loc[0, "CPA"] == 20.0
    assert result.loc[0, "ROAS"] == 5.0
    assert result.loc[0, "Profit"] == 800.0


def test_compute_kpis_handles_zero_denominators():
    df = pd.DataFrame(
        {
            "Impressions": [0],
            "Clicks": [0],
            "Spend": [0.0],
            "Conversions": [0.0],
            "ConversionValue": [0.0],
        }
    )

    result = compute_kpis(df)

    assert result.loc[0, "CTR"] == 0
    assert result.loc[0, "CPC"] == 0
    assert result.loc[0, "ConvRate"] == 0
    assert result.loc[0, "CPA"] == 0
    assert result.loc[0, "ROAS"] == 0
    assert result.loc[0, "Profit"] == 0


def test_compute_kpis_requires_expected_columns():
    df = pd.DataFrame(
        {
            "Spend": [100.0],
            "Clicks": [10],
        }
    )

    with pytest.raises(
        KeyError,
        match="Missing required KPI columns",
    ):
        compute_kpis(df)


def test_compute_roas_target_gap():
    df = pd.DataFrame(
        {
            "ROAS": [4.0],
        }
    )

    result = compute_roas_target_gap(
        df,
        target_roas=3.0,
    )

    assert result.loc[0, "TargetROAS"] == 3.0
    assert result.loc[0, "ROASGap"] == 1.0
    assert round(result.loc[0, "ROASGapPct"], 2) == 33.33
    assert result.loc[0, "ROASStatus"] == "Above Target"


def test_compute_roas_target_gap_statuses():
    df = pd.DataFrame(
        {
            "ROAS": [4.0, 3.0, 2.0],
        }
    )

    result = compute_roas_target_gap(
        df,
        target_roas=3.0,
    )

    assert result["ROASStatus"].tolist() == [
        "Above Target",
        "On Target",
        "Below Target",
    ]


def test_compute_roas_target_gap_requires_roas():
    with pytest.raises(
        KeyError,
        match="ROAS column is required",
    ):
        compute_roas_target_gap(
            pd.DataFrame(
                {
                    "Value": [1],
                }
            ),
            target_roas=3.0,
        )


def test_add_time_features():
    df = pd.DataFrame(
        {
            "Date": ["2026-01-15"],
        }
    )

    result = add_time_features(df)

    assert result.loc[0, "MonthNum"] == 1
    assert result.loc[0, "DayOfMonth"] == 15
    assert result.loc[0, "Quarter"] == 1
    assert result.loc[0, "IsWeekend"] == 0
    assert result.loc[0, "DayOfWeek"] == 3


def test_add_time_features_detects_weekend():
    df = pd.DataFrame(
        {
            "Date": ["2026-01-17"],
        }
    )

    result = add_time_features(df)

    assert result.loc[0, "IsWeekend"] == 1
    assert result.loc[0, "DayOfWeek"] == 5


def test_add_time_features_requires_date():
    with pytest.raises(
        KeyError,
        match="Date column is required",
    ):
        add_time_features(
            pd.DataFrame(
                {
                    "Value": [1],
                }
            )
        )


def test_get_turkey_public_holidays_contains_known_dates():
    result = get_turkey_public_holidays(2026)

    assert isinstance(result, dict)
    assert "2026-01-01" in result
    assert "2026-03-20" in result
    assert "2026-05-27" in result


def test_get_turkey_public_holidays_contains_religious_holidays():
    result = get_turkey_public_holidays(2025)

    assert result["2025-03-30"] == "Eid al-Fitr"
    assert result["2025-06-06"] == "Eid al-Adha"


def test_build_holiday_map_single_year():
    result = build_holiday_map(
        "2026-01-01",
        "2026-12-31",
    )

    assert isinstance(result, dict)
    assert "2026-01-01" in result
    assert "2026-10-29" in result


def test_build_holiday_map_multiple_years():
    result = build_holiday_map(
        "2025-12-01",
        "2026-01-31",
    )

    assert "2025-01-01" in result
    assert "2026-01-01" in result


def test_build_holiday_map_requires_date_from():
    with pytest.raises(
        ValueError,
        match="date_from and date_to are required",
    ):
        build_holiday_map(
            "",
            "2026-01-31",
        )


def test_build_holiday_map_requires_date_to():
    with pytest.raises(
        ValueError,
        match="date_from and date_to are required",
    ):
        build_holiday_map(
            "2026-01-01",
            "",
        )


def test_build_holiday_map_rejects_reversed_years():
    with pytest.raises(
        ValueError,
        match="date_from cannot be later than date_to",
    ):
        build_holiday_map(
            "2027-01-01",
            "2026-12-31",
        )


def test_add_calendar_context_features():
    df = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "2026-03-18",
                "2026-07-01",
            ]
        }
    )

    holiday_map = {
        "2026-01-01": "New Year",
        "2026-03-20": "Eid al-Fitr",
    }

    result = add_calendar_context_features(
        df,
        holiday_map,
    )

    assert result.loc[0, "IsHoliday"] == 1
    assert result.loc[0, "HolidayName"] == "New Year"

    assert result.loc[1, "IsPreHoliday"] == 1
    assert result.loc[2, "IsHoliday"] == 0
    assert result.loc[2, "IsPreHoliday"] == 0

    assert result.loc[0, "Season"] == "Winter"
    assert result.loc[1, "Season"] == "Spring"
    assert result.loc[2, "Season"] == "Summer"

    assert result.loc[0, "SeasonEN"] == "winter"
    assert result.loc[1, "SeasonEN"] == "spring"
    assert result.loc[2, "SeasonEN"] == "summer"

    assert result.loc[0, "SeasonROASMultiplier"] == 1.15
    assert result.loc[2, "SeasonROASMultiplier"] == 0.95

    assert result.loc[0, "ExpectedROASMultiplier"] == pytest.approx(
        1.20 * 1.15
    )

    assert result.loc[1, "ExpectedROASMultiplier"] == pytest.approx(
        1.20 * 1.05
    )

    assert result.loc[2, "ExpectedROASMultiplier"] == pytest.approx(
        0.95
    )


def test_add_calendar_context_features_handles_invalid_date():
    df = pd.DataFrame(
        {
            "Date": ["invalid-date"],
        }
    )

    result = add_calendar_context_features(
        df,
        holiday_map={},
    )

    assert pd.isna(result.loc[0, "Date"])
    assert result.loc[0, "IsHoliday"] == 0
    assert result.loc[0, "IsPreHoliday"] == 0
    assert result.loc[0, "Season"] == "Unknown"
    assert result.loc[0, "SeasonEN"] == "unknown"
    assert result.loc[0, "SeasonROASMultiplier"] == 1.0
    assert result.loc[0, "ExpectedROASMultiplier"] == 1.0


def test_add_calendar_context_features_requires_date():
    with pytest.raises(
        KeyError,
        match="Date column is required",
    ):
        add_calendar_context_features(
            pd.DataFrame(
                {
                    "Value": [1],
                }
            ),
            {},
        )


def test_add_lag_features():
    df = pd.DataFrame(
        {
            "CampaignId": [1, 1, 1],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "Spend": [100.0, 120.0, 140.0],
            "Clicks": [10, 12, 14],
            "Conversions": [1.0, 2.0, 3.0],
            "ConversionValue": [300.0, 500.0, 700.0],
            "ROAS": [3.0, 4.17, 5.0],
            "CPA": [100.0, 60.0, 46.67],
            "CTR": [0.1, 0.12, 0.14],
            "ConvRate": [0.1, 0.17, 0.21],
        }
    )

    result = add_lag_features(df)

    assert result.loc[0, "Spend_lag_1"] == 0
    assert result.loc[1, "Spend_lag_1"] == 100.0
    assert result.loc[2, "Clicks_lag_1"] == 12

    assert result.loc[0, "Spend_lag_7_avg"] == 100.0
    assert result.loc[1, "Spend_lag_7_avg"] == 110.0
    assert result.loc[2, "Spend_lag_7_avg"] == 120.0

    assert "ConversionValue_lag_7_avg" in result.columns
    assert "Conversions_lag_1" in result.columns
    assert "ROAS_lag_1" in result.columns


def test_add_lag_features_separates_campaigns():
    df = pd.DataFrame(
        {
            "CampaignId": [1, 1, 2, 2],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-01",
                "2026-01-02",
            ],
            "Spend": [100.0, 120.0, 200.0, 250.0],
            "Clicks": [10, 12, 20, 25],
            "Conversions": [1.0, 2.0, 3.0, 4.0],
            "ConversionValue": [300.0, 500.0, 700.0, 900.0],
            "ROAS": [3.0, 4.17, 3.5, 3.6],
            "CPA": [100.0, 60.0, 66.67, 62.5],
            "CTR": [0.1, 0.12, 0.2, 0.25],
            "ConvRate": [0.1, 0.17, 0.15, 0.16],
        }
    )

    result = add_lag_features(df)

    campaign_1 = result[
        result["CampaignId"] == 1
    ].reset_index(drop=True)

    campaign_2 = result[
        result["CampaignId"] == 2
    ].reset_index(drop=True)

    assert campaign_1.loc[0, "Spend_lag_1"] == 0
    assert campaign_1.loc[1, "Spend_lag_1"] == 100.0

    assert campaign_2.loc[0, "Spend_lag_1"] == 0
    assert campaign_2.loc[1, "Spend_lag_1"] == 200.0


def test_add_lag_features_requires_expected_columns():
    df = pd.DataFrame(
        {
            "CampaignId": [1],
            "Date": ["2026-01-01"],
            "Spend": [100.0],
        }
    )

    with pytest.raises(
        KeyError,
        match="Missing required lag columns",
    ):
        add_lag_features(df)


def test_prepare_training_data():
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 1, 1],
            "Campaign": [
                "Brand Campaign",
                "Brand Campaign",
                "Brand Campaign",
            ],
            "Channel": [
                "SEARCH",
                "SEARCH",
                "SEARCH",
            ],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "Spend": [100.0, 120.0, 140.0],
            "Impressions": [1000, 1200, 1400],
            "Clicks": [100, 120, 140],
            "Conversions": [10.0, 12.0, 14.0],
            "ConversionValue": [500.0, 600.0, 700.0],
        }
    )

    result = prepare_training_data(
        ads_raw,
        holiday_map={},
    )

    assert len(result) == 2
    assert "Target_Conversions_Next" in result.columns
    assert "Target_Revenue_Next" in result.columns

    assert result.iloc[0]["Target_Conversions_Next"] == 12.0
    assert result.iloc[0]["Target_Revenue_Next"] == 600.0

    assert result.iloc[1]["Target_Conversions_Next"] == 14.0
    assert result.iloc[1]["Target_Revenue_Next"] == 700.0


def test_prepare_training_data_returns_empty_for_empty_input():
    result = prepare_training_data(
        pd.DataFrame(),
        holiday_map={},
    )

    assert result.empty


def test_get_feature_columns_returns_expected_features():
    result = get_feature_columns()

    assert isinstance(result, list)
    assert len(result) > 20

    assert "CampaignId" in result
    assert "Spend" in result
    assert "ROAS" in result
    assert "ExpectedROASMultiplier" in result
    assert "Spend_lag_1" in result
    assert "Spend_lag_7_avg" in result
    assert "ConversionValue_lag_7_avg" in result


def test_get_feature_columns_has_no_duplicates():
    result = get_feature_columns()

    assert len(result) == len(set(result))


def test_get_latest_campaign_state():
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 1, 2, 2],
            "Campaign": [
                "Brand",
                "Brand",
                "Shopping",
                "Shopping",
            ],
            "Channel": [
                "SEARCH",
                "SEARCH",
                "SHOPPING",
                "SHOPPING",
            ],
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-01",
                "2026-01-02",
            ],
            "Spend": [100.0, 110.0, 200.0, 210.0],
            "Impressions": [1000, 1100, 2000, 2100],
            "Clicks": [100, 110, 200, 210],
            "Conversions": [10.0, 11.0, 20.0, 21.0],
            "ConversionValue": [500.0, 550.0, 900.0, 950.0],
        }
    )

    result = get_latest_campaign_state(
        ads_raw,
        holiday_map={},
    )

    assert len(result) == 2
    assert set(result["CampaignId"]) == {
        1,
        2,
    }

    latest_dates = result.set_index(
        "CampaignId"
    )["Date"]

    assert str(
        latest_dates.loc[1].date()
    ) == "2026-01-01"

    assert str(
        latest_dates.loc[2].date()
    ) == "2026-01-01"


def test_get_latest_campaign_state_returns_empty_for_empty_input():
    result = get_latest_campaign_state(
        pd.DataFrame(),
        holiday_map={},
    )

    assert result.empty
from typing import Dict, Tuple

import pandas as pd

from src.features.feature_engineering import (
    add_calendar_context_features,
    compute_kpis,
)


def build_category_summary(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    if ads_raw.empty:
        return pd.DataFrame()

    df = compute_kpis(ads_raw.copy())
    df = add_calendar_context_features(df, holiday_map)

    category_df = df.groupby(
        ["Category", "Season"],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    category_df = compute_kpis(category_df)

    return category_df.sort_values(
        "ConversionValue",
        ascending=False,
    ).reset_index(drop=True)


def build_product_summary(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    if ads_raw.empty:
        return pd.DataFrame()

    df = compute_kpis(ads_raw.copy())
    df = add_calendar_context_features(df, holiday_map)

    product_df = df.groupby(
        ["Category", "ProductGroup", "Campaign"],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    product_df = compute_kpis(product_df)

    return product_df.sort_values(
        "ConversionValue",
        ascending=False,
    ).reset_index(drop=True)


def build_holiday_impact_summary(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    if ads_raw.empty:
        return pd.DataFrame()

    df = compute_kpis(ads_raw.copy())
    df = add_calendar_context_features(df, holiday_map)

    impact_df = df.groupby(
        ["IsHoliday", "IsPreHoliday"],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
        DayCount=("Date", "nunique"),
    )

    impact_df["AvgDailySpend"] = (
        impact_df["Spend"]
        / impact_df["DayCount"].replace(0, pd.NA)
    )

    impact_df["AvgDailyRevenue"] = (
        impact_df["ConversionValue"]
        / impact_df["DayCount"].replace(0, pd.NA)
    )

    impact_df["ROAS"] = (
        impact_df["ConversionValue"]
        / impact_df["Spend"].replace(0, pd.NA)
    )

    impact_df["PeriodLabel"] = impact_df.apply(
        lambda row: (
            "Holiday"
            if row["IsHoliday"]
            else (
                "Pre-Holiday"
                if row["IsPreHoliday"]
                else "Normal Day"
            )
        ),
        axis=1,
    )

    return impact_df.fillna(0).reset_index(drop=True)


def build_daily_weekly_monthly_outputs(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ads_raw.empty:
        empty_df = pd.DataFrame()
        return empty_df, empty_df.copy(), empty_df.copy()

    df = add_calendar_context_features(
        ads_raw.copy(),
        holiday_map,
    )

    daily_df = df.groupby(
        [
            "Date",
            "Campaign",
            "Channel",
            "Category",
            "IsHoliday",
            "HolidayName",
            "Season",
        ],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    daily_df = compute_kpis(daily_df)

    temp = df.copy()

    temp["Week"] = (
        temp["Date"]
        .dt.to_period("W")
        .astype(str)
    )

    temp["Month"] = (
        temp["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    weekly_df = temp.groupby(
        [
            "Week",
            "Campaign",
            "Channel",
            "Category",
            "Season",
        ],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    weekly_df = compute_kpis(weekly_df)

    monthly_df = temp.groupby(
        [
            "Month",
            "Campaign",
            "Channel",
            "Category",
            "Season",
        ],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    monthly_df = compute_kpis(monthly_df)

    return (
        daily_df.reset_index(drop=True),
        weekly_df.reset_index(drop=True),
        monthly_df.reset_index(drop=True),
    )


def build_zero_activity_report(
    ads_raw: pd.DataFrame,
) -> pd.DataFrame:
    if ads_raw.empty:
        return pd.DataFrame()

    campaign_totals = ads_raw.groupby(
        ["CampaignId", "Campaign", "Channel"],
        as_index=False,
    ).agg(
        TotalImpressions=("Impressions", "sum"),
        TotalClicks=("Clicks", "sum"),
        TotalSpend=("Spend", "sum"),
        TotalConversions=("Conversions", "sum"),
    )

    zero_df = campaign_totals[
        (campaign_totals["TotalImpressions"] == 0)
        | (campaign_totals["TotalClicks"] == 0)
    ].copy()

    zero_df["IssueType"] = zero_df.apply(
        lambda row: (
            "Zero Impressions"
            if row["TotalImpressions"] == 0
            else "Zero Clicks"
        ),
        axis=1,
    )

    zero_df["Recommendation"] = (
        "Review campaign targeting, bids, and ad approvals."
    )

    return zero_df.reset_index(drop=True)
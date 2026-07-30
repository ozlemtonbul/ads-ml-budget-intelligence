from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd


PROPOSAL_COLUMNS = [
    "ProposalId",
    "ProposedCampaignName",
    "ProposalType",
    "Channel",
    "Category",
    "ProductGroup",
    "SuggestedDailyBudget",
    "MinimumDailyBudget",
    "MaximumDailyBudget",
    "TestDurationDays",
    "ProposedStartDate",
    "ProposedEndDate",
    "ExpectedDailyConversions",
    "ExpectedDailyRevenue",
    "ExpectedDailyProfit",
    "ExpectedROAS",
    "TargetROAS",
    "ConfidenceLevel",
    "RiskLevel",
    "SourceCampaignCount",
    "HistoryDays",
    "HistoricalROAS",
    "OpportunityScore",
    "HolidayStrategy",
    "RecommendationReason",
    "SuccessCriteria",
    "Guardrails",
]


def _empty_proposals() -> pd.DataFrame:
    return pd.DataFrame(columns=PROPOSAL_COLUMNS)


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    translation = str.maketrans(
        {
            "ç": "c",
            "ğ": "g",
            "ı": "i",
            "ö": "o",
            "ş": "s",
            "ü": "u",
        }
    )
    text = text.translate(translation)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:42] or "genel"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _confidence(
    history_days: int,
    campaign_count: int,
    conversions: float,
) -> str:
    if history_days >= 30 and campaign_count >= 2 and conversions >= 10:
        return "High"
    if history_days >= 14 and conversions >= 3:
        return "Medium"
    return "Low"


def _risk(confidence: str, expected_roas: float, target_roas: float) -> str:
    if confidence == "High" and expected_roas >= target_roas:
        return "Low"
    if confidence != "Low" and expected_roas >= target_roas * 0.85:
        return "Medium"
    return "High"


def _next_holiday_strategy(
    analysis_end: date,
    holiday_map: Dict[str, str] | None,
) -> str:
    if not holiday_map:
        return "Standard 14-day controlled test"

    upcoming: list[tuple[date, str]] = []
    for value, name in holiday_map.items():
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            continue
        holiday_date = parsed.date()
        if analysis_end < holiday_date <= analysis_end + timedelta(days=60):
            upcoming.append((holiday_date, str(name)))

    if not upcoming:
        return "Standard 14-day controlled test"

    holiday_date, holiday_name = min(upcoming, key=lambda item: item[0])
    return (
        f"{holiday_name} öncesinde kontrollü test; "
        f"en geç {holiday_date.isoformat()} tarihinden 14 gün önce başlatın"
    )


def build_new_campaign_proposals(
    ads_df: pd.DataFrame,
    target_roas: float,
    holiday_map: Dict[str, str] | None = None,
    max_proposals: int = 5,
    test_duration_days: int = 14,
) -> pd.DataFrame:
    """Build evidence-based proposals for new, controlled campaign tests."""
    if ads_df.empty:
        return _empty_proposals()

    required = {
        "Date",
        "Campaign",
        "Channel",
        "Spend",
        "Conversions",
        "ConversionValue",
    }
    if not required.issubset(ads_df.columns):
        return _empty_proposals()

    df = ads_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return _empty_proposals()

    for column in ("Spend", "Conversions", "ConversionValue"):
        df[column] = _numeric(df[column])

    if "Category" not in df.columns:
        df["Category"] = "All"
    if "ProductGroup" not in df.columns:
        df["ProductGroup"] = df["Category"]

    df["Category"] = df["Category"].fillna("All").astype(str)
    df["ProductGroup"] = df["ProductGroup"].fillna(df["Category"]).astype(str)

    analysis_start = df["Date"].min().date()
    analysis_end = df["Date"].max().date()
    analysis_days = max((analysis_end - analysis_start).days + 1, 1)
    minimum_history_days = max(4, min(14, analysis_days // 2))

    group_columns = ["Channel", "Category", "ProductGroup"]
    grouped = (
        df.groupby(group_columns, dropna=False)
        .agg(
            TotalSpend=("Spend", "sum"),
            TotalConversions=("Conversions", "sum"),
            TotalRevenue=("ConversionValue", "sum"),
            HistoryDays=("Date", "nunique"),
            SourceCampaignCount=("Campaign", "nunique"),
        )
        .reset_index()
    )

    grouped["HistoricalROAS"] = np.where(
        grouped["TotalSpend"] > 0,
        grouped["TotalRevenue"] / grouped["TotalSpend"],
        0.0,
    )
    grouped = grouped[
        (grouped["TotalSpend"] > 0)
        & (grouped["TotalConversions"] > 0)
        & (grouped["HistoryDays"] >= minimum_history_days)
    ].copy()
    if grouped.empty:
        return _empty_proposals()

    grouped["DailySpend"] = grouped["TotalSpend"] / analysis_days
    grouped["AverageOrderValue"] = (
        grouped["TotalRevenue"]
        / grouped["TotalConversions"].replace(0, np.nan)
    ).fillna(0.0)

    def normalize(series: pd.Series) -> pd.Series:
        maximum = float(series.max())
        if maximum <= 0:
            return pd.Series(0.0, index=series.index)
        return series / maximum

    grouped["OpportunityScore"] = 100 * (
        0.40 * normalize(grouped["TotalRevenue"])
        + 0.25 * normalize(grouped["TotalConversions"])
        + 0.20 * normalize(grouped["HistoricalROAS"].clip(upper=target_roas * 3))
        + 0.15 * normalize(grouped["HistoryDays"])
    )
    grouped = grouped.sort_values(
        ["OpportunityScore", "TotalRevenue"],
        ascending=False,
    ).head(max(1, int(max_proposals)))

    start_date = analysis_end + timedelta(days=1)
    end_date = start_date + timedelta(days=test_duration_days - 1)
    holiday_strategy = _next_holiday_strategy(analysis_end, holiday_map)
    proposals: list[dict[str, object]] = []

    for rank, row in enumerate(grouped.itertuples(index=False), start=1):
        historical_roas = float(row.HistoricalROAS)
        expected_roas = min(
            max(0.0, historical_roas * 0.75),
            max(float(target_roas) * 2.5, float(target_roas)),
        )
        suggested_budget = max(10.0, min(500.0, float(row.DailySpend) * 0.20))
        minimum_budget = max(5.0, suggested_budget * 0.75)
        maximum_budget = suggested_budget * 1.25
        expected_revenue = suggested_budget * expected_roas
        expected_conversions = (
            expected_revenue / float(row.AverageOrderValue)
            if float(row.AverageOrderValue) > 0
            else 0.0
        )
        confidence = _confidence(
            int(row.HistoryDays),
            int(row.SourceCampaignCount),
            float(row.TotalConversions),
        )
        risk = _risk(confidence, expected_roas, target_roas)
        channel = str(row.Channel)
        category = str(row.Category)
        product_group = str(row.ProductGroup)
        proposal_name = (
            f"new-{_slug(channel)}-{_slug(category)}-"
            f"{_slug(product_group)}-test"
        )[:120]

        proposals.append(
            {
                "ProposalId": f"NCP-{analysis_end:%Y%m%d}-{rank:02d}",
                "ProposedCampaignName": proposal_name,
                "ProposalType": "New Campaign Test",
                "Channel": channel,
                "Category": category,
                "ProductGroup": product_group,
                "SuggestedDailyBudget": round(suggested_budget, 2),
                "MinimumDailyBudget": round(minimum_budget, 2),
                "MaximumDailyBudget": round(maximum_budget, 2),
                "TestDurationDays": int(test_duration_days),
                "ProposedStartDate": start_date.isoformat(),
                "ProposedEndDate": end_date.isoformat(),
                "ExpectedDailyConversions": round(expected_conversions, 4),
                "ExpectedDailyRevenue": round(expected_revenue, 2),
                "ExpectedDailyProfit": round(
                    expected_revenue - suggested_budget,
                    2,
                ),
                "ExpectedROAS": round(expected_roas, 4),
                "TargetROAS": round(float(target_roas), 4),
                "ConfidenceLevel": confidence,
                "RiskLevel": risk,
                "SourceCampaignCount": int(row.SourceCampaignCount),
                "HistoryDays": int(row.HistoryDays),
                "HistoricalROAS": round(historical_roas, 4),
                "OpportunityScore": round(float(row.OpportunityScore), 2),
                "HolidayStrategy": holiday_strategy,
                "RecommendationReason": (
                    f"{category} / {product_group} segmentinde "
                    f"{int(row.HistoryDays)} günlük ve "
                    f"{int(row.SourceCampaignCount)} kampanyalık kanıt bulundu. "
                    "Öneri, ayrı ve kontrollü bir genişleme testidir."
                ),
                "SuccessCriteria": (
                    f"14 gün sonunda ROAS ≥ {target_roas:.2f}, "
                    "harcama ve dönüşüm takibi; bütçe artışı yalnızca "
                    "hedef sağlanırsa."
                ),
                "Guardrails": (
                    "İlk 7 gün bütçeyi artırmayın; izleme ve dönüşüm "
                    "kurulumunu doğrulayın; maksimum bütçe sınırını aşmayın."
                ),
            }
        )

    return pd.DataFrame(proposals, columns=PROPOSAL_COLUMNS)

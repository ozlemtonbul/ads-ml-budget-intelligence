# -*- coding: utf-8 -*-
"""
 Ads ML Budget Intelligence Pipeline — v2

This project is a Python-based predictive decision-support pipeline that integrates
Google Ads data to evaluate campaign performance, forecast future results, simulate
budget scenarios, validate model quality, explain decision drivers, and recommend
optimized campaign and portfolio-level budget allocation.

The pipeline now extends its core ML engine with product and category-level
granularity, calendar-aware seasonality modelling, ROAS target compliance
tracking, and LLM-generated executive commentary — enabling both automated
budget decisions and human-readable justification in a single run.

Core Capabilities:
- Google Ads API integration at campaign and ad group level
- Product and category breakdown extracted from ad group naming conventions
- KPI engineering including CTR, CPC, Conversion Rate, CPA, ROAS, and Profit
- Turkish public holidays and religious holidays calendar (Eid al-Fitr, Eid al-Adha)
- Pre-holiday (eve) effect detection for 1-3 days prior to each holiday
- Seasonal ROAS multiplier coefficients (Winter, Spring, Summer, Autumn)
- Combined holiday + season expected ROAS multiplier per campaign row
- Predictive modelling for next-period conversions and revenue
- Model validation (MAE, RMSE, R²) with train/test split reporting
- Feature importance reporting across both conversion and revenue models
- Budget scenario simulation (50%, 75%, 100%, 120%, 150%) with season adjustment
- ROAS target compliance check with gap percentage and status label
- Campaign-level optimization logic with ROAS-gated budget increase decisions
- Portfolio-level budget allocation normalized to total current spend
- Confidence scoring (High / Medium / Low) based on history depth and model R²
- Built-in guardrails, fallback logic, and confidence-based decision filtering
- LLM-generated executive commentary per campaign via Anthropic Claude API
- LLM-generated portfolio-level summary commentary saved as a separate file
- Daily, weekly, and monthly aggregated output reports with calendar context
- Category-level and product group-level KPI summary reports
- Holiday vs normal day performance impact analysis

Business Purpose:
This pipeline moves beyond static reporting by combining analytics, prediction,
simulation, optimization, validation, and explainability to support smarter budget
allocation and revenue-focused campaign decisions. The addition of seasonality and
holiday modelling ensures that budget recommendations account for known demand
patterns in the Turkish e-commerce calendar. The LLM commentary layer translates
model outputs into plain-language justifications that non-technical stakeholders
can act on directly.

Environment Variables:
- GOOGLE_ADS_DEVELOPER_TOKEN    Google Ads API developer token (required)
- GOOGLE_ADS_CLIENT_ID          OAuth2 client ID (required)
- GOOGLE_ADS_CLIENT_SECRET      OAuth2 client secret (required)
- GOOGLE_ADS_REFRESH_TOKEN      OAuth2 refresh token (required)
- GOOGLE_ADS_LOGIN_CUSTOMER_ID  MCC / manager account ID (required)
- GOOGLE_ADS_CUSTOMER_ID        Target customer account ID (required)
- DATE_MODE                     yesterday | last_30_days | last_60_days | custom (default: last_60_days)
- DATE_FROM                     Start date in YYYY-MM-DD format (required when DATE_MODE=custom)
- DATE_TO                       End date in YYYY-MM-DD format (required when DATE_MODE=custom)
- VICCO_OUTPUT_DIR              Output directory path (default: ./output)
- TARGET_ROAS                   ROAS target threshold for compliance checks (default: 3.0)
- ANTHROPIC_API_KEY             Anthropic API key for LLM commentary (optional)
- LLM_LANG                      Language for LLM prompts: en (default: en)
- LLM_MAX_CAMPAIGNS             Max campaigns to generate LLM commentary for (default: 20)

Output Files:
- ads_budget_scenarios.csv                      All simulated budget scenarios per campaign
- ads_budget_optimization_recommendations.csv   Per-campaign optimization recommendations
- ads_portfolio_budget_allocation.csv           Portfolio-normalized budget allocation
- ads_recommendation_summary.csv                Final summary with LLM commentary column
- ads_model_validation_metrics.csv              MAE, RMSE, R² for both models
- ads_feature_importance.csv                    Feature importances for both models
- ads_category_summary.csv                      KPI aggregation by product category and season
- ads_product_summary.csv                       KPI aggregation by product group and campaign
- ads_holiday_impact.csv                        Holiday vs pre-holiday vs normal day comparison
- ads_daily_fact.csv                            Daily KPIs with calendar context per campaign
- ads_weekly_campaign_summary.csv               Weekly KPIs per campaign and category
- ads_monthly_campaign_summary.csv              Monthly KPIs per campaign and category
- ads_portfolio_executive_commentary.txt        LLM-generated portfolio summary (plain text)
- ads_rule_based_fallback_recommendations.csv   Rule-based output when ML data is insufficient

Ad Group Naming Convention:
For category and product extraction to work correctly, ad group names should follow
the format: "Category | Product Group"
Example: "Shoes | Nike Air Max" extracts Category=Shoes, ProductGroup=Nike Air Max
If no pipe separator is found, the first word of the ad group name is used as Category.

Author: Ozlem Tonbul
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import anthropic
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("vicco_ads_ml_budget_intelligence")


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing environment variable: {name}")
    return value.strip()


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

def resolve_date_range() -> Tuple[str, str]:
    mode = os.getenv("DATE_MODE", "last_60_days").lower().strip()

    if mode == "yesterday":
        target_date = datetime.now().date() - timedelta(days=1)
        date_str = target_date.strftime("%Y-%m-%d")
        return date_str, date_str

    if mode == "last_30_days":
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    if mode == "last_60_days":
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=59)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    if mode == "custom":
        return env("DATE_FROM"), env("DATE_TO")

    raise ValueError(
        "DATE_MODE must be one of: yesterday, last_30_days, last_60_days, custom"
    )


# ---------------------------------------------------------------------------
# TURKEY PUBLIC HOLIDAYS & RELIGIOUS HOLIDAYS
# ---------------------------------------------------------------------------

def get_turkey_public_holidays(year: int) -> Dict[str, str]:
    """
    Fixed Turkish public holidays and estimated religious holidays.
    Religious holidays (Ramadan / Eid) shift each year; approximate
    dates for 2024-2026 are hardcoded here.
    In production, use a holidays library or API to keep these current.
    """
    fixed = {
        f"{year}-01-01": "New Year",
        f"{year}-04-23": "National Sovereignty and Children Day",
        f"{year}-05-01": "Labour Day",
        f"{year}-05-19": "Commemoration of Ataturk / Youth Day",
        f"{year}-07-15": "Democracy Day",
        f"{year}-08-30": "Victory Day",
        f"{year}-10-29": "Republic Day",
    }

    # Approximate Eid al-Fitr dates (eve day included)
    ramazan = {
        2024: ["2024-04-09", "2024-04-10", "2024-04-11"],
        2025: ["2025-03-29", "2025-03-30", "2025-03-31"],
        2026: ["2026-03-19", "2026-03-20", "2026-03-21"],
    }
    # Approximate Eid al-Adha dates (eve day included)
    kurban = {
        2024: ["2024-06-15", "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19"],
        2025: ["2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09"],
        2026: ["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30"],
    }

    result = dict(fixed)
    for d in ramazan.get(year, []):
        result[d] = "Eid al-Fitr"
    for d in kurban.get(year, []):
        result[d] = "Eid al-Adha"

    return result


def build_holiday_map(date_from: str, date_to: str) -> Dict[str, str]:
    """Returns holiday map for all years in the given date range."""
    start_year = int(date_from[:4])
    end_year = int(date_to[:4])
    holiday_map: Dict[str, str] = {}
    for y in range(start_year, end_year + 1):
        holiday_map.update(get_turkey_public_holidays(y))
    return holiday_map


# ---------------------------------------------------------------------------
# SEASONAL EFFECT
# ---------------------------------------------------------------------------

SEASON_MAP = {
    12: ("Winter", "winter"), 1: ("Winter", "winter"), 2: ("Winter", "winter"),
    3: ("Spring", "spring"), 4: ("Spring", "spring"), 5: ("Spring", "spring"),
    6: ("Summer", "summer"), 7: ("Summer", "summer"), 8: ("Summer", "summer"),
    9: ("Autumn", "autumn"), 10: ("Autumn", "autumn"), 11: ("Autumn", "autumn"),
}

# General seasonal ROAS multiplier expectation for e-commerce
SEASON_ROAS_MULTIPLIER = {
    "winter": 1.15,   # Q4 peak shopping period
    "spring": 1.05,
    "summer": 0.95,   # Summer tends to be slower in Turkey
    "autumn": 1.10,   # Back-to-school + new season
}

# Expected ROAS multiplier during holidays
HOLIDAY_ROAS_MULTIPLIER = 1.20


def add_calendar_context_features(df: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    """
    Adds calendar context to each row:
    - IsHoliday / HolidayName
    - IsPreHoliday (1-3 days before a holiday)
    - Season / SeasonEN
    - SeasonROASMultiplier
    - ExpectedROASMultiplier (holiday + season combined)
    """
    df = df.copy()

    date_str = df["Date"].dt.strftime("%Y-%m-%d")
    df["IsHoliday"] = date_str.map(lambda d: d in holiday_map).astype(int)
    df["HolidayName"] = date_str.map(lambda d: holiday_map.get(d, ""))

    # Pre-holiday days (1-3 days prior, eve effect)
    holiday_dates = {datetime.strptime(d, "%Y-%m-%d").date() for d in holiday_map}
    df["IsPreHoliday"] = df["Date"].dt.date.map(
        lambda d: int(any((d + timedelta(days=i)) in holiday_dates for i in range(1, 4)))
    )

    df["Season"] = df["Date"].dt.month.map(lambda m: SEASON_MAP[m][0])
    df["SeasonEN"] = df["Date"].dt.month.map(lambda m: SEASON_MAP[m][1])
    df["SeasonROASMultiplier"] = df["SeasonEN"].map(SEASON_ROAS_MULTIPLIER)
    df["ExpectedROASMultiplier"] = df.apply(
        lambda r: HOLIDAY_ROAS_MULTIPLIER * r["SeasonROASMultiplier"]
        if r["IsHoliday"] or r["IsPreHoliday"]
        else r["SeasonROASMultiplier"],
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"] = df["Clicks"] / df["Impressions"].replace(0, np.nan)
    df["CPC"] = df["Spend"] / df["Clicks"].replace(0, np.nan)
    df["ConvRate"] = df["Conversions"] / df["Clicks"].replace(0, np.nan)
    df["CPA"] = df["Spend"] / df["Conversions"].replace(0, np.nan)
    df["ROAS"] = df["ConversionValue"] / df["Spend"].replace(0, np.nan)
    df["Profit"] = df["ConversionValue"] - df["Spend"]
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ---------------------------------------------------------------------------
# ROAS target compliance check
# ---------------------------------------------------------------------------

def compute_roas_target_gap(df: pd.DataFrame, target_roas: float) -> pd.DataFrame:
    """
    Adds ROAS target gap and compliance status for each campaign.
    target_roas comes from the TARGET_ROAS env variable (default 3.0).
    """
    df = df.copy()
    df["TargetROAS"] = target_roas
    df["ROASGap"] = df["ROAS"] - target_roas
    df["ROASGapPct"] = np.where(
        target_roas > 0,
        (df["ROASGap"] / target_roas) * 100,
        0,
    )
    df["ROASStatus"] = df["ROAS"].apply(
        lambda r: "Above Target" if r >= target_roas * 1.10
        else ("On Target" if r >= target_roas * 0.90 else "Below Target")
    )
    return df


# ---------------------------------------------------------------------------
# Google Ads API — product & category breakdown
# ---------------------------------------------------------------------------

def build_google_ads_client() -> GoogleAdsClient:
    config = {
        "developer_token": env("GOOGLE_ADS_DEVELOPER_TOKEN"),
        "client_id": env("GOOGLE_ADS_CLIENT_ID"),
        "client_secret": env("GOOGLE_ADS_CLIENT_SECRET"),
        "refresh_token": env("GOOGLE_ADS_REFRESH_TOKEN"),
        "login_customer_id": env("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
        "use_proto_plus": True,
    }
    return GoogleAdsClient.load_from_dict(config)


def fetch_ads_purchase_only(date_from: str, date_to: str) -> pd.DataFrame:
    """
    Fetches PURCHASE conversion data at campaign + ad group level.
    ad_group.name is used to extract product/category information.
    """
    customer_id = env("GOOGLE_ADS_CUSTOMER_ID")
    client = build_google_ads_client()
    service = client.get_service("GoogleAdsService")

    query = f"""
        SELECT
          segments.date,
          campaign.id,
          campaign.name,
          campaign.advertising_channel_type,
          ad_group.id,
          ad_group.name,
          metrics.impressions,
          metrics.clicks,
          metrics.conversions,
          metrics.conversions_value,
          metrics.cost_micros
        FROM ad_group
        WHERE
          segments.date BETWEEN '{date_from}' AND '{date_to}'
          AND segments.conversion_action_category = 'PURCHASE'
    """

    rows = []
    try:
        response = service.search(customer_id=customer_id, query=query)
        for row in response:
            rows.append({
                "Date": str(row.segments.date),
                "CampaignId": int(row.campaign.id or 0),
                "Campaign": row.campaign.name or "UNKNOWN",
                "Channel": str(row.campaign.advertising_channel_type),
                "AdGroupId": int(row.ad_group.id or 0),
                "AdGroup": row.ad_group.name or "UNKNOWN",
                "Impressions": int(row.metrics.impressions or 0),
                "Clicks": int(row.metrics.clicks or 0),
                "Conversions": float(row.metrics.conversions or 0.0),
                "ConversionValue": float(row.metrics.conversions_value or 0.0),
                "Spend": float(row.metrics.cost_micros or 0.0) / 1_000_000.0,
            })
    except GoogleAdsException as exc:
        logger.error("Google Ads API error: %s", exc)
        raise

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["CampaignId", "AdGroupId", "Impressions", "Clicks",
                    "Conversions", "ConversionValue", "Spend"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["Campaign", "Channel", "AdGroup"]:
        df[col] = df[col].replace(["", "None", None], "UNKNOWN")

    # Product & category extraction
    # Rule: if AdGroup name contains "|", format "Category | Product" is expected
    # Example: "Shoes | Nike Air Max" -> Category=Shoes, Product=Nike Air Max
    # Otherwise, first word of ad group name is used as category
    df["Category"] = df["AdGroup"].apply(_extract_category)
    df["ProductGroup"] = df["AdGroup"].apply(_extract_product)

    return df


def _extract_category(adgroup_name: str) -> str:
    if "|" in adgroup_name:
        return adgroup_name.split("|")[0].strip()
    # First word of ad group name assumed as category
    return adgroup_name.split()[0].strip() if adgroup_name else "UNKNOWN"


def _extract_product(adgroup_name: str) -> str:
    if "|" in adgroup_name:
        return adgroup_name.split("|", 1)[1].strip()
    return adgroup_name.strip()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["MonthNum"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CampaignId", "Date"]).copy()
    group = df.groupby("CampaignId")

    for col in ["Spend", "Clicks", "Conversions", "ConversionValue", "ROAS", "CPA", "CTR", "ConvRate"]:
        df[f"{col}_lag_1"] = group[col].shift(1)

    for col in ["Spend", "Clicks", "Conversions", "ConversionValue"]:
        df[f"{col}_lag_7_avg"] = (
            group[col].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    return df.fillna(0)


def prepare_training_data(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    df = ads_raw.copy()
    df = compute_kpis(df)
    df = add_time_features(df)
    df = add_calendar_context_features(df, holiday_map)
    df = add_lag_features(df)

    df = df.sort_values(["CampaignId", "Date"]).copy()
    group = df.groupby("CampaignId")
    df["Target_Conversions_Next"] = group["Conversions"].shift(-1)
    df["Target_Revenue_Next"] = group["ConversionValue"].shift(-1)

    df = df.dropna(subset=["Target_Conversions_Next", "Target_Revenue_Next"]).copy()
    return df


def get_feature_columns() -> List[str]:
    base = [
        "CampaignId",
        "Spend", "Impressions", "Clicks", "Conversions", "ConversionValue",
        "CTR", "CPC", "ConvRate", "CPA", "ROAS", "Profit",
        "DayOfWeek", "DayOfMonth", "MonthNum", "WeekOfYear", "IsWeekend", "Quarter",
        "IsHoliday", "IsPreHoliday",
        "SeasonROASMultiplier", "ExpectedROASMultiplier",
        "Spend_lag_1", "Spend_lag_7_avg",
        "Clicks_lag_1", "Clicks_lag_7_avg",
        "Conversions_lag_1", "Conversions_lag_7_avg",
        "ConversionValue_lag_1", "ConversionValue_lag_7_avg",
        "ROAS_lag_1", "CPA_lag_1", "CTR_lag_1", "ConvRate_lag_1",
    ]
    return base


# ---------------------------------------------------------------------------
# Model training & validation
# ---------------------------------------------------------------------------

def train_and_validate_models(train_df: pd.DataFrame):
    feature_cols = get_feature_columns()
    # Use only available columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    X = train_df[feature_cols]

    y_conv = train_df["Target_Conversions_Next"]
    y_rev = train_df["Target_Revenue_Next"]

    X_train_conv, X_test_conv, y_train_conv, y_test_conv = train_test_split(
        X, y_conv, test_size=0.2, random_state=42
    )
    X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
        X, y_rev, test_size=0.2, random_state=42
    )

    model_conv = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=2, random_state=42)
    model_rev = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=2, random_state=42)

    model_conv.fit(X_train_conv, y_train_conv)
    model_rev.fit(X_train_rev, y_train_rev)

    pred_conv = model_conv.predict(X_test_conv)
    pred_rev = model_rev.predict(X_test_rev)

    metrics_df = pd.DataFrame([
        {
            "Model": "Conversions",
            "MAE": float(mean_absolute_error(y_test_conv, pred_conv)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_conv, pred_conv))),
            "R2": float(r2_score(y_test_conv, pred_conv)),
            "TrainRows": len(X_train_conv),
            "TestRows": len(X_test_conv),
        },
        {
            "Model": "Revenue",
            "MAE": float(mean_absolute_error(y_test_rev, pred_rev)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_rev, pred_rev))),
            "R2": float(r2_score(y_test_rev, pred_rev)),
            "TrainRows": len(X_train_rev),
            "TestRows": len(X_test_rev),
        },
    ])

    importance_conv = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_conv.feature_importances_,
        "Model": "Conversions",
    }).sort_values("Importance", ascending=False)

    importance_rev = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model_rev.feature_importances_,
        "Model": "Revenue",
    }).sort_values("Importance", ascending=False)

    feature_importance_df = pd.concat([importance_conv, importance_rev], ignore_index=True)
    return model_conv, model_rev, feature_cols, metrics_df, feature_importance_df


# ---------------------------------------------------------------------------
# Latest campaign state
# ---------------------------------------------------------------------------

def get_latest_campaign_state(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    df = prepare_training_data(ads_raw, holiday_map)
    if df.empty:
        return df

    latest_df = (
        df.sort_values(["CampaignId", "Date"])
        .groupby(["CampaignId", "Campaign", "Channel"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest_df


def safe_prediction(value: float) -> float:
    if value is None or np.isnan(value):
        return 0.0
    return max(0.0, float(value))


# ---------------------------------------------------------------------------
# Budget scenario simulation (season/holiday adjusted)
# ---------------------------------------------------------------------------

def simulate_budget_scenarios(
    latest_df: pd.DataFrame,
    model_conv,
    model_rev,
    feature_cols: List[str],
) -> pd.DataFrame:
    scenario_factors = [0.50, 0.75, 1.00, 1.20, 1.50]
    results = []

    for _, row in latest_df.iterrows():
        for factor in scenario_factors:
            sim_row = row.copy()
            sim_row["ScenarioFactor"] = factor
            sim_row["ScenarioSpend"] = row["Spend"] * factor

            sim_row["Spend"] = sim_row["ScenarioSpend"]
            sim_row["CPC"] = sim_row["Spend"] / row["Clicks"] if row["Clicks"] > 0 else 0
            sim_row["CPA"] = sim_row["Spend"] / row["Conversions"] if row["Conversions"] > 0 else 0
            sim_row["ROAS"] = row["ConversionValue"] / sim_row["Spend"] if sim_row["Spend"] > 0 else 0
            sim_row["Profit"] = row["ConversionValue"] - sim_row["Spend"]

            X_input = pd.DataFrame([sim_row])
            avail_cols = [c for c in feature_cols if c in X_input.columns]
            X_input = X_input[avail_cols]

            pred_conv = safe_prediction(model_conv.predict(X_input)[0])
            pred_rev = safe_prediction(model_rev.predict(X_input)[0])

            # Season/holiday multiplier adjusted prediction
            season_mult = float(row.get("ExpectedROASMultiplier", 1.0))
            pred_rev_adjusted = pred_rev * season_mult
            pred_profit = pred_rev_adjusted - sim_row["ScenarioSpend"]
            pred_roas = pred_rev_adjusted / sim_row["ScenarioSpend"] if sim_row["ScenarioSpend"] > 0 else 0

            results.append({
                "CampaignId": row["CampaignId"],
                "Campaign": row["Campaign"],
                "Channel": row["Channel"],
                "Category": row.get("Category", ""),
                "ProductGroup": row.get("ProductGroup", ""),
                "Season": row.get("Season", ""),
                "IsHoliday": int(row.get("IsHoliday", 0)),
                "IsPreHoliday": int(row.get("IsPreHoliday", 0)),
                "HolidayName": row.get("HolidayName", ""),
                "ExpectedROASMultiplier": round(season_mult, 3),
                "CurrentSpend": round(float(row["Spend"]), 2),
                "ScenarioFactor": factor,
                "ScenarioSpend": round(float(sim_row["ScenarioSpend"]), 2),
                "PredictedConversions": round(pred_conv, 2),
                "PredictedRevenue": round(pred_rev_adjusted, 2),
                "PredictedProfit": round(pred_profit, 2),
                "PredictedROAS": round(pred_roas, 4),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Optimal scenario & baseline uplift
# ---------------------------------------------------------------------------

def choose_optimal_scenario(sim_df: pd.DataFrame) -> pd.DataFrame:
    if sim_df.empty:
        return sim_df

    sim_df = sim_df.copy()
    sim_df["OptimizationScore"] = (
        (sim_df["PredictedRevenue"] * 0.45) +
        (sim_df["PredictedProfit"] * 0.35) +
        (sim_df["PredictedROAS"] * 100 * 0.20)
    )

    best_df = (
        sim_df.sort_values(["CampaignId", "OptimizationScore"], ascending=[True, False])
        .groupby("CampaignId", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_df


def add_baseline_uplift(best_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
    df = best_df.copy()
    baseline_df = sim_df[sim_df["ScenarioFactor"] == 1.0][
        ["CampaignId", "PredictedRevenue", "PredictedProfit", "PredictedConversions"]
    ].rename(columns={
        "PredictedRevenue": "BaselinePredictedRevenue",
        "PredictedProfit": "BaselinePredictedProfit",
        "PredictedConversions": "BaselinePredictedConversions",
    })

    df = df.merge(baseline_df, on="CampaignId", how="left")
    df["RevenueUplift"] = df["PredictedRevenue"] - df["BaselinePredictedRevenue"]
    df["ProfitUplift"] = df["PredictedProfit"] - df["BaselinePredictedProfit"]
    df["ConversionUplift"] = df["PredictedConversions"] - df["BaselinePredictedConversions"]
    df["RevenueUpliftPct"] = np.where(
        df["BaselinePredictedRevenue"] > 0,
        (df["RevenueUplift"] / df["BaselinePredictedRevenue"]) * 100,
        0,
    )
    return df


# ---------------------------------------------------------------------------
# Action recommendation (ROAS target integrated)
# ---------------------------------------------------------------------------

def build_action_recommendation(best_df: pd.DataFrame, target_roas: float) -> pd.DataFrame:
    df = best_df.copy()

    def decide_action(row):
        current_spend = row["CurrentSpend"]
        optimal_spend = row["ScenarioSpend"]
        pred_roas = row["PredictedROAS"]

        if current_spend <= 0:
            return "Review", "No active spend detected; manual review required."

        if row["PredictedConversions"] <= 0 and optimal_spend < current_spend:
            return "Pause / Review", "Predicted value remains weak even under lower spend scenarios."

        ratio = optimal_spend / current_spend if current_spend > 0 else 1

        # ROAS target check
        roas_ok = pred_roas >= target_roas * 0.90

        if ratio >= 1.15 and roas_ok:
            return "Increase Budget", (
                f"Predicted performance suggests scaling is likely to improve results. "
                f"Predicted ROAS ({pred_roas:.2f}) is above target ({target_roas:.1f})."
            )
        if ratio >= 1.15 and not roas_ok:
            return "Increase Budget (ROAS Risk)", (
                f"Budget increase potential exists but predicted ROAS ({pred_roas:.2f}) "
                f"does not meet target ({target_roas:.1f}). Proceed with caution."
            )
        if ratio <= 0.85:
            return "Reduce Budget", (
                f"Predicted return suggests the campaign is overfunded at its current level."
            )
        return "Maintain", "Predicted performance supports keeping the budget near its current level."

    decisions = df.apply(decide_action, axis=1)
    df["RecommendedAction"] = decisions.apply(lambda x: x[0])
    df["RecommendationReason"] = decisions.apply(lambda x: x[1])

    df["BudgetChange"] = df["ScenarioSpend"] - df["CurrentSpend"]
    df["BudgetChangePct"] = np.where(
        df["CurrentSpend"] > 0,
        (df["BudgetChange"] / df["CurrentSpend"]) * 100,
        0,
    )
    df = df.rename(columns={"ScenarioSpend": "RecommendedBudget"})
    return df


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def build_confidence_scores(
    recommendation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    df = recommendation_df.copy()
    campaign_history = (
        train_df.groupby(["CampaignId"], as_index=False)
        .size()
        .rename(columns={"size": "HistoryRows"})
    )
    df = df.merge(campaign_history, on="CampaignId", how="left")
    df["HistoryRows"] = df["HistoryRows"].fillna(0)

    conv_r2 = float(metrics_df.loc[metrics_df["Model"] == "Conversions", "R2"].iloc[0])
    rev_r2 = float(metrics_df.loc[metrics_df["Model"] == "Revenue", "R2"].iloc[0])
    avg_r2 = (conv_r2 + rev_r2) / 2

    def confidence_label(row):
        if row["HistoryRows"] >= 20 and avg_r2 >= 0.60:
            return "High"
        if row["HistoryRows"] >= 10 and avg_r2 >= 0.30:
            return "Medium"
        return "Low"

    df["ConfidenceLevel"] = df.apply(confidence_label, axis=1)
    return df


def apply_confidence_guardrail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    low_conf_mask = df["ConfidenceLevel"] == "Low"
    df.loc[low_conf_mask, "RecommendedAction"] = "Review"
    df.loc[low_conf_mask, "RecommendationReason"] = (
        "Low confidence prediction. Manual validation is recommended before taking action."
    )
    return df


# ---------------------------------------------------------------------------
# Portfolio budget allocation
# ---------------------------------------------------------------------------

def build_portfolio_allocation(recommendation_df: pd.DataFrame) -> pd.DataFrame:
    df = recommendation_df.copy()
    if df.empty:
        return df

    total_current = df["CurrentSpend"].sum()
    total_recommended = df["RecommendedBudget"].sum()

    if total_recommended <= 0:
        df["OptimizedPortfolioBudget"] = df["CurrentSpend"]
    else:
        df["OptimizedPortfolioBudget"] = (
            df["RecommendedBudget"] / total_recommended
        ) * total_current

    df["OptimizedPortfolioBudget"] = df["OptimizedPortfolioBudget"].round(2)
    df["PortfolioBudgetChange"] = (df["OptimizedPortfolioBudget"] - df["CurrentSpend"]).round(2)
    return df


# ---------------------------------------------------------------------------
# Category & product breakdown summaries
# ---------------------------------------------------------------------------

def build_category_summary(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    """KPI summary by category."""
    df = ads_raw.copy()
    df = compute_kpis(df)
    df = add_calendar_context_features(df, holiday_map)

    cat_df = df.groupby(["Category", "Season"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    cat_df = compute_kpis(cat_df)
    cat_df = cat_df.sort_values("ConversionValue", ascending=False)
    return cat_df


def build_product_summary(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    """KPI summary by product group."""
    df = ads_raw.copy()
    df = compute_kpis(df)
    df = add_calendar_context_features(df, holiday_map)

    prod_df = df.groupby(["Category", "ProductGroup", "Campaign"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    prod_df = compute_kpis(prod_df)
    prod_df = prod_df.sort_values("ConversionValue", ascending=False)
    return prod_df


def build_holiday_impact_summary(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]) -> pd.DataFrame:
    """Performance difference on holidays vs normal days."""
    df = ads_raw.copy()
    df = compute_kpis(df)
    df = add_calendar_context_features(df, holiday_map)

    impact_df = df.groupby(["IsHoliday", "IsPreHoliday"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
        DayCount=("Date", "nunique"),
    )
    impact_df["AvgDailySpend"] = impact_df["Spend"] / impact_df["DayCount"]
    impact_df["AvgDailyRevenue"] = impact_df["ConversionValue"] / impact_df["DayCount"]
    impact_df["ROAS"] = impact_df["ConversionValue"] / impact_df["Spend"].replace(0, np.nan)
    impact_df = impact_df.fillna(0)
    impact_df["PeriodLabel"] = impact_df.apply(
        lambda r: "Holiday" if r["IsHoliday"] else ("Pre-Holiday" if r["IsPreHoliday"] else "Normal Day"),
        axis=1,
    )
    return impact_df


# ---------------------------------------------------------------------------
# LLM EXECUTIVE COMMENTARY
# ---------------------------------------------------------------------------

def _build_llm_campaign_prompt(row: pd.Series, target_roas: float) -> str:
    """Builds the LLM prompt for a single campaign."""
    season = row.get("Season", "")
    holiday_ctx = ""
    if row.get("IsHoliday", 0):
        holiday_ctx = f"Holiday effect active: {row.get('HolidayName', '')}."
    elif row.get("IsPreHoliday", 0):
        holiday_ctx = "Pre-holiday period (eve effect expected)."

    roas_status = row.get("ROASStatus", "")
    roas_gap_pct = row.get("ROASGapPct", 0)

    prompt = f"""
You are a digital marketing budget analyst. Write a concise, executive-level commentary
(max 3 sentences) based on the campaign data below. Include: current status, reason for
the outcome, and the budget recommendation.

Campaign: {row.get('Campaign', '')}
Category: {row.get('Category', '')} / Product: {row.get('ProductGroup', '')}
Channel: {row.get('Channel', '')}
Season: {season}
{holiday_ctx}
Current daily spend: {row.get('CurrentSpend', 0):.2f}
Recommended budget: {row.get('RecommendedBudget', 0):.2f}
Budget change: {row.get('BudgetChangePct', 0):.1f}%
Predicted ROAS: {row.get('PredictedROAS', 0):.2f} (target: {target_roas:.1f})
ROAS status: {roas_status} (gap vs target: {roas_gap_pct:.1f}%)
Predicted revenue: {row.get('PredictedRevenue', 0):.2f}
Predicted profit: {row.get('PredictedProfit', 0):.2f}
Confidence level: {row.get('ConfidenceLevel', '')}
Recommended action: {row.get('RecommendedAction', '')}

Commentary (3 sentences, executive summary):
"""
    return prompt.strip()


def generate_llm_commentary(
    summary_df: pd.DataFrame,
    target_roas: float,
    lang: str = "tr",
    max_campaigns: int = 20,
) -> pd.DataFrame:
    """
    Calls Claude API for each campaign to generate executive commentary.
    ANTHROPIC_API_KEY env variable is required.
    max_campaigns: limit to control API cost.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY missing. Skipping LLM commentary.")
        summary_df["ExecutiveCommentary"] = "LLM commentary skipped (API key missing)."
        return summary_df

    client = anthropic.Anthropic(api_key=api_key)
    commentaries = []

    rows_to_process = summary_df.head(max_campaigns)
    logger.info("Generating LLM commentary for %d campaigns...", len(rows_to_process))

    for _, row in rows_to_process.iterrows():
        prompt = _build_llm_campaign_prompt(row, target_roas)
        try:
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            commentary = message.content[0].text.strip()
        except Exception as e:
            logger.warning("LLM error for campaign %s: %s", row.get("Campaign", ""), e)
            commentary = "Commentary could not be generated."

        commentaries.append(commentary)

    # Empty commentary for remaining rows
    remaining = len(summary_df) - len(rows_to_process)
    commentaries.extend(["Commentary limit reached (max_campaigns)."] * remaining)

    summary_df = summary_df.copy()
    summary_df["ExecutiveCommentary"] = commentaries
    return summary_df


def generate_portfolio_summary_commentary(
    portfolio_df: pd.DataFrame,
    category_df: pd.DataFrame,
    target_roas: float,
    lang: str = "tr",
) -> str:
    """
    Generates a single overall summary commentary for the entire portfolio.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "LLM portfolio commentary skipped (API key missing)."

    client = anthropic.Anthropic(api_key=api_key)

    total_spend = portfolio_df["CurrentSpend"].sum()
    total_recommended = portfolio_df["OptimizedPortfolioBudget"].sum()
    total_pred_rev = portfolio_df.get("PredictedRevenue", pd.Series([0])).sum()
    top_category = category_df.iloc[0]["Category"] if not category_df.empty else "N/A"
    top_category_roas = category_df.iloc[0]["ROAS"] if not category_df.empty else 0
    campaign_count = len(portfolio_df)
    increase_count = portfolio_df["RecommendedAction"].str.contains("Increase", na=False).sum()
    reduce_count = portfolio_df["RecommendedAction"].str.contains("Reduce", na=False).sum()

    prompt = f"""
You are a digital marketing director. Write a concise, executive-level portfolio summary
(max 5 sentences) based on the data below. Include: overall status, top performing
category, and budget recommendation direction.

Total campaigns: {campaign_count}
Total current daily spend: {total_spend:.2f}
Total recommended budget: {total_recommended:.2f}
Campaigns recommended to increase: {increase_count}
Campaigns recommended to reduce: {reduce_count}
Predicted total revenue: {total_pred_rev:.2f}
ROAS target: {target_roas:.1f}
Top performing category: {top_category} (ROAS: {top_category_roas:.2f})

Portfolio summary (5 sentences):
"""

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt.strip()}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        logger.warning("Portfolio LLM error: %s", e)
        return "Portfolio commentary could not be generated."


# ---------------------------------------------------------------------------
# Recommendation summary
# ---------------------------------------------------------------------------

def build_recommendation_summary(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "CampaignId", "Campaign", "Channel",
        "Category", "ProductGroup",
        "Season", "IsHoliday", "IsPreHoliday", "HolidayName",
        "ExpectedROASMultiplier",
        "CurrentSpend", "RecommendedBudget", "OptimizedPortfolioBudget",
        "RecommendedAction", "ConfidenceLevel",
        "PredictedConversions", "PredictedRevenue", "PredictedProfit", "PredictedROAS",
        "TargetROAS", "ROASStatus", "ROASGap", "ROASGapPct",
        "BaselinePredictedRevenue", "RevenueUplift", "RevenueUpliftPct",
        "RecommendationReason",
    ]
    existing_cols = [c for c in summary_cols if c in portfolio_df.columns]
    return portfolio_df[existing_cols].copy()


# ---------------------------------------------------------------------------
# Rule-based fallback (insufficient data)
# ---------------------------------------------------------------------------

def build_rule_based_fallback(ads_raw: pd.DataFrame, target_roas: float) -> pd.DataFrame:
    campaign_df = ads_raw.groupby(["CampaignId", "Campaign", "Channel", "Category", "ProductGroup"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    campaign_df = compute_kpis(campaign_df)
    campaign_df = compute_roas_target_gap(campaign_df, target_roas)

    def fallback_action(row):
        if row["Conversions"] == 0 and row["Spend"] > 0:
            return "Pause / Review", "No conversions observed in available history."
        if row["ROAS"] >= target_roas * 1.10 and row["Profit"] > 0:
            return "Maintain / Slight Increase", "Strong historical efficiency in fallback mode."
        if row["ROAS"] < target_roas * 0.90 or row["Profit"] < 0:
            return "Reduce Budget", "Weak historical efficiency in fallback mode."
        return "Maintain", "Rule-based recommendation."

    decisions = campaign_df.apply(fallback_action, axis=1)
    campaign_df["RecommendedAction"] = decisions.apply(lambda x: x[0])
    campaign_df["RecommendationReason"] = decisions.apply(lambda x: x[1])
    campaign_df["ConfidenceLevel"] = "Low"
    campaign_df["RecommendedBudget"] = campaign_df["Spend"]
    campaign_df["CurrentSpend"] = campaign_df["Spend"]
    return campaign_df


# ---------------------------------------------------------------------------
# Daily / weekly / monthly outputs
# ---------------------------------------------------------------------------

def build_daily_weekly_monthly_outputs(ads_raw: pd.DataFrame, holiday_map: Dict[str, str]):
    df = ads_raw.copy()
    df = add_calendar_context_features(df, holiday_map)

    daily_summary = df.groupby(["Date", "Campaign", "Channel", "Category", "IsHoliday", "HolidayName", "Season"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    daily_summary = compute_kpis(daily_summary)

    temp = df.copy()
    temp["Week"] = temp["Date"].dt.to_period("W").astype(str)
    temp["Month"] = temp["Date"].dt.to_period("M").astype(str)

    weekly_summary = temp.groupby(["Week", "Campaign", "Channel", "Category", "Season"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    weekly_summary = compute_kpis(weekly_summary)

    monthly_summary = temp.groupby(["Month", "Campaign", "Channel", "Category", "Season"], as_index=False).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )
    monthly_summary = compute_kpis(monthly_summary)

    return daily_summary, weekly_summary, monthly_summary


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = os.getenv("VICCO_OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    target_roas = env_float("TARGET_ROAS", 3.0)
    llm_lang = os.getenv("LLM_LANG", "en")       # "tr" or "en"
    llm_max = int(os.getenv("LLM_MAX_CAMPAIGNS", "20"))

    logger.info("ROAS target: %.1f | LLM language: %s | LLM campaign limit: %d",
                target_roas, llm_lang, llm_max)

    date_from, date_to = resolve_date_range()
    logger.info("Date range: %s → %s", date_from, date_to)

    # Build holiday map
    holiday_map = build_holiday_map(date_from, date_to)
    logger.info("Holiday/public holiday count (in range): %d", len(holiday_map))

    # Fetch data
    ads_raw = fetch_ads_purchase_only(date_from, date_to)
    if ads_raw.empty:
        logger.warning("Google Ads returned no data.")
        return

    # Category & product summaries
    category_df = build_category_summary(ads_raw, holiday_map)
    product_df = build_product_summary(ads_raw, holiday_map)
    holiday_impact_df = build_holiday_impact_summary(ads_raw, holiday_map)

    # Time series outputs
    daily_df, weekly_df, monthly_df = build_daily_weekly_monthly_outputs(ads_raw, holiday_map)

    # Training data
    train_df = prepare_training_data(ads_raw, holiday_map)

    if train_df.empty or len(train_df) < 20:
        logger.warning("Not enough data for ML. Using rule-based fallback.")
        fallback_df = build_rule_based_fallback(ads_raw, target_roas)

        # LLM commentary for fallback
        fallback_df = generate_llm_commentary(fallback_df, target_roas, lang=llm_lang, max_campaigns=llm_max)

        fallback_df.to_csv(os.path.join(output_dir, "ads_rule_based_fallback_recommendations.csv"), index=False)
        category_df.to_csv(os.path.join(output_dir, "ads_category_summary.csv"), index=False)
        product_df.to_csv(os.path.join(output_dir, "ads_product_summary.csv"), index=False)
        holiday_impact_df.to_csv(os.path.join(output_dir, "ads_holiday_impact.csv"), index=False)
        daily_df.to_csv(os.path.join(output_dir, "ads_daily_fact.csv"), index=False)
        weekly_df.to_csv(os.path.join(output_dir, "ads_weekly_campaign_summary.csv"), index=False)
        monthly_df.to_csv(os.path.join(output_dir, "ads_monthly_campaign_summary.csv"), index=False)
        return

    # ML pipeline
    model_conv, model_rev, feature_cols, metrics_df, feature_importance_df = train_and_validate_models(train_df)
    latest_df = get_latest_campaign_state(ads_raw, holiday_map)

    sim_df = simulate_budget_scenarios(
        latest_df=latest_df,
        model_conv=model_conv,
        model_rev=model_rev,
        feature_cols=feature_cols,
    )

    best_df = choose_optimal_scenario(sim_df)
    best_df = add_baseline_uplift(best_df, sim_df)
    recommendation_df = build_action_recommendation(best_df, target_roas)
    recommendation_df = build_confidence_scores(recommendation_df, metrics_df, train_df)
    recommendation_df = apply_confidence_guardrail(recommendation_df)

    # Add ROAS target gap
    recommendation_df = compute_roas_target_gap(recommendation_df, target_roas)

    portfolio_df = build_portfolio_allocation(recommendation_df)
    summary_df = build_recommendation_summary(portfolio_df)

    # LLM commentaries — per campaign
    summary_df = generate_llm_commentary(summary_df, target_roas, lang=llm_lang, max_campaigns=llm_max)

    # LLM portfolio overall summary
    portfolio_commentary = generate_portfolio_summary_commentary(
        portfolio_df, category_df, target_roas, lang=llm_lang
    )
    logger.info("=== PORTFOLIO EXECUTIVE SUMMARY COMMENTARY ===\n%s\n===", portfolio_commentary)

    # Save portfolio commentary to separate file
    with open(os.path.join(output_dir, "ads_portfolio_executive_commentary.txt"), "w", encoding="utf-8") as f:
        f.write(portfolio_commentary)

    # CSV outputs
    sim_df.to_csv(os.path.join(output_dir, "ads_budget_scenarios.csv"), index=False)
    recommendation_df.to_csv(os.path.join(output_dir, "ads_budget_optimization_recommendations.csv"), index=False)
    portfolio_df.to_csv(os.path.join(output_dir, "ads_portfolio_budget_allocation.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "ads_recommendation_summary.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "ads_model_validation_metrics.csv"), index=False)
    feature_importance_df.to_csv(os.path.join(output_dir, "ads_feature_importance.csv"), index=False)
    category_df.to_csv(os.path.join(output_dir, "ads_category_summary.csv"), index=False)
    product_df.to_csv(os.path.join(output_dir, "ads_product_summary.csv"), index=False)
    holiday_impact_df.to_csv(os.path.join(output_dir, "ads_holiday_impact.csv"), index=False)
    daily_df.to_csv(os.path.join(output_dir, "ads_daily_fact.csv"), index=False)
    weekly_df.to_csv(os.path.join(output_dir, "ads_weekly_campaign_summary.csv"), index=False)
    monthly_df.to_csv(os.path.join(output_dir, "ads_monthly_campaign_summary.csv"), index=False)

    logger.info("Vicco Ads ML Budget Intelligence v2 pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

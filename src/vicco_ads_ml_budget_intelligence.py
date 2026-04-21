
"""
Vicco Ads ML Budget Intelligence Pipeline

This project is a Python-based predictive decision-support pipeline that integrates
Google Ads data to evaluate campaign performance, forecast future results, simulate
budget scenarios, validate model quality, explain decision drivers, and recommend
optimized campaign and portfolio-level budget allocation.

Core Capabilities:
- Google Ads API integration
- KPI engineering including CTR, CPC, Conversion Rate, CPA, ROAS, and Profit
- Predictive modeling for next-period conversions and revenue
- Model validation (MAE, RMSE, R²)
- Feature importance reporting
- Budget scenario simulation
- Campaign-level optimization logic
- Portfolio-level budget allocation
- Confidence scoring and recommendation summary
- Built-in guardrails, fallback logic, and confidence-based decision filtering

Business Purpose:
This pipeline moves beyond static reporting by combining analytics, prediction,
simulation, optimization, validation, and explainability to support smarter budget
allocation and revenue-focused campaign decisions.

Author: Ozlem Tonbul
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd
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


def env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing environment variable: {name}")
    return value.strip()


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


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CTR"] = df["Clicks"] / df["Impressions"].replace(0, np.nan)
    df["CPC"] = df["Spend"] / df["Clicks"].replace(0, np.nan)
    df["ConvRate"] = df["Conversions"] / df["Clicks"].replace(0, np.nan)
    df["CPA"] = df["Spend"] / df["Conversions"].replace(0, np.nan)
    df["ROAS"] = df["ConversionValue"] / df["Spend"].replace(0, np.nan)
    df["Profit"] = df["ConversionValue"] - df["Spend"]

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


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
    customer_id = env("GOOGLE_ADS_CUSTOMER_ID")

    client = build_google_ads_client()
    service = client.get_service("GoogleAdsService")

    query = f"""
        SELECT
          segments.date,
          campaign.id,
          campaign.name,
          campaign.advertising_channel_type,
          metrics.impressions,
          metrics.clicks,
          metrics.conversions,
          metrics.conversions_value,
          metrics.cost_micros
        FROM campaign
        WHERE
          segments.date BETWEEN '{date_from}' AND '{date_to}'
          AND segments.conversion_action_category = 'PURCHASE'
    """

    rows = []

    try:
        response = service.search(customer_id=customer_id, query=query)
        for row in response:
            rows.append(
                {
                    "Date": str(row.segments.date),
                    "CampaignId": int(row.campaign.id or 0),
                    "Campaign": row.campaign.name or "UNKNOWN",
                    "Channel": str(row.campaign.advertising_channel_type),
                    "Impressions": int(row.metrics.impressions or 0),
                    "Clicks": int(row.metrics.clicks or 0),
                    "Conversions": float(row.metrics.conversions or 0.0),
                    "ConversionValue": float(row.metrics.conversions_value or 0.0),
                    "Spend": float(row.metrics.cost_micros or 0.0) / 1_000_000.0,
                }
            )
    except GoogleAdsException as exc:
        logger.error("Google Ads API error: %s", exc)
        raise

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "CampaignId", "Impressions", "Clicks", "Conversions", "ConversionValue", "Spend"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Campaign"] = df["Campaign"].replace(["", "None", None], "UNKNOWN")
    df["Channel"] = df["Channel"].replace(["", "None", None], "UNKNOWN")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["MonthNum"] = df["Date"].dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CampaignId", "Date"]).copy()
    group = df.groupby("CampaignId")

    df["Spend_lag_1"] = group["Spend"].shift(1)
    df["Spend_lag_7_avg"] = (
        group["Spend"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["Clicks_lag_1"] = group["Clicks"].shift(1)
    df["Clicks_lag_7_avg"] = (
        group["Clicks"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["Conversions_lag_1"] = group["Conversions"].shift(1)
    df["Conversions_lag_7_avg"] = (
        group["Conversions"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["Revenue_lag_1"] = group["ConversionValue"].shift(1)
    df["Revenue_lag_7_avg"] = (
        group["ConversionValue"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["ROAS_lag_1"] = group["ROAS"].shift(1)
    df["CPA_lag_1"] = group["CPA"].shift(1)
    df["CTR_lag_1"] = group["CTR"].shift(1)
    df["ConvRate_lag_1"] = group["ConvRate"].shift(1)

    return df.fillna(0)


def prepare_training_data(ads_raw: pd.DataFrame) -> pd.DataFrame:
    df = ads_raw.copy()
    df = compute_kpis(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    df = df.sort_values(["CampaignId", "Date"]).copy()
    group = df.groupby("CampaignId")

    df["Target_Conversions_Next"] = group["Conversions"].shift(-1)
    df["Target_Revenue_Next"] = group["ConversionValue"].shift(-1)

    df = df.dropna(subset=["Target_Conversions_Next", "Target_Revenue_Next"]).copy()
    return df


def get_feature_columns() -> List[str]:
    return [
        "CampaignId",
        "Spend",
        "Impressions",
        "Clicks",
        "Conversions",
        "ConversionValue",
        "CTR",
        "CPC",
        "ConvRate",
        "CPA",
        "ROAS",
        "Profit",
        "DayOfWeek",
        "DayOfMonth",
        "MonthNum",
        "Spend_lag_1",
        "Spend_lag_7_avg",
        "Clicks_lag_1",
        "Clicks_lag_7_avg",
        "Conversions_lag_1",
        "Conversions_lag_7_avg",
        "Revenue_lag_1",
        "Revenue_lag_7_avg",
        "ROAS_lag_1",
        "CPA_lag_1",
        "CTR_lag_1",
        "ConvRate_lag_1",
    ]


def train_and_validate_models(train_df: pd.DataFrame):
    feature_cols = get_feature_columns()
    X = train_df[feature_cols]

    y_conv = train_df["Target_Conversions_Next"]
    y_rev = train_df["Target_Revenue_Next"]

    X_train_conv, X_test_conv, y_train_conv, y_test_conv = train_test_split(
        X, y_conv, test_size=0.2, random_state=42
    )
    X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
        X, y_rev, test_size=0.2, random_state=42
    )

    model_conv = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )
    model_rev = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )

    model_conv.fit(X_train_conv, y_train_conv)
    model_rev.fit(X_train_rev, y_train_rev)

    pred_conv = model_conv.predict(X_test_conv)
    pred_rev = model_rev.predict(X_test_rev)

    metrics_df = pd.DataFrame(
        [
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
        ]
    )

    importance_conv = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model_conv.feature_importances_,
            "Model": "Conversions",
        }
    ).sort_values("Importance", ascending=False)

    importance_rev = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model_rev.feature_importances_,
            "Model": "Revenue",
        }
    ).sort_values("Importance", ascending=False)

    feature_importance_df = pd.concat([importance_conv, importance_rev], ignore_index=True)

    return model_conv, model_rev, feature_cols, metrics_df, feature_importance_df


def get_latest_campaign_state(ads_raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare_training_data(ads_raw)
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


def simulate_budget_scenarios(
    latest_df: pd.DataFrame,
    model_conv,
    model_rev,
    feature_cols: List[str]
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

            X_input = pd.DataFrame([sim_row])[feature_cols]

            pred_conv = safe_prediction(model_conv.predict(X_input)[0])
            pred_rev = safe_prediction(model_rev.predict(X_input)[0])
            pred_profit = pred_rev - sim_row["ScenarioSpend"]
            pred_roas = pred_rev / sim_row["ScenarioSpend"] if sim_row["ScenarioSpend"] > 0 else 0

            results.append(
                {
                    "CampaignId": row["CampaignId"],
                    "Campaign": row["Campaign"],
                    "Channel": row["Channel"],
                    "CurrentSpend": round(float(row["Spend"]), 2),
                    "ScenarioFactor": factor,
                    "ScenarioSpend": round(float(sim_row["ScenarioSpend"]), 2),
                    "PredictedConversions": round(pred_conv, 2),
                    "PredictedRevenue": round(pred_rev, 2),
                    "PredictedProfit": round(pred_profit, 2),
                    "PredictedROAS": round(pred_roas, 4),
                }
            )

    return pd.DataFrame(results)


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
    ].rename(
        columns={
            "PredictedRevenue": "BaselinePredictedRevenue",
            "PredictedProfit": "BaselinePredictedProfit",
            "PredictedConversions": "BaselinePredictedConversions",
        }
    )

    df = df.merge(baseline_df, on="CampaignId", how="left")

    df["RevenueUplift"] = df["PredictedRevenue"] - df["BaselinePredictedRevenue"]
    df["ProfitUplift"] = df["PredictedProfit"] - df["BaselinePredictedProfit"]
    df["ConversionUplift"] = df["PredictedConversions"] - df["BaselinePredictedConversions"]

    df["RevenueUpliftPct"] = np.where(
        df["BaselinePredictedRevenue"] > 0,
        (df["RevenueUplift"] / df["BaselinePredictedRevenue"]) * 100,
        0
    )

    return df


def build_action_recommendation(best_df: pd.DataFrame) -> pd.DataFrame:
    df = best_df.copy()

    def decide_action(row):
        current_spend = row["CurrentSpend"]
        optimal_spend = row["ScenarioSpend"]

        if current_spend <= 0:
            return "Review", "No active spend detected; manual review required."

        if row["PredictedConversions"] <= 0 and optimal_spend < current_spend:
            return "Pause / Review", "Predicted value remains weak even under lower spend scenarios."

        ratio = optimal_spend / current_spend if current_spend > 0 else 1

        if ratio >= 1.15:
            return "Increase Budget", "Predicted performance suggests scaling is likely to improve results."
        if ratio <= 0.85:
            return "Reduce Budget", "Predicted return suggests the campaign is overfunded at its current level."
        return "Maintain", "Predicted performance supports keeping the budget near its current level."

    decisions = df.apply(decide_action, axis=1)
    df["RecommendedAction"] = decisions.apply(lambda x: x[0])
    df["RecommendationReason"] = decisions.apply(lambda x: x[1])

    df["BudgetChange"] = df["ScenarioSpend"] - df["CurrentSpend"]
    df["BudgetChangePct"] = np.where(
        df["CurrentSpend"] > 0,
        (df["BudgetChange"] / df["CurrentSpend"]) * 100,
        0
    )

    df = df.rename(columns={"ScenarioSpend": "RecommendedBudget"})
    return df


def build_confidence_scores(
    recommendation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    train_df: pd.DataFrame
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


def build_portfolio_allocation(recommendation_df: pd.DataFrame) -> pd.DataFrame:
    df = recommendation_df.copy()

    if df.empty:
        return df

    total_current_budget = df["CurrentSpend"].sum()
    total_recommended_budget = df["RecommendedBudget"].sum()

    if total_recommended_budget <= 0:
        df["OptimizedPortfolioBudget"] = df["CurrentSpend"]
    else:
        df["OptimizedPortfolioBudget"] = (
            df["RecommendedBudget"] / total_recommended_budget
        ) * total_current_budget

    df["OptimizedPortfolioBudget"] = df["OptimizedPortfolioBudget"].round(2)
    df["PortfolioBudgetChange"] = (
        df["OptimizedPortfolioBudget"] - df["CurrentSpend"]
    ).round(2)

    return df


def build_recommendation_summary(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "CampaignId",
        "Campaign",
        "Channel",
        "CurrentSpend",
        "RecommendedBudget",
        "OptimizedPortfolioBudget",
        "RecommendedAction",
        "ConfidenceLevel",
        "PredictedConversions",
        "PredictedRevenue",
        "PredictedProfit",
        "PredictedROAS",
        "BaselinePredictedRevenue",
        "RevenueUplift",
        "RevenueUpliftPct",
        "RecommendationReason",
    ]

    existing_cols = [col for col in summary_cols if col in portfolio_df.columns]
    return portfolio_df[existing_cols].copy()


def build_rule_based_fallback(ads_raw: pd.DataFrame) -> pd.DataFrame:
    campaign_df = ads_raw.groupby(["CampaignId", "Campaign", "Channel"], as_index=False).agg(
        {
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum",
            "ConversionValue": "sum",
        }
    )
    campaign_df = compute_kpis(campaign_df)

    def fallback_action(row):
        if row["Conversions"] == 0 and row["Spend"] > 0:
            return "Pause / Review", "No conversions observed in available history."
        if row["ROAS"] >= 3.0 and row["Profit"] > 0:
            return "Maintain / Slight Increase", "Strong historical efficiency in fallback mode."
        if row["ROAS"] < 1.5 or row["Profit"] < 0:
            return "Reduce Budget", "Weak historical efficiency in fallback mode."
        return "Maintain", "Fallback rule-based recommendation."

    decisions = campaign_df.apply(fallback_action, axis=1)
    campaign_df["RecommendedAction"] = decisions.apply(lambda x: x[0])
    campaign_df["RecommendationReason"] = decisions.apply(lambda x: x[1])
    campaign_df["ConfidenceLevel"] = "Low"
    campaign_df["RecommendedBudget"] = campaign_df["Spend"]
    campaign_df["CurrentSpend"] = campaign_df["Spend"]

    return campaign_df


def build_daily_weekly_monthly_outputs(ads_raw: pd.DataFrame):
    daily_summary = ads_raw.groupby(["Date", "Campaign", "Channel"], as_index=False).agg(
        {
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum",
            "ConversionValue": "sum",
        }
    )
    daily_summary = compute_kpis(daily_summary)

    temp = ads_raw.copy()
    temp["Week"] = temp["Date"].dt.to_period("W").astype(str)
    temp["Month"] = temp["Date"].dt.to_period("M").astype(str)

    weekly_summary = temp.groupby(["Week", "Campaign", "Channel"], as_index=False).agg(
        {
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum",
            "ConversionValue": "sum",
        }
    )
    weekly_summary = compute_kpis(weekly_summary)

    monthly_summary = temp.groupby(["Month", "Campaign", "Channel"], as_index=False).agg(
        {
            "Spend": "sum",
            "Clicks": "sum",
            "Impressions": "sum",
            "Conversions": "sum",
            "ConversionValue": "sum",
        }
    )
    monthly_summary = compute_kpis(monthly_summary)

    return daily_summary, weekly_summary, monthly_summary


def main() -> None:
    output_dir = os.getenv("VICCO_OUTPUT_DIR", "./output")
    os.makedirs(output_dir, exist_ok=True)

    date_from, date_to = resolve_date_range()
    logger.info("Date range: %s to %s", date_from, date_to)

    ads_raw = fetch_ads_purchase_only(date_from, date_to)
    if ads_raw.empty:
        logger.warning("Google Ads returned no data.")
        return

    train_df = prepare_training_data(ads_raw)
    daily_df, weekly_df, monthly_df = build_daily_weekly_monthly_outputs(ads_raw)

    if train_df.empty or len(train_df) < 20:
        logger.warning("Not enough historical data for robust ML. Using rule-based fallback.")
        fallback_df = build_rule_based_fallback(ads_raw)
        fallback_df.to_csv(
            os.path.join(output_dir, "ads_rule_based_fallback_recommendations.csv"),
            index=False
        )
        daily_df.to_csv(os.path.join(output_dir, "ads_daily_fact.csv"), index=False)
        weekly_df.to_csv(os.path.join(output_dir, "ads_weekly_campaign_summary.csv"), index=False)
        monthly_df.to_csv(os.path.join(output_dir, "ads_monthly_campaign_summary.csv"), index=False)
        return

    model_conv, model_rev, feature_cols, metrics_df, feature_importance_df = train_and_validate_models(train_df)
    latest_df = get_latest_campaign_state(ads_raw)

    sim_df = simulate_budget_scenarios(
        latest_df=latest_df,
        model_conv=model_conv,
        model_rev=model_rev,
        feature_cols=feature_cols
    )

    best_df = choose_optimal_scenario(sim_df)
    best_df = add_baseline_uplift(best_df, sim_df)
    recommendation_df = build_action_recommendation(best_df)
    recommendation_df = build_confidence_scores(recommendation_df, metrics_df, train_df)
    recommendation_df = apply_confidence_guardrail(recommendation_df)
    portfolio_df = build_portfolio_allocation(recommendation_df)
    summary_df = build_recommendation_summary(portfolio_df)

    sim_df.to_csv(os.path.join(output_dir, "ads_budget_scenarios.csv"), index=False)
    recommendation_df.to_csv(
        os.path.join(output_dir, "ads_budget_optimization_recommendations.csv"),
        index=False
    )
    portfolio_df.to_csv(
        os.path.join(output_dir, "ads_portfolio_budget_allocation.csv"),
        index=False
    )
    summary_df.to_csv(
        os.path.join(output_dir, "ads_recommendation_summary.csv"),
        index=False
    )
    metrics_df.to_csv(
        os.path.join(output_dir, "ads_model_validation_metrics.csv"),
        index=False
    )
    feature_importance_df.to_csv(
        os.path.join(output_dir, "ads_feature_importance.csv"),
        index=False
    )
    daily_df.to_csv(os.path.join(output_dir, "ads_daily_fact.csv"), index=False)
    weekly_df.to_csv(os.path.join(output_dir, "ads_weekly_campaign_summary.csv"), index=False)
    monthly_df.to_csv(os.path.join(output_dir, "ads_monthly_campaign_summary.csv"), index=False)

    logger.info("Vicco Ads ML Budget Intelligence pipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
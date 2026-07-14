from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import get_feature_columns


CAMPAIGN_TYPE_KEYWORDS = {
    "Brand": ["brand", "marka", "branded"],
    "Shopping": ["shopping", "pla", "alisveris", "alışveriş"],
    "Performance Max": [
        "pmax",
        "performance max",
        "performans max",
    ],
    "Generic": [],
}

CAMPAIGN_TYPE_SCENARIO_FACTORS = {
    "Brand": [0.75, 0.90, 1.00, 1.10, 1.25],
    "Shopping": [0.50, 0.75, 1.00, 1.20, 1.50],
    "Performance Max": [0.60, 0.80, 1.00, 1.25, 1.50],
    "Generic": [0.50, 0.75, 1.00, 1.20, 1.50],
}


def classify_campaign_type(campaign_name: str) -> str:
    lower_name = str(campaign_name or "").lower().strip()

    for campaign_type, keywords in CAMPAIGN_TYPE_KEYWORDS.items():
        if campaign_type == "Generic":
            continue

        if any(keyword in lower_name for keyword in keywords):
            return campaign_type

    return "Generic"


def add_campaign_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "Campaign" not in df.columns:
        raise KeyError("Campaign column is required.")

    result_df = df.copy()

    result_df["CampaignType"] = (
        result_df["Campaign"]
        .apply(classify_campaign_type)
    )

    return result_df


def train_and_validate_models(
    train_df: pd.DataFrame,
) -> Tuple[
    RandomForestRegressor,
    RandomForestRegressor,
    List[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    required_target_columns = [
        "Target_Conversions_Next",
        "Target_Revenue_Next",
    ]

    missing_targets = [
        column
        for column in required_target_columns
        if column not in train_df.columns
    ]

    if missing_targets:
        raise KeyError(
            "Missing target columns: "
            + ", ".join(missing_targets)
        )

    feature_cols = [
        column
        for column in get_feature_columns()
        if column in train_df.columns
    ]

    if not feature_cols:
        raise ValueError(
            "No valid model feature columns were found."
        )

    model_df = train_df.copy()

    model_df[feature_cols] = (
        model_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model_df["Target_Conversions_Next"] = pd.to_numeric(
        model_df["Target_Conversions_Next"],
        errors="coerce",
    )

    model_df["Target_Revenue_Next"] = pd.to_numeric(
        model_df["Target_Revenue_Next"],
        errors="coerce",
    )

    model_df = model_df.dropna(
        subset=[
            "Target_Conversions_Next",
            "Target_Revenue_Next",
        ]
    )

    if len(model_df) < 10:
        raise ValueError(
            "At least 10 valid training rows are required."
        )

    X = model_df[feature_cols]

    y_conv = model_df["Target_Conversions_Next"]
    y_rev = model_df["Target_Revenue_Next"]

    (
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
    ) = train_test_split(
        X,
        y_conv,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
    ) = train_test_split(
        X,
        y_rev,
        test_size=0.2,
        random_state=42,
    )

    model_conv = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model_rev = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model_conv.fit(X_train_c, y_train_c)
    model_rev.fit(X_train_r, y_train_r)

    pred_conv = model_conv.predict(X_test_c)
    pred_rev = model_rev.predict(X_test_r)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Conversions",
                "MAE": float(
                    mean_absolute_error(
                        y_test_c,
                        pred_conv,
                    )
                ),
                "RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            y_test_c,
                            pred_conv,
                        )
                    )
                ),
                "R2": float(
                    r2_score(
                        y_test_c,
                        pred_conv,
                    )
                ),
                "TrainRows": len(X_train_c),
                "TestRows": len(X_test_c),
            },
            {
                "Model": "Revenue",
                "MAE": float(
                    mean_absolute_error(
                        y_test_r,
                        pred_rev,
                    )
                ),
                "RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            y_test_r,
                            pred_rev,
                        )
                    )
                ),
                "R2": float(
                    r2_score(
                        y_test_r,
                        pred_rev,
                    )
                ),
                "TrainRows": len(X_train_r),
                "TestRows": len(X_test_r),
            },
        ]
    )

    importance_conv = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model_conv.feature_importances_,
            "Model": "Conversions",
        }
    )

    importance_rev = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model_rev.feature_importances_,
            "Model": "Revenue",
        }
    )

    feature_importance_df = (
        pd.concat(
            [
                importance_conv,
                importance_rev,
            ],
            ignore_index=True,
        )
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        model_conv,
        model_rev,
        feature_cols,
        metrics_df,
        feature_importance_df,
    )


def safe_prediction(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, numeric_value)


def simulate_budget_scenarios(
    latest_df: pd.DataFrame,
    model_conv: RandomForestRegressor,
    model_rev: RandomForestRegressor,
    feature_cols: List[str],
) -> pd.DataFrame:
    if latest_df.empty:
        return pd.DataFrame()

    if not feature_cols:
        raise ValueError(
            "feature_cols cannot be empty."
        )

    results = []

    for _, row in latest_df.iterrows():
        campaign_type = row.get(
            "CampaignType",
            "Generic",
        )

        scenario_factors = (
            CAMPAIGN_TYPE_SCENARIO_FACTORS.get(
                campaign_type,
                CAMPAIGN_TYPE_SCENARIO_FACTORS["Generic"],
            )
        )

        current_spend = float(
            row.get("Spend", 0) or 0
        )

        clicks = float(
            row.get("Clicks", 0) or 0
        )

        conversions = float(
            row.get("Conversions", 0) or 0
        )

        conversion_value = float(
            row.get("ConversionValue", 0) or 0
        )

        for factor in scenario_factors:
            sim_row = row.copy()

            scenario_spend = current_spend * factor

            sim_row["Spend"] = scenario_spend

            sim_row["CPC"] = (
                scenario_spend / clicks
                if clicks > 0
                else 0
            )

            sim_row["CPA"] = (
                scenario_spend / conversions
                if conversions > 0
                else 0
            )

            sim_row["ROAS"] = (
                conversion_value / scenario_spend
                if scenario_spend > 0
                else 0
            )

            sim_row["Profit"] = (
                conversion_value - scenario_spend
            )

            X_input = pd.DataFrame(
                [sim_row]
            )

            for feature in feature_cols:
                if feature not in X_input.columns:
                    X_input[feature] = 0

            X_input = (
                X_input[feature_cols]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0)
            )

            pred_conv = safe_prediction(
                model_conv.predict(X_input)[0]
            )

            pred_rev = safe_prediction(
                model_rev.predict(X_input)[0]
            )

            season_multiplier = safe_prediction(
                row.get(
                    "ExpectedROASMultiplier",
                    1.0,
                )
            )

            if season_multiplier <= 0:
                season_multiplier = 1.0

            pred_rev_adjusted = (
                pred_rev * season_multiplier
            )

            pred_profit = (
                pred_rev_adjusted
                - scenario_spend
            )

            pred_roas = (
                pred_rev_adjusted / scenario_spend
                if scenario_spend > 0
                else 0
            )

            results.append(
                {
                    "CampaignId": row.get(
                        "CampaignId",
                        0,
                    ),
                    "Campaign": row.get(
                        "Campaign",
                        "UNKNOWN",
                    ),
                    "Channel": row.get(
                        "Channel",
                        "UNKNOWN",
                    ),
                    "CampaignType": campaign_type,
                    "Category": row.get(
                        "Category",
                        "",
                    ),
                    "ProductGroup": row.get(
                        "ProductGroup",
                        "",
                    ),
                    "Season": row.get(
                        "Season",
                        "",
                    ),
                    "IsHoliday": int(
                        row.get(
                            "IsHoliday",
                            0,
                        )
                    ),
                    "IsPreHoliday": int(
                        row.get(
                            "IsPreHoliday",
                            0,
                        )
                    ),
                    "HolidayName": row.get(
                        "HolidayName",
                        "",
                    ),
                    "ExpectedROASMultiplier": round(
                        season_multiplier,
                        3,
                    ),
                    "CurrentSpend": round(
                        current_spend,
                        2,
                    ),
                    "ScenarioFactor": factor,
                    "ScenarioSpend": round(
                        scenario_spend,
                        2,
                    ),
                    "PredictedConversions": round(
                        pred_conv,
                        2,
                    ),
                    "PredictedRevenue": round(
                        pred_rev_adjusted,
                        2,
                    ),
                    "PredictedProfit": round(
                        pred_profit,
                        2,
                    ),
                    "PredictedROAS": round(
                        pred_roas,
                        4,
                    ),
                }
            )

    return pd.DataFrame(results)


def choose_optimal_scenario(
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    if sim_df.empty:
        return sim_df.copy()

    required_columns = [
        "CampaignId",
        "PredictedRevenue",
        "PredictedProfit",
        "PredictedROAS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in sim_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing simulation columns: "
            + ", ".join(missing_columns)
        )

    df = sim_df.copy()

    df["OptimizationScore"] = (
        df["PredictedRevenue"] * 0.45
        + df["PredictedProfit"] * 0.35
        + df["PredictedROAS"] * 100 * 0.20
    )

    best_df = (
        df.sort_values(
            [
                "CampaignId",
                "OptimizationScore",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "CampaignId",
            as_index=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    return best_df


def add_baseline_uplift(
    best_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    if best_df.empty:
        return best_df.copy()

    if sim_df.empty:
        return best_df.copy()

    df = best_df.copy()

    baseline = (
        sim_df[
            sim_df["ScenarioFactor"] == 1.0
        ][
            [
                "CampaignId",
                "PredictedRevenue",
                "PredictedProfit",
                "PredictedConversions",
            ]
        ]
        .drop_duplicates(
            subset=["CampaignId"]
        )
        .rename(
            columns={
                "PredictedRevenue": (
                    "BaselinePredictedRevenue"
                ),
                "PredictedProfit": (
                    "BaselinePredictedProfit"
                ),
                "PredictedConversions": (
                    "BaselinePredictedConversions"
                ),
            }
        )
    )

    df = df.merge(
        baseline,
        on="CampaignId",
        how="left",
    )

    baseline_columns = [
        "BaselinePredictedRevenue",
        "BaselinePredictedProfit",
        "BaselinePredictedConversions",
    ]

    for column in baseline_columns:
        if column not in df.columns:
            df[column] = 0

        df[column] = df[column].fillna(0)

    df["RevenueUplift"] = (
        df["PredictedRevenue"]
        - df["BaselinePredictedRevenue"]
    )

    df["ProfitUplift"] = (
        df["PredictedProfit"]
        - df["BaselinePredictedProfit"]
    )

    df["ConversionUplift"] = (
        df["PredictedConversions"]
        - df["BaselinePredictedConversions"]
    )

    df["RevenueUpliftPct"] = np.where(
        df["BaselinePredictedRevenue"] > 0,
        (
            df["RevenueUplift"]
            / df["BaselinePredictedRevenue"]
        )
        * 100,
        0,
    )

    return df
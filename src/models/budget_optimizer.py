from __future__ import annotations

from time import perf_counter
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import get_feature_columns


CAMPAIGN_TYPE_KEYWORDS = {
    "Brand": [
        "brand",
        "marka",
        "branded",
    ],
    "Shopping": [
        "shopping",
        "pla",
        "alisveris",
        "alışveriş",
    ],
    "Performance Max": [
        "pmax",
        "performance max",
        "performans max",
    ],
    "Generic": [],
}


CAMPAIGN_TYPE_SCENARIO_FACTORS = {
    "Brand": [
        0.75,
        0.90,
        1.00,
        1.10,
        1.25,
    ],
    "Shopping": [
        0.50,
        0.75,
        1.00,
        1.20,
        1.50,
    ],
    "Performance Max": [
        0.60,
        0.80,
        1.00,
        1.25,
        1.50,
    ],
    "Generic": [
        0.50,
        0.75,
        1.00,
        1.20,
        1.50,
    ],
}


DEFAULT_TARGET_ROAS = 3.0
DEFAULT_MIN_HISTORY_ROWS = 30
DEFAULT_MAX_BUDGET_INCREASE_PCT = 50.0
DEFAULT_MAX_BUDGET_DECREASE_PCT = 50.0


def safe_float(
    value: object,
    default: float = 0.0,
) -> float:
    """Convert an arbitrary value to a finite float."""
    if value is None:
        return default

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(numeric_value):
        return default

    return numeric_value


def safe_prediction(value: object) -> float:
    """Return a non-negative finite prediction."""
    return max(
        0.0,
        safe_float(value),
    )


def safe_percentage_change(
    current_value: object,
    comparison_value: object,
) -> float:
    """Calculate a safe percentage change."""
    current = safe_float(current_value)
    comparison = safe_float(comparison_value)

    if comparison == 0:
        return 0.0

    return (
        (current - comparison)
        / abs(comparison)
    ) * 100.0


def classify_campaign_type(
    campaign_name: str,
) -> str:
    """Classify a campaign from its name."""
    lower_name = str(
        campaign_name or ""
    ).lower().strip()

    for campaign_type, keywords in (
        CAMPAIGN_TYPE_KEYWORDS.items()
    ):
        if campaign_type == "Generic":
            continue

        if any(
            keyword in lower_name
            for keyword in keywords
        ):
            return campaign_type

    return "Generic"


def add_campaign_type(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add CampaignType without changing the source dataframe."""
    if df.empty:
        return df.copy()

    if "Campaign" not in df.columns:
        raise KeyError(
            "Campaign column is required."
        )

    result_df = df.copy()

    result_df["CampaignType"] = (
        result_df["Campaign"]
        .apply(classify_campaign_type)
    )

    return result_df


def _get_analysis_period(
    dataframe: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Resolve analysis start and end dates."""
    if dataframe.empty:
        return None, None

    date_candidates = [
        "Date",
        "AnalysisDate",
        "AnalysisStartDate",
    ]

    date_column = next(
        (
            column
            for column in date_candidates
            if column in dataframe.columns
        ),
        None,
    )

    if date_column is None:
        return None, None

    dates = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return None, None

    return (
        dates.min().date().isoformat(),
        dates.max().date().isoformat(),
    )


def _prepare_model_dataframe(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Clean model features and target columns."""
    model_df = train_df.copy()

    model_df[list(feature_cols)] = (
        model_df[list(feature_cols)]
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

    model_df["Target_Conversions_Next"] = (
        pd.to_numeric(
            model_df["Target_Conversions_Next"],
            errors="coerce",
        )
    )

    model_df["Target_Revenue_Next"] = (
        pd.to_numeric(
            model_df["Target_Revenue_Next"],
            errors="coerce",
        )
    )

    model_df = model_df.dropna(
        subset=[
            "Target_Conversions_Next",
            "Target_Revenue_Next",
        ]
    )

    return model_df


def _build_candidate_models() -> dict[str, Any]:
    """Build fresh candidate regressors for one prediction target."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def _evaluate_candidate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str,
    analysis_start: str | None,
    analysis_end: str | None,
) -> tuple[Any, list[dict[str, object]]]:
    """
    Train all candidate models on the same split and select the best one.

    Primary selection metric: minimum RMSE.
    Tie-breakers: minimum MAE, then maximum R².
    """
    evaluated: list[
        tuple[
            str,
            Any,
            dict[str, object],
        ]
    ] = []

    for algorithm, model in _build_candidate_models().items():
        started = perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            perf_counter() - started
        )

        predictions = model.predict(
            X_test
        )

        mae = float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        r2 = float(
            r2_score(
                y_test,
                predictions,
            )
        )

        row = {
            "Model": target_name,
            "Algorithm": algorithm,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "TrainingSeconds": round(
                training_seconds,
                4,
            ),
            "Selected": False,
            "SelectionMetric": "RMSE",
            "TrainRows": len(X_train),
            "TestRows": len(X_test),
            "AnalysisStartDate": analysis_start,
            "AnalysisEndDate": analysis_end,
        }

        evaluated.append(
            (
                algorithm,
                model,
                row,
            )
        )

    best_algorithm, best_model, _ = min(
        evaluated,
        key=lambda item: (
            safe_float(
                item[2]["RMSE"],
                default=float("inf"),
            ),
            safe_float(
                item[2]["MAE"],
                default=float("inf"),
            ),
            -safe_float(
                item[2]["R2"],
                default=float("-inf"),
            ),
        ),
    )

    metric_rows: list[
        dict[str, object]
    ] = []

    for algorithm, _, row in evaluated:
        row["Selected"] = (
            algorithm == best_algorithm
        )
        metric_rows.append(row)

    return best_model, metric_rows


def _build_feature_importance_dataframe(
    model: Any,
    feature_cols: Sequence[str],
    target_name: str,
    algorithm: str,
    analysis_start: str | None,
    analysis_end: str | None,
) -> pd.DataFrame:
    """Create a normalized feature-importance table for the selected model."""
    importances = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importances is None:
        importances = np.zeros(
            len(feature_cols),
            dtype=float,
        )

    return pd.DataFrame(
        {
            "Feature": list(feature_cols),
            "Importance": np.asarray(
                importances,
                dtype=float,
            ),
            "Model": target_name,
            "Algorithm": algorithm,
            "Selected": True,
            "AnalysisStartDate": analysis_start,
            "AnalysisEndDate": analysis_end,
        }
    )


def train_and_validate_models(
    train_df: pd.DataFrame,
) -> Tuple[
    Any,
    Any,
    List[str],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Benchmark Random Forest, XGBoost and LightGBM.

    The three algorithms use the same train/test split. Revenue and
    conversion targets are evaluated independently, and the model with
    the lowest RMSE is selected for each target.
    """
    if train_df.empty:
        raise ValueError(
            "Training dataframe is empty."
        )

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

    model_df = _prepare_model_dataframe(
        train_df,
        feature_cols,
    )

    if len(model_df) < 10:
        raise ValueError(
            "At least 10 valid training rows are required."
        )

    analysis_start, analysis_end = (
        _get_analysis_period(model_df)
    )

    X = model_df[feature_cols]

    y_conv = model_df[
        "Target_Conversions_Next"
    ]

    y_rev = model_df[
        "Target_Revenue_Next"
    ]

    (
        X_train,
        X_test,
        y_train_c,
        y_test_c,
        y_train_r,
        y_test_r,
    ) = train_test_split(
        X,
        y_conv,
        y_rev,
        test_size=0.2,
        random_state=42,
    )

    (
        model_conv,
        conversion_metrics,
    ) = _evaluate_candidate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_c,
        y_test=y_test_c,
        target_name="Conversions",
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )

    (
        model_rev,
        revenue_metrics,
    ) = _evaluate_candidate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_r,
        y_test=y_test_r,
        target_name="Revenue",
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )

    metrics_df = pd.DataFrame(
        conversion_metrics
        + revenue_metrics
    )

    metrics_df = (
        metrics_df
        .sort_values(
            [
                "Model",
                "RMSE",
                "MAE",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    selected_conv_algorithm = str(
        metrics_df.loc[
            (
                (metrics_df["Model"] == "Conversions")
                & metrics_df["Selected"]
            ),
            "Algorithm",
        ].iloc[0]
    )

    selected_rev_algorithm = str(
        metrics_df.loc[
            (
                (metrics_df["Model"] == "Revenue")
                & metrics_df["Selected"]
            ),
            "Algorithm",
        ].iloc[0]
    )

    importance_conv = (
        _build_feature_importance_dataframe(
            model=model_conv,
            feature_cols=feature_cols,
            target_name="Conversions",
            algorithm=selected_conv_algorithm,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
    )

    importance_rev = (
        _build_feature_importance_dataframe(
            model=model_rev,
            feature_cols=feature_cols,
            target_name="Revenue",
            algorithm=selected_rev_algorithm,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
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


def calculate_model_r2(
    metrics_df: pd.DataFrame,
) -> float:
    """Calculate average R² using selected models when available."""
    if (
        metrics_df.empty
        or "R2" not in metrics_df.columns
    ):
        return 0.0

    selected_df = metrics_df

    if "Selected" in metrics_df.columns:
        selected_mask = (
            metrics_df["Selected"]
            .fillna(False)
            .astype(bool)
        )

        if selected_mask.any():
            selected_df = metrics_df.loc[
                selected_mask
            ]

    r2_values = pd.to_numeric(
        selected_df["R2"],
        errors="coerce",
    ).dropna()

    if r2_values.empty:
        return 0.0

    return float(
        r2_values.mean()
    )


def calculate_confidence_level(
    history_rows: object,
    model_r2: object,
) -> str:
    """Calculate recommendation confidence."""
    history = int(
        max(
            0,
            safe_float(history_rows),
        )
    )

    r2_value = safe_float(model_r2)

    if history >= 90 and r2_value >= 0.60:
        return "High"

    if history >= 30 and r2_value >= 0.30:
        return "Medium"

    return "Low"


def _prepare_prediction_input(
    row: pd.Series,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Create a numeric model input with the expected columns."""
    X_input = pd.DataFrame(
        [row]
    )

    for feature in feature_cols:
        if feature not in X_input.columns:
            X_input[feature] = 0.0

    X_input = (
        X_input[list(feature_cols)]
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

    return X_input



def _model_algorithm_name(model: Any) -> str:
    """Return a stable human-readable algorithm name."""
    name = model.__class__.__name__
    mapping = {
        "RandomForestRegressor": "Random Forest",
        "XGBRegressor": "XGBoost",
        "LGBMRegressor": "LightGBM",
    }
    return mapping.get(name, name)


def _build_shap_explainer(model: Any) -> Any | None:
    """Build a SHAP TreeExplainer without making SHAP a hard failure point."""
    try:
        import shap

        return shap.TreeExplainer(model)
    except Exception:
        return None


def _extract_top_shap_drivers(
    explainer: Any | None,
    model_input: pd.DataFrame,
    feature_cols: Sequence[str],
    top_n: int = 3,
) -> list[dict[str, object]]:
    """Return the strongest local SHAP contributions for one prediction."""
    if explainer is None or model_input.empty:
        return []

    try:
        explanation = explainer(model_input)
        values = np.asarray(explanation.values, dtype=float)

        if values.ndim == 3:
            values = values[:, :, 0]

        if values.ndim == 2:
            shap_values = values[0]
        else:
            shap_values = values.reshape(-1)

        feature_values = model_input.iloc[0].to_numpy(dtype=float)

        driver_rows: list[dict[str, object]] = []
        for index, feature in enumerate(feature_cols):
            if index >= len(shap_values):
                break

            shap_value = safe_float(shap_values[index])
            feature_value = safe_float(feature_values[index])

            driver_rows.append(
                {
                    "Feature": str(feature),
                    "FeatureValue": feature_value,
                    "SHAPValue": shap_value,
                    "AbsSHAPValue": abs(shap_value),
                    "Direction": (
                        "Positive"
                        if shap_value > 0
                        else "Negative"
                        if shap_value < 0
                        else "Neutral"
                    ),
                }
            )

        driver_rows.sort(
            key=lambda row: safe_float(row["AbsSHAPValue"]),
            reverse=True,
        )

        return driver_rows[: max(int(top_n), 0)]

    except Exception:
        return []


def _format_shap_drivers(
    drivers: Sequence[dict[str, object]],
) -> str:
    """Format local SHAP evidence for dashboards and LLM prompts."""
    if not drivers:
        return ""

    parts: list[str] = []
    for driver in drivers:
        parts.append(
            f"{driver.get('Feature', '')} "
            f"({driver.get('Direction', 'Neutral')}, "
            f"SHAP={safe_float(driver.get('SHAPValue', 0.0)):.4f})"
        )

    return "; ".join(parts)


def _attach_driver_columns(
    row: dict[str, object],
    prefix: str,
    drivers: Sequence[dict[str, object]],
) -> None:
    """Attach structured Top-3 SHAP fields to one scenario row."""
    row[f"{prefix}TopDrivers"] = _format_shap_drivers(drivers)

    for rank in range(1, 4):
        driver = drivers[rank - 1] if rank <= len(drivers) else {}
        row[f"{prefix}Driver{rank}Feature"] = driver.get("Feature", "")
        row[f"{prefix}Driver{rank}FeatureValue"] = safe_float(
            driver.get("FeatureValue", 0.0)
        )
        row[f"{prefix}Driver{rank}SHAP"] = safe_float(
            driver.get("SHAPValue", 0.0)
        )
        row[f"{prefix}Driver{rank}Direction"] = driver.get(
            "Direction", ""
        )


def build_shap_explanation_table(
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert selected-scenario SHAP driver columns into an audit table."""
    if best_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for _, source_row in best_df.iterrows():
        for target, prefix, algorithm_column in [
            ("Conversions", "Conversion", "ConversionModelAlgorithm"),
            ("Revenue", "Revenue", "RevenueModelAlgorithm"),
        ]:
            for rank in range(1, 4):
                feature = str(
                    source_row.get(f"{prefix}Driver{rank}Feature", "") or ""
                ).strip()

                if not feature:
                    continue

                rows.append(
                    {
                        "CampaignId": source_row.get("CampaignId", 0),
                        "Campaign": source_row.get("Campaign", ""),
                        "Target": target,
                        "Algorithm": source_row.get(algorithm_column, ""),
                        "Feature": feature,
                        "FeatureValue": safe_float(
                            source_row.get(
                                f"{prefix}Driver{rank}FeatureValue", 0.0
                            )
                        ),
                        "SHAPValue": safe_float(
                            source_row.get(
                                f"{prefix}Driver{rank}SHAP", 0.0
                            )
                        ),
                        "Direction": source_row.get(
                            f"{prefix}Driver{rank}Direction", ""
                        ),
                        "Rank": rank,
                    }
                )

    return pd.DataFrame(rows)

def simulate_budget_scenarios(
    latest_df: pd.DataFrame,
    model_conv: Any,
    model_rev: Any,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Generate safe budget scenarios for every campaign."""
    if latest_df.empty:
        return pd.DataFrame()

    if not feature_cols:
        raise ValueError(
            "feature_cols cannot be empty."
        )

    results: list[dict[str, object]] = []

    conversion_explainer = _build_shap_explainer(model_conv)
    revenue_explainer = _build_shap_explainer(model_rev)
    conversion_algorithm = _model_algorithm_name(model_conv)
    revenue_algorithm = _model_algorithm_name(model_rev)

    for _, source_row in latest_df.iterrows():
        row = source_row.copy()

        campaign_type = str(
            row.get(
                "CampaignType",
                "Generic",
            )
        )

        scenario_factors = (
            CAMPAIGN_TYPE_SCENARIO_FACTORS.get(
                campaign_type,
                CAMPAIGN_TYPE_SCENARIO_FACTORS[
                    "Generic"
                ],
            )
        )

        current_spend = max(
            0.0,
            safe_float(
                row.get("Spend", 0)
            ),
        )

        clicks = max(
            0.0,
            safe_float(
                row.get("Clicks", 0)
            ),
        )

        conversions = max(
            0.0,
            safe_float(
                row.get(
                    "Conversions",
                    0,
                )
            ),
        )

        conversion_value = max(
            0.0,
            safe_float(
                row.get(
                    "ConversionValue",
                    0,
                )
            ),
        )

        history_rows = int(
            max(
                0.0,
                safe_float(
                    row.get(
                        "HistoryRows",
                        0,
                    )
                ),
            )
        )

        season_multiplier = safe_float(
            row.get(
                "ExpectedROASMultiplier",
                1.0,
            ),
            default=1.0,
        )

        if season_multiplier <= 0:
            season_multiplier = 1.0

        is_holiday = int(
            safe_float(
                row.get(
                    "IsHoliday",
                    0,
                )
            )
            > 0
        )

        is_pre_holiday = int(
            safe_float(
                row.get(
                    "IsPreHoliday",
                    0,
                )
            )
            > 0
        )

        for factor in scenario_factors:
            sim_row = row.copy()

            scenario_spend = (
                current_spend
                * safe_float(
                    factor,
                    default=1.0,
                )
            )

            sim_row["Spend"] = scenario_spend

            sim_row["CPC"] = (
                scenario_spend / clicks
                if clicks > 0
                else 0.0
            )

            sim_row["CPA"] = (
                scenario_spend / conversions
                if conversions > 0
                else 0.0
            )

            sim_row["ROAS"] = (
                conversion_value
                / scenario_spend
                if scenario_spend > 0
                else 0.0
            )

            sim_row["Profit"] = (
                conversion_value
                - scenario_spend
            )

            X_input = _prepare_prediction_input(
                sim_row,
                feature_cols,
            )

            pred_conv = safe_prediction(
                model_conv.predict(
                    X_input
                )[0]
            )

            pred_rev = safe_prediction(
                model_rev.predict(
                    X_input
                )[0]
            )

            pred_rev_adjusted = (
                pred_rev
                * season_multiplier
            )

            if scenario_spend <= 0:
                pred_conv = 0.0
                pred_rev_adjusted = 0.0
                pred_profit = 0.0
                pred_roas = 0.0
            else:
                pred_profit = (
                    pred_rev_adjusted
                    - scenario_spend
                )

                pred_roas = (
                    pred_rev_adjusted
                    / scenario_spend
                )

            conversion_drivers = _extract_top_shap_drivers(
                conversion_explainer,
                X_input,
                feature_cols,
                top_n=3,
            )

            revenue_drivers = _extract_top_shap_drivers(
                revenue_explainer,
                X_input,
                feature_cols,
                top_n=3,
            )

            result_row: dict[str, object] = {
                    "ConversionModelAlgorithm": conversion_algorithm,
                    "RevenueModelAlgorithm": revenue_algorithm,
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
                    "IsHoliday": is_holiday,
                    "IsPreHoliday": (
                        is_pre_holiday
                    ),
                    "HolidayName": row.get(
                        "HolidayName",
                        "",
                    ),
                    "ExpectedROASMultiplier": (
                        round(
                            season_multiplier,
                            3,
                        )
                    ),
                    "HistoryRows": history_rows,
                    "CurrentSpend": round(
                        current_spend,
                        2,
                    ),
                    "ScenarioFactor": (
                        safe_float(factor)
                    ),
                    "ScenarioSpend": round(
                        scenario_spend,
                        2,
                    ),
                    "PredictedConversions": (
                        round(
                            pred_conv,
                            4,
                        )
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
                    "AnalysisStartDate": row.get(
                        "AnalysisStartDate",
                        "",
                    ),
                    "AnalysisEndDate": row.get(
                        "AnalysisEndDate",
                        "",
                    ),
                }

            _attach_driver_columns(
                result_row,
                "Conversion",
                conversion_drivers,
            )
            _attach_driver_columns(
                result_row,
                "Revenue",
                revenue_drivers,
            )

            results.append(result_row)

    return pd.DataFrame(results)


def choose_optimal_scenario(
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Select the highest scoring scenario per campaign."""
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

    numeric_columns = [
        "PredictedRevenue",
        "PredictedProfit",
        "PredictedROAS",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    df["OptimizationScore"] = (
        df["PredictedRevenue"] * 0.45
        + df["PredictedProfit"] * 0.35
        + df["PredictedROAS"]
        * 100
        * 0.20
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


def apply_budget_guardrails(
    dataframe: pd.DataFrame,
    current_spend_column: str = "CurrentSpend",
    budget_column: str = "ScenarioSpend",
    max_increase_pct: float = (
        DEFAULT_MAX_BUDGET_INCREASE_PCT
    ),
    max_decrease_pct: float = (
        DEFAULT_MAX_BUDGET_DECREASE_PCT
    ),
) -> pd.DataFrame:
    """Prevent unsafe budget jumps outside configured limits."""
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    if current_spend_column not in result.columns:
        result[current_spend_column] = 0.0

    if budget_column not in result.columns:
        result[budget_column] = 0.0

    current_values = pd.to_numeric(
        result[current_spend_column],
        errors="coerce",
    ).fillna(0).clip(lower=0)

    planned_values = pd.to_numeric(
        result[budget_column],
        errors="coerce",
    ).fillna(0).clip(lower=0)

    min_factor = max(
        0.0,
        1.0
        - safe_float(
            max_decrease_pct
        )
        / 100.0,
    )

    max_factor = (
        1.0
        + max(
            0.0,
            safe_float(
                max_increase_pct
            ),
        )
        / 100.0
    )

    lower_bounds = (
        current_values * min_factor
    )

    upper_bounds = (
        current_values * max_factor
    )

    guarded_values = planned_values.copy()

    active_mask = current_values > 0

    guarded_values.loc[active_mask] = np.minimum(
        np.maximum(
            planned_values.loc[
                active_mask
            ],
            lower_bounds.loc[
                active_mask
            ],
        ),
        upper_bounds.loc[
            active_mask
        ],
    )

    guarded_values.loc[~active_mask] = 0.0

    result["OriginalPlannedBudget"] = (
        planned_values.round(2)
    )

    result[budget_column] = (
        guarded_values.round(2)
    )

    result["BudgetGuardrailApplied"] = (
        (
            planned_values
            - guarded_values
        )
        .abs()
        .gt(0.01)
    )

    result["BudgetChange"] = (
        guarded_values
        - current_values
    ).round(2)

    result["BudgetChangePct"] = np.where(
        current_values > 0,
        (
            result["BudgetChange"]
            / current_values
        )
        * 100,
        0.0,
    )

    result["BudgetChangePct"] = (
        pd.to_numeric(
            result["BudgetChangePct"],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            0,
        )
        .fillna(0)
        .round(2)
    )

    return result


def add_baseline_uplift(
    best_df: pd.DataFrame,
    sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the selected scenario with the 1.00 baseline."""
    if best_df.empty:
        return best_df.copy()

    if sim_df.empty:
        return best_df.copy()

    df = best_df.copy()

    baseline = (
        sim_df[
            pd.to_numeric(
                sim_df["ScenarioFactor"],
                errors="coerce",
            ).round(4)
            == 1.0
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
            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    df["RevenueUplift"] = (
        pd.to_numeric(
            df["PredictedRevenue"],
            errors="coerce",
        ).fillna(0)
        - df["BaselinePredictedRevenue"]
    )

    df["ProfitUplift"] = (
        pd.to_numeric(
            df["PredictedProfit"],
            errors="coerce",
        ).fillna(0)
        - df["BaselinePredictedProfit"]
    )

    df["ConversionUplift"] = (
        pd.to_numeric(
            df["PredictedConversions"],
            errors="coerce",
        ).fillna(0)
        - df[
            "BaselinePredictedConversions"
        ]
    )

    df["RevenueUpliftPct"] = np.where(
        df["BaselinePredictedRevenue"] > 0,
        (
            df["RevenueUplift"]
            / df[
                "BaselinePredictedRevenue"
            ]
        )
        * 100,
        0.0,
    )

    df["RevenueUpliftPct"] = (
        pd.to_numeric(
            df["RevenueUpliftPct"],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            0,
        )
        .fillna(0)
        .round(2)
    )

    return df


def aggregate_campaign_period(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate one analysis period by campaign."""
    if dataframe.empty:
        return pd.DataFrame()

    required_columns = [
        "Campaign",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing period columns: "
            + ", ".join(missing_columns)
        )

    df = dataframe.copy()

    metric_columns = {
        "Spend": "Spend",
        "ConversionValue": "Revenue",
        "Conversions": "Conversions",
        "Clicks": "Clicks",
        "Impressions": "Impressions",
    }

    for source_column in metric_columns:
        if source_column not in df.columns:
            df[source_column] = 0.0

        df[source_column] = pd.to_numeric(
            df[source_column],
            errors="coerce",
        ).fillna(0)

    grouping_columns = [
        "Campaign",
    ]

    if "CampaignId" in df.columns:
        grouping_columns.insert(
            0,
            "CampaignId",
        )

    aggregated = (
        df.groupby(
            grouping_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            Spend=("Spend", "sum"),
            Revenue=(
                "ConversionValue",
                "sum",
            ),
            Conversions=(
                "Conversions",
                "sum",
            ),
            Clicks=("Clicks", "sum"),
            Impressions=(
                "Impressions",
                "sum",
            ),
        )
    )

    aggregated["ROAS"] = np.where(
        aggregated["Spend"] > 0,
        aggregated["Revenue"]
        / aggregated["Spend"],
        0.0,
    )

    aggregated["CPA"] = np.where(
        aggregated["Conversions"] > 0,
        aggregated["Spend"]
        / aggregated["Conversions"],
        0.0,
    )

    rename_map = {
        column: f"{prefix}{column}"
        for column in [
            "Spend",
            "Revenue",
            "Conversions",
            "Clicks",
            "Impressions",
            "ROAS",
            "CPA",
        ]
    }

    return aggregated.rename(
        columns=rename_map
    )


def add_comparison_signals(
    current_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add current-versus-comparison campaign signals."""
    current_summary = aggregate_campaign_period(
        current_df,
        prefix="Current",
    )

    if current_summary.empty:
        return current_summary

    comparison_summary = (
        aggregate_campaign_period(
            comparison_df,
            prefix="Comparison",
        )
        if not comparison_df.empty
        else pd.DataFrame()
    )

    join_columns = [
        column
        for column in [
            "CampaignId",
            "Campaign",
        ]
        if column in current_summary.columns
        and (
            comparison_summary.empty
            or column
            in comparison_summary.columns
        )
    ]

    if not join_columns:
        join_columns = ["Campaign"]

    if comparison_summary.empty:
        result = current_summary.copy()

        for metric in [
            "Spend",
            "Revenue",
            "Conversions",
            "Clicks",
            "Impressions",
            "ROAS",
            "CPA",
        ]:
            result[
                f"Comparison{metric}"
            ] = 0.0
    else:
        result = current_summary.merge(
            comparison_summary,
            on=join_columns,
            how="left",
        )

    for metric in [
        "Spend",
        "Revenue",
        "Conversions",
        "Clicks",
        "Impressions",
        "ROAS",
        "CPA",
    ]:
        comparison_column = (
            f"Comparison{metric}"
        )

        current_column = (
            f"Current{metric}"
        )

        result[comparison_column] = (
            pd.to_numeric(
                result.get(
                    comparison_column,
                    0,
                ),
                errors="coerce",
            )
            .fillna(0)
        )

        result[
            f"{metric}ChangePct"
        ] = np.where(
            result[comparison_column] != 0,
            (
                result[current_column]
                - result[
                    comparison_column
                ]
            )
            / result[
                comparison_column
            ].abs()
            * 100,
            0.0,
        )

        result[
            f"{metric}ChangePct"
        ] = (
            pd.to_numeric(
                result[
                    f"{metric}ChangePct"
                ],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                0,
            )
            .fillna(0)
            .round(2)
        )

    result["ComparisonDataAvailable"] = (
        result["ComparisonSpend"] > 0
    )

    return result


def build_reference_benchmarks(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build reference benchmarks for the future new-campaign engine.

    This does not create a new campaign by itself. The next engine will
    use these benchmarks to propose campaign type, starting daily budget,
    target ROAS and test duration.
    """
    if dataframe.empty:
        return pd.DataFrame()

    df = dataframe.copy()

    if "CampaignType" not in df.columns:
        if "Campaign" not in df.columns:
            return pd.DataFrame()

        df = add_campaign_type(df)

    grouping_columns = [
        "CampaignType",
    ]

    for optional_column in [
        "Channel",
        "Category",
        "ProductGroup",
        "Season",
        "IsHoliday",
        "HolidayName",
    ]:
        if optional_column in df.columns:
            grouping_columns.append(
                optional_column
            )

    for column in [
        "Spend",
        "ConversionValue",
        "Conversions",
        "Clicks",
        "Impressions",
    ]:
        if column not in df.columns:
            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

    aggregated = (
        df.groupby(
            grouping_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            TotalSpend=(
                "Spend",
                "sum",
            ),
            TotalRevenue=(
                "ConversionValue",
                "sum",
            ),
            TotalConversions=(
                "Conversions",
                "sum",
            ),
            TotalClicks=(
                "Clicks",
                "sum",
            ),
            TotalImpressions=(
                "Impressions",
                "sum",
            ),
            DataRows=(
                "CampaignType",
                "size",
            ),
        )
    )

    if (
        "Date" in df.columns
        and df["Date"].notna().any()
    ):
        day_counts = (
            df.groupby(
                grouping_columns,
                as_index=False,
                dropna=False,
            )["Date"]
            .nunique()
            .rename(
                columns={
                    "Date": "DataDays",
                }
            )
        )

        aggregated = aggregated.merge(
            day_counts,
            on=grouping_columns,
            how="left",
        )
    else:
        aggregated["DataDays"] = 0

    aggregated["ROAS"] = np.where(
        aggregated["TotalSpend"] > 0,
        aggregated["TotalRevenue"]
        / aggregated["TotalSpend"],
        0.0,
    )

    aggregated["CPA"] = np.where(
        aggregated[
            "TotalConversions"
        ]
        > 0,
        aggregated["TotalSpend"]
        / aggregated[
            "TotalConversions"
        ],
        0.0,
    )

    aggregated["ConversionRate"] = np.where(
        aggregated["TotalClicks"] > 0,
        aggregated[
            "TotalConversions"
        ]
        / aggregated["TotalClicks"],
        0.0,
    )

    aggregated["AverageDailySpend"] = np.where(
        aggregated["DataDays"] > 0,
        aggregated["TotalSpend"]
        / aggregated["DataDays"],
        0.0,
    )

    aggregated[
        "SuggestedStartingDailyBudget"
    ] = (
        aggregated["AverageDailySpend"]
        .clip(lower=0)
        .round(2)
    )

    aggregated["ReferenceScore"] = (
        aggregated["ROAS"] * 0.45
        + np.log1p(
            aggregated["TotalRevenue"]
        )
        * 0.30
        + np.log1p(
            aggregated["TotalConversions"]
        )
        * 0.15
        + np.log1p(
            aggregated["DataDays"]
        )
        * 0.10
    )

    return (
        aggregated
        .sort_values(
            "ReferenceScore",
            ascending=False,
        )
        .reset_index(drop=True)
    )
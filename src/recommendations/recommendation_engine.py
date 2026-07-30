from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    compute_kpis,
    compute_roas_target_gap,
)
from src.llm.manager import generate_text
from src.models.budget_optimizer import add_campaign_type
from src.utils.logger import get_logger


logger = get_logger(__name__)


def add_budget_spike_flag(
    df: pd.DataFrame,
    spike_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Flag unusually large budget increases.

    A budget spike is recorded when the recommended budget is greater
    than the current spend by more than the configured threshold.

    Parameters
    ----------
    df:
        Recommendation dataframe.
    spike_threshold:
        Maximum accepted increase ratio. The default value of 0.50
        represents a 50 percent increase.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe containing BudgetSpike and
        BudgetSpikeWarning columns.
    """
    if df.empty:
        return df.copy()

    result_df = df.copy()

    if "CurrentSpend" not in result_df.columns:
        raise KeyError(
            "CurrentSpend column is required."
        )

    if "ScenarioSpend" in result_df.columns:
        recommended_spend = pd.to_numeric(
            result_df["ScenarioSpend"],
            errors="coerce",
        ).fillna(0.0)

    elif "RecommendedBudget" in result_df.columns:
        recommended_spend = pd.to_numeric(
            result_df["RecommendedBudget"],
            errors="coerce",
        ).fillna(0.0)

    else:
        raise KeyError(
            "ScenarioSpend or RecommendedBudget column is required."
        )

    current_spend = pd.to_numeric(
        result_df["CurrentSpend"],
        errors="coerce",
    ).fillna(0.0)

    budget_change_ratio = (
        recommended_spend - current_spend
    ) / current_spend.replace(0.0, np.nan)

    result_df["BudgetSpike"] = (
        budget_change_ratio > float(spike_threshold)
    ).fillna(False)

    warning_text = (
        "Warning: Budget increase exceeds "
        f"{float(spike_threshold) * 100:.0f}%. "
        "Monitor closely for the first 7 days."
    )

    result_df["BudgetSpikeWarning"] = np.where(
        result_df["BudgetSpike"],
        warning_text,
        "",
    )

    return result_df


def build_action_recommendation(
    best_df: pd.DataFrame,
    target_roas: float,
) -> pd.DataFrame:
    """
    Convert model scenarios into campaign-level budget actions.

    Parameters
    ----------
    best_df:
        Dataframe containing the selected scenario for each campaign.
    target_roas:
        Business ROAS target.

    Returns
    -------
    pd.DataFrame
        Recommendation dataframe containing actions, reasons and
        budget-change metrics.
    """
    if best_df.empty:
        return best_df.copy()

    required_columns = [
        "CurrentSpend",
        "ScenarioSpend",
        "PredictedROAS",
        "PredictedConversions",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in best_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing recommendation columns: "
            + ", ".join(missing_columns)
        )

    df = best_df.copy()

    numeric_columns = [
        "CurrentSpend",
        "ScenarioSpend",
        "PredictedROAS",
        "PredictedConversions",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    resolved_target_roas = max(
        float(target_roas),
        0.0,
    )

    def decide_action(
        row: pd.Series,
    ) -> tuple[str, str]:
        current_spend = float(
            row.get("CurrentSpend", 0.0) or 0.0
        )

        optimal_spend = float(
            row.get("ScenarioSpend", 0.0) or 0.0
        )

        predicted_roas = float(
            row.get("PredictedROAS", 0.0) or 0.0
        )

        predicted_conversions = float(
            row.get(
                "PredictedConversions",
                0.0,
            )
            or 0.0
        )

        if current_spend <= 0:
            return (
                "Review",
                "No active spend detected. "
                "Manual review is required.",
            )

        if (
            predicted_conversions <= 0
            and optimal_spend < current_spend
        ):
            return (
                "Pause / Review",
                "Predicted value remains weak even under "
                "lower-spend scenarios.",
            )

        spend_ratio = (
            optimal_spend / current_spend
        )

        roas_ok = (
            predicted_roas
            >= resolved_target_roas * 0.90
        )

        if spend_ratio >= 1.15 and roas_ok:
            return (
                "Increase Budget",
                "Predicted ROAS meets the target. "
                "Controlled scaling is recommended.",
            )

        if spend_ratio >= 1.15 and not roas_ok:
            return (
                "Increase Budget With ROAS Risk",
                "Budget increase potential exists, but "
                "predicted ROAS remains below target.",
            )

        if spend_ratio <= 0.85:
            return (
                "Reduce Budget",
                "Predicted return suggests that the "
                "campaign is overfunded.",
            )

        return (
            "Maintain",
            "Predicted performance supports keeping the "
            "budget near its current level.",
        )

    decisions = df.apply(
        decide_action,
        axis=1,
    )

    df["RecommendedAction"] = decisions.apply(
        lambda decision: decision[0]
    )

    df["RecommendationReason"] = decisions.apply(
        lambda decision: decision[1]
    )

    df["BudgetChange"] = (
        df["ScenarioSpend"]
        - df["CurrentSpend"]
    )

    df["BudgetChangePct"] = np.where(
        df["CurrentSpend"] > 0,
        (
            df["BudgetChange"]
            / df["CurrentSpend"]
        )
        * 100.0,
        0.0,
    )

    df["BudgetChange"] = pd.to_numeric(
        df["BudgetChange"],
        errors="coerce",
    ).fillna(0.0).round(2)

    df["BudgetChangePct"] = pd.to_numeric(
        df["BudgetChangePct"],
        errors="coerce",
    ).fillna(0.0).round(2)

    df = df.rename(
        columns={
            "ScenarioSpend": "RecommendedBudget",
        }
    )

    return df


def build_confidence_scores(
    recommendation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate recommendation confidence using model quality and history.

    The confidence calculation combines:

    - Number of historical campaign observations
    - Conversion-model R²
    - Revenue-model R²

    Existing HistoryRows values are retained as a fallback when the
    training dataframe cannot provide campaign-level history.

    Parameters
    ----------
    recommendation_df:
        Campaign recommendation dataframe.
    metrics_df:
        Model-evaluation dataframe containing Model and R2 columns.
    train_df:
        Model-training dataframe.

    Returns
    -------
    pd.DataFrame
        Recommendation dataframe containing HistoryRows,
        AverageModelR2 and ConfidenceLevel.
    """
    if recommendation_df.empty:
        return recommendation_df.copy()

    df = recommendation_df.copy()

    if "HistoryRows" in df.columns:
        existing_history = pd.to_numeric(
            df["HistoryRows"],
            errors="coerce",
        )
    else:
        existing_history = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    if (
        train_df.empty
        or "CampaignId" not in train_df.columns
        or "CampaignId" not in df.columns
    ):
        calculated_history = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    else:
        campaign_history = (
            train_df.groupby(
                "CampaignId",
                dropna=False,
            )
            .size()
        )

        calculated_history = pd.to_numeric(
            df["CampaignId"].map(
                campaign_history
            ),
            errors="coerce",
        )

    df["HistoryRows"] = (
        calculated_history
        .fillna(existing_history)
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(int)
    )

    conversion_r2 = 0.0
    revenue_r2 = 0.0

    if (
        not metrics_df.empty
        and "Model" in metrics_df.columns
        and "R2" in metrics_df.columns
    ):
        model_names = (
            metrics_df["Model"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        r2_values = pd.to_numeric(
            metrics_df["R2"],
            errors="coerce",
        )

        conversion_rows = r2_values[
            model_names.eq("conversions")
        ].dropna()

        revenue_rows = r2_values[
            model_names.eq("revenue")
        ].dropna()

        if not conversion_rows.empty:
            conversion_r2 = float(
                conversion_rows.iloc[0]
            )

        if not revenue_rows.empty:
            revenue_r2 = float(
                revenue_rows.iloc[0]
            )

    conversion_r2 = float(
        np.clip(
            conversion_r2,
            -1.0,
            1.0,
        )
    )

    revenue_r2 = float(
        np.clip(
            revenue_r2,
            -1.0,
            1.0,
        )
    )

    average_r2 = (
        conversion_r2 + revenue_r2
    ) / 2.0

    df["ConversionModelR2"] = round(
        conversion_r2,
        4,
    )

    df["RevenueModelR2"] = round(
        revenue_r2,
        4,
    )

    df["AverageModelR2"] = round(
        average_r2,
        4,
    )

    def confidence_label(
        row: pd.Series,
    ) -> str:
        history_rows = int(
            pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "HistoryRows",
                            0,
                        )
                    ]
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[0]
        )

        if (
            history_rows >= 20
            and average_r2 >= 0.60
        ):
            return "High"

        if (
            history_rows >= 10
            and average_r2 >= 0.30
        ):
            return "Medium"

        return "Low"

    df["ConfidenceLevel"] = df.apply(
        confidence_label,
        axis=1,
    )

    return df
def apply_confidence_guardrail(
    df: pd.DataFrame,
    target_roas: float | None = None,
) -> pd.DataFrame:
    """
    Apply conservative rules to low-confidence recommendations.

    High- and medium-confidence recommendations remain unchanged.
    Low-confidence recommendations are restricted to safer actions.

    Parameters
    ----------
    df:
        Recommendation dataframe.
    target_roas:
        Optional business ROAS target.

    Returns
    -------
    pd.DataFrame
        Dataframe containing guarded actions, original model actions,
        recommendation reasons and decision-basis information.
    """
    if df.empty:
        return df.copy()

    required_columns = [
        "ConfidenceLevel",
        "RecommendedAction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing confidence guardrail columns: "
            + ", ".join(missing_columns)
        )

    result_df = df.copy()

    if "RecommendationReason" not in result_df.columns:
        result_df["RecommendationReason"] = ""

    result_df["ModelRecommendedAction"] = (
        result_df["RecommendedAction"]
        .fillna("Review")
        .astype(str)
    )

    result_df["DecisionBasis"] = (
        "ML prediction"
    )

    has_current_spend = (
        "CurrentSpend" in result_df.columns
    )

    has_recommended_budget = (
        "RecommendedBudget" in result_df.columns
    )

    has_predicted_roas = (
        "PredictedROAS" in result_df.columns
    )

    has_predicted_conversions = (
        "PredictedConversions"
        in result_df.columns
    )

    low_confidence_mask = (
        result_df["ConfidenceLevel"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("low")
    )

    for index in result_df.index[
        low_confidence_mask
    ]:
        row = result_df.loc[index]

        model_action = str(
            row.get(
                "RecommendedAction",
                "Review",
            )
            or "Review"
        ).strip()

        if not has_current_spend:
            result_df.at[
                index,
                "RecommendedAction",
            ] = "Review"

            result_df.at[
                index,
                "RecommendationReason",
            ] = (
                "Low confidence prediction. "
                "Manual validation is recommended "
                "before taking action."
            )

            result_df.at[
                index,
                "DecisionBasis",
            ] = (
                "Conservative low-confidence "
                "guardrail"
            )

            continue

        current_spend = float(
            pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            "CurrentSpend",
                            0.0,
                        )
                    ]
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[0]
        )

        if has_recommended_budget:
            recommended_budget = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "RecommendedBudget",
                                current_spend,
                            )
                        ]
                    ),
                    errors="coerce",
                )
                .fillna(current_spend)
                .iloc[0]
            )
        else:
            recommended_budget = current_spend

        if has_predicted_roas:
            predicted_roas = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "PredictedROAS",
                                0.0,
                            )
                        ]
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )
        else:
            predicted_roas = 0.0

        if has_predicted_conversions:
            predicted_conversions = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "PredictedConversions",
                                0.0,
                            )
                        ]
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )
        else:
            predicted_conversions = None

        if current_spend <= 0:
            action = "Review"

            reason = (
                "Low confidence prediction. "
                "No active spend was detected. "
                "Manual validation is required before "
                "launching or funding the campaign."
            )

        elif (
            has_predicted_conversions
            and predicted_conversions is not None
            and predicted_conversions <= 0
        ):
            action = "Pause / Review"

            reason = (
                "Low confidence prediction. "
                "The model predicts no conversions. "
                "Validate conversion tracking before "
                "changing the budget."
            )

        elif model_action == "Reduce Budget":
            action = "Reduce Budget"

            reason = (
                "Low confidence prediction. "
                "Apply the budget reduction gradually "
                "and monitor performance daily."
            )

        elif model_action in {
            "Maintain",
            "Pause / Review",
            "Review",
        }:
            action = model_action

            reason = (
                "Low confidence prediction. "
                "The conservative action is retained "
                "until more historical observations "
                "become available."
            )

        elif model_action.startswith(
            "Increase Budget"
        ):
            resolved_target_roas = (
                float(target_roas)
                if target_roas is not None
                else None
            )

            roas_is_strong = (
                resolved_target_roas is not None
                and predicted_roas
                >= resolved_target_roas * 1.10
            )

            increase_ratio = (
                recommended_budget
                / current_spend
                if current_spend > 0
                else 0.0
            )

            if (
                roas_is_strong
                and increase_ratio <= 1.25
            ):
                action = "Increase Budget"

                reason = (
                    "Low confidence prediction. "
                    "ROAS is materially above target. "
                    "Use only a capped test increase "
                    "and monitor performance for at "
                    "least 7 days."
                )

            else:
                action = "Maintain"

                reason = (
                    "Low confidence prediction. "
                    "The model suggests an increase, "
                    "but the available history is not "
                    "sufficient. Keep the current budget "
                    "until more observations are available."
                )

        else:
            action = "Review"

            reason = (
                "Low confidence prediction. "
                "The recommendation could not be "
                "validated with enough history. "
                "Manual validation is recommended."
            )

        result_df.at[
            index,
            "RecommendedAction",
        ] = action

        result_df.at[
            index,
            "RecommendationReason",
        ] = reason

        result_df.at[
            index,
            "DecisionBasis",
        ] = (
            "Conservative low-confidence "
            "guardrail"
        )

    return result_df


def build_portfolio_allocation(
    recommendation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Redistribute the current total budget across campaigns.

    The total portfolio budget remains equal to the existing total
    spend. Recommended budgets are used as allocation weights.

    Parameters
    ----------
    recommendation_df:
        Campaign-level recommendation dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe containing optimized portfolio budgets and changes.
    """
    if recommendation_df.empty:
        return recommendation_df.copy()

    required_columns = [
        "CurrentSpend",
        "RecommendedBudget",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in recommendation_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing portfolio columns: "
            + ", ".join(missing_columns)
        )

    df = recommendation_df.copy()

    df["CurrentSpend"] = (
        pd.to_numeric(
            df["CurrentSpend"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df["RecommendedBudget"] = (
        pd.to_numeric(
            df["RecommendedBudget"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    total_current = float(
        df["CurrentSpend"].sum()
    )

    total_recommended = float(
        df["RecommendedBudget"].sum()
    )

    if total_current <= 0:
        df["OptimizedPortfolioBudget"] = 0.0
        df["PortfolioBudgetChange"] = 0.0
        df["PortfolioBudgetChangePct"] = 0.0

        return df

    if total_recommended <= 0:
        df["OptimizedPortfolioBudget"] = (
            df["CurrentSpend"]
        )

    else:
        df["OptimizedPortfolioBudget"] = (
            df["RecommendedBudget"]
            / total_recommended
        ) * total_current

    df["OptimizedPortfolioBudget"] = (
        df["OptimizedPortfolioBudget"]
        .round(2)
    )

    rounding_difference = round(
        total_current
        - float(
            df["OptimizedPortfolioBudget"].sum()
        ),
        2,
    )

    if (
        rounding_difference != 0
        and not df.empty
    ):
        largest_budget_index = (
            df["OptimizedPortfolioBudget"]
            .idxmax()
        )

        df.at[
            largest_budget_index,
            "OptimizedPortfolioBudget",
        ] = round(
            float(
                df.at[
                    largest_budget_index,
                    "OptimizedPortfolioBudget",
                ]
            )
            + rounding_difference,
            2,
        )

    df["PortfolioBudgetChange"] = (
        df["OptimizedPortfolioBudget"]
        - df["CurrentSpend"]
    ).round(2)

    df["PortfolioBudgetChangePct"] = np.where(
        df["CurrentSpend"] > 0,
        (
            df["PortfolioBudgetChange"]
            / df["CurrentSpend"]
        )
        * 100.0,
        0.0,
    )

    df["PortfolioBudgetChangePct"] = (
        pd.to_numeric(
            df["PortfolioBudgetChangePct"],
            errors="coerce",
        )
        .fillna(0.0)
        .round(2)
    )

    return df


def build_recommendation_summary(
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final campaign recommendation output table.

    Only columns available in the dataframe are included. This keeps
    the function compatible with both ML and rule-based outputs.
    """
    if portfolio_df.empty:
        return portfolio_df.copy()

    summary_columns = [
        "CampaignId",
        "Campaign",
        "Channel",
        "CampaignType",
        "Category",
        "ProductGroup",
        "Season",
        "IsHoliday",
        "IsPreHoliday",
        "HolidayName",
        "ExpectedROASMultiplier",
        "CurrentSpend",
        "RecommendedBudget",
        "OptimizedPortfolioBudget",
        "BudgetChange",
        "BudgetChangePct",
        "PortfolioBudgetChange",
        "PortfolioBudgetChangePct",
        "RecommendedAction",
        "ModelRecommendedAction",
        "DecisionBasis",
        "ConfidenceLevel",
        "HistoryRows",
        "ConversionModelR2",
        "RevenueModelR2",
        "AverageModelR2",
        "PredictedConversions",
        "PredictedRevenue",
        "PredictedProfit",
        "PredictedROAS",
        "TargetROAS",
        "ROASStatus",
        "ROASGap",
        "ROASGapPct",
        "BaselinePredictedRevenue",
        "RevenueUplift",
        "RevenueUpliftPct",
        "BudgetSpike",
        "BudgetSpikeWarning",
        "RecommendationReason",
        "ExecutiveCommentary",
    ]

    available_columns = [
        column
        for column in summary_columns
        if column in portfolio_df.columns
    ]

    return (
        portfolio_df[available_columns]
        .copy()
        .reset_index(drop=True)
    )


def build_rule_based_fallback(
    ads_raw: pd.DataFrame,
    target_roas: float,
) -> pd.DataFrame:
    """
    Build campaign recommendations without machine-learning outputs.

    This fallback is used when the training set or trained model is not
    available. It relies on historical campaign KPIs and conservative
    business rules.

    Parameters
    ----------
    ads_raw:
        Raw Google Ads dataframe.
    target_roas:
        Business ROAS target.

    Returns
    -------
    pd.DataFrame
        Campaign-level fallback recommendation dataframe.
    """
    if ads_raw.empty:
        return pd.DataFrame()

    required_columns = [
        "CampaignId",
        "Campaign",
        "Channel",
        "Category",
        "ProductGroup",
        "Spend",
        "Clicks",
        "Impressions",
        "Conversions",
        "ConversionValue",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in ads_raw.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing fallback columns: "
            + ", ".join(missing_columns)
        )

    source_df = ads_raw.copy()

    numeric_columns = [
        "Spend",
        "Clicks",
        "Impressions",
        "Conversions",
        "ConversionValue",
    ]

    for column in numeric_columns:
        source_df[column] = pd.to_numeric(
            source_df[column],
            errors="coerce",
        ).fillna(0.0)

    campaign_df = source_df.groupby(
        [
            "CampaignId",
            "Campaign",
            "Channel",
            "Category",
            "ProductGroup",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=(
            "ConversionValue",
            "sum",
        ),
    )

    campaign_df = compute_kpis(
        campaign_df
    )

    campaign_df = compute_roas_target_gap(
        campaign_df,
        float(target_roas),
    )

    campaign_df = add_campaign_type(
        campaign_df
    )

    def fallback_action(
        row: pd.Series,
    ) -> tuple[str, str]:
        conversions = float(
            row.get(
                "Conversions",
                0.0,
            )
            or 0.0
        )

        spend = float(
            row.get(
                "Spend",
                0.0,
            )
            or 0.0
        )

        roas = float(
            row.get(
                "ROAS",
                0.0,
            )
            or 0.0
        )

        profit = float(
            row.get(
                "Profit",
                0.0,
            )
            or 0.0
        )

        if spend <= 0:
            return (
                "Review",
                "No active spend is available for "
                "rule-based evaluation.",
            )

        if conversions <= 0:
            return (
                "Pause / Review",
                "No conversions were observed in "
                "the available campaign history.",
            )

        if (
            roas >= float(target_roas) * 1.10
            and profit > 0
        ):
            return (
                "Maintain / Slight Increase",
                "Historical ROAS is materially above "
                "target. A limited test increase may "
                "be considered.",
            )

        if (
            roas < float(target_roas) * 0.90
            or profit < 0
        ):
            return (
                "Reduce Budget",
                "Historical efficiency is below the "
                "target or the campaign is unprofitable.",
            )

        return (
            "Maintain",
            "Historical performance is close to the "
            "business target.",
        )

    decisions = campaign_df.apply(
        fallback_action,
        axis=1,
    )

    campaign_df["RecommendedAction"] = (
        decisions.apply(
            lambda decision: decision[0]
        )
    )

    campaign_df["RecommendationReason"] = (
        decisions.apply(
            lambda decision: decision[1]
        )
    )

    campaign_df["ConfidenceLevel"] = "Low"

    campaign_df["RecommendedBudget"] = (
        campaign_df["Spend"]
    )

    campaign_df["CurrentSpend"] = (
        campaign_df["Spend"]
    )

    campaign_df["BudgetChange"] = 0.0
    campaign_df["BudgetChangePct"] = 0.0

    campaign_df["ModelRecommendedAction"] = (
        campaign_df["RecommendedAction"]
    )

    campaign_df["DecisionBasis"] = (
        "Rule-based fallback"
    )

    campaign_df["HistoryRows"] = 0
    campaign_df["ConversionModelR2"] = 0.0
    campaign_df["RevenueModelR2"] = 0.0
    campaign_df["AverageModelR2"] = 0.0

    return campaign_df
def build_llm_campaign_prompt(
    row: pd.Series,
    target_roas: float,
) -> str:
    """
    Build an executive-level LLM prompt for one campaign.

    Parameters
    ----------
    row:
        Campaign recommendation row.
    target_roas:
        Business ROAS target.

    Returns
    -------
    str
        Prompt used for campaign commentary generation.
    """
    is_holiday = bool(
        row.get(
            "IsHoliday",
            False,
        )
    )

    is_pre_holiday = bool(
        row.get(
            "IsPreHoliday",
            False,
        )
    )

    holiday_name = str(
        row.get(
            "HolidayName",
            "",
        )
        or ""
    ).strip()

    if is_holiday:
        holiday_context = (
            "Holiday effect is currently active"
        )

        if holiday_name:
            holiday_context += (
                f": {holiday_name}"
            )

        holiday_context += "."

    elif is_pre_holiday:
        holiday_context = (
            "The campaign is in a pre-holiday "
            "period and demand may increase."
        )

    else:
        holiday_context = (
            "No active holiday effect was detected."
        )

    current_spend = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "CurrentSpend",
                        0.0,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    recommended_budget = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "RecommendedBudget",
                        current_spend,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(current_spend)
        .iloc[0]
    )

    budget_change_pct = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "BudgetChangePct",
                        0.0,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    predicted_roas = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "PredictedROAS",
                        0.0,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    predicted_revenue = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "PredictedRevenue",
                        0.0,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    predicted_profit = float(
        pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "PredictedProfit",
                        0.0,
                    )
                ]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .iloc[0]
    )

    confidence_level = str(
        row.get(
            "ConfidenceLevel",
            "Unknown",
        )
        or "Unknown"
    ).strip()

    recommended_action = str(
        row.get(
            "RecommendedAction",
            "Review",
        )
        or "Review"
    ).strip()

    recommendation_reason = str(
        row.get(
            "RecommendationReason",
            "",
        )
        or ""
    ).strip()

    decision_basis = str(
        row.get(
            "DecisionBasis",
            "ML prediction",
        )
        or "ML prediction"
    ).strip()

    return f"""
You are a senior digital marketing budget analyst.

Write a concise executive-level campaign commentary in English.

Rules:
- Use no more than 3 sentences.
- Do not invent data.
- Do not promise guaranteed performance.
- Clearly mention risk when confidence is Low.
- Treat the recommendation as decision support, not an automatic action.
- Prefer controlled testing and monitoring for budget increases.
- Do not use markdown headings or bullet points.

Campaign: {row.get('Campaign', '')}
Campaign ID: {row.get('CampaignId', '')}
Campaign type: {row.get('CampaignType', '')}
Category: {row.get('Category', '')}
Product group: {row.get('ProductGroup', '')}
Channel: {row.get('Channel', '')}
Season: {row.get('Season', '')}
Holiday context: {holiday_context}

Current daily spend: {current_spend:.2f}
Recommended daily budget: {recommended_budget:.2f}
Budget change: {budget_change_pct:.2f}%
Predicted ROAS: {predicted_roas:.2f}
Target ROAS: {float(target_roas):.2f}
Predicted revenue: {predicted_revenue:.2f}
Predicted profit: {predicted_profit:.2f}
Confidence level: {confidence_level}
Decision basis: {decision_basis}
Recommended action: {recommended_action}
Recommendation reason: {recommendation_reason}

Executive commentary:
""".strip()


def generate_llm_commentary(
    summary_df: pd.DataFrame,
    target_roas: float,
    max_campaigns: int = 20,
) -> pd.DataFrame:
    """
    Generate LLM commentary for campaign recommendations.

    LLM failures do not interrupt the recommendation pipeline.
    A deterministic fallback commentary is used when necessary.

    Parameters
    ----------
    summary_df:
        Final campaign recommendation dataframe.
    target_roas:
        Business ROAS target.
    max_campaigns:
        Maximum number of campaigns sent to the LLM.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe containing ExecutiveCommentary.
    """
    if summary_df.empty:
        return summary_df.copy()

    result_df = summary_df.copy()

    safe_max_campaigns = max(
        int(max_campaigns),
        0,
    )

    commentaries: list[str] = []

    for position, (_, row) in enumerate(
        result_df.iterrows()
    ):
        if position >= safe_max_campaigns:
            commentaries.append(
                "AI commentary was not generated because "
                "the campaign processing limit was reached. "
                "Use the recommendation action, reason and "
                "confidence level for decision-making."
            )

            continue

        prompt = build_llm_campaign_prompt(
            row=row,
            target_roas=float(target_roas),
        )

        try:
            commentary = generate_text(
                prompt=prompt,
                max_tokens=300,
                temperature=0.2,
            )

        except Exception as exc:
            logger.warning(
                "LLM commentary failed for campaign %s: %s",
                row.get(
                    "Campaign",
                    "",
                ),
                exc,
            )

            commentary = None

        if commentary is None or not str(
            commentary
        ).strip():
            logger.warning(
                "LLM commentary unavailable for campaign %s. "
                "Using safe fallback commentary.",
                row.get(
                    "Campaign",
                    "",
                ),
            )

            commentary = (
                "AI commentary is unavailable. "
                "Use the rule-based recommendation and confidence level "
                "for decision-making."
            )

        commentaries.append(
            str(commentary).strip()
        )

    result_df["ExecutiveCommentary"] = (
        commentaries
    )

    return result_df


def generate_portfolio_summary_commentary(
    portfolio_df: pd.DataFrame,
    category_df: pd.DataFrame,
    target_roas: float,
) -> str:
    """
    Generate an executive-level portfolio summary.

    The function sends portfolio KPIs to the configured LLM provider.
    If the provider is unavailable, a deterministic summary is returned.

    Parameters
    ----------
    portfolio_df:
        Campaign recommendation and portfolio allocation dataframe.
    category_df:
        Category-level performance dataframe.
    target_roas:
        Business ROAS target.

    Returns
    -------
    str
        Executive portfolio commentary.
    """
    if portfolio_df.empty:
        return (
            "Portfolio commentary was skipped because "
            "the portfolio dataset is empty."
        )

    def numeric_series(
        column_name: str,
        default_value: float = 0.0,
    ) -> pd.Series:
        if column_name in portfolio_df.columns:
            source = portfolio_df[
                column_name
            ]
        else:
            source = pd.Series(
                default_value,
                index=portfolio_df.index,
                dtype=float,
            )

        return pd.to_numeric(
            source,
            errors="coerce",
        ).fillna(default_value)

    total_spend = float(
        numeric_series(
            "CurrentSpend"
        ).sum()
    )

    if (
        "OptimizedPortfolioBudget"
        in portfolio_df.columns
    ):
        total_recommended = float(
            numeric_series(
                "OptimizedPortfolioBudget"
            ).sum()
        )

    else:
        total_recommended = float(
            numeric_series(
                "RecommendedBudget"
            ).sum()
        )

    total_predicted_revenue = float(
        numeric_series(
            "PredictedRevenue"
        ).sum()
    )

    total_predicted_profit = float(
        numeric_series(
            "PredictedProfit"
        ).sum()
    )

    total_predicted_conversions = float(
        numeric_series(
            "PredictedConversions"
        ).sum()
    )

    campaign_count = int(
        len(portfolio_df)
    )

    recommended_actions = (
        portfolio_df.get(
            "RecommendedAction",
            pd.Series(
                "",
                index=portfolio_df.index,
                dtype=str,
            ),
        )
        .fillna("")
        .astype(str)
        .str.strip()
    )

    increase_count = int(
        recommended_actions.str.contains(
            "Increase",
            case=False,
            na=False,
        ).sum()
    )

    reduce_count = int(
        recommended_actions.str.contains(
            "Reduce",
            case=False,
            na=False,
        ).sum()
    )

    maintain_count = int(
        recommended_actions.str.contains(
            "Maintain",
            case=False,
            na=False,
        ).sum()
    )

    review_count = int(
        recommended_actions.str.contains(
            "Review|Pause",
            case=False,
            na=False,
            regex=True,
        ).sum()
    )

    confidence_values = (
        portfolio_df.get(
            "ConfidenceLevel",
            pd.Series(
                "",
                index=portfolio_df.index,
                dtype=str,
            ),
        )
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    high_confidence_count = int(
        confidence_values.eq(
            "high"
        ).sum()
    )

    medium_confidence_count = int(
        confidence_values.eq(
            "medium"
        ).sum()
    )

    low_confidence_count = int(
        confidence_values.eq(
            "low"
        ).sum()
    )

    spike_values = portfolio_df.get(
        "BudgetSpike",
        pd.Series(
            False,
            index=portfolio_df.index,
            dtype=bool,
        ),
    )

    spike_count = int(
        spike_values
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if (
        total_recommended > 0
        and total_predicted_revenue > 0
    ):
        portfolio_predicted_roas = (
            total_predicted_revenue
            / total_recommended
        )

    else:
        portfolio_predicted_roas = 0.0

    top_category = "N/A"
    top_category_roas = 0.0

    if not category_df.empty:
        category_source = (
            category_df.copy()
        )

        if "ROAS" in category_source.columns:
            category_source["ROAS"] = (
                pd.to_numeric(
                    category_source["ROAS"],
                    errors="coerce",
                )
                .fillna(0.0)
            )

            category_source = (
                category_source.sort_values(
                    "ROAS",
                    ascending=False,
                )
            )

        top_category_row = (
            category_source.iloc[0]
        )

        top_category = str(
            top_category_row.get(
                "Category",
                "N/A",
            )
            or "N/A"
        )

        top_category_roas = float(
            pd.to_numeric(
                pd.Series(
                    [
                        top_category_row.get(
                            "ROAS",
                            0.0,
                        )
                    ]
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[0]
        )

    prompt = f"""
You are a senior digital marketing director.

Write a concise executive-level portfolio summary in English.

Rules:
- Use no more than 5 sentences.
- Do not invent data.
- Do not guarantee future performance.
- Mention material risks and low-confidence recommendations.
- Explain that recommendations require monitoring before implementation.
- Do not use markdown headings or bullet points.

Total campaigns: {campaign_count}
Total current daily spend: {total_spend:.2f}
Total optimized daily portfolio budget: {total_recommended:.2f}
Predicted total conversions: {total_predicted_conversions:.2f}
Predicted total revenue: {total_predicted_revenue:.2f}
Predicted total profit: {total_predicted_profit:.2f}
Predicted portfolio ROAS: {portfolio_predicted_roas:.2f}
ROAS target: {float(target_roas):.2f}

Campaigns recommended to increase: {increase_count}
Campaigns recommended to reduce: {reduce_count}
Campaigns recommended to maintain: {maintain_count}
Campaigns requiring pause or review: {review_count}

High-confidence campaigns: {high_confidence_count}
Medium-confidence campaigns: {medium_confidence_count}
Low-confidence campaigns: {low_confidence_count}
Campaigns with budget-spike warnings: {spike_count}

Top-performing category: {top_category}
Top-category ROAS: {top_category_roas:.2f}

Portfolio executive summary:
""".strip()

    try:
        commentary = generate_text(
            prompt=prompt,
            max_tokens=500,
            temperature=0.2,
        )

    except Exception as exc:
        logger.warning(
            "Portfolio LLM commentary failed: %s",
            exc,
        )

        commentary = None

    if commentary is None or not str(
        commentary
    ).strip():
        logger.warning(
            "Portfolio LLM commentary is unavailable. "
            "Using deterministic fallback summary."
        )

        budget_change = (
            total_recommended
            - total_spend
        )

        budget_change_pct = (
            budget_change
            / total_spend
            * 100.0
            if total_spend > 0
            else 0.0
        )

        return (
            f"Portfolio contains {campaign_count} campaigns "
            f"with a current daily spend of {total_spend:.2f} "
            f"and an optimized daily budget of "
            f"{total_recommended:.2f}, representing a "
            f"{budget_change_pct:.2f}% change. "
            f"The model predicts total revenue of "
            f"{total_predicted_revenue:.2f}, total profit of "
            f"{total_predicted_profit:.2f} and a portfolio ROAS "
            f"of {portfolio_predicted_roas:.2f}, compared with a "
            f"ROAS target of {float(target_roas):.1f}. "
            f"{increase_count} campaigns are marked for an "
            f"increase, {reduce_count} for a reduction and "
            f"{review_count} require pause or review. "
            f"{low_confidence_count} campaigns have low confidence "
            f"and {spike_count} campaigns have budget-spike "
            f"warnings, so changes should be implemented gradually "
            f"and monitored. "
            f"The top category is {top_category} "
            f"with a ROAS of {top_category_roas:.2f}."
        )

    return str(
        commentary
    ).strip()
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
    if df.empty:
        return df.copy()

    result_df = df.copy()

    if "CurrentSpend" not in result_df.columns:
        raise KeyError("CurrentSpend column is required.")

    if "ScenarioSpend" in result_df.columns:
        recommended_spend = result_df["ScenarioSpend"]
    elif "RecommendedBudget" in result_df.columns:
        recommended_spend = result_df["RecommendedBudget"]
    else:
        raise KeyError(
            "ScenarioSpend or RecommendedBudget column is required."
        )

    budget_change_ratio = (
        recommended_spend - result_df["CurrentSpend"]
    ) / result_df["CurrentSpend"].replace(0, np.nan)

    result_df["BudgetSpike"] = (
        budget_change_ratio > spike_threshold
    ).fillna(False)

    result_df["BudgetSpikeWarning"] = result_df["BudgetSpike"].apply(
        lambda is_spike: (
            "Warning: Budget increase exceeds 50%. "
            "Monitor closely for the first 7 days."
            if is_spike
            else ""
        )
    )

    return result_df


def build_action_recommendation(
    best_df: pd.DataFrame,
    target_roas: float,
) -> pd.DataFrame:
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

    def decide_action(row):
        current_spend = float(row.get("CurrentSpend", 0) or 0)
        optimal_spend = float(row.get("ScenarioSpend", 0) or 0)
        predicted_roas = float(row.get("PredictedROAS", 0) or 0)
        predicted_conversions = float(
            row.get("PredictedConversions", 0) or 0
        )

        if current_spend <= 0:
            return (
                "Review",
                "No active spend detected. Manual review is required.",
            )

        if (
            predicted_conversions <= 0
            and optimal_spend < current_spend
        ):
            return (
                "Pause / Review",
                "Predicted value remains weak even under lower "
                "spend scenarios.",
            )

        spend_ratio = optimal_spend / current_spend
        roas_ok = predicted_roas >= target_roas * 0.90

        if spend_ratio >= 1.15 and roas_ok:
            return (
                "Increase Budget",
                "Predicted ROAS meets the target. "
                "Scaling is recommended.",
            )

        if spend_ratio >= 1.15 and not roas_ok:
            return (
                "Increase Budget With ROAS Risk",
                "Budget increase potential exists, but predicted "
                "ROAS is below target.",
            )

        if spend_ratio <= 0.85:
            return (
                "Reduce Budget",
                "Predicted return suggests that the campaign "
                "is overfunded.",
            )

        return (
            "Maintain",
            "Predicted performance supports keeping the budget "
            "near the current level.",
        )

    decisions = df.apply(decide_action, axis=1)

    df["RecommendedAction"] = decisions.apply(
        lambda decision: decision[0]
    )

    df["RecommendationReason"] = decisions.apply(
        lambda decision: decision[1]
    )

    df["BudgetChange"] = (
        df["ScenarioSpend"] - df["CurrentSpend"]
    )

    df["BudgetChangePct"] = np.where(
        df["CurrentSpend"] > 0,
        (
            df["BudgetChange"]
            / df["CurrentSpend"]
        )
        * 100,
        0,
    )

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
    if recommendation_df.empty:
        return recommendation_df.copy()

    df = recommendation_df.copy()

    if train_df.empty or "CampaignId" not in train_df.columns:
        df["HistoryRows"] = 0
    else:
        campaign_history = (
            train_df.groupby(
                "CampaignId",
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size": "HistoryRows",
                }
            )
        )

        df = df.merge(
            campaign_history,
            on="CampaignId",
            how="left",
        )

        df["HistoryRows"] = (
            df["HistoryRows"]
            .fillna(0)
            .astype(int)
        )

    conversion_r2_rows = metrics_df.loc[
        metrics_df["Model"] == "Conversions",
        "R2",
    ]

    revenue_r2_rows = metrics_df.loc[
        metrics_df["Model"] == "Revenue",
        "R2",
    ]

    conversion_r2 = (
        float(conversion_r2_rows.iloc[0])
        if not conversion_r2_rows.empty
        else 0.0
    )

    revenue_r2 = (
        float(revenue_r2_rows.iloc[0])
        if not revenue_r2_rows.empty
        else 0.0
    )

    average_r2 = (
        conversion_r2 + revenue_r2
    ) / 2

    def confidence_label(row):
        history_rows = int(row.get("HistoryRows", 0) or 0)

        if history_rows >= 20 and average_r2 >= 0.60:
            return "High"

        if history_rows >= 10 and average_r2 >= 0.30:
            return "Medium"

        return "Low"

    df["ConfidenceLevel"] = df.apply(
        confidence_label,
        axis=1,
    )

    return df


def apply_confidence_guardrail(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if "ConfidenceLevel" not in df.columns:
        raise KeyError("ConfidenceLevel column is required.")

    result_df = df.copy()

    low_confidence_mask = (
        result_df["ConfidenceLevel"] == "Low"
    )

    result_df.loc[
        low_confidence_mask,
        "RecommendedAction",
    ] = "Review"

    result_df.loc[
        low_confidence_mask,
        "RecommendationReason",
    ] = (
        "Low confidence prediction. Manual validation is "
        "recommended before taking action."
    )

    return result_df


def build_portfolio_allocation(
    recommendation_df: pd.DataFrame,
) -> pd.DataFrame:
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

    total_current = float(
        df["CurrentSpend"].sum()
    )

    total_recommended = float(
        df["RecommendedBudget"].sum()
    )

    if total_current <= 0:
        df["OptimizedPortfolioBudget"] = 0.0
        df["PortfolioBudgetChange"] = 0.0
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

    df["PortfolioBudgetChange"] = (
        df["OptimizedPortfolioBudget"]
        - df["CurrentSpend"]
    ).round(2)

    return df


def build_recommendation_summary(
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
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
        "RecommendedAction",
        "ConfidenceLevel",
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
    if ads_raw.empty:
        return pd.DataFrame()

    campaign_df = ads_raw.groupby(
        [
            "CampaignId",
            "Campaign",
            "Channel",
            "Category",
            "ProductGroup",
        ],
        as_index=False,
    ).agg(
        Spend=("Spend", "sum"),
        Clicks=("Clicks", "sum"),
        Impressions=("Impressions", "sum"),
        Conversions=("Conversions", "sum"),
        ConversionValue=("ConversionValue", "sum"),
    )

    campaign_df = compute_kpis(campaign_df)

    campaign_df = compute_roas_target_gap(
        campaign_df,
        target_roas,
    )

    campaign_df = add_campaign_type(
        campaign_df
    )

    def fallback_action(row):
        conversions = float(
            row.get("Conversions", 0) or 0
        )

        spend = float(
            row.get("Spend", 0) or 0
        )

        roas = float(
            row.get("ROAS", 0) or 0
        )

        profit = float(
            row.get("Profit", 0) or 0
        )

        if conversions == 0 and spend > 0:
            return (
                "Pause / Review",
                "No conversions observed in the available history.",
            )

        if (
            roas >= target_roas * 1.10
            and profit > 0
        ):
            return (
                "Maintain / Slight Increase",
                "Strong historical efficiency in fallback mode.",
            )

        if (
            roas < target_roas * 0.90
            or profit < 0
        ):
            return (
                "Reduce Budget",
                "Weak historical efficiency in fallback mode.",
            )

        return (
            "Maintain",
            "Rule-based recommendation.",
        )

    decisions = campaign_df.apply(
        fallback_action,
        axis=1,
    )

    campaign_df["RecommendedAction"] = decisions.apply(
        lambda decision: decision[0]
    )

    campaign_df["RecommendationReason"] = decisions.apply(
        lambda decision: decision[1]
    )

    campaign_df["ConfidenceLevel"] = "Low"
    campaign_df["RecommendedBudget"] = campaign_df["Spend"]
    campaign_df["CurrentSpend"] = campaign_df["Spend"]

    return campaign_df


def build_llm_campaign_prompt(
    row: pd.Series,
    target_roas: float,
) -> str:
    holiday_context = ""

    if row.get("IsHoliday", 0):
        holiday_context = (
            f"Holiday effect active: "
            f"{row.get('HolidayName', '')}."
        )

    elif row.get("IsPreHoliday", 0):
        holiday_context = (
            "Pre-holiday period effect expected."
        )

    return f"""
You are a digital marketing budget analyst.
Write a concise executive-level commentary in English.
Use maximum 3 sentences.

Campaign: {row.get('Campaign', '')}
Campaign type: {row.get('CampaignType', '')}
Category: {row.get('Category', '')}
Product group: {row.get('ProductGroup', '')}
Channel: {row.get('Channel', '')}
Season: {row.get('Season', '')}
{holiday_context}

Current daily spend: {row.get('CurrentSpend', 0):.2f}
Recommended budget: {row.get('RecommendedBudget', 0):.2f}
Budget change percentage: {row.get('BudgetChangePct', 0):.1f}%
Predicted ROAS: {row.get('PredictedROAS', 0):.2f}
Target ROAS: {target_roas:.1f}
Predicted revenue: {row.get('PredictedRevenue', 0):.2f}
Predicted profit: {row.get('PredictedProfit', 0):.2f}
Confidence level: {row.get('ConfidenceLevel', '')}
Recommended action: {row.get('RecommendedAction', '')}
Recommendation reason: {row.get('RecommendationReason', '')}

Executive commentary:
""".strip()


def generate_llm_commentary(
    summary_df: pd.DataFrame,
    target_roas: float,
    max_campaigns: int = 20,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    result_df = summary_df.copy()
    commentaries = []

    rows_to_process = result_df.head(
        max_campaigns
    )

    for _, row in rows_to_process.iterrows():
        prompt = build_llm_campaign_prompt(
            row,
            target_roas,
        )

        commentary = generate_text(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2,
        )

        if commentary is None:
            logger.warning(
                "LLM commentary unavailable for campaign %s. "
                "Using safe fallback commentary.",
                row.get("Campaign", ""),
            )

            commentary = (
                "AI commentary is unavailable. "
                "Use the rule-based recommendation and confidence "
                "level for decision-making."
            )

        commentaries.append(commentary)

    remaining_count = (
        len(result_df)
        - len(rows_to_process)
    )

    commentaries.extend(
        ["Commentary limit reached."]
        * remaining_count
    )

    result_df["ExecutiveCommentary"] = commentaries

    return result_df


def generate_portfolio_summary_commentary(
    portfolio_df: pd.DataFrame,
    category_df: pd.DataFrame,
    target_roas: float,
) -> str:
    if portfolio_df.empty:
        return (
            "Portfolio commentary skipped because "
            "portfolio data is empty."
        )

    total_spend = float(
        portfolio_df.get(
            "CurrentSpend",
            pd.Series(dtype=float),
        ).sum()
    )

    total_recommended = float(
        portfolio_df.get(
            "OptimizedPortfolioBudget",
            pd.Series(dtype=float),
        ).sum()
    )

    total_predicted_revenue = float(
        portfolio_df.get(
            "PredictedRevenue",
            pd.Series(dtype=float),
        ).sum()
    )

    if not category_df.empty:
        top_category = category_df.iloc[0].get(
            "Category",
            "N/A",
        )

        top_category_roas = float(
            category_df.iloc[0].get(
                "ROAS",
                0,
            )
            or 0
        )
    else:
        top_category = "N/A"
        top_category_roas = 0.0

    campaign_count = len(portfolio_df)

    recommended_actions = portfolio_df.get(
        "RecommendedAction",
        pd.Series(dtype=str),
    ).astype(str)

    increase_count = int(
        recommended_actions.str.contains(
            "Increase",
            na=False,
        ).sum()
    )

    reduce_count = int(
        recommended_actions.str.contains(
            "Reduce",
            na=False,
        ).sum()
    )

    spike_count = int(
        portfolio_df.get(
            "BudgetSpike",
            pd.Series(
                False,
                index=portfolio_df.index,
            ),
        ).fillna(False).sum()
    )

    prompt = f"""
You are a digital marketing director.
Write a concise executive-level portfolio summary in English.
Use maximum 5 sentences.

Total campaigns: {campaign_count}
Total current daily spend: {total_spend:.2f}
Total optimized portfolio budget: {total_recommended:.2f}
Campaigns recommended to increase: {increase_count}
Campaigns recommended to reduce: {reduce_count}
Campaigns with budget spike warning: {spike_count}
Predicted total revenue: {total_predicted_revenue:.2f}
ROAS target: {target_roas:.1f}
Top performing category: {top_category}
Top category ROAS: {top_category_roas:.2f}

Portfolio executive summary:
""".strip()

    commentary = generate_text(
        prompt=prompt,
        max_tokens=500,
        temperature=0.2,
    )

    if commentary is None:
        logger.warning(
            "Portfolio LLM commentary is unavailable. "
            "Using safe fallback summary."
        )

        return (
            f"Portfolio contains {campaign_count} campaigns with "
            f"a current daily spend of {total_spend:.2f} and an "
            f"optimized budget of {total_recommended:.2f}. "
            f"{increase_count} campaigns are recommended for an "
            f"increase and {reduce_count} for a reduction. "
            f"Predicted total revenue is "
            f"{total_predicted_revenue:.2f}, against a ROAS target "
            f"of {target_roas:.1f}. "
            f"The top category is {top_category} with a ROAS of "
            f"{top_category_roas:.2f}."
        )

    return commentary


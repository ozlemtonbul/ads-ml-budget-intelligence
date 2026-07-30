from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import TARGET_ROAS
from dashboard.utils import find_first_column


@dataclass(frozen=True)
class ActionSummary:
    """Executive recommendation and risk counts."""

    increase_count: int
    reduce_count: int
    maintain_count: int
    review_count: int
    high_risk_count: int
    insufficient_data_count: int
    high_confidence_count: int
    total_recommendation_count: int

    @property
    def confidence_ratio(self) -> float:
        if self.total_recommendation_count == 0:
            return 0.0

        return (
            self.high_confidence_count
            / self.total_recommendation_count
        )


def _numeric_series(
    dataframe: pd.DataFrame,
    column: str | None,
    default: float = 0.0,
) -> pd.Series:
    """Return a numeric float series safely."""

    if column is None:
        return pd.Series(
            default,
            index=dataframe.index,
            dtype="float64",
        )

    return (
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(float(default))
        .astype("float64")
    )


def _boolean_series(
    dataframe: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    """Return a normalized boolean series."""

    if column is None:
        return pd.Series(
            False,
            index=dataframe.index,
            dtype="bool",
        )

    normalized = (
        dataframe[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "evet",
        }
    )


def _confidence_score(
    dataframe: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    """Convert numeric or categorical confidence to 0–1."""

    if column is None:
        return pd.Series(
            0.0,
            index=dataframe.index,
            dtype="float64",
        )

    numeric_values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    if numeric_values.notna().any():
        return (
            numeric_values
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
            .astype("float64")
        )

    normalized = (
        dataframe[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "high": 0.90,
                "medium": 0.65,
                "low": 0.35,
                "yüksek": 0.90,
                "orta": 0.65,
                "düşük": 0.35,
            }
        )
    )

    return (
        pd.to_numeric(
            normalized,
            errors="coerce",
        )
        .fillna(0.0)
        .astype("float64")
    )


def normalize_action(
    value: object,
) -> str:
    """Return a canonical recommendation action."""

    normalized = str(value).strip().lower()

    if "increase" in normalized or "artır" in normalized:
        return "increase"

    if (
        "reduce" in normalized
        or "decrease" in normalized
        or "azalt" in normalized
    ):
        return "reduce"

    if "maintain" in normalized or "koru" in normalized:
        return "maintain"

    if "review" in normalized or "incele" in normalized:
        return "review"

    return "unknown"


def enrich_recommendations(
    dataframe: pd.DataFrame,
    target_roas: float = TARGET_ROAS,
) -> pd.DataFrame:
    """
    Add canonical fields used by cards, tables and charts.

    Source columns are preserved.
    """

    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    campaign_column = find_first_column(
        result,
        ["Campaign", "CampaignName"],
    )

    campaign_type_column = find_first_column(
        result,
        ["CampaignType", "Type"],
    )

    current_spend_column = find_first_column(
        result,
        [
            "CurrentSpend",
            "CurrentBudget",
            "BaselineBudget",
        ],
    )

    recommended_budget_column = find_first_column(
        result,
        [
            "RecommendedBudget",
            "AllocatedBudget",
        ],
    )

    predicted_roas_column = find_first_column(
        result,
        ["PredictedROAS", "ROAS"],
    )

    action_column = find_first_column(
        result,
        [
            "RecommendedAction",
            "Recommendation",
            "Action",
            "BudgetAction",
        ],
    )

    confidence_column = find_first_column(
        result,
        [
            "ConfidenceLevel",
            "ConfidenceScore",
            "Confidence",
        ],
    )

    history_column = find_first_column(
        result,
        ["HistoryRows", "DataRows"],
    )

    budget_spike_column = find_first_column(
        result,
        ["BudgetSpike"],
    )

    opportunity_column = find_first_column(
        result,
        [
            "OptimizationScore",
            "PredictedProfit",
            "RevenueUpliftPct",
        ],
    )

    result["CampaignCanonical"] = (
        result[campaign_column].astype(str)
        if campaign_column
        else ""
    )

    result["CampaignTypeCanonical"] = (
        result[campaign_type_column].astype(str)
        if campaign_type_column
        else ""
    )

    result["CurrentSpendCanonical"] = (
        _numeric_series(
            result,
            current_spend_column,
        )
    )

    result["RecommendedBudgetCanonical"] = (
        _numeric_series(
            result,
            recommended_budget_column,
        )
    )

    result["PredictedROASCanonical"] = (
        _numeric_series(
            result,
            predicted_roas_column,
        )
    )

    result["HistoryRowsCanonical"] = (
        _numeric_series(
            result,
            history_column,
        )
    )

    result["ConfidenceScoreCanonical"] = (
        _confidence_score(
            result,
            confidence_column,
        )
    )

    result["BudgetSpikeCanonical"] = (
        _boolean_series(
            result,
            budget_spike_column,
        )
    )

    result["ActionCanonical"] = (
        result[action_column].map(
            normalize_action
        )
        if action_column
        else "unknown"
    )

    result["OpportunityScoreCanonical"] = (
        _numeric_series(
            result,
            opportunity_column,
        )
    )

    result["BudgetChangeCanonical"] = (
        result["RecommendedBudgetCanonical"]
        - result["CurrentSpendCanonical"]
    )

    result["BudgetChangePctCanonical"] = (
        result["BudgetChangeCanonical"]
        .div(
            result[
                "CurrentSpendCanonical"
            ].where(
                result[
                    "CurrentSpendCanonical"
                ] > 0
            )
        )
        .mul(100)
        .fillna(0.0)
        .astype("float64")
    )

    active_mask = (
        result["CurrentSpendCanonical"] > 0
    )

    if history_column is not None:
        active_mask &= (
            result["HistoryRowsCanonical"] > 0
        )

    active_mask &= (
        result["PredictedROASCanonical"] > 0
    )

    result["IsActiveCanonical"] = active_mask

    risk_level = pd.Series(
        "medium",
        index=result.index,
        dtype="object",
    )

    risk_level.loc[
        active_mask
        & (
            result["PredictedROASCanonical"]
            >= target_roas
        )
    ] = "low"

    risk_level.loc[
        active_mask
        & (
            result["PredictedROASCanonical"]
            < target_roas * 0.80
        )
    ] = "high"

    risk_level.loc[
        active_mask
        & (
            result[
                "ConfidenceScoreCanonical"
            ] < 0.50
        )
    ] = "high"

    risk_level.loc[
        active_mask
        & result["BudgetSpikeCanonical"]
    ] = "high"

    risk_level.loc[
        ~active_mask
    ] = "insufficient"

    result["RiskLevelCanonical"] = risk_level

    target_gap_ratio = (
        target_roas
        - result["PredictedROASCanonical"]
    ).div(
        target_roas
        if target_roas > 0
        else 1.0
    )

    target_gap_ratio = (
        target_gap_ratio
        .clip(lower=0.0)
        .astype("float64")
    )

    result["RiskScoreCanonical"] = (
        target_gap_ratio * 100
        + result["BudgetSpikeCanonical"].astype(int) * 25
        + (
            1
            - result["ConfidenceScoreCanonical"]
        ) * 20
    )

    result.loc[
        ~active_mask,
        "RiskScoreCanonical",
    ] = 0.0

    return result


def build_action_summary(
    enriched: pd.DataFrame,
) -> ActionSummary:
    """Calculate action, risk and confidence counts."""

    if enriched.empty:
        return ActionSummary(
            increase_count=0,
            reduce_count=0,
            maintain_count=0,
            review_count=0,
            high_risk_count=0,
            insufficient_data_count=0,
            high_confidence_count=0,
            total_recommendation_count=0,
        )

    actions = enriched["ActionCanonical"]
    risks = enriched["RiskLevelCanonical"]
    confidence = enriched[
        "ConfidenceScoreCanonical"
    ]

    return ActionSummary(
        increase_count=int(
            actions.eq("increase").sum()
        ),
        reduce_count=int(
            actions.eq("reduce").sum()
        ),
        maintain_count=int(
            actions.eq("maintain").sum()
        ),
        review_count=int(
            actions.eq("review").sum()
        ),
        high_risk_count=int(
            risks.eq("high").sum()
        ),
        insufficient_data_count=int(
            risks.eq("insufficient").sum()
        ),
        high_confidence_count=int(
            confidence.ge(0.80).sum()
        ),
        total_recommendation_count=len(
            enriched
        ),
    )


def get_top_opportunities(
    enriched: pd.DataFrame,
    limit: int = 3,
) -> pd.DataFrame:
    """Return highest-scoring active opportunities."""

    if enriched.empty:
        return enriched.copy()

    active = enriched.loc[
        enriched["IsActiveCanonical"]
    ].copy()

    return (
        active.sort_values(
            [
                "OpportunityScoreCanonical",
                "PredictedROASCanonical",
            ],
            ascending=[False, False],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def get_top_risks(
    enriched: pd.DataFrame,
    limit: int = 3,
) -> pd.DataFrame:
    """Return highest-risk active campaigns."""

    if enriched.empty:
        return enriched.copy()

    active = enriched.loc[
        enriched["IsActiveCanonical"]
    ].copy()

    return (
        active.sort_values(
            [
                "RiskScoreCanonical",
                "PredictedROASCanonical",
            ],
            ascending=[False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def localize_action(
    value: str,
    language: str,
) -> str:
    """Localize a canonical action value."""

    labels = {
        "tr": {
            "increase": "Bütçeyi Artır",
            "reduce": "Bütçeyi Azalt",
            "maintain": "Bütçeyi Koru",
            "review": "İncele",
            "unknown": "Belirsiz",
        },
        "en": {
            "increase": "Increase Budget",
            "reduce": "Reduce Budget",
            "maintain": "Maintain Budget",
            "review": "Review",
            "unknown": "Unknown",
        },
    }

    return labels.get(
        language,
        labels["en"],
    ).get(
        value,
        value,
    )


def localize_risk(
    value: str,
    language: str,
) -> str:
    """Localize a canonical risk value."""

    labels = {
        "tr": {
            "low": "Düşük",
            "medium": "Orta",
            "high": "Yüksek",
            "insufficient": "Veri Yetersiz",
        },
        "en": {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "insufficient": "Insufficient Data",
        },
    }

    return labels.get(
        language,
        labels["en"],
    ).get(
        value,
        value,
    )


def build_display_table(
    enriched: pd.DataFrame,
    language: str,
) -> pd.DataFrame:
    """
    Build a localized recommendation display table.

    Numeric source values remain numeric so Streamlit
    can format them correctly.
    """

    if enriched.empty:
        return pd.DataFrame()

    campaign_type_labels = {
        "Brand": (
            "Marka"
            if language == "tr"
            else "Brand"
        ),
        "Generic": (
            "Genel"
            if language == "tr"
            else "Generic"
        ),
    }

    display = pd.DataFrame(
        {
            (
                "Kampanya"
                if language == "tr"
                else "Campaign"
            ): enriched[
                "CampaignCanonical"
            ],
            (
                "Kampanya Türü"
                if language == "tr"
                else "Campaign Type"
            ): enriched[
                "CampaignTypeCanonical"
            ].replace(
                campaign_type_labels
            ),
            (
                "Mevcut Harcama"
                if language == "tr"
                else "Current Spend"
            ): enriched[
                "CurrentSpendCanonical"
            ],
            (
                "Önerilen Bütçe"
                if language == "tr"
                else "Recommended Budget"
            ): enriched[
                "RecommendedBudgetCanonical"
            ],
            (
                "Bütçe Değişimi %"
                if language == "tr"
                else "Budget Change %"
            ): enriched[
                "BudgetChangePctCanonical"
            ],
            (
                "Tahmini ROAS"
                if language == "tr"
                else "Predicted ROAS"
            ): enriched[
                "PredictedROASCanonical"
            ],
            (
                "Öneri Güveni"
                if language == "tr"
                else "Recommendation Confidence"
            ): enriched[
                "ConfidenceScoreCanonical"
            ],
            (
                "Önerilen Aksiyon"
                if language == "tr"
                else "Recommended Action"
            ): enriched[
                "ActionCanonical"
            ].map(
                lambda value: localize_action(
                    value,
                    language,
                )
            ),
            (
                "Risk Seviyesi"
                if language == "tr"
                else "Risk Level"
            ): enriched[
                "RiskLevelCanonical"
            ].map(
                lambda value: localize_risk(
                    value,
                    language,
                )
            ),
            (
                "Fırsat Skoru"
                if language == "tr"
                else "Opportunity Score"
            ): enriched[
                "OpportunityScoreCanonical"
            ],
        }
    )

    return display
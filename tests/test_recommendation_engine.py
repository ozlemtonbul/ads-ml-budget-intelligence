import os
from unittest.mock import MagicMock, patch

import pandas as pd

from src.recommendations.recommendation_engine import (
    add_budget_spike_flag,
    apply_confidence_guardrail,
    build_action_recommendation,
    build_confidence_scores,
    build_llm_campaign_prompt,
    build_portfolio_allocation,
    build_recommendation_summary,
    build_rule_based_fallback,
    generate_llm_commentary,
    generate_portfolio_summary_commentary,
)


def test_add_budget_spike_flag_marks_large_increase():
    df = pd.DataFrame(
        {
            "CurrentSpend": [100.0, 100.0],
            "ScenarioSpend": [160.0, 120.0],
        }
    )

    result = add_budget_spike_flag(df)

    assert result["BudgetSpike"].tolist() == [True, False]
    assert result.loc[0, "BudgetSpikeWarning"] != ""
    assert result.loc[1, "BudgetSpikeWarning"] == ""


def test_build_action_recommendation():
    df = pd.DataFrame(
        {
            "CurrentSpend": [100.0, 100.0, 100.0, 0.0],
            "ScenarioSpend": [120.0, 70.0, 100.0, 50.0],
            "PredictedROAS": [4.0, 2.0, 3.0, 2.0],
            "PredictedConversions": [10.0, 5.0, 8.0, 0.0],
        }
    )

    result = build_action_recommendation(
        df,
        target_roas=3.0,
    )

    assert result.loc[0, "RecommendedAction"] == "Increase Budget"
    assert result.loc[1, "RecommendedAction"] == "Reduce Budget"
    assert result.loc[2, "RecommendedAction"] == "Maintain"
    assert result.loc[3, "RecommendedAction"] == "Review"

    assert "RecommendedBudget" in result.columns
    assert "BudgetChange" in result.columns
    assert "BudgetChangePct" in result.columns


def test_build_confidence_scores():
    recommendation_df = pd.DataFrame(
        {
            "CampaignId": [1, 2, 3],
        }
    )

    metrics_df = pd.DataFrame(
        {
            "Model": ["Conversions", "Revenue"],
            "R2": [0.70, 0.80],
        }
    )

    train_df = pd.DataFrame(
        {
            "CampaignId": ([1] * 25) + ([2] * 12) + ([3] * 3),
        }
    )

    result = build_confidence_scores(
        recommendation_df,
        metrics_df,
        train_df,
    )

    assert result.loc[
        result["CampaignId"] == 1,
        "ConfidenceLevel",
    ].iloc[0] == "High"

    assert result.loc[
        result["CampaignId"] == 2,
        "ConfidenceLevel",
    ].iloc[0] == "Medium"

    assert result.loc[
        result["CampaignId"] == 3,
        "ConfidenceLevel",
    ].iloc[0] == "Low"


def test_apply_confidence_guardrail():
    df = pd.DataFrame(
        {
            "ConfidenceLevel": ["High", "Low"],
            "RecommendedAction": [
                "Increase Budget",
                "Reduce Budget",
            ],
            "RecommendationReason": [
                "Strong prediction.",
                "Weak performance.",
            ],
        }
    )

    result = apply_confidence_guardrail(df)

    assert result.loc[0, "RecommendedAction"] == "Increase Budget"
    assert result.loc[1, "RecommendedAction"] == "Review"
    assert "Low confidence prediction" in result.loc[
        1,
        "RecommendationReason",
    ]


def test_build_portfolio_allocation_preserves_total_budget():
    df = pd.DataFrame(
        {
            "CurrentSpend": [100.0, 200.0],
            "RecommendedBudget": [150.0, 150.0],
        }
    )

    result = build_portfolio_allocation(df)

    assert round(
        result["OptimizedPortfolioBudget"].sum(),
        2,
    ) == 300.0

    assert "PortfolioBudgetChange" in result.columns


def test_build_recommendation_summary_returns_available_columns():
    df = pd.DataFrame(
        {
            "CampaignId": [1],
            "Campaign": ["Brand Search"],
            "CurrentSpend": [100.0],
            "RecommendedBudget": [120.0],
            "PredictedROAS": [4.0],
            "ConfidenceLevel": ["High"],
            "ExtraColumn": ["ignore"],
        }
    )

    result = build_recommendation_summary(df)

    assert "CampaignId" in result.columns
    assert "Campaign" in result.columns
    assert "CurrentSpend" in result.columns
    assert "RecommendedBudget" in result.columns
    assert "PredictedROAS" in result.columns
    assert "ExtraColumn" not in result.columns


def test_build_rule_based_fallback():
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "Campaign": [
                "Brand Campaign",
                "Generic Campaign",
            ],
            "Channel": ["SEARCH", "SEARCH"],
            "Category": ["Shoes", "Shoes"],
            "ProductGroup": ["Kids", "Adult"],
            "Spend": [100.0, 100.0],
            "Clicks": [100, 100],
            "Impressions": [1000, 1000],
            "Conversions": [10.0, 0.0],
            "ConversionValue": [500.0, 0.0],
        }
    )

    result = build_rule_based_fallback(
        ads_raw,
        target_roas=3.0,
    )

    assert len(result) == 2
    assert "RecommendedAction" in result.columns
    assert "RecommendationReason" in result.columns
    assert "ConfidenceLevel" in result.columns
    assert "CampaignType" in result.columns

    pause_action = result.loc[
        result["CampaignId"] == 2,
        "RecommendedAction",
    ].iloc[0]

    assert pause_action == "Pause / Review"


def test_build_llm_campaign_prompt():
    row = pd.Series(
        {
            "Campaign": "Brand Search",
            "CampaignType": "Brand",
            "Category": "Shoes",
            "ProductGroup": "Kids",
            "Channel": "SEARCH",
            "Season": "Winter",
            "CurrentSpend": 100.0,
            "RecommendedBudget": 120.0,
            "BudgetChangePct": 20.0,
            "PredictedROAS": 4.0,
            "PredictedRevenue": 500.0,
            "PredictedProfit": 400.0,
            "ConfidenceLevel": "High",
            "RecommendedAction": "Increase Budget",
            "RecommendationReason": "Strong forecast.",
            "IsHoliday": 1,
            "HolidayName": "New Year",
        }
    )

    prompt = build_llm_campaign_prompt(
        row,
        target_roas=3.0,
    )

    assert "Brand Search" in prompt
    assert "Increase Budget" in prompt
    assert "New Year" in prompt
    assert "Target ROAS: 3.0" in prompt


def test_generate_llm_commentary_without_api_key():
    df = pd.DataFrame(
        {
            "Campaign": ["Brand Search"],
        }
    )

    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": ""},
        clear=False,
    ):
        result = generate_llm_commentary(
            df,
            target_roas=3.0,
        )

    assert "ExecutiveCommentary" in result.columns
    assert (
        result.loc[0, "ExecutiveCommentary"]
        == "LLM commentary skipped because API key is missing."
    )


@patch(
    "src.recommendations.recommendation_engine.anthropic.Anthropic"
)
def test_generate_llm_commentary_with_mock(
    mock_anthropic,
):
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    mock_message = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Increase budget cautiously."
    mock_message.content = [mock_content]

    mock_client.messages.create.return_value = mock_message

    df = pd.DataFrame(
        {
            "Campaign": ["Brand Search"],
            "CampaignType": ["Brand"],
            "Category": ["Shoes"],
            "ProductGroup": ["Kids"],
            "Channel": ["SEARCH"],
            "Season": ["Winter"],
            "CurrentSpend": [100.0],
            "RecommendedBudget": [120.0],
            "BudgetChangePct": [20.0],
            "PredictedROAS": [4.0],
            "PredictedRevenue": [500.0],
            "PredictedProfit": [400.0],
            "ConfidenceLevel": ["High"],
            "RecommendedAction": ["Increase Budget"],
            "RecommendationReason": ["Strong forecast."],
        }
    )

    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-key"},
        clear=False,
    ):
        result = generate_llm_commentary(
            df,
            target_roas=3.0,
        )

    assert (
        result.loc[0, "ExecutiveCommentary"]
        == "Increase budget cautiously."
    )

    mock_client.messages.create.assert_called_once()


def test_generate_portfolio_summary_without_api_key():
    portfolio_df = pd.DataFrame(
        {
            "CurrentSpend": [100.0],
        }
    )

    category_df = pd.DataFrame(
        {
            "Category": ["Shoes"],
            "ROAS": [4.0],
        }
    )

    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": ""},
        clear=False,
    ):
        result = generate_portfolio_summary_commentary(
            portfolio_df,
            category_df,
            target_roas=3.0,
        )

    assert "API key is missing" in result
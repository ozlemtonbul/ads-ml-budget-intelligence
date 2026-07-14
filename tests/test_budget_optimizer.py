import pandas as pd

from src.models.budget_optimizer import (
    add_campaign_type,
    classify_campaign_type,
    choose_optimal_scenario,
)


def test_classify_campaign_type():
    assert classify_campaign_type("Brand Search Campaign") == "Brand"
    assert classify_campaign_type("Shopping Shoes") == "Shopping"
    assert classify_campaign_type("Performance Max Kids") == "Performance Max"
    assert classify_campaign_type("Generic Search") == "Generic"


def test_add_campaign_type():
    df = pd.DataFrame(
        {
            "Campaign": [
                "Brand Search Campaign",
                "Shopping Shoes",
                "Performance Max Kids",
                "Generic Search",
            ]
        }
    )

    result = add_campaign_type(df)

    assert result["CampaignType"].tolist() == [
        "Brand",
        "Shopping",
        "Performance Max",
        "Generic",
    ]


def test_choose_optimal_scenario():
    df = pd.DataFrame(
        {
            "CampaignId": [1, 1, 2, 2],
            "PredictedRevenue": [1000.0, 1200.0, 800.0, 700.0],
            "PredictedProfit": [500.0, 450.0, 300.0, 350.0],
            "PredictedROAS": [5.0, 4.0, 3.0, 4.0],
            "ScenarioFactor": [1.0, 1.2, 1.0, 0.8],
        }
    )

    result = choose_optimal_scenario(df)

    assert len(result) == 2
    assert set(result["CampaignId"]) == {1, 2}
    assert "OptimizationScore" in result.columns
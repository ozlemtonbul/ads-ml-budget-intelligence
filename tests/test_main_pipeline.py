from unittest.mock import MagicMock

import pandas as pd
import pytest

import main as pipeline


# ============================================================
# DATE RANGE TESTS
# ============================================================

def test_resolve_date_range_custom(monkeypatch):
    monkeypatch.setattr(pipeline, "DATE_MODE", "custom")
    monkeypatch.setattr(pipeline, "DATE_FROM", "2026-01-01")
    monkeypatch.setattr(pipeline, "DATE_TO", "2026-01-31")

    result = pipeline.resolve_date_range()

    assert result == ("2026-01-01", "2026-01-31")


def test_resolve_date_range_custom_requires_dates(monkeypatch):
    monkeypatch.setattr(pipeline, "DATE_MODE", "custom")
    monkeypatch.setattr(pipeline, "DATE_FROM", "")
    monkeypatch.setattr(pipeline, "DATE_TO", "")

    with pytest.raises(
        ValueError,
        match="DATE_FROM and DATE_TO are required",
    ):
        pipeline.resolve_date_range()


def test_resolve_date_range_rejects_invalid_format(monkeypatch):
    monkeypatch.setattr(pipeline, "DATE_MODE", "custom")
    monkeypatch.setattr(pipeline, "DATE_FROM", "01-01-2026")
    monkeypatch.setattr(pipeline, "DATE_TO", "31-01-2026")

    with pytest.raises(
        ValueError,
        match="YYYY-MM-DD",
    ):
        pipeline.resolve_date_range()


def test_resolve_date_range_rejects_reversed_dates(monkeypatch):
    monkeypatch.setattr(pipeline, "DATE_MODE", "custom")
    monkeypatch.setattr(pipeline, "DATE_FROM", "2026-02-01")
    monkeypatch.setattr(pipeline, "DATE_TO", "2026-01-01")

    with pytest.raises(
        ValueError,
        match="DATE_FROM cannot be later than DATE_TO",
    ):
        pipeline.resolve_date_range()


def test_resolve_date_range_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "DATE_MODE", "invalid_mode")

    with pytest.raises(
        ValueError,
        match="DATE_MODE must be one of",
    ):
        pipeline.resolve_date_range()


# ============================================================
# CSV EXPORT TEST
# ============================================================

def test_export_csv_writes_only_valid_dataframe(tmp_path):
    outputs = {
        "valid_table": pd.DataFrame(
            {
                "CampaignId": [1, 2],
                "Spend": [100.0, 200.0],
            }
        ),
        "empty_table": pd.DataFrame(),
        "none_table": None,
        "invalid_table": "not-a-dataframe",
    }

    pipeline.export_csv(
        outputs=outputs,
        output_dir=str(tmp_path),
    )

    valid_path = tmp_path / "valid_table.csv"

    assert valid_path.exists()
    assert not (tmp_path / "empty_table.csv").exists()
    assert not (tmp_path / "none_table.csv").exists()
    assert not (tmp_path / "invalid_table.csv").exists()

    exported_df = pd.read_csv(valid_path)

    assert len(exported_df) == 2
    assert exported_df["CampaignId"].tolist() == [1, 2]


# ============================================================
# GA4 LOADING TESTS
# ============================================================

def test_load_ga4_data_returns_dataframe(monkeypatch):
    expected_df = pd.DataFrame(
        {
            "Campaign": ["Brand Campaign"],
            "Sessions": [100.0],
        }
    )

    mock_extractor = MagicMock()
    mock_extractor.fetch_campaign_performance.return_value = expected_df

    mock_extractor_class = MagicMock(
        return_value=mock_extractor
    )

    monkeypatch.setattr(
        pipeline,
        "GA4Extractor",
        mock_extractor_class,
    )

    result = pipeline.load_ga4_data(
        "2026-01-01",
        "2026-01-31",
    )

    pd.testing.assert_frame_equal(
        result,
        expected_df,
    )

    mock_extractor.fetch_campaign_performance.assert_called_once_with(
        "2026-01-01",
        "2026-01-31",
    )


def test_load_ga4_data_returns_empty_on_error(monkeypatch):
    mock_extractor_class = MagicMock(
        side_effect=RuntimeError("GA4 unavailable")
    )

    monkeypatch.setattr(
        pipeline,
        "GA4Extractor",
        mock_extractor_class,
    )

    result = pipeline.load_ga4_data(
        "2026-01-01",
        "2026-01-31",
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ============================================================
# EMPTY ADS PIPELINE TEST
# ============================================================

def test_main_stops_when_ads_data_is_empty(monkeypatch):
    mock_export_csv = MagicMock()
    mock_postgres_export = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "resolve_date_range",
        MagicMock(
            return_value=("2026-01-01", "2026-01-31")
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "build_holiday_map",
        MagicMock(return_value={}),
    )

    monkeypatch.setattr(
        pipeline,
        "fetch_ads_purchase_only",
        MagicMock(return_value=pd.DataFrame()),
    )

    monkeypatch.setattr(
        pipeline,
        "export_csv",
        mock_export_csv,
    )

    monkeypatch.setattr(
        pipeline,
        "write_outputs_to_postgres",
        mock_postgres_export,
    )

    result = pipeline.main()

    assert result is None
    mock_export_csv.assert_not_called()
    mock_postgres_export.assert_not_called()


# ============================================================
# FALLBACK PIPELINE TEST
# ============================================================

def test_main_runs_rule_based_fallback_pipeline(monkeypatch):
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "Campaign": [
                "Brand Campaign",
                "Generic Campaign",
            ],
            "Channel": ["SEARCH", "SEARCH"],
            "Spend": [100.0, 200.0],
        }
    )

    ads_typed = ads_raw.copy()
    ads_typed["CampaignType"] = ["Brand", "Generic"]

    ga4_df = pd.DataFrame(
        {
            "Campaign": ["Brand Campaign"],
            "Sessions": [100.0],
        }
    )

    fallback_df = pd.DataFrame(
        {
            "CampaignId": [1],
            "RecommendedAction": ["Maintain"],
        }
    )

    fallback_with_commentary = fallback_df.copy()
    fallback_with_commentary["ExecutiveCommentary"] = [
        "Maintain current investment."
    ]

    category_df = pd.DataFrame(
        {
            "Category": ["Shoes"],
            "ROAS": [4.0],
        }
    )

    product_df = pd.DataFrame(
        {
            "ProductGroup": ["Kids"],
        }
    )

    holiday_df = pd.DataFrame(
        {
            "PeriodLabel": ["Normal Day"],
        }
    )

    daily_df = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
        }
    )

    weekly_df = pd.DataFrame(
        {
            "Week": ["2025-12-29/2026-01-04"],
        }
    )

    monthly_df = pd.DataFrame(
        {
            "Month": ["2026-01"],
        }
    )

    zero_df = pd.DataFrame(
        {
            "CampaignId": [2],
        }
    )

    short_train_df = pd.DataFrame(
        {
            "CampaignId": [1, 1, 2],
        }
    )

    mock_export_csv = MagicMock()
    mock_get_engine = MagicMock(return_value="mock-engine")
    mock_postgres_export = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "resolve_date_range",
        MagicMock(
            return_value=("2026-01-01", "2026-01-31")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_holiday_map",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_ads_purchase_only",
        MagicMock(return_value=ads_raw),
    )
    monkeypatch.setattr(
        pipeline,
        "load_ga4_data",
        MagicMock(return_value=ga4_df),
    )
    monkeypatch.setattr(
        pipeline,
        "add_campaign_type",
        MagicMock(return_value=ads_typed),
    )
    monkeypatch.setattr(
        pipeline,
        "build_zero_activity_report",
        MagicMock(return_value=zero_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_category_summary",
        MagicMock(return_value=category_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_product_summary",
        MagicMock(return_value=product_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_holiday_impact_summary",
        MagicMock(return_value=holiday_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_daily_weekly_monthly_outputs",
        MagicMock(
            return_value=(
                daily_df,
                weekly_df,
                monthly_df,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_training_data",
        MagicMock(return_value=short_train_df),
    )

    mock_fallback = MagicMock(
        return_value=fallback_df
    )
    mock_llm = MagicMock(
        return_value=fallback_with_commentary
    )

    monkeypatch.setattr(
        pipeline,
        "build_rule_based_fallback",
        mock_fallback,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_llm_commentary",
        mock_llm,
    )
    monkeypatch.setattr(
        pipeline,
        "export_csv",
        mock_export_csv,
    )
    monkeypatch.setattr(
        pipeline,
        "get_postgres_engine",
        mock_get_engine,
    )
    monkeypatch.setattr(
        pipeline,
        "write_outputs_to_postgres",
        mock_postgres_export,
    )

    pipeline.main()

    mock_fallback.assert_called_once_with(
        ads_typed,
        pipeline.TARGET_ROAS,
    )

    mock_llm.assert_called_once()

    exported_outputs = mock_export_csv.call_args.args[0]

    assert (
        "ads_rule_based_fallback_recommendations"
        in exported_outputs
    )
    assert "ga4_campaign_performance" in exported_outputs
    assert "ads_category_summary" in exported_outputs
    assert "ads_daily_fact" in exported_outputs

    mock_export_csv.assert_called_once()
    mock_get_engine.assert_called_once()

    mock_postgres_export.assert_called_once_with(
        exported_outputs,
        "mock-engine",
        if_exists=pipeline.POSTGRES_IF_EXISTS,
    )


# ============================================================
# ML PIPELINE TEST
# ============================================================

def test_main_runs_machine_learning_pipeline(
    monkeypatch,
    tmp_path,
):
    ads_raw = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "Campaign": [
                "Brand Campaign",
                "Shopping Campaign",
            ],
            "Channel": ["SEARCH", "SHOPPING"],
            "Spend": [100.0, 200.0],
        }
    )

    ads_typed = ads_raw.copy()
    ads_typed["CampaignType"] = [
        "Brand",
        "Shopping",
    ]

    train_df = pd.DataFrame(
        {
            "CampaignId": [1] * 10 + [2] * 10,
            "Feature": list(range(20)),
        }
    )

    latest_df = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "Campaign": [
                "Brand Campaign",
                "Shopping Campaign",
            ],
            "Channel": ["SEARCH", "SHOPPING"],
            "Spend": [100.0, 200.0],
        }
    )

    metrics_df = pd.DataFrame(
        {
            "Model": ["Conversions", "Revenue"],
            "R2": [0.70, 0.75],
        }
    )

    feature_importance_df = pd.DataFrame(
        {
            "Feature": ["Spend"],
            "Importance": [1.0],
        }
    )

    sim_df = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "ScenarioFactor": [1.0, 1.0],
        }
    )

    best_df = pd.DataFrame(
        {
            "CampaignId": [1, 2],
        }
    )

    uplift_df = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "CurrentSpend": [100.0, 200.0],
            "ScenarioSpend": [120.0, 180.0],
            "PredictedROAS": [4.0, 3.5],
            "PredictedConversions": [10.0, 15.0],
        }
    )

    recommendation_df = pd.DataFrame(
        {
            "CampaignId": [1, 2],
            "CurrentSpend": [100.0, 200.0],
            "RecommendedBudget": [120.0, 180.0],
            "PredictedROAS": [4.0, 3.5],
            "RecommendedAction": [
                "Increase Budget",
                "Maintain",
            ],
        }
    )

    confidence_df = recommendation_df.copy()
    confidence_df["ConfidenceLevel"] = [
        "High",
        "Medium",
    ]

    guarded_df = confidence_df.copy()

    roas_gap_df = guarded_df.copy()
    roas_gap_df["ROAS"] = roas_gap_df[
        "PredictedROAS"
    ]
    roas_gap_df["TargetROAS"] = pipeline.TARGET_ROAS

    spike_df = roas_gap_df.copy()
    spike_df["BudgetSpike"] = [False, False]

    portfolio_df = spike_df.copy()
    portfolio_df["OptimizedPortfolioBudget"] = [
        110.0,
        190.0,
    ]

    summary_df = portfolio_df.copy()

    summary_with_llm = summary_df.copy()
    summary_with_llm["ExecutiveCommentary"] = [
        "Scale cautiously.",
        "Maintain budget.",
    ]

    category_df = pd.DataFrame(
        {
            "Category": ["Shoes"],
            "ROAS": [4.0],
        }
    )

    empty_or_simple_df = pd.DataFrame(
        {
            "Value": [1],
        }
    )

    mock_model_conv = MagicMock()
    mock_model_rev = MagicMock()

    mock_export_csv = MagicMock()
    mock_postgres_export = MagicMock()

    monkeypatch.setattr(
        pipeline,
        "OUTPUT_DIR",
        str(tmp_path),
    )

    monkeypatch.setattr(
        pipeline,
        "resolve_date_range",
        MagicMock(
            return_value=("2026-01-01", "2026-01-31")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_holiday_map",
        MagicMock(return_value={}),
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_ads_purchase_only",
        MagicMock(return_value=ads_raw),
    )
    monkeypatch.setattr(
        pipeline,
        "load_ga4_data",
        MagicMock(return_value=pd.DataFrame()),
    )
    monkeypatch.setattr(
        pipeline,
        "add_campaign_type",
        MagicMock(return_value=ads_typed),
    )
    monkeypatch.setattr(
        pipeline,
        "build_zero_activity_report",
        MagicMock(return_value=pd.DataFrame()),
    )
    monkeypatch.setattr(
        pipeline,
        "build_category_summary",
        MagicMock(return_value=category_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_product_summary",
        MagicMock(return_value=empty_or_simple_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_holiday_impact_summary",
        MagicMock(return_value=empty_or_simple_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_daily_weekly_monthly_outputs",
        MagicMock(
            return_value=(
                empty_or_simple_df,
                empty_or_simple_df,
                empty_or_simple_df,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_training_data",
        MagicMock(return_value=train_df),
    )
    monkeypatch.setattr(
        pipeline,
        "train_and_validate_models",
        MagicMock(
            return_value=(
                mock_model_conv,
                mock_model_rev,
                ["Spend"],
                metrics_df,
                feature_importance_df,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "get_latest_campaign_state",
        MagicMock(return_value=latest_df),
    )
    monkeypatch.setattr(
        pipeline,
        "simulate_budget_scenarios",
        MagicMock(return_value=sim_df),
    )
    monkeypatch.setattr(
        pipeline,
        "choose_optimal_scenario",
        MagicMock(return_value=best_df),
    )
    monkeypatch.setattr(
        pipeline,
        "add_baseline_uplift",
        MagicMock(return_value=uplift_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_action_recommendation",
        MagicMock(return_value=recommendation_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_confidence_scores",
        MagicMock(return_value=confidence_df),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_confidence_guardrail",
        MagicMock(return_value=guarded_df),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_roas_target_gap",
        MagicMock(return_value=roas_gap_df),
    )
    monkeypatch.setattr(
        pipeline,
        "add_budget_spike_flag",
        MagicMock(return_value=spike_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_portfolio_allocation",
        MagicMock(return_value=portfolio_df),
    )
    monkeypatch.setattr(
        pipeline,
        "build_recommendation_summary",
        MagicMock(return_value=summary_df),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_llm_commentary",
        MagicMock(return_value=summary_with_llm),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_portfolio_summary_commentary",
        MagicMock(
            return_value="Portfolio performance is stable."
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "export_csv",
        mock_export_csv,
    )
    monkeypatch.setattr(
        pipeline,
        "get_postgres_engine",
        MagicMock(return_value="mock-engine"),
    )
    monkeypatch.setattr(
        pipeline,
        "write_outputs_to_postgres",
        mock_postgres_export,
    )

    pipeline.main()

    commentary_path = (
        tmp_path
        / "ads_portfolio_executive_commentary.txt"
    )

    assert commentary_path.exists()
    assert commentary_path.read_text(
        encoding="utf-8"
    ) == "Portfolio performance is stable."

    exported_outputs = mock_export_csv.call_args.args[0]

    assert "ads_budget_scenarios" in exported_outputs
    assert (
        "ads_budget_optimization_recommendations"
        in exported_outputs
    )
    assert (
        "ads_portfolio_budget_allocation"
        in exported_outputs
    )
    assert "ads_recommendation_summary" in exported_outputs
    assert "ads_model_validation_metrics" in exported_outputs
    assert "ads_feature_importance" in exported_outputs

    mock_export_csv.assert_called_once()

    mock_postgres_export.assert_called_once_with(
        exported_outputs,
        "mock-engine",
        if_exists=pipeline.POSTGRES_IF_EXISTS,
    )
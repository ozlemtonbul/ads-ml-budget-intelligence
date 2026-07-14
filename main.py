import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd

from config.settings import (
    DATE_FROM,
    DATE_MODE,
    DATE_TO,
    LLM_MAX_CAMPAIGNS,
    OUTPUT_DIR,
    POSTGRES_IF_EXISTS,
    TARGET_ROAS,
)
from src.extract.ads_extractor import fetch_ads_purchase_only
from src.extract.ga4_extractor import GA4Extractor
from src.features.feature_engineering import (
    build_holiday_map,
    compute_roas_target_gap,
    get_latest_campaign_state,
    prepare_training_data,
)
from src.features.reporting import (
    build_category_summary,
    build_daily_weekly_monthly_outputs,
    build_holiday_impact_summary,
    build_product_summary,
    build_zero_activity_report,
)
from src.models.budget_optimizer import (
    add_baseline_uplift,
    add_campaign_type,
    choose_optimal_scenario,
    simulate_budget_scenarios,
    train_and_validate_models,
)
from src.recommendations.recommendation_engine import (
    add_budget_spike_flag,
    apply_confidence_guardrail,
    build_action_recommendation,
    build_confidence_scores,
    build_portfolio_allocation,
    build_recommendation_summary,
    build_rule_based_fallback,
    generate_llm_commentary,
    generate_portfolio_summary_commentary,
)
from src.utils.logger import get_logger
from src.warehouse.postgres_manager import (
    get_postgres_engine,
    write_outputs_to_postgres,
)


logger = get_logger(__name__)


def resolve_date_range() -> Tuple[str, str]:
    mode = DATE_MODE.lower().strip()

    if mode == "yesterday":
        date_value = datetime.now().date() - timedelta(days=1)
        formatted_date = date_value.strftime("%Y-%m-%d")
        return formatted_date, formatted_date

    if mode == "last_30_days":
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=29)

        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if mode == "last_60_days":
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=59)

        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

    if mode == "custom":
        if not DATE_FROM or not DATE_TO:
            raise ValueError(
                "DATE_FROM and DATE_TO are required when DATE_MODE=custom."
            )

        try:
            start_date = datetime.strptime(
                DATE_FROM,
                "%Y-%m-%d",
            ).date()

            end_date = datetime.strptime(
                DATE_TO,
                "%Y-%m-%d",
            ).date()

        except ValueError as exc:
            raise ValueError(
                "DATE_FROM and DATE_TO must use YYYY-MM-DD format."
            ) from exc

        if start_date > end_date:
            raise ValueError(
                "DATE_FROM cannot be later than DATE_TO."
            )

        return DATE_FROM, DATE_TO

    raise ValueError(
        "DATE_MODE must be one of: "
        "yesterday, last_30_days, last_60_days, custom"
    )


def export_csv(
    outputs: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    for table_name, dataframe in outputs.items():
        if dataframe is None:
            logger.warning(
                "CSV export skipped for '%s' because dataframe is None.",
                table_name,
            )
            continue

        if not isinstance(dataframe, pd.DataFrame):
            logger.warning(
                "CSV export skipped for '%s' because value is not a dataframe.",
                table_name,
            )
            continue

        if dataframe.empty:
            logger.warning(
                "CSV export skipped for '%s' because dataframe is empty.",
                table_name,
            )
            continue

        csv_path = os.path.join(
            output_dir,
            f"{table_name}.csv",
        )

        dataframe.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "CSV written: %s | Rows: %d",
            csv_path,
            len(dataframe),
        )


def load_ga4_data(
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    try:
        extractor = GA4Extractor()

        ga4_raw = extractor.fetch_campaign_performance(
            date_from,
            date_to,
        )

        if ga4_raw.empty:
            logger.warning(
                "GA4 returned no data for the selected date range."
            )
        else:
            logger.info(
                "GA4 data loaded successfully. Rows: %d",
                len(ga4_raw),
            )

        return ga4_raw

    except Exception as exc:
        logger.warning(
            "GA4 could not be loaded: %s",
            exc,
        )

        return pd.DataFrame()


def main() -> None:
    logger.info(
        "Ads Budget Intelligence pipeline started."
    )

    date_from, date_to = resolve_date_range()

    logger.info(
        "Date range: %s to %s",
        date_from,
        date_to,
    )

    logger.info(
        "Target ROAS: %.2f",
        TARGET_ROAS,
    )

    holiday_map = build_holiday_map(
        date_from,
        date_to,
    )

    logger.info(
        "Holiday map created. Holiday count: %d",
        len(holiday_map),
    )

    ads_raw = fetch_ads_purchase_only(
        date_from,
        date_to,
    )

    if ads_raw.empty:
        logger.warning(
            "Google Ads API returned no data. Pipeline stopped."
        )
        return

    logger.info(
        "Google Ads data loaded successfully. Rows: %d",
        len(ads_raw),
    )

    ga4_raw = load_ga4_data(
        date_from,
        date_to,
    )

    ads_raw = add_campaign_type(
        ads_raw
    )

    zero_activity_df = build_zero_activity_report(
        ads_raw
    )

    category_df = build_category_summary(
        ads_raw,
        holiday_map,
    )

    product_df = build_product_summary(
        ads_raw,
        holiday_map,
    )

    holiday_impact_df = build_holiday_impact_summary(
        ads_raw,
        holiday_map,
    )

    (
        daily_df,
        weekly_df,
        monthly_df,
    ) = build_daily_weekly_monthly_outputs(
        ads_raw,
        holiday_map,
    )

    train_df = prepare_training_data(
        ads_raw,
        holiday_map,
    )

    common_outputs = {
        "ads_category_summary": category_df,
        "ads_product_summary": product_df,
        "ads_holiday_impact": holiday_impact_df,
        "ads_daily_fact": daily_df,
        "ads_weekly_campaign_summary": weekly_df,
        "ads_monthly_campaign_summary": monthly_df,
        "ads_zero_activity_campaigns": zero_activity_df,
    }

    if not ga4_raw.empty:
        common_outputs["ga4_campaign_performance"] = ga4_raw

    if train_df.empty or len(train_df) < 20:
        logger.warning(
            "Not enough data for ML. "
            "Rule-based fallback mode will be used."
        )

        fallback_df = build_rule_based_fallback(
            ads_raw,
            TARGET_ROAS,
        )

        fallback_df = generate_llm_commentary(
            fallback_df,
            TARGET_ROAS,
            max_campaigns=LLM_MAX_CAMPAIGNS,
        )

        outputs = {
            "ads_rule_based_fallback_recommendations": fallback_df,
            **common_outputs,
        }

    else:
        logger.info(
            "ML training started. Training rows: %d",
            len(train_df),
        )

        (
            model_conv,
            model_rev,
            feature_cols,
            metrics_df,
            feature_importance_df,
        ) = train_and_validate_models(
            train_df
        )

        latest_df = get_latest_campaign_state(
            ads_raw,
            holiday_map,
        )

        if latest_df.empty:
            raise ValueError(
                "Latest campaign state could not be created."
            )

        type_map = (
            ads_raw[
                [
                    "CampaignId",
                    "CampaignType",
                ]
            ]
            .drop_duplicates(
                subset=["CampaignId"]
            )
        )

        if "CampaignType" in latest_df.columns:
            latest_df = latest_df.drop(
                columns=["CampaignType"]
            )

        latest_df = latest_df.merge(
            type_map,
            on="CampaignId",
            how="left",
        )

        latest_df["CampaignType"] = (
            latest_df["CampaignType"]
            .fillna("Generic")
        )

        sim_df = simulate_budget_scenarios(
            latest_df,
            model_conv,
            model_rev,
            feature_cols,
        )

        if sim_df.empty:
            raise ValueError(
                "Budget scenario simulation returned no data."
            )

        best_df = choose_optimal_scenario(
            sim_df
        )

        best_df = add_baseline_uplift(
            best_df,
            sim_df,
        )

        recommendation_df = build_action_recommendation(
            best_df,
            TARGET_ROAS,
        )

        recommendation_df = build_confidence_scores(
            recommendation_df,
            metrics_df,
            train_df,
        )

        recommendation_df = apply_confidence_guardrail(
            recommendation_df
        )

        recommendation_df["ROAS"] = (
            recommendation_df["PredictedROAS"]
        )

        recommendation_df = compute_roas_target_gap(
            recommendation_df,
            TARGET_ROAS,
        )

        recommendation_df = add_budget_spike_flag(
            recommendation_df
        )

        portfolio_df = build_portfolio_allocation(
            recommendation_df
        )

        summary_df = build_recommendation_summary(
            portfolio_df
        )

        summary_df = generate_llm_commentary(
            summary_df,
            TARGET_ROAS,
            max_campaigns=LLM_MAX_CAMPAIGNS,
        )

        portfolio_commentary = (
            generate_portfolio_summary_commentary(
                portfolio_df,
                category_df,
                TARGET_ROAS,
            )
        )

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        commentary_path = os.path.join(
            OUTPUT_DIR,
            "ads_portfolio_executive_commentary.txt",
        )

        with open(
            commentary_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                portfolio_commentary
            )

        logger.info(
            "Portfolio commentary written: %s",
            commentary_path,
        )

        outputs = {
            "ads_budget_scenarios": sim_df,
            "ads_budget_optimization_recommendations": recommendation_df,
            "ads_portfolio_budget_allocation": portfolio_df,
            "ads_recommendation_summary": summary_df,
            "ads_model_validation_metrics": metrics_df,
            "ads_feature_importance": feature_importance_df,
            **common_outputs,
        }

    export_csv(
        outputs,
        OUTPUT_DIR,
    )

    postgres_engine = get_postgres_engine()

    write_outputs_to_postgres(
        outputs,
        postgres_engine,
        if_exists=POSTGRES_IF_EXISTS,
    )

    logger.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        logger.exception(
            "Pipeline failed because of an unexpected error."
        )
        raise